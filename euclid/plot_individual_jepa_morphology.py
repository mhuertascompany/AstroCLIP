"""Compare morphology of one SFH's JEPA neighbours with physical controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PARAMETERS = (
    ("zoobot_smooth_probability", "ZooBot P(smooth)"),
    ("zoobot_featured_probability", "ZooBot P(featured)"),
    ("zoobot_spiral_probability", "ZooBot P(spiral arms)"),
    ("zoobot_bar_probability", "ZooBot P(bar)"),
    ("zoobot_merger_probability", "ZooBot P(merger)"),
    ("zoobot_edge_on_probability", "ZooBot P(edge-on)"),
    ("sersic_index", "Sersic index n"),
    ("axis_ratio", "Axis ratio b/a"),
    ("concentration", "Concentration"),
    ("asymmetry", "Asymmetry"),
    ("gini", "Gini coefficient"),
    ("sersic_radius", "Sersic radius [arcsec]"),
)


def _align_properties(path: Path, galaxy_ids: np.ndarray):
    with np.load(path) as archive:
        source_ids = np.asarray(archive["galaxy_id"], dtype=np.int64)
        position = {int(value): index for index, value in enumerate(source_ids)}
        indices = np.array(
            [position.get(int(value), -1) for value in galaxy_ids], dtype=np.int64,
        )
        matched = indices >= 0
        output = {}
        for name in archive.files:
            values = np.asarray(archive[name])
            if (
                name != "galaxy_id" and values.ndim == 1
                and len(values) == len(source_ids)
                and values.dtype.kind in "biuf"
            ):
                selected = np.full(len(galaxy_ids), np.nan, dtype=np.float64)
                selected[matched] = values[indices[matched]]
                output[name] = selected
    return output, matched


def _log_sfr100(properties: dict[str, np.ndarray]) -> np.ndarray:
    required = {"log_stellar_mass", "sfh_log_sfr_per_stellar_mass_100myr_r0"}
    if not required.issubset(properties):
        raise ValueError(
            "Properties require log_stellar_mass and "
            "sfh_log_sfr_per_stellar_mass_100myr_r0"
        )
    return (
        np.asarray(properties["log_stellar_mass"], dtype=float)
        + np.asarray(
            properties["sfh_log_sfr_per_stellar_mass_100myr_r0"], dtype=float,
        )
    )


def _matched_control_indices(
    query: int, retrieved: np.ndarray, mass: np.ndarray, redshift: np.ndarray,
    log_sfr: np.ndarray, count: int,
    scales=(0.1, 0.05, 0.2), calipers=(0.15, 0.075, 0.3),
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest independent controls in mass, redshift, and SFH SFR100."""
    valid = np.isfinite(mass) & np.isfinite(redshift) & np.isfinite(log_sfr)
    valid[query] = False
    valid[np.asarray(retrieved, dtype=int)] = False
    delta = np.column_stack((
        mass - mass[query], redshift - redshift[query], log_sfr - log_sfr[query],
    ))
    within = (
        valid
        & (np.abs(delta[:, 0]) <= calipers[0])
        & (np.abs(delta[:, 1]) <= calipers[1])
        & (np.abs(delta[:, 2]) <= calipers[2])
    )
    candidates = np.flatnonzero(within)
    if len(candidates) < count:
        candidates = np.flatnonzero(valid)
    distance = np.sqrt(np.sum((delta / np.asarray(scales)) ** 2, axis=1))
    order = candidates[np.argsort(distance[candidates])]
    if len(order) < count:
        raise ValueError(f"Only {len(order)} valid physical controls for {count} neighbours")
    selected = order[:count]
    return selected, delta[selected]


def _representative_query(
    rank: np.ndarray, properties: dict[str, np.ndarray], covered: np.ndarray,
) -> int:
    mass = np.asarray(properties["log_stellar_mass"], dtype=float)
    redshift = np.asarray(properties["redshift"], dtype=float)
    log_sfr = _log_sfr100(properties)
    eligible = (
        covered & np.isfinite(mass) & np.isfinite(redshift) & np.isfinite(log_sfr)
        & (mass >= 9.5) & (mass <= 11.0)
        & (redshift >= 0.2) & (redshift <= 1.0)
        & (log_sfr >= -3.0) & (log_sfr <= 1.5)
    )
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        raise ValueError("No eligible representative query")
    target = np.median(rank[candidates])
    return int(candidates[np.argmin(np.abs(rank[candidates] - target))])


def _histogram_edges(control, retrieved, key, bins=10):
    combined = np.concatenate((control, retrieved))
    combined = combined[np.isfinite(combined)]
    if key.startswith("zoobot_") or key == "axis_ratio":
        return np.linspace(0.0, 1.0, bins + 1)
    lower, upper = np.nanmin(combined), np.nanmax(combined)
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 0.05)
    else:
        padding = 0.04 * (upper - lower)
    return np.linspace(lower - padding, upper + padding, bins + 1)


def _scatter68(values):
    """Robust one-sigma-like scatter: half the 16th-to-84th percentile range."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    lower, upper = np.percentile(values, (16.0, 84.0))
    return 0.5 * (upper - lower)


def _scatter_statistics(control, retrieved, rng, bootstrap_samples=3000):
    control = np.asarray(control, dtype=float)
    retrieved = np.asarray(retrieved, dtype=float)
    control = control[np.isfinite(control)]
    retrieved = retrieved[np.isfinite(retrieved)]
    control_scatter = _scatter68(control)
    retrieved_scatter = _scatter68(retrieved)
    ratio = (
        retrieved_scatter / control_scatter
        if np.isfinite(control_scatter) and control_scatter > 0.0 else np.nan
    )
    bootstrap = np.full(bootstrap_samples, np.nan, dtype=float)
    for index in range(bootstrap_samples):
        control_draw = rng.choice(control, size=len(control), replace=True)
        retrieved_draw = rng.choice(retrieved, size=len(retrieved), replace=True)
        denominator = _scatter68(control_draw)
        if np.isfinite(denominator) and denominator > 0.0:
            bootstrap[index] = _scatter68(retrieved_draw) / denominator
    valid = bootstrap[np.isfinite(bootstrap) & (bootstrap > 0.0)]
    if len(valid):
        ratio_lower, ratio_upper = np.percentile(valid, (16.0, 84.0))
    else:
        ratio_lower = ratio_upper = np.nan
    return {
        "control_scatter68": float(control_scatter),
        "retrieved_scatter68": float(retrieved_scatter),
        "scatter_ratio": float(ratio),
        "scatter_ratio_p16": float(ratio_lower),
        "scatter_ratio_p84": float(ratio_upper),
    }


def _plot_parameter_distribution(
    axis, control, retrieved, query_value, key, title, rng,
):
    control = np.asarray(control, dtype=float)
    retrieved = np.asarray(retrieved, dtype=float)
    control = control[np.isfinite(control)]
    retrieved = retrieved[np.isfinite(retrieved)]
    edges = _histogram_edges(control, retrieved, key)
    axis.hist(
        control, bins=edges, density=True, histtype="stepfilled", alpha=0.28,
        color="0.45", edgecolor="0.35", linewidth=1.5, label="M,z,SFR controls",
    )
    axis.hist(
        retrieved, bins=edges, density=True, histtype="step", color="#0072B2",
        linewidth=2.2, label="JEPA-CSLS neighbours",
    )
    control_median = np.nanmedian(control)
    retrieved_median = np.nanmedian(retrieved)
    axis.axvline(control_median, color="0.25", lw=1.5, ls=":")
    axis.axvline(retrieved_median, color="#0072B2", lw=1.5, ls=":")
    if np.isfinite(query_value):
        axis.axvline(
            query_value, color="#D55E00", lw=1.5, ls="--",
            label="query's true VIS",
        )
    statistics = _scatter_statistics(control, retrieved, rng)
    scatter_ratio = statistics["scatter_ratio"]
    ratio_text = f"{scatter_ratio:.2f}x" if np.isfinite(scatter_ratio) else "undefined"
    axis.set_title(
        f"{title}\nmedian difference = {retrieved_median - control_median:+.3f}\n"
        f"scatter $\\sigma_{{68}}$: C={statistics['control_scatter68']:.3f}, "
        f"N={statistics['retrieved_scatter68']:.3f} ({ratio_text})",
        fontsize=8.7,
    )
    axis.set_ylabel("Density")
    axis.grid(axis="y", alpha=0.18)
    return statistics


def _scatter_summary_figure(statistics, query_id):
    labels = [row["label"] for row in statistics]
    ratios = np.asarray([row["scatter_ratio"] for row in statistics], dtype=float)
    lower = np.asarray([row["scatter_ratio_p16"] for row in statistics], dtype=float)
    upper = np.asarray([row["scatter_ratio_p84"] for row in statistics], dtype=float)
    valid = (
        np.isfinite(ratios) & (ratios > 0.0)
        & np.isfinite(lower) & (lower > 0.0)
        & np.isfinite(upper) & (upper > 0.0)
    )
    figure, axis = plt.subplots(figsize=(9.2, 7.0))
    y = np.arange(len(labels))[::-1]
    x = np.log2(ratios[valid])
    interval_lower = np.minimum(lower[valid], ratios[valid])
    interval_upper = np.maximum(upper[valid], ratios[valid])
    xerr = np.vstack((
        x - np.log2(interval_lower), np.log2(interval_upper) - x,
    ))
    colors = np.where(ratios[valid] >= 1.0, "#D55E00", "#0072B2")
    axis.errorbar(
        x, y[valid], xerr=xerr, fmt="none", ecolor="0.45", elinewidth=1.4,
        capsize=3, zorder=1,
    )
    axis.scatter(x, y[valid], color=colors, s=58, zorder=2)
    axis.axvline(0.0, color="black", lw=1.2, ls="--")
    axis.set_yticks(y, labels)
    ticks = np.arange(-3, 4, dtype=float)
    axis.set_xticks(ticks, ["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    finite_bounds = np.concatenate((np.log2(interval_lower), np.log2(interval_upper)))
    bound = max(3.0, np.nanmax(np.abs(finite_bounds)) + 0.35) if len(finite_bounds) else 3.0
    axis.set_xlim(-bound, bound)
    axis.set_xlabel(
        "Scatter ratio: JEPA-CSLS neighbours / matched controls\n"
        "points: observed $\\sigma_{68}$ ratio; bars: bootstrap 16-84% interval"
    )
    axis.set_title(
        f"Morphological scatter selected from one SFH\nQuery ID {query_id}",
        fontsize=14,
    )
    axis.grid(axis="x", alpha=0.2)
    figure.text(
        0.5, 0.02,
        "Orange: retrieved neighbours are more dispersed. Blue: controls are more dispersed. "
        "$\\sigma_{68}=(P_{84}-P_{16})/2$.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return figure


def _read_stamp(stamps: Path, galaxy_id: int) -> np.ndarray | None:
    candidates = (
        stamps / f"VIS_{galaxy_id}.jpg",
        stamps / f"VIS_{galaxy_id}.jpeg",
        stamps / f"VIS_{galaxy_id}.png",
        stamps / f"{galaxy_id}.jpg",
        stamps / f"{galaxy_id}.png",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None
    image = np.asarray(plt.imread(path), dtype=float)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=-1)
    if np.nanmax(image) > 1.0:
        image /= 255.0
    return np.clip(np.nan_to_num(image), 0.0, 1.0)


def _read_query_sfh(h5_path: Path, galaxy_id: int):
    with h5py.File(h5_path, "r") as source:
        ids = np.asarray(source["galaxy_id"], dtype=np.int64)
        matches = np.flatnonzero(ids == galaxy_id)
        if len(matches) != 1:
            raise ValueError(f"Galaxy ID {galaxy_id} is absent or duplicated in {h5_path}")
        row = int(matches[0])
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))
        output = {
            "time": np.asarray(source["sfh_time_grid"], dtype=float),
            "age_gyr": float(source["sfh_time_norm"][row]) / 1000.0,
            "median": np.asarray(source["sfh"][row], dtype=float),
            "p16": (
                np.asarray(source["sfh_p16"][row], dtype=float)
                if "sfh_p16" in source else None
            ),
            "p84": (
                np.asarray(source["sfh_p84"][row], dtype=float)
                if "sfh_p84" in source else None
            ),
            "epsilon": epsilon,
        }
    return output


def _sfh_density(log_weights, time, age_gyr, epsilon):
    weights = np.maximum(np.power(10.0, log_weights) - epsilon, 0.0)
    edges = np.empty(len(time) + 1, dtype=float)
    edges[1:-1] = 0.5 * (time[:-1] + time[1:])
    edges[0], edges[-1] = 0.0, 1.0
    widths = np.diff(edges) * age_gyr
    return weights / np.maximum(widths, np.finfo(float).eps)


def _query_figure(
    h5_path: Path, stamps: Path | None, galaxy_id: int, mass: float,
    redshift: float, log_sfr: float, paired_rank: int, gallery_size: int,
    properties: dict[str, np.ndarray], query: int,
):
    sfh = _read_query_sfh(h5_path, galaxy_id)
    lookback = sfh["time"] * sfh["age_gyr"]
    median = _sfh_density(
        sfh["median"], sfh["time"], sfh["age_gyr"], sfh["epsilon"],
    )
    lower = upper = None
    if sfh["p16"] is not None and sfh["p84"] is not None:
        lower = _sfh_density(
            sfh["p16"], sfh["time"], sfh["age_gyr"], sfh["epsilon"],
        )
        upper = _sfh_density(
            sfh["p84"], sfh["time"], sfh["age_gyr"], sfh["epsilon"],
        )

    figure = plt.figure(figsize=(12.5, 6.5))
    grid = figure.add_gridspec(1, 2, width_ratios=(2.15, 1.0), wspace=0.18)
    sfh_axis = figure.add_subplot(grid[0, 0])
    image_axis = figure.add_subplot(grid[0, 1])
    if lower is not None:
        sfh_axis.fill_between(
            lookback, lower, upper, color="#56B4E9", alpha=0.25,
            label="posterior 16-84%",
        )
    sfh_axis.plot(
        lookback, median, color="#0072B2", lw=2.2,
        label="median normalized SFH",
    )
    sfh_axis.axvspan(0.0, min(0.1, sfh["age_gyr"]), color="#E69F00", alpha=0.2,
                     label="latest 100 Myr")
    sfh_axis.set_xlabel("Lookback time [Gyr]")
    sfh_axis.set_ylabel("Normalized star-formation rate [Gyr$^{-1}$]")
    sfh_axis.set_xlim(0.0, sfh["age_gyr"])
    sfh_axis.set_ylim(bottom=0.0)
    sfh_axis.grid(alpha=0.2)
    sfh_axis.legend(loc="best", fontsize=9)
    sfh_axis.set_title("Query star-formation history")

    stamp = _read_stamp(stamps, galaxy_id) if stamps is not None else None
    if stamp is None:
        image_axis.text(0.5, 0.5, "true VIS stamp unavailable", ha="center", va="center")
    else:
        image_axis.imshow(stamp, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    image_axis.set_title("True paired VIS morphology")
    image_axis.set_xticks([])
    image_axis.set_yticks([])

    extra = []
    for key, label, unit in (
        ("sfh_mass_weighted_age_gyr", "mass-weighted age", "Gyr"),
        ("sfh_t50_lookback", "$t_{50}/t_H$", ""),
        ("sfh_duration_80", "$(t_{80}-t_{20})/t_H$", ""),
    ):
        if key in properties and np.isfinite(properties[key][query]):
            suffix = f" {unit}" if unit else ""
            extra.append(f"{label}={properties[key][query]:.2f}{suffix}")
    figure.suptitle(
        f"Individual SFH-to-morphology diagnostic | ID {galaxy_id}\n"
        f"log(M*/Msun)={mass:.3f}, z={redshift:.3f}, "
        f"log(SFR100/[Msun yr^-1])={log_sfr:.3f} | "
        f"paired rank {paired_rank:,}/{gallery_size:,}",
        fontsize=15,
    )
    if extra:
        figure.text(0.5, 0.025, " | ".join(extra), ha="center", fontsize=10)
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.78, wspace=0.22)
    return figure


def _stamp_figure(
    stamps: Path, galaxy_ids: np.ndarray, indices: np.ndarray,
    mass: np.ndarray, redshift: np.ndarray, log_sfr: np.ndarray,
    query_id: int, title: str,
):
    columns = 5
    rows = int(np.ceil(len(indices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12.5, 2.55 * rows))
    axes = np.asarray(axes).reshape(rows, columns)
    for rank, (axis, index) in enumerate(zip(axes.flat, indices), start=1):
        galaxy_id = int(galaxy_ids[index])
        image = _read_stamp(stamps, galaxy_id)
        if image is None:
            axis.text(0.5, 0.5, "stamp missing", ha="center", va="center")
        else:
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        axis.set_title(
            f"#{rank}  ID {galaxy_id}\n"
            f"M={mass[index]:.2f}, z={redshift[index]:.2f}, "
            f"SFR={log_sfr[index]:.2f}",
            fontsize=7,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(indices):]:
        axis.axis("off")
    figure.suptitle(
        f"{title}\nSFH query ID {query_id} | stored VIS scaling fixed to [0, 1]",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.9, w_pad=0.35)
    return figure


def plot(args):
    with np.load(args.csls) as archive:
        galaxy_ids = np.asarray(archive["galaxy_id"], dtype=np.int64)
        ranks = np.asarray(archive[f"csls_k{args.csls_k}_rank"], dtype=np.int64)
        top = np.asarray(archive[f"csls_k{args.csls_k}_top_index"], dtype=np.int64)
    properties, covered = _align_properties(args.properties, galaxy_ids)
    mass = np.asarray(properties["log_stellar_mass"], dtype=float)
    redshift = np.asarray(properties["redshift"], dtype=float)
    log_sfr = _log_sfr100(properties)

    if args.galaxy_id is None:
        query = _representative_query(ranks, properties, covered)
        selection = "representative median-rank validation query"
    else:
        matches = np.flatnonzero(galaxy_ids == args.galaxy_id)
        if len(matches) != 1:
            raise ValueError(f"Galaxy ID {args.galaxy_id} is absent or duplicated")
        query = int(matches[0])
        selection = "requested galaxy ID"
    retrieved = top[query, :args.neighbours]
    control, control_delta = _matched_control_indices(
        query, retrieved, mass, redshift, log_sfr, args.neighbours,
    )

    available = [(key, label) for key, label in PARAMETERS if key in properties]
    figure, axes = plt.subplots(4, 3, figsize=(14, 15), squeeze=False)
    rng = np.random.default_rng(args.seed)
    scatter_statistics = []
    for axis, (key, label) in zip(axes.flat, available):
        values = np.asarray(properties[key], dtype=float)
        statistics = _plot_parameter_distribution(
            axis, values[control], values[retrieved], values[query], key, label, rng,
        )
        statistics.update({"parameter": key, "label": label})
        scatter_statistics.append(statistics)
    for axis in axes.flat[len(available):]:
        axis.axis("off")
    axes.flat[0].legend(loc="best", fontsize=7)
    median_delta = np.median(np.abs(control_delta), axis=0)
    figure.suptitle(
        "Morphology distributions selected from one SFH\n"
        f"ID {galaxy_ids[query]} | CSLS k={args.csls_k} | "
        f"paired rank {ranks[query]:,}/{len(galaxy_ids):,}\n"
        f"log(M*/Msun)={mass[query]:.3f}, z={redshift[query]:.3f}, "
        f"log(SFR100/[Msun yr^-1])={log_sfr[query]:.3f}",
        fontsize=15,
    )
    figure.text(
        0.5, 0.018,
        f"Blue: {args.neighbours} SFH-to-VIS neighbours. Gray: {args.neighbours} "
        "independent nearest controls in mass, redshift, and SFH SFR100. "
        f"Median absolute control offsets: {median_delta[0]:.3f} dex, "
        f"{median_delta[1]:.3f} in z, {median_delta[2]:.3f} dex in SFR.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.935))

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"individual_sfh_morphology_{int(galaxy_ids[query])}"
    pdf = args.output / f"{stem}.pdf"
    png = args.output / f"{stem}.png"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    scatter_png = args.output / f"{stem}_scatter.png"
    scatter_figure = _scatter_summary_figure(
        scatter_statistics, int(galaxy_ids[query]),
    )
    scatter_figure.savefig(scatter_png, dpi=180, bbox_inches="tight")
    retrieved_stamps_png = args.output / f"{stem}_retrieved_stamps.png"
    control_stamps_png = args.output / f"{stem}_control_stamps.png"
    query_png = args.output / f"{stem}_query_sfh.png"
    query_figure = None
    if args.h5 is not None:
        query_figure = _query_figure(
            args.h5, args.stamps, int(galaxy_ids[query]), mass[query],
            redshift[query], log_sfr[query], int(ranks[query]), len(galaxy_ids),
            properties, query,
        )
        query_figure.savefig(query_png, dpi=180, bbox_inches="tight")
    stamp_figures = []
    if args.stamps is not None:
        stamp_figures = [
            _stamp_figure(
                args.stamps, galaxy_ids, retrieved, mass, redshift, log_sfr,
                int(galaxy_ids[query]), "JEPA-CSLS retrieved VIS neighbours",
            ),
            _stamp_figure(
                args.stamps, galaxy_ids, control, mass, redshift, log_sfr,
                int(galaxy_ids[query]), "Mass-redshift-SFR matched VIS controls",
            ),
        ]
        stamp_figures[0].savefig(retrieved_stamps_png, dpi=180, bbox_inches="tight")
        stamp_figures[1].savefig(control_stamps_png, dpi=180, bbox_inches="tight")
    with PdfPages(pdf) as document:
        if query_figure is not None:
            document.savefig(query_figure, bbox_inches="tight")
        document.savefig(figure, bbox_inches="tight")
        document.savefig(scatter_figure, bbox_inches="tight")
        for stamp_figure in stamp_figures:
            document.savefig(stamp_figure, bbox_inches="tight")
    plt.close(figure)
    plt.close(scatter_figure)
    if query_figure is not None:
        plt.close(query_figure)
    for stamp_figure in stamp_figures:
        plt.close(stamp_figure)

    rows = []
    for group, indices in (("retrieved", retrieved), ("physical_control", control)):
        for index in indices:
            row = {
                "group": group,
                "galaxy_id": int(galaxy_ids[index]),
                "log_stellar_mass": float(mass[index]),
                "redshift": float(redshift[index]),
                "log_sfr100": float(log_sfr[index]),
            }
            row.update({
                key: float(properties[key][index])
                for key, _ in available
            })
            rows.append(row)
    csv_path = args.output / f"{stem}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scatter_csv_path = args.output / f"{stem}_scatter.csv"
    with scatter_csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scatter_statistics[0]))
        writer.writeheader()
        writer.writerows(scatter_statistics)
    metadata = {
        "query_galaxy_id": int(galaxy_ids[query]),
        "query_index": query,
        "selection": selection,
        "csls_k": args.csls_k,
        "paired_rank": int(ranks[query]),
        "gallery_size": int(len(galaxy_ids)),
        "neighbours_per_group": args.neighbours,
        "query_log_stellar_mass": float(mass[query]),
        "query_redshift": float(redshift[query]),
        "query_log_sfr100": float(log_sfr[query]),
        "sfr_definition": (
            "log_stellar_mass + sfh_log_sfr_per_stellar_mass_100myr_r0"
        ),
        "median_absolute_control_offsets": {
            "log_mass_dex": float(median_delta[0]),
            "redshift": float(median_delta[1]),
            "log_sfr100_dex": float(median_delta[2]),
        },
        "retrieved_ids": galaxy_ids[retrieved].astype(int).tolist(),
        "control_ids": galaxy_ids[control].astype(int).tolist(),
        "pdf": str(pdf),
        "png": str(png),
        "scatter_png": str(scatter_png),
        "scatter_csv": str(scatter_csv_path),
        "scatter_definition": "sigma68 = (84th percentile - 16th percentile) / 2",
        "scatter_statistics": scatter_statistics,
        "query_sfh_png": str(query_png) if args.h5 is not None else None,
        "retrieved_stamps_png": (
            str(retrieved_stamps_png) if args.stamps is not None else None
        ),
        "control_stamps_png": (
            str(control_stamps_png) if args.stamps is not None else None
        ),
        "csv": str(csv_path),
    }
    metadata_path = args.output / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csls", type=Path, required=True)
    parser.add_argument("--properties", type=Path, required=True)
    parser.add_argument("--h5", type=Path)
    parser.add_argument("--stamps", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/pdf"))
    parser.add_argument("--galaxy-id", type=int)
    parser.add_argument("--csls-k", type=int, default=100)
    parser.add_argument("--neighbours", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plot(args)


if __name__ == "__main__":
    main()
