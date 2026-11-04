"""Compare morphology of one SFH's JEPA neighbours with physical controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


def _plot_parameter_distribution(
    axis, control, retrieved, query_value, key, title,
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
    axis.set_title(
        f"{title}\nmedian difference = {retrieved_median - control_median:+.3f}",
        fontsize=10,
    )
    axis.set_ylabel("Density")
    axis.grid(axis="y", alpha=0.18)


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
    for axis, (key, label) in zip(axes.flat, available):
        values = np.asarray(properties[key], dtype=float)
        _plot_parameter_distribution(
            axis, values[control], values[retrieved], values[query], key, label,
        )
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
    retrieved_stamps_png = args.output / f"{stem}_retrieved_stamps.png"
    control_stamps_png = args.output / f"{stem}_control_stamps.png"
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
        document.savefig(figure, bbox_inches="tight")
        for stamp_figure in stamp_figures:
            document.savefig(stamp_figure, bbox_inches="tight")
    plt.close(figure)
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
