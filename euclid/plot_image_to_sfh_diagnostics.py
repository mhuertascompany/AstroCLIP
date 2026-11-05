"""Diagnose SFHs retrieved from morphology-selected VIS images.

For each high-probability spiral-arm or bar query, the script compares the
SFHs of its reverse-JEPA CSLS neighbours with an equal-size control selected
only in stellar mass and redshift.  It emphasizes robust scatter rather than
only median differences.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .plot_individual_jepa_morphology import (
    _align_properties,
    _histogram_edges,
    _read_stamp,
    _scatter68,
    _scatter_statistics,
)


SFH_PARAMETERS = (
    ("sfh_mass_weighted_age_gyr", "Mass-weighted age [Gyr]"),
    ("sfh_t50_lookback", "$t_{50}/t_H$"),
    ("sfh_duration_80", "$(t_{80}-t_{20})/t_H$"),
    ("sfh_recent_mass_fraction_100myr", "Mass fraction in latest 100 Myr"),
    ("sfh_peak_age_gyr", "Peak lookback time [Gyr]"),
    ("sfh_entropy", "SFH entropy"),
)

FEATURES = {
    "spiral": ("zoobot_spiral_probability", "P(spiral arms)"),
    "bar": ("zoobot_bar_probability", "P(bar)"),
}


def _fractional_widths(time: np.ndarray) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    edges = np.empty(len(time) + 1, dtype=float)
    edges[1:-1] = 0.5 * (time[:-1] + time[1:])
    edges[0], edges[-1] = 0.0, 1.0
    return np.diff(edges)


def _sfh_fractional_density(log_weights, time, epsilon):
    """Convert log bin masses to a unit-integral density over t_lookback/t_H."""
    weights = np.maximum(np.power(10.0, log_weights) - epsilon, 0.0)
    normalization = weights.sum(axis=-1, keepdims=True)
    weights = weights / np.maximum(normalization, np.finfo(float).eps)
    return weights / _fractional_widths(time)


def _load_sfhs(path: Path, galaxy_ids: np.ndarray):
    with h5py.File(path, "r") as source:
        source_ids = np.asarray(source["galaxy_id"], dtype=np.int64)
        time = np.asarray(source["sfh_time_grid"], dtype=float)
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))
        position = {int(value): index for index, value in enumerate(source_ids)}
        rows = np.array(
            [position.get(int(value), -1) for value in galaxy_ids], dtype=np.int64,
        )
        if np.any(rows < 0):
            missing = galaxy_ids[rows < 0]
            raise ValueError(f"SFH file is missing {len(missing)} retrieval galaxies")
        log_weights = np.asarray(source["sfh"][rows], dtype=float)
    return time, _sfh_fractional_density(log_weights, time, epsilon)


def _retrieved_without_pair(top: np.ndarray, query: int, count: int) -> np.ndarray:
    selected = np.asarray(top[query], dtype=np.int64)
    selected = selected[selected != query]
    selected = selected[:count]
    if len(selected) != count:
        raise ValueError(
            f"Only {len(selected)} non-paired neighbours available; requested {count}"
        )
    return selected


def _mass_redshift_controls(
    query: int, retrieved: np.ndarray, mass: np.ndarray, redshift: np.ndarray,
    count: int, scales=(0.1, 0.05), calipers=(0.2, 0.1),
):
    """Nearest independent controls using only stellar mass and redshift."""
    valid = np.isfinite(mass) & np.isfinite(redshift)
    valid[query] = False
    valid[np.asarray(retrieved, dtype=int)] = False
    delta = np.column_stack((mass - mass[query], redshift - redshift[query]))
    within = (
        valid
        & (np.abs(delta[:, 0]) <= calipers[0])
        & (np.abs(delta[:, 1]) <= calipers[1])
    )
    candidates = np.flatnonzero(within)
    if len(candidates) < count:
        candidates = np.flatnonzero(valid)
    distance = np.sqrt(np.sum((delta / np.asarray(scales)) ** 2, axis=1))
    ordered = candidates[np.argsort(distance[candidates])]
    if len(ordered) < count:
        raise ValueError(f"Only {len(ordered)} valid mass-redshift controls")
    selected = ordered[:count]
    return selected, delta[selected]


def _diverse_high_probability_queries(
    score: np.ndarray, mass: np.ndarray, redshift: np.ndarray,
    eligible: np.ndarray, count: int,
) -> np.ndarray:
    """Pick high-score queries while spanning the eligible mass distribution."""
    candidates = np.flatnonzero(
        eligible & np.isfinite(score) & np.isfinite(mass) & np.isfinite(redshift),
    )
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible morphology queries")
    edges = np.quantile(mass[candidates], np.linspace(0.0, 1.0, count + 1))
    selected = []
    for index in range(count):
        upper = mass[candidates] <= edges[index + 1] if index == count - 1 else mass[candidates] < edges[index + 1]
        inside = candidates[(mass[candidates] >= edges[index]) & upper]
        if len(inside):
            selected.append(int(inside[np.argmax(score[inside])]))
    if len(selected) < count:
        remainder = candidates[np.argsort(score[candidates])[::-1]]
        selected.extend(
            int(value) for value in remainder if int(value) not in selected
        )
    return np.asarray(selected[:count], dtype=np.int64)


def _smooth(values: np.ndarray, width: int = 11) -> np.ndarray:
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def _sfh_overview_figure(
    time, sfhs, query, retrieved, controls, stamp, feature, feature_label,
    feature_score, galaxy_id, mass, redshift,
):
    retrieved_values = sfhs[retrieved]
    control_values = sfhs[controls]
    retrieved_q = np.percentile(retrieved_values, (16, 50, 84), axis=0)
    control_q = np.percentile(control_values, (16, 50, 84), axis=0)
    retrieved_scatter = 0.5 * (retrieved_q[2] - retrieved_q[0])
    control_scatter = 0.5 * (control_q[2] - control_q[0])
    ratio = _smooth(retrieved_scatter) / np.maximum(
        _smooth(control_scatter), np.finfo(float).eps,
    )

    figure = plt.figure(figsize=(13.5, 9.5))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(0.8, 2.4), height_ratios=(1.35, 1.0),
        hspace=0.3, wspace=0.22,
    )
    image_axis = figure.add_subplot(grid[0, 0])
    ratio_axis = figure.add_subplot(grid[1, 0])
    curve_axis = figure.add_subplot(grid[0, 1])
    scatter_axis = figure.add_subplot(grid[1, 1])

    if stamp is None:
        image_axis.text(0.5, 0.5, "VIS stamp unavailable", ha="center", va="center")
    else:
        image_axis.imshow(stamp, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    image_axis.set_title(
        f"Morphology-selected query\n{feature_label}={feature_score:.3f}",
        fontsize=11,
    )
    image_axis.set_xticks([])
    image_axis.set_yticks([])

    curve_axis.fill_between(
        time, control_q[0], control_q[2], color="0.55", alpha=0.23,
        label="mass-z controls: 16-84%",
    )
    curve_axis.plot(time, control_q[1], color="0.35", lw=2, label="control median")
    curve_axis.fill_between(
        time, retrieved_q[0], retrieved_q[2], color="#56B4E9", alpha=0.28,
        label="JEPA-CSLS neighbours: 16-84%",
    )
    curve_axis.plot(
        time, retrieved_q[1], color="#0072B2", lw=2.2,
        label="retrieved median",
    )
    curve_axis.plot(
        time, sfhs[query], color="#D55E00", lw=1.7, ls="--",
        label="true paired SFH",
    )
    curve_axis.set(
        xlabel="Fractional lookback time $t_{LB}/t_H$",
        ylabel="Normalized SFH density",
        title="Median reconstruction and population width",
        xlim=(0, 1), ylim=(0, None),
    )
    curve_axis.legend(fontsize=8, ncol=2)
    curve_axis.grid(alpha=0.18)

    scatter_axis.plot(
        time, control_scatter, color="0.45", lw=1.5, alpha=0.7,
        label="mass-z control $\\sigma_{68}$",
    )
    scatter_axis.plot(
        time, retrieved_scatter, color="#0072B2", lw=1.5, alpha=0.8,
        label="retrieved $\\sigma_{68}$",
    )
    scatter_axis.plot(time, _smooth(control_scatter), color="0.2", lw=2.3)
    scatter_axis.plot(time, _smooth(retrieved_scatter), color="#0072B2", lw=2.3)
    scatter_axis.set(
        xlabel="Fractional lookback time $t_{LB}/t_H$",
        ylabel="$\\sigma_{68}$ of SFH density",
        title="Time-dependent SFH scatter (thin: raw; thick: 11-bin smooth)",
        xlim=(0, 1), ylim=(0, None),
    )
    scatter_axis.legend(fontsize=8)
    scatter_axis.grid(alpha=0.18)

    ratio_axis.axhline(1.0, color="black", lw=1.2, ls="--")
    ratio_axis.plot(time, ratio, color="#CC79A7", lw=2)
    ratio_axis.fill_between(
        time, 1.0, ratio, where=ratio < 1.0,
        color="#0072B2", alpha=0.18,
    )
    ratio_axis.fill_between(
        time, 1.0, ratio, where=ratio >= 1.0,
        color="#D55E00", alpha=0.18,
    )
    ratio_axis.set(
        xlabel="$t_{LB}/t_H$", ylabel="Scatter ratio",
        title="retrieved / control",
        xlim=(0, 1), ylim=(0, min(4.0, max(2.0, np.nanpercentile(ratio, 98)))),
    )
    ratio_axis.grid(alpha=0.18)

    figure.suptitle(
        f"VIS-to-SFH reverse retrieval: high-{feature} query | ID {galaxy_id}\n"
        f"log(M*/Msun)={mass:.3f}, z={redshift:.3f} | paired object excluded from neighbours",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.88)
    return figure, retrieved_scatter, control_scatter, ratio


def _summary_distribution_figure(
    properties, query, retrieved, controls, feature, galaxy_id, rng,
):
    available = [(key, label) for key, label in SFH_PARAMETERS if key in properties]
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), squeeze=False)
    statistics = []
    for axis, (key, label) in zip(axes.flat, available):
        values = np.asarray(properties[key], dtype=float)
        control = values[controls]
        selected = values[retrieved]
        control_finite = control[np.isfinite(control)]
        selected_finite = selected[np.isfinite(selected)]
        edges = _histogram_edges(control_finite, selected_finite, key, bins=9)
        axis.hist(
            control_finite, bins=edges, density=True, histtype="stepfilled",
            color="0.55", edgecolor="0.4", alpha=0.25, label="mass-z controls",
        )
        axis.hist(
            selected_finite, bins=edges, density=True, histtype="step",
            color="#0072B2", lw=2.2, label="JEPA-CSLS neighbours",
        )
        if np.isfinite(values[query]):
            axis.axvline(values[query], color="#D55E00", lw=1.5, ls="--")
        result = _scatter_statistics(control_finite, selected_finite, rng)
        result.update({"parameter": key, "label": label})
        statistics.append(result)
        axis.set_title(
            f"{label}\n"
            f"$\\sigma_{{68}}$: N={result['retrieved_scatter68']:.3g}, "
            f"C={result['control_scatter68']:.3g}; "
            f"ratio={result['scatter_ratio']:.2f}",
            fontsize=9,
        )
        axis.set_ylabel("Density")
        axis.grid(axis="y", alpha=0.18)
    for axis in axes.flat[len(available):]:
        axis.axis("off")
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(
        f"SFH summary distributions for high-{feature} image | ID {galaxy_id}\n"
        "The reported width is sigma68=(P84-P16)/2; both groups have equal N",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure, statistics


def _scatter_ratio_figure(statistics, feature, galaxy_id):
    labels = [row["label"] for row in statistics]
    ratio = np.asarray([row["scatter_ratio"] for row in statistics], dtype=float)
    lower = np.asarray([row["scatter_ratio_p16"] for row in statistics], dtype=float)
    upper = np.asarray([row["scatter_ratio_p84"] for row in statistics], dtype=float)
    valid = (
        np.isfinite(ratio) & (ratio > 0) & np.isfinite(lower) & (lower > 0)
        & np.isfinite(upper) & (upper > 0)
    )
    y = np.arange(len(labels))[::-1]
    figure, axis = plt.subplots(figsize=(9.5, 6.4))
    x = np.log2(ratio[valid])
    lo = np.minimum(lower[valid], ratio[valid])
    hi = np.maximum(upper[valid], ratio[valid])
    axis.errorbar(
        x, y[valid],
        xerr=np.vstack((x - np.log2(lo), np.log2(hi) - x)),
        fmt="none", ecolor="0.45", capsize=3,
    )
    colors = np.where(ratio[valid] < 1.0, "#0072B2", "#D55E00")
    axis.scatter(x, y[valid], c=colors, s=65, zorder=3)
    axis.axvline(0, color="black", ls="--", lw=1.2)
    axis.set_yticks(y, labels)
    ticks = np.arange(-3, 4, dtype=float)
    axis.set_xticks(ticks, ("1/8", "1/4", "1/2", "1", "2", "4", "8"))
    axis.set_xlabel(
        "Scatter ratio: JEPA-CSLS neighbours / mass-z controls\n"
        "error bars: bootstrap 16-84% interval"
    )
    axis.set_title(
        f"How strongly does a high-{feature} image constrain SFH?\nID {galaxy_id}",
        fontsize=14,
    )
    axis.grid(axis="x", alpha=0.2)
    figure.text(
        0.5, 0.025,
        "Blue (<1): retrieved SFHs are tighter than physical controls. "
        "Orange (>1): retrieved SFHs are broader.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.93))
    return figure


def _individual_curve_figure(time, sfhs, retrieved, controls, feature, galaxy_id):
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharex=True, sharey=True)
    for axis, indices, color, title in (
        (axes[0], retrieved, "#0072B2", "JEPA-CSLS SFH neighbours"),
        (axes[1], controls, "0.45", "Mass-redshift controls"),
    ):
        axis.plot(time, sfhs[indices].T, color=color, alpha=0.18, lw=0.8)
        axis.plot(time, np.median(sfhs[indices], axis=0), color=color, lw=2.6)
        axis.set(xlabel="$t_{LB}/t_H$", title=title, xlim=(0, 1), ylim=(0, None))
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Normalized SFH density")
    figure.suptitle(
        f"Individual normalized SFHs for high-{feature} query | ID {galaxy_id}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def plot(args):
    with np.load(args.csls) as archive:
        galaxy_ids = np.asarray(archive["galaxy_id"], dtype=np.int64)
        ranks = np.asarray(archive[f"csls_k{args.csls_k}_rank"], dtype=np.int64)
        top = np.asarray(archive[f"csls_k{args.csls_k}_top_index"], dtype=np.int64)
    properties, covered = _align_properties(args.properties, galaxy_ids)
    mass = np.asarray(properties["log_stellar_mass"], dtype=float)
    redshift = np.asarray(properties["redshift"], dtype=float)
    time, sfhs = _load_sfhs(args.h5, galaxy_ids)

    args.output.mkdir(parents=True, exist_ok=True)
    selected_queries = []
    for feature in args.feature:
        key, label = FEATURES[feature]
        score = np.asarray(properties[key], dtype=float)
        threshold = args.spiral_threshold if feature == "spiral" else args.bar_threshold
        eligible = covered & (score >= threshold)
        if "zoobot_featured_probability" in properties:
            eligible &= np.asarray(properties["zoobot_featured_probability"]) >= args.featured_threshold
        if "zoobot_edge_on_probability" in properties:
            eligible &= np.asarray(properties["zoobot_edge_on_probability"]) <= args.edge_on_maximum
        queries = _diverse_high_probability_queries(
            score, mass, redshift, eligible, args.queries_per_feature,
        )
        selected_queries.extend((feature, key, label, int(query)) for query in queries)

    outputs = []
    for feature, key, feature_label, query in selected_queries:
        retrieved = _retrieved_without_pair(top, query, args.neighbours)
        controls, control_delta = _mass_redshift_controls(
            query, retrieved, mass, redshift, args.neighbours,
        )
        galaxy_id = int(galaxy_ids[query])
        stamp = _read_stamp(args.stamps, galaxy_id) if args.stamps is not None else None
        overview, time_retrieved_scatter, time_control_scatter, time_ratio = (
            _sfh_overview_figure(
                time, sfhs, query, retrieved, controls, stamp,
                feature, feature_label, properties[key][query], galaxy_id,
                mass[query], redshift[query],
            )
        )
        rng = np.random.default_rng(args.seed + query)
        distributions, statistics = _summary_distribution_figure(
            properties, query, retrieved, controls, feature, galaxy_id, rng,
        )
        scatter = _scatter_ratio_figure(statistics, feature, galaxy_id)
        individual = _individual_curve_figure(
            time, sfhs, retrieved, controls, feature, galaxy_id,
        )

        stem = f"image_to_sfh_{feature}_{galaxy_id}"
        pdf_path = args.output / f"{stem}.pdf"
        with PdfPages(pdf_path) as document:
            for figure in (overview, distributions, scatter, individual):
                document.savefig(figure, bbox_inches="tight")
        for figure in (overview, distributions, scatter, individual):
            plt.close(figure)

        rows = []
        for group, indices in (("retrieved", retrieved), ("mass_redshift_control", controls)):
            for index in indices:
                row = {
                    "group": group,
                    "galaxy_id": int(galaxy_ids[index]),
                    "log_stellar_mass": float(mass[index]),
                    "redshift": float(redshift[index]),
                }
                row.update({
                    parameter: float(properties[parameter][index])
                    for parameter, _ in SFH_PARAMETERS if parameter in properties
                })
                rows.append(row)
        csv_path = args.output / f"{stem}.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        median_delta = np.median(np.abs(control_delta), axis=0)
        metadata = {
            "direction": "vis_to_sfh",
            "query_galaxy_id": galaxy_id,
            "feature": feature,
            "feature_probability_key": key,
            "feature_probability": float(properties[key][query]),
            "paired_rank": int(ranks[query]),
            "gallery_size": int(len(galaxy_ids)),
            "csls_k": args.csls_k,
            "paired_object_excluded": True,
            "neighbours_per_group": args.neighbours,
            "control_matching": "nearest independent galaxies in log stellar mass and redshift only",
            "median_absolute_control_offsets": {
                "log_mass_dex": float(median_delta[0]),
                "redshift": float(median_delta[1]),
            },
            "time_scatter_definition": "sigma68=(P84-P16)/2 across normalized SFH densities",
            "median_time_scatter_ratio": float(np.nanmedian(time_ratio)),
            "retrieved_ids": galaxy_ids[retrieved].astype(int).tolist(),
            "control_ids": galaxy_ids[controls].astype(int).tolist(),
            "sfh_summary_scatter": statistics,
            "pdf": str(pdf_path),
            "csv": str(csv_path),
        }
        metadata_path = args.output / f"{stem}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        outputs.append(metadata)

    summary_path = args.output / "image_to_sfh_selected_queries.json"
    summary_path.write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps({
        "queries": len(outputs),
        "pdfs": [record["pdf"] for record in outputs],
        "summary": str(summary_path),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csls", type=Path, required=True)
    parser.add_argument("--properties", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--stamps", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/pdf"))
    parser.add_argument(
        "--feature", action="append", choices=tuple(FEATURES),
        default=None,
    )
    parser.add_argument("--queries-per-feature", type=int, default=3)
    parser.add_argument("--spiral-threshold", type=float, default=0.9)
    parser.add_argument("--bar-threshold", type=float, default=0.8)
    parser.add_argument("--featured-threshold", type=float, default=0.5)
    parser.add_argument("--edge-on-maximum", type=float, default=0.5)
    parser.add_argument("--csls-k", type=int, default=100)
    parser.add_argument("--neighbours", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.feature is None:
        args.feature = ["spiral", "bar"]
    plot(args)


if __name__ == "__main__":
    main()
