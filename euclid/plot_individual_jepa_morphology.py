"""Compare morphology of one SFH's JEPA neighbours with physical controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


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


def _plot_parameter(axis, control, retrieved, query_value, title, rng):
    groups = (np.asarray(control, dtype=float), np.asarray(retrieved, dtype=float))
    colors = ("0.55", "#0072B2")
    for x, (values, color) in enumerate(zip(groups, colors)):
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        jitter = rng.uniform(-0.12, 0.12, len(values))
        axis.scatter(
            np.full(len(values), x) + jitter, values, s=22, alpha=0.7,
            color=color, edgecolor="white", linewidth=0.3, zorder=2,
        )
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        axis.vlines(x, q25, q75, color="black", lw=3, zorder=3)
        axis.scatter(x, median, marker="_", s=180, color="black", lw=2, zorder=4)
    if np.isfinite(query_value):
        axis.axhline(
            query_value, color="#D55E00", lw=1.5, ls="--",
            label="query's true VIS",
        )
    retrieved_median = np.nanmedian(groups[1])
    control_median = np.nanmedian(groups[0])
    axis.set_title(
        f"{title}\nmedian difference = {retrieved_median - control_median:+.3f}",
        fontsize=10,
    )
    axis.set_xticks((0, 1), ("M,z,SFR\ncontrols", "JEPA-CSLS\nneighbours"))
    axis.grid(axis="y", alpha=0.18)


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
    for axis, (key, label) in zip(axes.flat, available):
        values = np.asarray(properties[key], dtype=float)
        _plot_parameter(
            axis, values[control], values[retrieved], values[query], label, rng,
        )
    for axis in axes.flat[len(available):]:
        axis.axis("off")
    axes.flat[0].legend(loc="best", fontsize=8)
    median_delta = np.median(np.abs(control_delta), axis=0)
    figure.suptitle(
        "Morphology selected from one SFH\n"
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
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(figure)

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
        "csv": str(csv_path),
    }
    metadata_path = args.output / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csls", type=Path, required=True)
    parser.add_argument("--properties", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/pdf"))
    parser.add_argument("--galaxy-id", type=int)
    parser.add_argument("--csls-k", type=int, default=100)
    parser.add_argument("--neighbours", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plot(args)


if __name__ == "__main__":
    main()
