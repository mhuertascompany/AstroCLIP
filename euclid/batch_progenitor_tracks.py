"""Compute progenitor-analogue tracks for every ID in an explorer selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize

from .progenitor_analogues import (
    DEFAULT_ARCHIVE,
    DEFAULT_BUNDLE,
    DEFAULT_CATALOG,
    DEFAULT_FORMED_FRACTIONS,
    _load_inputs,
    _plot_cluster_background,
    find_analogues,
)


DEFAULT_FRACTIONS = DEFAULT_FORMED_FRACTIONS


def _population_report(
    data: dict[str, np.ndarray | float],
    descendants: pd.DataFrame,
    checkpoints: pd.DataFrame,
    output_pdf: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    xy = np.asarray(data["xy_joint"])
    masses = np.asarray(data["mass"])
    delta = descendants.phz_delta_ms.to_numpy(float)
    norm = Normalize(vmin=np.nanpercentile(delta, 2), vmax=np.nanpercentile(delta, 98))
    cmap = plt.get_cmap("coolwarm")
    small_sample = len(descendants) <= 20
    descendant_size = 42 if small_sample else 13
    track_alpha = 0.58 if small_sample else 0.10
    track_width = 1.25 if small_sample else 0.7
    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")
        _plot_cluster_background(axes[0], data, alpha=0.10)
        indices = descendants.bundle_index.to_numpy(int)
        points = axes[0].scatter(xy[indices, 0], xy[indices, 1], c=delta, cmap=cmap,
                                 norm=norm, s=descendant_size, alpha=0.9,
                                 linewidths=0,
                                 rasterized=True)
        fig.colorbar(points, ax=axes[0], label="Descendant PHZ deltaMS [dex]")
        axes[0].set(xlabel="Joint-average UMAP 1", ylabel="Joint-average UMAP 2",
                    title=f"All {len(descendants):,} selected descendants")
        axes[1].scatter(descendants.log_stellar_mass, descendants.phz_delta_ms,
                        c=descendants.redshift, cmap="viridis",
                        s=descendant_size, alpha=0.85,
                        linewidths=0, rasterized=True)
        axes[1].axhline(0, color="k", ls="--", lw=1)
        axes[1].set(xlabel="PHZ log stellar mass", ylabel="PHZ deltaMS [dex]",
                    title="No descendant mass cut")
        axes[1].grid(alpha=0.15)
        fig.suptitle("Selected descendants used for progenitor-analogue tracks", fontsize=15)
        pdf.savefig(fig, dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
        _plot_cluster_background(ax, data, alpha=0.08)
        descendant_lookup = descendants.set_index("galaxy_id")
        for galaxy_id, group in checkpoints.groupby("descendant_id", sort=False):
            row = descendant_lookup.loc[int(galaxy_id)]
            ordered = group.sort_values("formed_mass_fraction")
            path_x = np.r_[ordered.analogue_centroid_x.to_numpy(float), row.joint_umap_x]
            path_y = np.r_[ordered.analogue_centroid_y.to_numpy(float), row.joint_umap_y]
            ax.plot(
                path_x,
                path_y,
                color=cmap(norm(row.phz_delta_ms)),
                alpha=track_alpha,
                lw=track_width,
            )
        ax.set(xlabel="Joint-average UMAP 1", ylabel="Joint-average UMAP 2",
               title="All analogue-centroid tracks\nColor encodes descendant PHZ deltaMS")
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(scalar, ax=ax, label="Descendant PHZ deltaMS [dex]")
        pdf.savefig(fig, dpi=180)
        plt.close(fig)

        grouped = checkpoints.groupby("formed_mass_fraction")
        fractions = np.array(sorted(grouped.groups, reverse=True))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
        for ax, column, ylabel in [
            (axes[0], "state_lookback_gyr", "Descendant lookback [Gyr]"),
            (axes[1], "median_cumulative_sfh_distance", "Median top-5 SFH distance"),
        ]:
            values = [checkpoints.loc[grouped.groups[f], column].dropna().to_numpy() for f in fractions]
            med = np.array([np.median(v) for v in values])
            p16 = np.array([np.percentile(v, 16) for v in values])
            p84 = np.array([np.percentile(v, 84) for v in values])
            ax.plot(fractions, med, marker="o", color="#355f8d")
            ax.fill_between(fractions, p16, p84, color="#355f8d", alpha=0.2)
            ax.set(xlabel="Formed-mass fraction f", ylabel=ylabel)
            ax.invert_xaxis()
            ax.grid(alpha=0.15)
        axes[0].set_title("Checkpoint timing across all descendants")
        axes[1].set_title("Analogue-match quality across all descendants")
        fig.suptitle("Median and 16th-84th percentiles; every available selected track", fontsize=14)
        pdf.savefig(fig, dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
        fraction_norm = Normalize(
            vmin=checkpoints.formed_mass_fraction.min(),
            vmax=checkpoints.formed_mass_fraction.max(),
        )
        fraction_cmap = plt.get_cmap("viridis")
        for fraction, group in checkpoints.groupby("formed_mass_fraction"):
            axes[0].scatter(
                group.predicted_log_stellar_mass,
                group.median_cumulative_sfh_distance,
                s=5, alpha=0.22, color=fraction_cmap(fraction_norm(fraction)),
                rasterized=True,
            )
        fraction_scalar = plt.cm.ScalarMappable(norm=fraction_norm, cmap=fraction_cmap)
        fig.colorbar(fraction_scalar, ax=axes[0], label="Formed-mass fraction f")
        axes[0].set(xlabel="Predicted log stellar mass", ylabel="Median top-5 SFH distance",
                    title="Match quality versus inferred progenitor mass")
        axes[0].grid(alpha=0.15)
        limited = checkpoints.shape_comparison_limited.astype(bool)
        axes[1].hist(checkpoints.comparison_history_gyr[~limited], bins=35,
                     alpha=0.7, label="adequate", color="#2c7fb8")
        axes[1].hist(checkpoints.comparison_history_gyr[limited], bins=35,
                     alpha=0.8, label="limited (<0.5 Gyr)", color="#d95f0e")
        axes[1].set(xlabel="Available comparison history [Gyr]", ylabel="Checkpoints",
                    title="History available for SFH matching")
        axes[1].set_yscale("log")
        axes[1].legend()
        fig.suptitle("Completeness and reliability diagnostics", fontsize=14)
        pdf.savefig(fig, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("euclid_selected_galaxies.csv"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path,
                        default=Path("euclid/diagnostics/all_selected_progenitor_tracks"))
    parser.add_argument("--pdf", type=Path,
                        default=Path("output/pdf/euclid_all_selected_progenitor_tracks.pdf"))
    parser.add_argument("--minimum-progenitor-mass", type=float, default=9.0)
    parser.add_argument("--mass-tolerance", type=float, default=0.15)
    parser.add_argument("--history-gyr", type=float, default=2.0)
    parser.add_argument("--n-analogues", type=int, default=5)
    parser.add_argument("--curve-points", type=int, default=128)
    parser.add_argument("--ms-sfr-offset", type=float, default=-0.93)
    parser.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS)
    args = parser.parse_args()

    selected = pd.read_csv(args.selection, dtype={"galaxy_id": str})
    if selected.galaxy_id.duplicated().any():
        raise ValueError("Selection contains duplicate galaxy IDs")
    data = _load_inputs(args.bundle, args.archive, args.catalog, args.ms_sfr_offset)
    ids = np.asarray(data["ids"])
    lookup = {str(int(g)): i for i, g in enumerate(ids)}
    missing = [g for g in selected.galaxy_id if g not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} selected IDs are absent from the explorer bundle")
    indices = np.array([lookup[g] for g in selected.galaxy_id], dtype=int)
    stamp_mask = np.array(
        [(args.bundle / "VIS" / f"VIS_{g}.jpg").is_file() for g in ids], dtype=bool
    )
    if not np.all(stamp_mask[indices]):
        raise ValueError("At least one selected descendant is missing its VIS stamp")

    descendant_rows = []
    checkpoint_frames = []
    candidate_frames = []
    for number, index in enumerate(indices, start=1):
        descendant_mass = float(np.asarray(data["mass"])[index])
        fractions = [
            f for f in args.fractions
            if descendant_mass + np.log10(f) >= args.minimum_progenitor_mass - 0.01
        ]
        candidates, census, _ = find_analogues(
            data,
            int(index),
            fractions,
            mass_tolerance=args.mass_tolerance,
            history_gyr=args.history_gyr,
            minimum_comparison_gyr=0.5,
            n_analogues=args.n_analogues,
            stamp_mask=stamp_mask,
            n_curve_points=args.curve_points,
        )
        descendant_id = int(ids[index])
        if not candidates.empty:
            candidates.insert(0, "descendant_id", descendant_id)
            candidates.insert(1, "descendant_bundle_index", int(index))
            candidates.insert(2, "descendant_log_stellar_mass", descendant_mass)
            candidate_frames.append(candidates)
            census.insert(0, "descendant_id", descendant_id)
            census.insert(1, "descendant_bundle_index", int(index))
            for stage, group in candidates.groupby("stage"):
                candidate_indices = group.bundle_index.to_numpy(int)
                mask = census.stage == stage
                census.loc[mask, "analogue_centroid_x"] = np.median(
                    np.asarray(data["xy_joint"])[candidate_indices, 0]
                )
                census.loc[mask, "analogue_centroid_y"] = np.median(
                    np.asarray(data["xy_joint"])[candidate_indices, 1]
                )
                census.loc[mask, "median_cumulative_sfh_distance"] = np.median(
                    group.cumulative_sfh_distance
                )
                census.loc[mask, "best_cumulative_sfh_distance"] = np.min(
                    group.cumulative_sfh_distance
                )
            checkpoint_frames.append(census)
        descendant_rows.append(
            {
                "galaxy_id": descendant_id,
                "bundle_index": int(index),
                "log_stellar_mass": descendant_mass,
                "redshift": float(np.asarray(data["redshift"])[index]),
                "phz_log_ssfr": float(np.asarray(data["catalog_log_ssfr"])[index]),
                "phz_delta_ms": float(np.asarray(data["catalog_delta_ms"])[index]),
                "joint_umap_x": float(np.asarray(data["xy_joint"])[index, 0]),
                "joint_umap_y": float(np.asarray(data["xy_joint"])[index, 1]),
                "n_checkpoints": len(census),
            }
        )
        if number % 25 == 0 or number == len(indices):
            print(f"Processed {number:,}/{len(indices):,} descendants", flush=True)

    descendants = pd.DataFrame(descendant_rows)
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    descendants.to_csv(args.output / "descendants.csv", index=False)
    checkpoints.to_csv(args.output / "progenitor_checkpoints.csv", index=False)
    candidates.to_csv(args.output / "analogue_candidates.csv", index=False)
    manifest = {
        "selection": str(args.selection),
        "n_selected_descendants": len(descendants),
        "descendant_mass_cut": None,
        "minimum_descendant_mass": float(descendants.log_stellar_mass.min()),
        "maximum_descendant_mass": float(descendants.log_stellar_mass.max()),
        "minimum_progenitor_mass": args.minimum_progenitor_mass,
        "fractions": args.fractions,
        "n_checkpoints": len(checkpoints),
        "n_ranked_analogues": len(candidates),
        "n_analogues_per_checkpoint": args.n_analogues,
        "mass_tolerance_dex": args.mass_tolerance,
        "history_gyr": args.history_gyr,
        "curve_points": args.curve_points,
        "selection_uses": ["predicted stellar mass", "renormalized cumulative SFH"],
        "selection_does_not_use": ["redshift", "morphology", "UMAP position", "sSFR"],
        "track_visualization_space": "xy_joint (normalized average of aligned image and SFH embeddings)",
        "summary_pdf": str(args.pdf),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _population_report(data, descendants, checkpoints, args.pdf)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
