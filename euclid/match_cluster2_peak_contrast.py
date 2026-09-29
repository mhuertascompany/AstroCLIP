"""Match cluster 2 to cluster 1 while maximizing smoothed SFH-peak contrast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from .match_cluster3_main_sequence import _cluster_labels, _read_rows
from .sfh_shape import sfh_recent_activity


MORPHOLOGY = {
    "Featured / disk probability": "zoobot_featured_conditional_fraction",
    "Spiral-arm probability": "zoobot_spiral_probability",
    "Merger / disturbed probability": "zoobot_merger_probability",
}


def _smoothed_peak_age(weights, time, age_gyr, smoothing_gyr):
    """Return the peak time after Gaussian smoothing in physical time."""
    step_fraction = float(np.median(np.diff(time)))
    result = np.full(len(weights), np.nan)
    for row in range(len(weights)):
        physical_step = step_fraction * age_gyr[row]
        if not np.isfinite(physical_step) or physical_step <= 0:
            continue
        sigma_bins = max(smoothing_gyr / physical_step, 0.5)
        smooth = gaussian_filter1d(weights[row], sigma=sigma_bins, mode="nearest")
        result[row] = time[int(np.nanargmax(smooth))] * age_gyr[row]
    return result


def run(args):
    with np.load(args.archive, allow_pickle=False) as archive:
        ids = np.asarray(archive["galaxy_id"], dtype=np.int64)
        mass = np.asarray(archive["log_stellar_mass"], dtype=float)
        redshift = np.asarray(archive["redshift"], dtype=float)
        labels = _cluster_labels(
            np.asarray(archive["sfh_embedding"], dtype=float),
            np.asarray(archive["sfh_mean_lookback"], dtype=float),
        )
        morphology = {
            label: np.asarray(archive[key], dtype=float)
            for label, key in MORPHOLOGY.items()
        }

    with h5py.File(args.bundle / "euclid_explorer.h5", "r") as source:
        source_ids = np.asarray(source["galaxy_id"][:], dtype=np.int64)
        lookup = {int(value): row for row, value in enumerate(source_ids)}
        rows = np.array([lookup[int(value)] for value in ids], dtype=np.int64)
        sfh_log = _read_rows(source["sfh"], rows).astype(np.float32)
        time = np.asarray(source["sfh_time_grid"][:], dtype=float)
        age_gyr = _read_rows(source["sfh_time_norm"], rows).astype(float) / 1000.0
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))

    weights = np.maximum(10.0 ** sfh_log.astype(float) - epsilon, 0.0)
    weights /= np.where(weights.sum(axis=1) > 0, weights.sum(axis=1), np.nan)[:, None]
    activity = sfh_recent_activity(sfh_log, time, epsilon, age_gyr * 1000.0)
    log_sfr = mass + np.asarray(
        activity["sfh_log_sfr_per_stellar_mass_100myr_r0"], dtype=float
    )
    peak_age = _smoothed_peak_age(weights, time, age_gyr, args.peak_smoothing_gyr)

    finite = np.isfinite(log_sfr + mass + redshift + peak_age)
    target = np.flatnonzero(
        finite & (labels == 2)
        & (mass >= args.mass_range[0]) & (mass <= args.mass_range[1])
    )
    controls = np.flatnonzero(finite & (labels == 1))
    features = np.c_[log_sfr, mass, redshift]
    scale = np.asarray(args.tolerances, dtype=float)
    tree = cKDTree(features[controls] / scale)
    neighbours = tree.query_ball_point(features[target] / scale, r=1.0, p=np.inf, workers=1)

    matched_target = []
    matched_control = []
    candidate_counts = []
    for target_index, local_candidates in zip(target, neighbours):
        if not local_candidates:
            continue
        candidate_indices = controls[np.asarray(local_candidates, dtype=int)]
        separation = np.abs(peak_age[candidate_indices] - peak_age[target_index])
        # Break exact peak-separation ties using the closest physical match.
        residual = np.sum(
            ((features[candidate_indices] - features[target_index]) / scale) ** 2, axis=1
        )
        order = np.lexsort((residual, -separation))
        matched_target.append(target_index)
        matched_control.append(int(candidate_indices[order[0]]))
        candidate_counts.append(len(candidate_indices))

    matched_target = np.asarray(matched_target, dtype=int)
    matched_control = np.asarray(matched_control, dtype=int)
    candidate_counts = np.asarray(candidate_counts, dtype=int)
    records = []
    for pair, (target_index, control_index, n_candidates) in enumerate(
        zip(matched_target, matched_control, candidate_counts), start=1
    ):
        for role, index in (("cluster2", target_index), ("cluster1_match", control_index)):
            record = {
                "pair": pair,
                "role": role,
                "galaxy_id": str(int(ids[index])),
                "cluster": int(labels[index]),
                "log_stellar_mass": float(mass[index]),
                "redshift": float(redshift[index]),
                "sfh_log_sfr_100myr_r0": float(log_sfr[index]),
                "smoothed_sfh_peak_age_gyr": float(peak_age[index]),
                "n_valid_cluster1_candidates": int(n_candidates),
            }
            for label, values in morphology.items():
                record[label.lower().replace(" / ", "_").replace(" ", "_")] = float(
                    values[index]
                )
            records.append(record)
    table = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    targets = table[table.role == "cluster2"].reset_index(drop=True)
    matches = table[table.role == "cluster1_match"].reset_index(drop=True)
    residuals = {
        "log_sfr": np.abs(
            targets.sfh_log_sfr_100myr_r0.to_numpy()
            - matches.sfh_log_sfr_100myr_r0.to_numpy()
        ),
        "log_mass": np.abs(
            targets.log_stellar_mass.to_numpy() - matches.log_stellar_mass.to_numpy()
        ),
        "redshift": np.abs(targets.redshift.to_numpy() - matches.redshift.to_numpy()),
    }
    peak_separation = np.abs(
        targets.smoothed_sfh_peak_age_gyr.to_numpy()
        - matches.smoothed_sfh_peak_age_gyr.to_numpy()
    )
    # The full target catalogue uses matching with replacement.  For a clear
    # distribution plot, retain one target-control pair per distinct control.
    # If a control was selected repeatedly, keep its largest peak-age contrast.
    plot_rows = (
        pd.DataFrame({
            "row": np.arange(len(matches)),
            "control_id": matches.galaxy_id.to_numpy(),
            "peak_separation": peak_separation,
        })
        .sort_values("peak_separation", ascending=False)
        .drop_duplicates("control_id")
        .sort_values("row")
        .row.to_numpy(dtype=int)
    )
    plot_targets = targets.iloc[plot_rows].reset_index(drop=True)
    plot_matches = matches.iloc[plot_rows].reset_index(drop=True)
    plot_sample = pd.concat([plot_targets, plot_matches], ignore_index=True)
    plot_sample.to_csv(
        args.output.with_name(args.output.stem + "_unique_plot_sample.csv"), index=False
    )
    report = {
        "n_cluster2_mass_selected": int(len(target)),
        "n_matched": int(len(matched_target)),
        "matched_fraction": float(len(matched_target) / len(target)),
        "n_cluster1_controls": int(len(controls)),
        "n_unique_cluster1_matches": int(len(np.unique(matched_control))),
        "n_unique_pairs_in_figure": int(len(plot_rows)),
        "duplicate_control_policy_in_figure": (
            "one pair per cluster-1 control; retain largest smoothed peak-age separation"
        ),
        "matching_with_replacement": True,
        "mass_range": args.mass_range,
        "tolerances": {
            "log_sfr_100myr": args.tolerances[0],
            "log_stellar_mass": args.tolerances[1],
            "redshift": args.tolerances[2],
        },
        "sfh_sfr": "normalized mass formed in last 100 Myr times observed Mstar / 1e8 yr; R=0",
        "peak_smoothing_gyr": args.peak_smoothing_gyr,
        "selection_objective": "largest absolute smoothed SFH peak-age difference among valid controls",
        "residuals": {
            name: {
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "maximum": float(np.max(values)),
            }
            for name, values in residuals.items()
        },
        "peak_age_separation_gyr": {
            "median": float(np.median(peak_separation)),
            "p16": float(np.percentile(peak_separation, 16)),
            "p84": float(np.percentile(peak_separation, 84)),
            "maximum": float(np.max(peak_separation)),
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.2), layout="constrained",
                             sharex=True, sharey=True)
    row_samples = (("Cluster 2", plot_targets), ("Matched cluster 1", plot_matches))
    for column_index, (title, field) in enumerate(MORPHOLOGY.items()):
        column = title.lower().replace(" / ", "_").replace(" ", "_")
        combined = np.r_[plot_targets[column].to_numpy(), plot_matches[column].to_numpy()]
        valid = combined[np.isfinite(combined)]
        low, high = np.percentile(valid, [2, 98]) if len(valid) else (0.0, 1.0)
        low, high = max(0.0, low), min(1.0, high)
        if high <= low:
            low, high = 0.0, 1.0
        scatter = None
        for row_index, (label, sample) in enumerate(row_samples):
            ax = axes[row_index, column_index]
            scatter = ax.scatter(
                sample.log_stellar_mass, sample.sfh_log_sfr_100myr_r0,
                c=sample[column], cmap="viridis", vmin=low, vmax=high,
                marker="o", s=22, alpha=0.72,
                linewidths=0, rasterized=True,
            )
            ax.grid(alpha=0.12)
            if row_index == 0:
                ax.set_title(title)
            if column_index == 0:
                ax.set_ylabel(
                    f"{label}\nSFH $\log_{{10}}(\mathrm{{SFR}}_{{100}}/"
                    r"M_\odot\,\mathrm{yr}^{-1})$, R=0"
                )
            if row_index == 1:
                ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        fig.colorbar(scatter, ax=axes[:, column_index], label=title, shrink=0.92)
    fig.suptitle(
        f"{len(plot_rows):,} distinct cluster-2 / cluster-1 pairs matched in mass, "
        "SFH SFR100, and redshift\n"
        f"Cluster-1 match maximizes the SFH peak-time difference after "
        f"{args.peak_smoothing_gyr * 1000:.0f} Myr smoothing; one pair per unique control",
        fontsize=14,
    )
    fig.savefig(args.pdf, dpi=180)
    fig.savefig(args.pdf.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--mass-range", type=float, nargs=2, default=[10.0, 11.0])
    parser.add_argument("--tolerances", type=float, nargs=3, default=[0.20, 0.12, 0.10],
                        metavar=("LOGSFR", "LOGMASS", "Z"))
    parser.add_argument("--peak-smoothing-gyr", type=float, default=0.30)
    args = parser.parse_args()
    if args.peak_smoothing_gyr <= 0 or any(value <= 0 for value in args.tolerances):
        parser.error("Smoothing scale and matching tolerances must be positive")
    run(args)


if __name__ == "__main__":
    main()
