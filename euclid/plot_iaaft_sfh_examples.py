"""Plot random Euclid SFHs with recent-preserved IAAFT surrogates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .sfh_surrogates import recent_preserved_iaaft


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--n-surrogates", type=int, default=3)
    parser.add_argument("--recent-fraction", type=float, default=0.1)
    parser.add_argument("--transition-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = _arguments()
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.h5, "r") as source:
        log_sfh = np.asarray(source["sfh"], dtype=np.float64)
        time = np.asarray(source["sfh_time_grid"], dtype=np.float64)
        galaxy_id = np.asarray(source["galaxy_id"], dtype=np.int64)
        redshift = np.asarray(source["redshift"], dtype=np.float64)
        age_myr = np.asarray(source["sfh_time_norm"], dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        weights = np.maximum(10.0 ** log_sfh - 1e-10, 0.0)
    totals = weights.sum(axis=1)
    valid = np.flatnonzero(
        np.all(np.isfinite(weights), axis=1)
        & np.isfinite(totals) & (totals > 0)
        & np.isfinite(redshift) & np.isfinite(age_myr) & (age_myr > 0)
    )
    if len(valid) < args.n_examples:
        raise ValueError(f"Only {len(valid)} valid histories for {args.n_examples} examples")
    selected = rng.choice(valid, size=args.n_examples, replace=False)

    n_columns = 2
    n_rows = int(np.ceil(args.n_examples / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(12.0, 3.0 * n_rows), sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.plasma(np.linspace(0.18, 0.78, args.n_surrogates))
    records = []

    for panel, row in enumerate(selected):
        ax = axes[panel]
        original = weights[row] / totals[row]
        ax.axvspan(0, args.recent_fraction, color="#d7f0e4", alpha=0.75,
                   label="preserved recent 10%" if panel == 0 else None)
        ax.plot(time, original, color="black", lw=2.0,
                label="original" if panel == 0 else None, zorder=5)
        correlations = []
        for number, color in enumerate(colors, start=1):
            surrogate, diagnostic = recent_preserved_iaaft(
                original, time, rng,
                recent_fraction=args.recent_fraction,
                transition_bins=args.transition_bins,
            )
            correlations.append(diagnostic["old_correlation"])
            record = {
                "galaxy_id": int(galaxy_id[row]),
                "h5_row": int(row),
                "surrogate": number,
                **diagnostic,
            }
            records.append(record)
            ax.plot(time, surrogate, color=color, lw=1.1, alpha=0.88,
                    label="IAAFT surrogates" if panel == 0 and number == 1 else None)

        ax.axvline(args.recent_fraction, color="#16845b", lw=1.0, ls="--")
        physical_recent = args.recent_fraction * age_myr[row] / 1e3
        ax.set_title(
            f"ID {galaxy_id[row]}  |  z={redshift[row]:.2f}  |  "
            f"0.1 age={physical_recent:.2f} Gyr\n"
            f"old-shape correlation: {np.min(correlations):.2f} to {np.max(correlations):.2f}",
            fontsize=9,
        )
        ax.grid(alpha=0.18, lw=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        if panel % n_columns == 0:
            ax.set_ylabel("Normalized SFH bin weight")
        if panel >= (n_rows - 1) * n_columns:
            ax.set_xlabel("Fractional lookback time")

    for ax in axes[args.n_examples:]:
        ax.set_visible(False)
    axes[0].legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)
    fig.suptitle(
        "Recent-preserved IAAFT SFH surrogates\n"
        "The latest 10% is exact; the older value distribution and integral are preserved",
        fontsize=14, y=0.992,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=1.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_h5": str(args.h5),
            "seed": args.seed,
            "recent_fraction": args.recent_fraction,
            "transition_bins_requested": args.transition_bins,
            "n_examples": args.n_examples,
            "n_surrogates_per_example": args.n_surrogates,
            "selected_galaxy_ids": [int(galaxy_id[row]) for row in selected],
            "maximum_absolute_old_integral_error": float(max(abs(x["old_integral_error"]) for x in records)),
            "old_correlation_median": float(np.nanmedian([x["old_correlation"] for x in records])),
            "old_correlation_range": [
                float(np.nanmin([x["old_correlation"] for x in records])),
                float(np.nanmax([x["old_correlation"] for x in records])),
            ],
            "per_surrogate": records,
        }
        args.report.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
