"""Plot independent-donor past-shape ablations before model training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .plot_null_sfh_resampling_examples import linear_sfh
from .sfh_surrogates import recent_preserved_independent_past


def diverse_rows(weights: np.ndarray, time: np.ndarray, count: int) -> np.ndarray:
    recent = time <= 0.1
    recent_mass = weights[:, recent].sum(axis=1)
    selected = []
    # Span the recent-fraction distribution directly.  This includes one
    # effectively quenched boundary and avoids filling the figure with the
    # many exactly-zero cases in the catalogue.
    for value in np.quantile(recent_mass, np.linspace(0.05, 0.95, count)):
        order = np.argsort(np.abs(recent_mass - value))
        selected.append(next(int(row) for row in order if int(row) not in selected))
    return np.asarray(selected, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recent-fraction", type=float, default=0.1)
    parser.add_argument("--transition-bins", type=int, default=8)
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as source:
        ids = np.asarray(source["galaxy_id"], dtype=np.int64)
        time = np.asarray(source["sfh_time_grid"], dtype=np.float64)
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))
        weights = linear_sfh(np.asarray(source["sfh"]), epsilon)
    targets = diverse_rows(weights, time, args.n_examples)
    rng = np.random.default_rng(args.seed)
    donors = np.empty(len(targets), dtype=np.int64)
    smooth = np.empty_like(weights[targets])
    raw = np.empty_like(smooth)
    diagnostics = []
    for position, target in enumerate(targets):
        while True:
            donor = int(rng.integers(len(weights)))
            if donor == target:
                continue
            try:
                curve, raw_curve, info = recent_preserved_independent_past(
                    weights[target], weights[donor], time,
                    recent_fraction=args.recent_fraction,
                    transition_bins=args.transition_bins,
                )
            except ValueError:
                continue
            donors[position] = donor
            smooth[position] = curve
            raw[position] = raw_curve
            diagnostics.append(info)
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(targets), 2, figsize=(12, 2.55 * len(targets)),
        gridspec_kw={"width_ratios": (2.25, 1)}, squeeze=False,
    )
    for row, target in enumerate(targets):
        info = diagnostics[row]
        start = info["old_start"]
        for axis in axes[row]:
            axis.plot(time, weights[target], color="black", lw=2.0,
                      label="original target")
            axis.plot(time, raw[row], color="#d95f02", lw=1.15, ls=":",
                      label="raw independent donor splice")
            axis.plot(time, smooth[row], color="#1b9e77", lw=1.7,
                      label="smoothed donor past")
            axis.axvline(args.recent_fraction, color="0.45", ls="--", lw=1)
            axis.axvspan(0, args.recent_fraction, color="#d8f0e5", alpha=.35)
            axis.set_ylim(bottom=0)
            axis.grid(alpha=.18)
        axes[row, 0].set_ylabel("Fractional SFR per bin")
        axes[row, 0].set_title(
            f"Target {ids[target]}  |  donor {ids[donors[row]]}  |  "
            f"$f_R$={info['recent_sum']:.3f}", loc="left", fontsize=10,
        )
        width = info["transition_bins"]
        left = max(0, start - 5)
        right = min(len(time) - 1, start + width + 5)
        axes[row, 1].set_xlim(time[left], time[right])
        local_max = max(
            weights[target, left:right + 1].max(),
            raw[row, left:right + 1].max(),
            smooth[row, left:right + 1].max(),
        )
        axes[row, 1].set_ylim(0, 1.12 * local_max if local_max > 0 else 1)
        axes[row, 1].set_title(
            f"Boundary zoom ({width}-bin bridge)", fontsize=10,
        )
    axes[-1, 0].set_xlabel("Fractional lookback time")
    axes[-1, 1].set_xlabel("Fractional lookback time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, .976))
    fig.suptitle(
        "Experiment 2 prototype: recent SFH preserved; independent past shape",
        y=.997, fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, .94))
    png = args.output.with_suffix(".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    with PdfPages(args.output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    table = args.output.with_suffix(".csv")
    with table.open("w", newline="") as stream:
        fields = [
            "target_row", "target_id", "donor_row", "donor_id",
            *diagnostics[0].keys(),
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for target, donor, info in zip(targets, donors, diagnostics):
            writer.writerow({
                "target_row": int(target), "target_id": int(ids[target]),
                "donor_row": int(donor), "donor_id": int(ids[donor]), **info,
            })
    np.savez_compressed(
        args.output.with_suffix(".npz"), time=time.astype(np.float32),
        target_rows=targets, target_ids=ids[targets], donor_rows=donors,
        donor_ids=ids[donors], original=weights[targets].astype(np.float32),
        raw_splice=raw.astype(np.float32), smooth_ablation=smooth.astype(np.float32),
        recent_fraction=np.float32(args.recent_fraction),
        transition_bins=np.int32(args.transition_bins),
    )
    print(f"Saved {args.output}")
    print(f"Saved {png}")
    print(f"Saved {table}")
    print(f"Maximum |integral-1|: {np.max(np.abs(smooth.sum(axis=1) - 1)):.3g}")
    print(
        "Maximum recent-segment error: "
        f"{max(item['recent_max_error'] for item in diagnostics):.3g}"
    )


if __name__ == "__main__":
    main()
