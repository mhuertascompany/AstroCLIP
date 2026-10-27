"""Visualize the zero-information SFH component-resampling null.

Each synthetic SFH combines three independently sampled galaxies: one supplies
the recent mass fraction, one the recent shape, and one the past shape.  No
component is taken from the target galaxy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def linear_sfh(log_sfh: np.ndarray, epsilon: float) -> np.ndarray:
    weights = np.maximum(10.0 ** np.asarray(log_sfh, dtype=np.float64) - epsilon, 0.0)
    totals = weights.sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("Every SFH must have a finite, positive integral.")
    return weights / totals


def decompose_sfh(weights: np.ndarray, time: np.ndarray, recent_limit: float):
    recent = np.asarray(time) <= recent_limit
    if recent.sum() < 1 or (~recent).sum() < 1:
        raise ValueError("The recent boundary must leave bins on both sides.")
    recent_mass = weights[:, recent].sum(axis=1)
    past_mass = weights[:, ~recent].sum(axis=1)
    recent_shape = np.zeros((len(weights), recent.sum()), dtype=np.float64)
    past_shape = np.zeros((len(weights), (~recent).sum()), dtype=np.float64)
    valid_recent = recent_mass > np.finfo(np.float64).eps
    valid_past = past_mass > np.finfo(np.float64).eps
    recent_shape[valid_recent] = weights[valid_recent][:, recent] / recent_mass[valid_recent, None]
    past_shape[valid_past] = weights[valid_past][:, ~recent] / past_mass[valid_past, None]
    return recent, recent_mass, recent_shape, past_shape, valid_recent, valid_past


def _draw_one(pool: np.ndarray, forbidden: set[int], rng: np.random.Generator) -> int:
    if len(pool) <= len(forbidden):
        available = np.asarray([value for value in pool if int(value) not in forbidden])
        if not len(available):
            raise ValueError("No eligible null donor remains after exclusions.")
        return int(rng.choice(available))
    while True:
        value = int(pool[rng.integers(len(pool))])
        if value not in forbidden:
            return value


def make_null_draws(
    weights: np.ndarray,
    time: np.ndarray,
    target_rows: np.ndarray,
    *,
    recent_limit: float = 0.1,
    draws_per_target: int = 4,
    seed: int = 42,
):
    """Return null SFHs and donor rows for the requested target rows."""
    (recent, recent_mass, recent_shape, past_shape,
     valid_recent, valid_past) = decompose_sfh(weights, time, recent_limit)
    rng = np.random.default_rng(seed)
    all_rows = np.arange(len(weights), dtype=np.int64)
    recent_pool = all_rows[valid_recent]
    past_pool = all_rows[valid_past]
    curves = np.empty((len(target_rows), draws_per_target, weights.shape[1]), dtype=np.float64)
    donors = np.empty((len(target_rows), draws_per_target, 3), dtype=np.int64)
    for target_position, target in enumerate(np.asarray(target_rows, dtype=np.int64)):
        for draw in range(draws_per_target):
            fraction_donor = _draw_one(all_rows, {int(target)}, rng)
            recent_donor = _draw_one(
                recent_pool, {int(target), fraction_donor}, rng,
            )
            past_donor = _draw_one(
                past_pool, {int(target), fraction_donor, recent_donor}, rng,
            )
            fraction = float(recent_mass[fraction_donor])
            curve = curves[target_position, draw]
            curve[recent] = fraction * recent_shape[recent_donor]
            curve[~recent] = (1.0 - fraction) * past_shape[past_donor]
            curve /= curve.sum()
            donors[target_position, draw] = (
                fraction_donor, recent_donor, past_donor,
            )
    return curves, donors


def make_full_donor_draws(
    weights: np.ndarray,
    target_rows: np.ndarray,
    *,
    draws_per_target: int = 4,
    seed: int = 42,
):
    """Draw complete observed SFHs independently of every target galaxy."""
    rng = np.random.default_rng(seed)
    all_rows = np.arange(len(weights), dtype=np.int64)
    curves = np.empty((len(target_rows), draws_per_target, weights.shape[1]), dtype=np.float64)
    donors = np.empty((len(target_rows), draws_per_target, 3), dtype=np.int64)
    for target_position, target in enumerate(np.asarray(target_rows, dtype=np.int64)):
        used = {int(target)}
        for draw in range(draws_per_target):
            donor = _draw_one(all_rows, used, rng)
            used.add(donor)
            curves[target_position, draw] = weights[donor]
            # Keep the same provenance schema as component resampling.  Equal
            # entries mean that the complete SFH came from this one donor.
            donors[target_position, draw] = donor
    return curves, donors


def select_diverse_targets(weights: np.ndarray, time: np.ndarray, count: int) -> np.ndarray:
    mean_lookback = weights @ np.asarray(time, dtype=np.float64)
    quantiles = np.linspace(0.05, 0.95, count)
    targets = []
    for value in np.quantile(mean_lookback, quantiles):
        order = np.argsort(np.abs(mean_lookback - value))
        targets.append(next(int(index) for index in order if int(index) not in targets))
    return np.asarray(targets, dtype=np.int64)


def write_donor_table(path: Path, ids: np.ndarray, targets: np.ndarray, donors: np.ndarray):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "target_row", "target_id", "draw", "fraction_donor_row",
            "fraction_donor_id", "recent_shape_donor_row", "recent_shape_donor_id",
            "past_shape_donor_row", "past_shape_donor_id",
        ))
        writer.writeheader()
        for target_position, target in enumerate(targets):
            for draw, (fraction, recent, past) in enumerate(donors[target_position]):
                writer.writerow({
                    "target_row": int(target), "target_id": int(ids[target]),
                    "draw": draw, "fraction_donor_row": int(fraction),
                    "fraction_donor_id": int(ids[fraction]),
                    "recent_shape_donor_row": int(recent),
                    "recent_shape_donor_id": int(ids[recent]),
                    "past_shape_donor_row": int(past),
                    "past_shape_donor_id": int(ids[past]),
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recent-limit", type=float, default=0.1)
    parser.add_argument("--n-targets", type=int, default=6)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode", choices=("full-donor", "independent-components"),
        default="full-donor",
        help=(
            "Use a complete random donor SFH for the strict chance null, or "
            "assemble fraction/recent/past components from separate donors."
        ),
    )
    args = parser.parse_args()
    if args.n_targets < 1 or args.draws < 1:
        raise ValueError("n-targets and draws must be positive.")

    with h5py.File(args.h5, "r") as source:
        ids = np.asarray(source["galaxy_id"], dtype=np.int64)
        time = np.asarray(source["sfh_time_grid"], dtype=np.float64)
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))
        weights = linear_sfh(np.asarray(source["sfh"]), epsilon)
    targets = select_diverse_targets(weights, time, args.n_targets)
    if args.mode == "full-donor":
        curves, donors = make_full_donor_draws(
            weights, targets, draws_per_target=args.draws, seed=args.seed,
        )
    else:
        curves, donors = make_null_draws(
            weights, time, targets, recent_limit=args.recent_limit,
            draws_per_target=args.draws, seed=args.seed,
        )
    integrals = curves.sum(axis=-1)
    if not np.allclose(integrals, 1.0, atol=1e-12):
        raise RuntimeError("A null SFH failed unit-integral validation.")
    for position, target in enumerate(targets):
        if np.any(donors[position] == target):
            raise RuntimeError("A target galaxy donated information to its own null SFH.")
        if (args.mode == "independent-components"
                and any(len(set(row)) != 3 for row in donors[position])):
            raise RuntimeError("Null components must use three distinct donors.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, args.draws))
    ncols = 2
    nrows = int(np.ceil(args.n_targets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.15 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for position, (axis, target) in enumerate(zip(axes, targets)):
        axis.plot(time, weights[target], color="black", lw=2.2,
                  label="Original target")
        for draw, color in enumerate(colors):
            axis.plot(time, curves[position, draw], color=color,
                      lw=1.15, alpha=0.82, label=f"Null draw {draw + 1}")
        recent_fraction = weights[target, time <= args.recent_limit].sum()
        mean_time = float(weights[target] @ time)
        axis.axvline(args.recent_limit, color="0.45", ls="--", lw=1)
        axis.set_title(
            f"ID {ids[target]}   original $f_R$={recent_fraction:.3f}, "
            f"$\\langle t\\rangle$={mean_time:.2f}"
        )
        axis.grid(alpha=0.18)
        axis.set_ylim(bottom=0)
        axis.set_ylabel("Fractional SFR per bin")
    for axis in axes[args.n_targets:]:
        axis.set_visible(False)
    for axis in axes[-ncols:]:
        if axis.get_visible():
            axis.set_xlabel("Fractional lookback time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6),
               frameon=False, bbox_to_anchor=(0.5, 1.005))
    title = (
        "Zero-information null: complete real SFH resampled independently of the image"
        if args.mode == "full-donor" else
        "Zero-information null: recent fraction, recent shape, and past shape "
        "come from independent donors"
    )
    fig.suptitle(title, y=1.035, fontsize=13)
    fig.tight_layout()
    png = args.output.with_suffix(".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    with PdfPages(args.output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    table = args.output.with_suffix(".csv")
    write_donor_table(table, ids, targets, donors)
    arrays = args.output.with_suffix(".npz")
    np.savez_compressed(
        arrays, galaxy_id=ids, time=time.astype(np.float32), target_rows=targets,
        target_ids=ids[targets], original=weights[targets].astype(np.float32),
        null_draws=curves.astype(np.float32), donor_rows=donors,
        donor_ids=ids[donors], recent_limit=np.float32(args.recent_limit),
        mode=np.asarray(args.mode),
    )
    print(f"Saved {args.output}")
    print(f"Saved {png}")
    print(f"Saved {table}")
    print(f"Saved {arrays}")
    print(f"Maximum |integral-1|: {np.max(np.abs(integrals - 1.0)):.3g}")


if __name__ == "__main__":
    main()
