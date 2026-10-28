"""Build SFHs whose recent fraction and two shapes come from independent donors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .sfh_surrogates import factorized_independent_sfh


def _draw_distinct_donors(n_rows, recent_pool, past_pool, rng):
    """Draw three donors per target, distinct from one another and the target."""
    target = np.arange(n_rows, dtype=np.int64)
    fraction = rng.integers(0, n_rows, size=n_rows, dtype=np.int64)
    while np.any(fraction == target):
        mask = fraction == target
        fraction[mask] = rng.integers(0, n_rows, size=int(mask.sum()))

    recent = rng.choice(recent_pool, size=n_rows, replace=True).astype(np.int64)
    while np.any((recent == target) | (recent == fraction)):
        mask = (recent == target) | (recent == fraction)
        recent[mask] = rng.choice(recent_pool, size=int(mask.sum()), replace=True)

    past = rng.choice(past_pool, size=n_rows, replace=True).astype(np.int64)
    while np.any((past == target) | (past == fraction) | (past == recent)):
        mask = (past == target) | (past == fraction) | (past == recent)
        past[mask] = rng.choice(past_pool, size=int(mask.sum()), replace=True)
    return fraction, recent, past


def _safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _t50(weights, time):
    indices = np.argmax(np.cumsum(weights, axis=1) >= 0.5, axis=1)
    return time[indices]


def build_dataset(
    source_path,
    output_path,
    *,
    seed=42,
    recent_fraction=0.1,
    transition_bins=8,
    integral_tolerance=2e-6,
    chunk_size=512,
    max_rows=None,
):
    source_path = Path(source_path)
    output_path = Path(output_path)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    if output_path.exists() or partial.exists():
        raise FileExistsError(f"Output or partial output already exists: {output_path}")
    if not 0 < recent_fraction < 1 or transition_bins < 1 or chunk_size < 1:
        raise ValueError("invalid recent fraction, transition width, or chunk size")

    with h5py.File(source_path, "r") as source:
        for name in ("galaxy_id", "sfh", "sfh_time_grid"):
            if name not in source:
                raise ValueError(f"Missing source dataset: {name}")
        n_total, n_bins = source["sfh"].shape
        n_rows = n_total if max_rows is None else min(int(max_rows), n_total)
        ids = np.asarray(source["galaxy_id"][:n_rows], dtype=np.int64)
        time = np.asarray(source["sfh_time_grid"], dtype=np.float64)
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))
        log_sfh = np.asarray(source["sfh"][:n_rows], dtype=np.float32)

    weights = np.maximum(10.0 ** log_sfh.astype(np.float64) - epsilon, 0.0)
    totals = weights.sum(axis=1)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        bad = int(np.flatnonzero((~np.isfinite(totals)) | (totals <= 0))[0])
        raise ValueError(f"Invalid source SFH integral at row {bad}: {totals[bad]}")
    weights /= totals[:, None]
    split = int(np.searchsorted(time, recent_fraction, side="right"))
    if split < 2 or n_bins - split < 2:
        raise ValueError("recent_fraction leaves too few bins in one interval")
    recent_mass = weights[:, :split].sum(axis=1)
    past_mass = weights[:, split:].sum(axis=1)
    recent_pool = np.flatnonzero(recent_mass > np.finfo(float).eps)
    past_pool = np.flatnonzero(past_mass > np.finfo(float).eps)
    if len(recent_pool) < 4 or len(past_pool) < 4:
        raise ValueError("Too few valid component donors")

    rng = np.random.default_rng(seed)
    fraction_donor, recent_donor, past_donor = _draw_distinct_donors(
        n_rows, recent_pool, past_pool, rng,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    maximum_error = 0.0
    null_recent_fraction = np.empty(n_rows, dtype=np.float64)
    null_mean_lookback = np.empty(n_rows, dtype=np.float64)
    bridge_width = np.empty(n_rows, dtype=np.int16)
    with h5py.File(partial, "w") as target:
        target.create_dataset("galaxy_id", data=ids)
        target.create_dataset("source_row", data=np.arange(n_rows, dtype=np.int64))
        target.create_dataset("sfh_time_grid", data=time.astype(np.float32))
        for name, values in (
            ("fraction_donor_row", fraction_donor),
            ("recent_shape_donor_row", recent_donor),
            ("past_shape_donor_row", past_donor),
        ):
            target.create_dataset(name, data=values)
            target.create_dataset(name.replace("_row", "_id"), data=ids[values])
        sfh_out = target.create_dataset(
            "sfh", shape=(n_rows, n_bins), dtype=np.float32,
            chunks=(min(chunk_size, n_rows), n_bins), compression="gzip",
            compression_opts=4, shuffle=True,
        )

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            stored_chunk = np.empty((stop - start, n_bins), dtype=np.float32)
            for offset, row in enumerate(range(start, stop)):
                generated, _, info = factorized_independent_sfh(
                    weights[fraction_donor[row]],
                    weights[recent_donor[row]],
                    weights[past_donor[row]],
                    time,
                    recent_fraction=recent_fraction,
                    transition_bins=transition_bins,
                )
                stored = np.log10(generated + epsilon).astype(np.float32)
                recovered = np.maximum(
                    10.0 ** stored.astype(np.float64) - epsilon, 0.0,
                )
                error = abs(float(recovered.sum()) - 1.0)
                if error > integral_tolerance:
                    raise ValueError(
                        f"Stored integral error {error:.3g} at row {row} exceeds "
                        f"{integral_tolerance:g}"
                    )
                maximum_error = max(maximum_error, error)
                stored_chunk[offset] = stored
                null_recent_fraction[row] = float(recovered[:split].sum())
                null_mean_lookback[row] = float(np.dot(recovered, time))
                bridge_width[row] = int(info["transition_bins"])
            sfh_out[start:stop] = stored_chunk
            if start == 0 or stop == n_rows or (start // chunk_size + 1) % 20 == 0:
                print(
                    f"Factorized-null rows {stop:,}/{n_rows:,}; "
                    f"max |sum-1|={maximum_error:.3g}", flush=True,
                )

        target.create_dataset("bridge_width", data=bridge_width)
        target.attrs.update({
            "n_galaxies": n_rows,
            "n_bins": n_bins,
            "sfh_log_epsilon": epsilon,
            "source_dataset": str(source_path),
            "transformation": "independent recent-fraction, recent-shape, and past-shape donors",
            "recent_fraction": recent_fraction,
            "transition_bins": transition_bins,
            "random_seed": seed,
            "integral_tolerance": integral_tolerance,
            "maximum_absolute_stored_integral_error": maximum_error,
        })

    original_mean = weights @ time
    original_t50 = _t50(weights, time)
    # Re-read the compact output only for t50; this also independently checks
    # that all compressed values were written successfully.
    with h5py.File(partial, "r") as target:
        null_weights = np.maximum(
            10.0 ** np.asarray(target["sfh"], dtype=np.float64) - epsilon, 0.0,
        )
    null_t50 = _t50(null_weights, time)
    report = {
        "source": str(source_path),
        "output": str(output_path),
        "n_galaxies": n_rows,
        "n_bins": n_bins,
        "seed": seed,
        "recent_fraction": recent_fraction,
        "transition_bins": transition_bins,
        "integral_tolerance": integral_tolerance,
        "maximum_absolute_stored_integral_error": maximum_error,
        "all_donors_distinct_from_target_and_each_other": bool(
            np.all(fraction_donor != np.arange(n_rows))
            and np.all(recent_donor != np.arange(n_rows))
            and np.all(past_donor != np.arange(n_rows))
            and np.all(fraction_donor != recent_donor)
            and np.all(fraction_donor != past_donor)
            and np.all(recent_donor != past_donor)
        ),
        "target_null_correlation_recent_fraction": _safe_correlation(
            recent_mass, null_recent_fraction,
        ),
        "target_null_correlation_mean_lookback": _safe_correlation(
            original_mean, null_mean_lookback,
        ),
        "target_null_correlation_t50": _safe_correlation(original_t50, null_t50),
        "null_recent_fraction_matches_fraction_donor_max_error": float(
            np.max(np.abs(null_recent_fraction - recent_mass[fraction_donor]))
        ),
        "bridge_width_minimum": int(bridge_width.min()),
        "bridge_width_maximum": int(bridge_width.max()),
    }
    partial.replace(output_path)
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recent-fraction", type=float, default=0.1)
    parser.add_argument("--transition-bins", type=int, default=8)
    parser.add_argument("--integral-tolerance", type=float, default=2e-6)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    build_dataset(
        args.source, args.output, seed=args.seed,
        recent_fraction=args.recent_fraction,
        transition_bins=args.transition_bins,
        integral_tolerance=args.integral_tolerance,
        chunk_size=args.chunk_size, max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
