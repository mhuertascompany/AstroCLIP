"""Preserve each galaxy's recent SFH and replace its past with a random donor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .sfh_surrogates import recent_preserved_independent_past


def _safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _row_correlation(left, right):
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(float).eps:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def _draw_past_donors(n_rows, valid_pool, rng):
    target = np.arange(n_rows, dtype=np.int64)
    donors = rng.choice(valid_pool, size=n_rows, replace=True).astype(np.int64)
    while np.any(donors == target):
        mask = donors == target
        donors[mask] = rng.choice(valid_pool, size=int(mask.sum()), replace=True)
    return donors


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
    past_mass = weights[:, split:].sum(axis=1)
    valid_pool = np.flatnonzero(past_mass > np.finfo(float).eps)
    if len(valid_pool) < 2:
        raise ValueError("Too few valid past-shape donors")
    donors = _draw_past_donors(n_rows, valid_pool, np.random.default_rng(seed))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    maximum_integral_error = 0.0
    maximum_recent_error = 0.0
    ablated_recent_fraction = np.empty(n_rows, dtype=np.float64)
    original_past_correlation = np.empty(n_rows, dtype=np.float32)
    donor_past_correlation = np.empty(n_rows, dtype=np.float32)
    bridge_width = np.empty(n_rows, dtype=np.int16)
    with h5py.File(partial, "w") as target:
        target.create_dataset("galaxy_id", data=ids)
        target.create_dataset("source_row", data=np.arange(n_rows, dtype=np.int64))
        target.create_dataset("sfh_time_grid", data=time.astype(np.float32))
        target.create_dataset("past_shape_donor_row", data=donors)
        target.create_dataset("past_shape_donor_id", data=ids[donors])
        sfh_out = target.create_dataset(
            "sfh", shape=(n_rows, n_bins), dtype=np.float32,
            chunks=(min(chunk_size, n_rows), n_bins), compression="gzip",
            compression_opts=4, shuffle=True,
        )

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            stored_chunk = np.empty((stop - start, n_bins), dtype=np.float32)
            for offset, row in enumerate(range(start, stop)):
                generated, _, info = recent_preserved_independent_past(
                    weights[row], weights[donors[row]], time,
                    recent_fraction=recent_fraction,
                    transition_bins=transition_bins,
                )
                stored = np.log10(generated + epsilon).astype(np.float32)
                recovered = np.maximum(
                    10.0 ** stored.astype(np.float64) - epsilon, 0.0,
                )
                integral_error = abs(float(recovered.sum()) - 1.0)
                recent_error = float(np.max(np.abs(
                    recovered[:split] - weights[row, :split]
                )))
                if integral_error > integral_tolerance or recent_error > integral_tolerance:
                    raise ValueError(
                        f"Validation failed at row {row}: integral={integral_error:.3g}, "
                        f"recent={recent_error:.3g}, tolerance={integral_tolerance:g}"
                    )
                maximum_integral_error = max(maximum_integral_error, integral_error)
                maximum_recent_error = max(maximum_recent_error, recent_error)
                stored_chunk[offset] = stored
                ablated_recent_fraction[row] = float(recovered[:split].sum())
                original_past_correlation[row] = _row_correlation(
                    weights[row, split:], recovered[split:],
                )
                donor_past_correlation[row] = _row_correlation(
                    weights[donors[row], split:], recovered[split:],
                )
                bridge_width[row] = int(info["transition_bins"])
            sfh_out[start:stop] = stored_chunk
            if start == 0 or stop == n_rows or (start // chunk_size + 1) % 20 == 0:
                print(
                    f"Past-ablation rows {stop:,}/{n_rows:,}; "
                    f"max |sum-1|={maximum_integral_error:.3g}; "
                    f"max recent error={maximum_recent_error:.3g}", flush=True,
                )

        target.create_dataset("bridge_width", data=bridge_width)
        target.create_dataset(
            "original_past_correlation", data=original_past_correlation,
        )
        target.create_dataset("donor_past_correlation", data=donor_past_correlation)
        target.attrs.update({
            "n_galaxies": n_rows,
            "n_bins": n_bins,
            "sfh_log_epsilon": epsilon,
            "source_dataset": str(source_path),
            "transformation": "target recent SFH preserved; independent past-shape donor",
            "recent_fraction": recent_fraction,
            "transition_bins": transition_bins,
            "random_seed": seed,
            "integral_tolerance": integral_tolerance,
            "maximum_absolute_stored_integral_error": maximum_integral_error,
            "maximum_recent_segment_error": maximum_recent_error,
        })

    original_recent_fraction = weights[:, :split].sum(axis=1)
    report = {
        "source": str(source_path),
        "output": str(output_path),
        "n_galaxies": n_rows,
        "n_bins": n_bins,
        "seed": seed,
        "recent_fraction": recent_fraction,
        "transition_bins": transition_bins,
        "integral_tolerance": integral_tolerance,
        "maximum_absolute_stored_integral_error": maximum_integral_error,
        "maximum_recent_segment_error": maximum_recent_error,
        "all_past_donors_distinct_from_target": bool(
            np.all(donors != np.arange(n_rows))
        ),
        "target_ablated_recent_fraction_correlation": _safe_correlation(
            original_recent_fraction, ablated_recent_fraction,
        ),
        "recent_fraction_maximum_error": float(np.max(np.abs(
            original_recent_fraction - ablated_recent_fraction
        ))),
        "median_original_past_correlation": float(
            np.median(original_past_correlation)
        ),
        "median_donor_past_correlation": float(np.median(donor_past_correlation)),
        "bridge_width_minimum": int(bridge_width.min()),
        "bridge_width_maximum": int(bridge_width.max()),
    }
    partial.replace(output_path)
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
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
