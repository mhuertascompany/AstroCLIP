"""Build a compact IAAFT temporal-ablation SFH training dataset."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from .sfh_surrogates import (
    past_preserved_iaaft, recent_preserved_iaaft, window_preserved_iaaft,
)


def _process_chunk(payload):
    (
        first_row, log_sfh, time, epsilon, seed, recent_fraction,
        transition_bins, candidates, max_iterations, integral_tolerance, preserve,
        window_start, window_end,
    ) = payload
    output = np.empty_like(log_sfh, dtype=np.float32)
    diagnostics = []
    for offset, values in enumerate(log_sfh):
        source_row = first_row + offset
        weights = np.maximum(10.0 ** values.astype(np.float64) - epsilon, 0.0)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Invalid SFH integral at source row {source_row}: {total}")
        weights /= total

        best = None
        for candidate in range(candidates):
            candidate_seed = np.random.SeedSequence([seed, source_row, candidate])
            rng = np.random.default_rng(candidate_seed)
            if preserve == "window":
                surrogate, info = window_preserved_iaaft(
                    weights, time, rng, window_start=window_start,
                    window_end=window_end, transition_bins=transition_bins,
                    max_iterations=max_iterations,
                )
                randomized_key = "outside_correlation"
            else:
                transform = (
                    recent_preserved_iaaft
                    if preserve == "recent" else past_preserved_iaaft
                )
                surrogate, info = transform(
                    weights, time, rng,
                    recent_fraction=recent_fraction,
                    transition_bins=transition_bins,
                    max_iterations=max_iterations,
                )
                randomized_key = (
                    "old_correlation" if preserve == "recent" else "recent_correlation"
                )
            score = abs(info[randomized_key])
            if best is None or score < best[0]:
                best = (score, surrogate, info, candidate)
        _, surrogate, info, candidate = best

        integral = float(surrogate.sum())
        if preserve == "window":
            first, last = info["window_start_index"], info["window_end_index"]
            preserved_error = float(
                np.max(np.abs(surrogate[first:last] - weights[first:last]))
            )
            randomized_correlation = float(info["outside_correlation"])
        elif preserve == "recent":
            recent_bins = int(info["n_recent_bins"])
            preserved_error = float(
                np.max(np.abs(surrogate[:recent_bins] - weights[:recent_bins]))
            )
            randomized_correlation = float(info["old_correlation"])
        else:
            recent_bins = int(info["n_recent_bins"])
            preserved_error = float(
                np.max(np.abs(surrogate[recent_bins:] - weights[recent_bins:]))
            )
            randomized_correlation = float(info["recent_correlation"])
        if abs(integral - 1.0) > integral_tolerance:
            raise ValueError(
                f"IAAFT integral is {integral:.12g} at source row {source_row}; "
                f"tolerance={integral_tolerance:g}"
            )
        if preserved_error > integral_tolerance:
            raise ValueError(
                f"Preserved SFH segment changed by {preserved_error:.3g} "
                f"at source row {source_row}"
            )

        stored = np.log10(surrogate + epsilon).astype(np.float32)
        recovered = np.maximum(10.0 ** stored.astype(np.float64) - epsilon, 0.0)
        stored_integral = float(recovered.sum())
        if abs(stored_integral - 1.0) > integral_tolerance:
            raise ValueError(
                f"Stored IAAFT integral is {stored_integral:.12g} at source row "
                f"{source_row}; tolerance={integral_tolerance:g}"
            )
        output[offset] = stored
        diagnostics.append((
            integral, stored_integral, preserved_error,
            randomized_correlation, float(info["spectral_error"]),
            int(info["iterations"]), int(candidate),
        ))
    return first_row, output, np.asarray(diagnostics, dtype=np.float64)


def build_dataset(
    source_path,
    output_path,
    *,
    seed=42,
    recent_fraction=0.1,
    transition_bins=10,
    candidates=4,
    max_iterations=1000,
    integral_tolerance=2e-6,
    workers=8,
    chunk_size=256,
    max_rows=None,
    preserve="recent",
    window_start=0.1,
    window_end=0.2,
):
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.exists() or output_path.with_suffix(output_path.suffix + ".partial").exists():
        raise FileExistsError(f"Output or partial output already exists: {output_path}")
    if workers < 1 or chunk_size < 1 or candidates < 1:
        raise ValueError("workers, chunk_size, and candidates must be positive")
    if not 0 < integral_tolerance < 1e-3:
        raise ValueError("integral_tolerance must lie in (0, 1e-3)")
    if preserve not in {"recent", "past", "window"}:
        raise ValueError("preserve must be 'recent', 'past', or 'window'")
    if preserve == "window" and not 0 <= window_start < window_end <= 1:
        raise ValueError("window_start and window_end must define a window within [0, 1]")

    with h5py.File(source_path, "r") as source:
        required = ("galaxy_id", "sfh", "sfh_time_grid")
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"Missing source datasets: {missing}")
        n_total, n_bins = source["sfh"].shape
        n_rows = n_total if max_rows is None else min(int(max_rows), n_total)
        ids = np.asarray(source["galaxy_id"][:n_rows], dtype=np.int64)
        time = np.asarray(source["sfh_time_grid"], dtype=np.float64)
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))
        chunks = [
            (
                start,
                np.asarray(source["sfh"][start:min(start + chunk_size, n_rows)]),
                time,
                epsilon,
                int(seed),
                float(recent_fraction),
                int(transition_bins),
                int(candidates),
                int(max_iterations),
                float(integral_tolerance),
                preserve,
                float(window_start),
                float(window_end),
            )
            for start in range(0, n_rows, chunk_size)
        ]

    partial = output_path.with_suffix(output_path.suffix + ".partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_diagnostics = []
    with h5py.File(partial, "w") as target:
        target.create_dataset("galaxy_id", data=ids)
        target.create_dataset("source_row", data=np.arange(n_rows, dtype=np.int64))
        target.create_dataset("sfh_time_grid", data=time.astype(np.float32))
        sfh_out = target.create_dataset(
            "sfh", shape=(n_rows, n_bins), dtype=np.float32,
            chunks=(min(chunk_size, n_rows), n_bins),
            compression="gzip", compression_opts=4, shuffle=True,
        )
        iterator = map(_process_chunk, chunks)
        if workers > 1:
            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(_process_chunk, chunks)
        try:
            for number, (start, values, diagnostics) in enumerate(iterator, start=1):
                sfh_out[start:start + len(values)] = values
                all_diagnostics.append(diagnostics)
                if number == 1 or number % 20 == 0 or number == len(chunks):
                    print(
                        f"IAAFT rows {start + len(values):,}/{n_rows:,}; "
                        f"max |sum-1|={np.max(np.abs(diagnostics[:, 1] - 1)):.3g}",
                        flush=True,
                    )
        finally:
            if workers > 1:
                executor.shutdown()

        diagnostics = np.concatenate(all_diagnostics)
        preserved_name = (
            "window" if preserve == "window" else
            ("recent" if preserve == "recent" else "old")
        )
        randomized_name = (
            "outside" if preserve == "window" else
            ("old" if preserve == "recent" else "recent")
        )
        attributes = {
            "n_galaxies": n_rows,
            "n_bins": n_bins,
            "sfh_log_epsilon": epsilon,
            "source_dataset": str(source_path),
            "transformation": f"{preserved_name}-preserved IAAFT",
            "preserved_segment": preserved_name,
            "recent_fraction": recent_fraction,
            "window_start": window_start if preserve == "window" else np.nan,
            "window_end": window_end if preserve == "window" else np.nan,
            "transition_bins": transition_bins,
            "candidates_per_galaxy": candidates,
            "random_seed": seed,
            "integral_tolerance": integral_tolerance,
            "minimum_stored_integral": float(np.min(diagnostics[:, 1])),
            "maximum_stored_integral": float(np.max(diagnostics[:, 1])),
            "maximum_absolute_stored_integral_error": float(
                np.max(np.abs(diagnostics[:, 1] - 1.0))
            ),
            "maximum_preserved_segment_error": float(np.max(diagnostics[:, 2])),
            f"maximum_{preserved_name}_segment_error": float(np.max(diagnostics[:, 2])),
            f"median_{randomized_name}_correlation": float(np.median(diagnostics[:, 3])),
            f"maximum_absolute_{randomized_name}_correlation": float(
                np.max(np.abs(diagnostics[:, 3]))
            ),
            "median_spectral_error": float(np.median(diagnostics[:, 4])),
        }
        target.attrs.update(attributes)
    partial.replace(output_path)

    report = {
        "source": str(source_path),
        "output": str(output_path),
        "n_galaxies": n_rows,
        "n_bins": n_bins,
        "recent_fraction": recent_fraction,
        "window_start": window_start if preserve == "window" else None,
        "window_end": window_end if preserve == "window" else None,
        "preserved_segment": preserved_name,
        "transition_bins": transition_bins,
        "candidates_per_galaxy": candidates,
        "seed": seed,
        "integral_tolerance": integral_tolerance,
        "stored_integral_minimum": float(np.min(diagnostics[:, 1])),
        "stored_integral_maximum": float(np.max(diagnostics[:, 1])),
        "maximum_absolute_stored_integral_error": float(
            np.max(np.abs(diagnostics[:, 1] - 1.0))
        ),
        "maximum_preserved_segment_error": float(np.max(diagnostics[:, 2])),
        f"maximum_{preserved_name}_segment_error": float(np.max(diagnostics[:, 2])),
        f"{randomized_name}_correlation_median": float(np.median(diagnostics[:, 3])),
        f"{randomized_name}_correlation_p05_p95": [
            float(np.quantile(diagnostics[:, 3], 0.05)),
            float(np.quantile(diagnostics[:, 3], 0.95)),
        ],
        "spectral_error_median": float(np.median(diagnostics[:, 4])),
    }
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
    parser.add_argument(
        "--preserve", choices=("recent", "past", "window"), default="recent",
        help="SFH segment copied exactly; IAAFT is applied to the other segment.",
    )
    parser.add_argument("--transition-bins", type=int, default=10)
    parser.add_argument("--window-start", type=float, default=0.1)
    parser.add_argument("--window-end", type=float, default=0.2)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--integral-tolerance", type=float, default=2e-6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    build_dataset(
        args.source, args.output,
        seed=args.seed,
        recent_fraction=args.recent_fraction,
        transition_bins=args.transition_bins,
        candidates=args.candidates,
        max_iterations=args.max_iterations,
        integral_tolerance=args.integral_tolerance,
        workers=args.workers,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        preserve=args.preserve,
        window_start=args.window_start,
        window_end=args.window_end,
    )


if __name__ == "__main__":
    main()
