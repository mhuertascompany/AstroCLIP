"""Create detailed progenitor-analogue reports across descendant mass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .progenitor_analogues import (
    DEFAULT_ARCHIVE,
    DEFAULT_BUNDLE,
    DEFAULT_CATALOG,
    DEFAULT_FORMED_FRACTIONS,
    _load_inputs,
    find_analogues,
    make_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("euclid_selected_galaxies.csv"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--output", type=Path,
        default=Path("euclid/diagnostics/selected_progenitor_mass_examples"),
    )
    parser.add_argument(
        "--mass-quantiles", type=float, nargs="+",
        default=[0.05, 0.20, 0.40, 0.60, 0.80, 0.95],
    )
    parser.add_argument("--minimum-progenitor-mass", type=float, default=9.0)
    parser.add_argument("--ms-sfr-offset", type=float, default=-0.93)
    args = parser.parse_args()

    selected = pd.read_csv(args.selection)
    id_column = "galaxy_id" if "galaxy_id" in selected else "object_id"
    selected_ids = selected[id_column].astype(np.int64).to_numpy()
    data = _load_inputs(args.bundle, args.archive, args.catalog, args.ms_sfr_offset)
    ids = np.asarray(data["ids"], dtype=np.int64)
    lookup = {int(galaxy_id): index for index, galaxy_id in enumerate(ids)}
    indices = np.array([lookup.get(int(galaxy_id), -1) for galaxy_id in selected_ids])
    if np.any(indices < 0):
        raise ValueError("At least one selected ID is absent from the explorer bundle")

    masses = np.asarray(data["mass"])[indices]
    chosen: list[int] = []
    records: list[dict[str, float | int | str]] = []
    stamps = args.bundle / "VIS"
    args.output.mkdir(parents=True, exist_ok=True)
    for quantile in args.mass_quantiles:
        target = float(np.quantile(masses, quantile))
        order = np.argsort(np.abs(masses - target))
        index = next(int(indices[position]) for position in order if int(indices[position]) not in chosen)
        chosen.append(index)
        galaxy_id = int(ids[index])
        descendant_mass = float(np.asarray(data["mass"])[index])
        fractions = [
            fraction for fraction in DEFAULT_FORMED_FRACTIONS
            if descendant_mass + np.log10(fraction) >= args.minimum_progenitor_mass - 0.01
        ]
        candidates, census, stages = find_analogues(
            data, index, fractions, stamps=stamps, n_analogues=5
        )
        destination = args.output / str(galaxy_id)
        destination.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(destination / "analogue_candidates.csv", index=False)
        census.to_csv(destination / "checkpoint_census.csv", index=False)
        pdf = args.output / f"mass_q{quantile:.2f}_{galaxy_id}.pdf"
        make_report(data, index, candidates, census, stages, stamps, pdf, destination)
        records.append(
            {
                "mass_quantile": quantile,
                "galaxy_id": galaxy_id,
                "log_stellar_mass": descendant_mass,
                "redshift": float(np.asarray(data["redshift"])[index]),
                "phz_delta_ms": float(np.asarray(data["catalog_delta_ms"])[index]),
                "n_checkpoints": len(stages),
                "pdf": str(pdf),
            }
        )
        print(f"q={quantile:.2f}: {galaxy_id}, logM={descendant_mass:.3f}")

    pd.DataFrame(records).to_csv(args.output / "selected_descendants.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "selection": str(args.selection),
                "selection_mass_cut": None,
                "minimum_progenitor_mass": args.minimum_progenitor_mass,
                "ms_sfr_offset_dex": args.ms_sfr_offset,
                "examples": records,
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
