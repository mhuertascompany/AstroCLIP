"""Augment the AstroCLIP dataset with Galaxy Zoo morphology labels.

This script loads the AstroCLIP dataset (e.g. ``EiffL/AstroCLIP``) from the
Hugging Face hub or a local ``save_to_disk`` directory, cross-matches sources
against the Galaxy Zoo DECaLS catalogue, and saves a new dataset with selected
morphology probabilities appended.

Example usage:
    python teaching_scripts/add_morphology_labels.py \
        --dataset EiffL/AstroCLIP \
        --output-dir /path/to/astroclip_with_morphology \
        --morphology-dataset BigBang/galaxyzoo-decals \
        --train-size 10000 --test-size 2000
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from astroclip.env import format_with_env

LOGGER = logging.getLogger("add_morphology_labels")

# Default Galaxy Zoo columns to keep. These capture the high-level questions used
# in the AstroCLIP morphology downstream task.
DEFAULT_MORPH_COLUMNS = [
    "smooth-or-featured_featured-or-disk_debiased",
    "smooth-or-featured_smooth_debiased",
    "disk-edge-on_yes_debiased",
    "disk-edge-on_no_debiased",
    "has-spiral-arms_yes_debiased",
    "has-spiral-arms_no_debiased",
    "bar_strong_debiased",
    "bar_weak_debiased",
    "bar_no_debiased",
    "bulge-size_large_debiased",
    "bulge-size_small_debiased",
    "bulge-size_none_debiased",
]


def _load_dataset(path_or_name: str) -> DatasetDict:
    if Path(path_or_name).exists():
        LOGGER.info("Loading dataset from disk: %s", path_or_name)
        ds = load_from_disk(path_or_name)
        if isinstance(ds, Dataset):
            ds = DatasetDict({"train": ds})
        return ds
    LOGGER.info("Loading dataset from Hugging Face hub: %s", path_or_name)
    ds = load_dataset(path_or_name)
    if isinstance(ds, Dataset):
        ds = DatasetDict({"train": ds})
    return ds


def _load_morphology_source(
    source: str,
    columns: List[str],
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    if Path(source).exists():
        path = Path(source)
        if path.suffix.lower() == ".csv":
            morph = pd.read_csv(path)
        elif path.suffix.lower() in {".h5", ".hdf5"}:
            morph = pd.read_hdf(path)
        else:
            raise ValueError(f"Unsupported morphology file type: {path.suffix}")
        LOGGER.info("Loaded morphology table with %d rows from %s", len(morph), source)
    else:
        LOGGER.info("Loading morphology dataset %s from Hugging Face hub", source)
        morph_ds = load_dataset(source, split="train", cache_dir=cache_dir)
        morph = morph_ds.to_pandas()
        LOGGER.info("Downloaded morphology dataset with %d rows", len(morph))

    required_cols = {"ra", "dec"}
    missing = required_cols - set(morph.columns)
    if missing:
        raise ValueError(
            f"Morphology table is missing required columns: {', '.join(sorted(missing))}"
        )

    if columns:
        missing_cols = set(columns) - set(morph.columns)
        if missing_cols:
            raise ValueError(
                "Requested morphology columns not found: "
                f"{', '.join(sorted(missing_cols))}"
            )
        morph = morph[["iauname", "ra", "dec", *columns]]
    else:
        numeric_cols = [
            c for c in morph.columns if c.endswith("_debiased") or c.endswith("_fraction")
        ]
        morph = morph[["iauname", "ra", "dec", *numeric_cols]]
    morph = morph.drop_duplicates("iauname")
    morph.reset_index(drop=True, inplace=True)
    return morph


def _prepare_subset(
    dataset: Dataset,
    columns: Iterable[str],
) -> pd.DataFrame:
    subset_columns = [col for col in columns if col in dataset.column_names]
    missing_cols = set(columns) - set(subset_columns)
    if missing_cols:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{', '.join(sorted(missing_cols))}. "
            "Ensure the dataset includes sky positions (ra/dec)."
        )
    frame = dataset.select_columns(subset_columns).to_pandas()
    LOGGER.info("Subset extracted with %d rows", len(frame))
    return frame


def _cross_match(
    astroclip_df: pd.DataFrame,
    morph_df: pd.DataFrame,
    max_sep_arcsec: float,
) -> pd.DataFrame:
    LOGGER.info(
        "Cross-matching %d AstroCLIP sources with %d Galaxy Zoo entries "
        "(max separation %.2f arcsec)",
        len(astroclip_df),
        len(morph_df),
        max_sep_arcsec,
    )
    astro_coords = SkyCoord(
        ra=astroclip_df["ra"].to_numpy() * u.deg,
        dec=astroclip_df["dec"].to_numpy() * u.deg,
    )
    morph_coords = SkyCoord(
        ra=morph_df["ra"].to_numpy() * u.deg,
        dec=morph_df["dec"].to_numpy() * u.deg,
    )
    idx, d2d, _ = astro_coords.match_to_catalog_sky(morph_coords)
    mask = d2d <= max_sep_arcsec * u.arcsec
    LOGGER.info("Matched %d sources within %.2f arcsec", mask.sum(), max_sep_arcsec)
    matched = astroclip_df.loc[mask].copy()
    matched["morph_index"] = idx[mask]
    return matched


def _build_lookup(
    cross_match_df: pd.DataFrame,
    morph_df: pd.DataFrame,
    columns: List[str],
) -> dict[int, np.ndarray]:
    values = {}
    for _, row in cross_match_df.iterrows():
        morph_row = morph_df.iloc[int(row["morph_index"])]
        tid = int(row["targetid"])
        data = morph_row[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        if np.any(np.isfinite(data)):
            values[tid] = data.astype(np.float32)
    LOGGER.info("Constructed lookup for %d targetids", len(values))
    return values


def _append_columns(
    dataset: Dataset,
    lookup: dict[int, np.ndarray],
    columns: List[str],
    dtype: np.dtype = np.float32,
) -> Dataset:
    missing_row = np.full(len(columns), np.nan, dtype=dtype)
    data_matrix = np.empty((len(dataset), len(columns)), dtype=dtype)
    target_ids: List[int] = dataset["targetid"]
    for i, tid in enumerate(target_ids):
        row = lookup.get(int(tid))
        if row is None:
            data_matrix[i] = missing_row
        else:
            data_matrix[i] = row

    updated = dataset
    for col_idx, column in enumerate(columns):
        updated = updated.add_column(column, data_matrix[:, col_idx].tolist())
    return updated


def augment_dataset(
    dataset: DatasetDict,
    morph_source: str,
    morph_columns: List[str],
    max_sep_arcsec: float,
    cache_dir: Optional[Path] = None,
) -> DatasetDict:
    morph_df = _load_morphology_source(morph_source, morph_columns, cache_dir)
    columns_needed = {"targetid", "ra", "dec"}
    augmented_splits = {}
    galaxy_lookup = None

    for split_name, split_dataset in dataset.items():
        LOGGER.info("Processing split '%s'", split_name)
        astro_df = _prepare_subset(split_dataset, columns_needed)
        if galaxy_lookup is None:
            cross_match_df = _cross_match(astro_df, morph_df, max_sep_arcsec)
            if cross_match_df.empty:
                raise RuntimeError(
                    "No matches were found. Check that the dataset includes "
                    "RA/Dec and that the separations/tolerances are correct."
                )
            galaxy_lookup = _build_lookup(cross_match_df, morph_df, morph_columns)

        augmented_splits[split_name] = _append_columns(
            split_dataset, galaxy_lookup, morph_columns
        )

    return DatasetDict(augmented_splits)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = format_with_env("{ASTROCLIP_ROOT}")
    parser.add_argument(
        "--dataset",
        type=str,
        default="EiffL/AstroCLIP",
        help="Base AstroCLIP dataset name or path to load_from_disk directory.",
    )
    parser.add_argument(
        "--morphology-dataset",
        type=str,
        default="BigBang/galaxyzoo-decals",
        help="Galaxy Zoo dataset identifier or path to CSV/HDF5 file.",
    )
    parser.add_argument(
        "--morphology-columns",
        nargs="+",
        default=DEFAULT_MORPH_COLUMNS,
        help="Morphology columns to append (default uses key debiased probabilities).",
    )
    parser.add_argument(
        "--max-separation-arcsec",
        type=float,
        default=1.0,
        help="Maximum separation to treat as a match (in arcseconds).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(default_root) / "teaching_demo" / "astroclip_with_morphology",
        help="Destination directory for the augmented dataset (save_to_disk).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for Hugging Face downloads.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dataset = _load_dataset(args.dataset)
    augmented = augment_dataset(
        dataset=dataset,
        morph_source=args.morphology_dataset,
        morph_columns=args.morphology_columns,
        max_sep_arcsec=args.max_separation_arcsec,
        cache_dir=args.cache_dir,
    )

    if args.output_dir.exists():
        LOGGER.info("Removing existing directory %s", args.output_dir)
        for child in sorted(args.output_dir.glob("**/*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        args.output_dir.rmdir()

    LOGGER.info("Saving augmented dataset to %s", args.output_dir)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    augmented.save_to_disk(args.output_dir)
    LOGGER.info("Completed augmentation successfully.")


if __name__ == "__main__":
    main()
