"""Utility script to stage paired image/spectrum data for the teaching pipeline.

This script relies on the in-repo HuggingFace dataset builder located at
`astroclip/data/dataset.py`. It can fetch the full dataset (≈60 GB) or derive a
smaller subset for quick experimentation. The resulting dataset is saved in the
standard HuggingFace `save_to_disk` format so that `AstroClipDataloader` can
load it without further changes.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import Dataset, DatasetDict, load_dataset

from astroclip.env import format_with_env

LOGGER = logging.getLogger("prepare_dataset")


def _load_splits(
    split_sizes: Dict[str, Optional[int]],
    shuffle: bool,
    seed: int,
    streaming: bool,
) -> DatasetDict:
    dataset_splits: Dict[str, Dataset] = {}

    dataset_module = "astroclip.data.dataset"
    for split_name, sample_size in split_sizes.items():
        LOGGER.info("Loading split '%s' (sample_size=%s)", split_name, sample_size)
        split_selector = split_name
        ds = load_dataset(
            dataset_module,
            name="joint",
            split=split_selector,
            streaming=streaming,
        )

        if streaming:
            if sample_size is None:
                raise ValueError(
                    "Streaming mode requires --sample-size/--train-size/--test-size."
                )
            LOGGER.info("Sampling %d elements from streaming dataset", sample_size)
            sampled_columns = _take(ds, sample_size)
            ds = Dataset.from_dict(sampled_columns)

        if sample_size is not None and not streaming:
            ds = ds.shuffle(seed=seed) if shuffle else ds
            sample_size = min(sample_size, len(ds))
            ds = ds.select(range(sample_size))
            LOGGER.info("Selected %d examples from split '%s'", len(ds), split_name)

        dataset_splits[split_name] = ds

    return DatasetDict(dataset_splits)


def _take(dataset: Iterable[Dict], n: int) -> Dict[str, Iterable]:
    """Collect `n` examples from an iterable dataset into column-major dict."""
    columns = None
    count = 0
    for item in dataset:
        if columns is None:
            columns = {k: [] for k in item.keys()}
        for key, value in item.items():
            columns[key].append(value)
        count += 1
        if count >= n:
            break
    if columns is None:
        raise ValueError("Dataset iterator produced no items.")
    return columns


def stage_dataset(
    output_dir: Path,
    train_size: Optional[int],
    test_size: Optional[int],
    overwrite: bool,
    shuffle: bool,
    seed: int,
    streaming: bool,
) -> Path:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already exists. "
                "Pass --overwrite to replace it."
            )
        LOGGER.warning("Removing existing directory %s", output_dir)
        shutil.rmtree(output_dir)

    split_sizes = {"train": train_size, "test": test_size}
    dataset = _load_splits(
        split_sizes=split_sizes,
        shuffle=shuffle,
        seed=seed,
        streaming=streaming,
    )

    LOGGER.info("Saving dataset dictionary to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_dir)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download/save a paired AstroCLIP dataset for teaching demos."
    )
    default_root = format_with_env("{ASTROCLIP_ROOT}")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(default_root) / "teaching_demo" / "astroclip_dataset",
        help="Destination folder for the prepared dataset.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=2048,
        help=(
            "Number of examples to keep in the training split. "
            "None keeps the full split."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=512,
        help=(
            "Number of examples to keep in the test split. "
            "None keeps the full split."
        ),
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before sampling subsets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling splits before sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Use HuggingFace streaming mode. Requires --train-size/--test-size "
            "to limit the number of downloaded examples."
        ),
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

    stage_dataset(
        output_dir=args.output_dir,
        train_size=None if args.train_size is None else args.train_size,
        test_size=None if args.test_size is None else args.test_size,
        overwrite=args.overwrite,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        streaming=args.streaming,
    )
    LOGGER.info("Dataset preparation completed successfully.")


if __name__ == "__main__":
    main()
