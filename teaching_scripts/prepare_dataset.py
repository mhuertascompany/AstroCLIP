"""Prepare a lightweight AstroCLIP dataset from the Hugging Face hub.

This script mirrors the setup used in the AstroCLIP tutorial notebook:
https://github.com/EiffL/Tutorials/blob/master/FoundationModels/AstroCLIPTutorial_solutions.ipynb

It downloads the community-hosted dataset ``EiffL/AstroCLIP`` and materialises a
train/test split locally so the rest of the teaching pipeline can run offline.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

from astroclip.env import format_with_env

LOGGER = logging.getLogger("prepare_dataset")


def _subset_dataset(
    dataset: Dataset,
    sample_size: Optional[int],
    shuffle: bool,
    seed: int,
) -> Dataset:
    if sample_size is None:
        return dataset
    sample_size = min(sample_size, len(dataset))
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(sample_size))


def stage_dataset(
    dataset_name: str,
    output_dir: Path,
    train_size: Optional[int],
    test_size: Optional[int],
    test_fraction: Optional[float],
    shuffle: bool,
    seed: int,
    overwrite: bool,
) -> Path:
    LOGGER.info("Loading dataset '%s'", dataset_name)
    base_dataset = load_dataset(dataset_name, split="train")
    LOGGER.info("Loaded %d examples from the hub", len(base_dataset))

    if test_size is not None and test_size >= len(base_dataset):
        raise ValueError("test_size must be smaller than the dataset size.")

    if test_fraction is None and test_size is None:
        test_fraction = 0.2

    if test_size is not None:
        split = base_dataset.train_test_split(test_size=test_size, seed=seed)
    elif test_fraction is not None and 0 < test_fraction < 1:
        split = base_dataset.train_test_split(test_size=test_fraction, seed=seed)
    else:
        split = {"train": base_dataset, "test": None}

    train_dataset = _subset_dataset(split["train"], train_size, shuffle, seed)
    test_dataset = split["test"]
    if test_dataset is not None:
        test_dataset = _subset_dataset(test_dataset, test_size, shuffle, seed)

    dataset_dict = DatasetDict({"train": train_dataset})
    if test_dataset is not None:
        dataset_dict["test"] = test_dataset

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    LOGGER.info("Saving dataset to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    LOGGER.info("Done. Train size=%d, Test size=%s", len(train_dataset), len(test_dataset) if test_dataset is not None else "N/A")
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the tutorial dataset from the Hugging Face hub and stage it "
            "for the AstroCLIP teaching pipeline."
        )
    )
    parser.add_argument(
        "--dataset",
        default="EiffL/AstroCLIP",
        help="Hugging Face dataset identifier to download.",
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
        help="Number of training examples to keep (None uses the entire split).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=512,
        help="Number of test examples to keep (set to None to use fraction instead).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of the dataset reserved for testing. "
            "Ignored if --test-size is provided."
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
        help="Random seed used for shuffling and splitting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the output directory if it already exists.",
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
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        train_size=None if args.train_size is None else args.train_size,
        test_size=None if args.test_size is None else args.test_size,
        test_fraction=args.test_fraction,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
