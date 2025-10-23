"""Utility script to sanity-check prepared AstroCLIP datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import torch

from astroclip.data.datamodule import AstroClipCollator, AstroClipDataloader
from astroclip.env import format_with_env

LOGGER = logging.getLogger("verify_dataset")


def _describe_tensor(name: str, tensor: torch.Tensor) -> None:
    LOGGER.info(
        "%s: shape=%s dtype=%s mean=%.4f std=%.4f min=%.4f max=%.4f",
        name,
        tuple(tensor.shape),
        tensor.dtype,
        tensor.float().mean().item(),
        tensor.float().std(unbiased=False).item(),
        tensor.float().min().item(),
        tensor.float().max().item(),
    )


def run_checks(
    dataset_dir: Path,
    batch_size: int,
    columns: List[str],
    center_crop: int,
) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

    collator = AstroClipCollator(center_crop=center_crop)
    datamodule = AstroClipDataloader(
        path=str(dataset_dir),
        batch_size=batch_size,
        columns=columns,
        num_workers=0,
        collate_fn=collator,
    )
    datamodule.setup(stage="fit")

    loader = datamodule.train_dataloader()
    batch = next(iter(loader))

    LOGGER.info("Loaded batch with keys: %s", list(batch.keys()))
    batch_size = None
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key == "image":
                batch_size = value.shape[0]
            _describe_tensor(key, value)
        else:
            LOGGER.info("%s: type=%s", key, type(value))

    if "image" in batch and "spectrum" in batch and batch_size is not None:
        image_norm = torch.norm(
            batch["image"].float().reshape(batch["image"].shape[0], -1), dim=1
        )
        spectrum_norm = torch.norm(
            batch["spectrum"].float().reshape(batch["spectrum"].shape[0], -1),
            dim=1,
        )
        LOGGER.info(
            "Average L2 norms – image: %.3f ± %.3f, spectrum: %.3f ± %.3f",
            image_norm.mean().item(),
            image_norm.std(unbiased=False).item(),
            spectrum_norm.mean().item(),
            spectrum_norm.std(unbiased=False).item(),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quick verification for AstroCLIP teaching datasets."
    )
    default_root = format_with_env("{ASTROCLIP_ROOT}")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(default_root) / "teaching_demo" / "astroclip_dataset",
        help="Path to the HuggingFace dataset directory produced by prepare_dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Mini-batch size to sample while verifying.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["image", "spectrum", "targetid", "redshift"],
        help="Dataset columns to request from the dataloader.",
    )
    parser.add_argument(
        "--center-crop",
        type=int,
        default=144,
        help="Image crop size passed to AstroClipCollator.",
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

    run_checks(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        columns=args.columns,
        center_crop=args.center_crop,
    )
    LOGGER.info("Dataset verification completed successfully.")


if __name__ == "__main__":
    main()
