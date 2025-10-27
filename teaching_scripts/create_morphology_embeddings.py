from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from teaching_scripts.data_utils import load_astroclip_dataset
from teaching_scripts.models import ImageAutoencoder

LOGGER = logging.getLogger("create_morphology_embeddings")

DEFAULT_LABELS = [
    "smooth-or-featured_featured-or-disk_debiased",
    "smooth-or-featured_smooth_debiased",
    "disk-edge-on_yes_debiased",
    "disk-edge-on_no_debiased",
    "has-spiral-arms_yes_debiased",
    "has-spiral-arms_no_debiased",
    "bar_strong_debiased",
    "bar_no_debiased",
    "bulge-size_large_debiased",
    "bulge-size_small_debiased",
    "bulge-size_none_debiased",
]


class GalaxyZooDataset(Dataset):
    def __init__(
        self,
        table: Table,
        label_columns: List[str],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.images = table["image"]
        self.label_columns = label_columns
        self.labels = table[label_columns]
        self.targetids = table["targetid"] if "targetid" in table.colnames else None
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = torch.tensor(np.asarray(self.images[idx]), dtype=self.dtype)
        label = torch.tensor(
            [float(self.labels[col][idx]) for col in self.label_columns],
            dtype=self.dtype,
        )
        targetid = (
            int(self.targetids[idx]) if self.targetids is not None else -1
        )
        return image, label, targetid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed Galaxy Zoo cross-matched images and save morphology labels."
    )
    parser.add_argument(
        "--crossmatch-file",
        type=Path,
        required=True,
        help="Path to gz5 cross-matched HDF5 (output of cross_match.py).",
    )
    parser.add_argument(
        "--image-ckpt",
        type=Path,
        required=True,
        help="Checkpoint for the image encoder (ImageAutoencoder).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination HDF5 for embeddings + labels.",
    )
    parser.add_argument(
        "--label-columns",
        nargs="+",
        default=None,
        help="Morphology columns to include. Defaults to common Galaxy Zoo questions.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    LOGGER.info("Loading cross-match table from %s", args.crossmatch_file)
    table = Table.read(args.crossmatch_file)
    label_columns = args.label_columns or [col for col in DEFAULT_LABELS if col in table.colnames]
    if not label_columns:
        raise ValueError("No label columns specified/found in the cross-match file.")

    dataset = GalaxyZooDataset(table, label_columns)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    LOGGER.info("Loading image encoder from %s", args.image_ckpt)
    model = ImageAutoencoder.load_from_checkpoint(args.image_ckpt)
    model.eval()
    device = (
        torch.device("cuda")
        if (args.accelerator == "cuda" or (args.accelerator == "auto" and torch.cuda.is_available()))
        else torch.device("cpu")
    )
    model.to(device)

    image_embeddings = []
    labels = []
    targetids = []

    with torch.no_grad():
        for images, batch_labels, batch_targetids in tqdm(
            dataloader, desc="Embedding Galaxy Zoo samples"
        ):
            images = images.to(device)
            embeddings = model.encode(images)
            image_embeddings.append(embeddings.cpu().numpy())
            labels.append(batch_labels.numpy())
            targetids.append(np.array(batch_targetids))

    image_embeddings = np.concatenate(image_embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    targetids = np.concatenate(targetids, axis=0)

    LOGGER.info("Saving embeddings to %s", args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("image_embeddings", data=image_embeddings)
        for idx, column in enumerate(label_columns):
            f.create_dataset(column, data=labels[:, idx])
        f.create_dataset("targetid", data=targetids)

    LOGGER.info("Done. Embedded %d samples.", image_embeddings.shape[0])


if __name__ == "__main__":
    main()
