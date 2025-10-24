"""Utility helpers for loading AstroCLIP teaching datasets."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, default_collate

from astroclip.data.datamodule import AstroClipCollator


def load_astroclip_dataset(path_or_name: str) -> DatasetDict:
    """Load a dataset from disk or the Hugging Face hub."""
    if path_or_name is None:
        raise ValueError("Dataset path/name must be provided.")

    if path_or_name.endswith(".arrow"):
        raise ValueError("Expected a HuggingFace save_to_disk directory, not an .arrow file.")

    if path_or_name.startswith("hf://") or not path_or_name.strip():
        raise ValueError("Provide a valid dataset name or filesystem path.")

    try:
        ds = load_from_disk(path_or_name)
    except (FileNotFoundError, ValueError):
        ds = load_dataset(path_or_name)

    if isinstance(ds, Dataset):
        return DatasetDict({"train": ds})
    return ds


def _base_collator(center_crop: int = 144, bands: Optional[List[str]] = None) -> Callable:
    collator = AstroClipCollator(center_crop=center_crop, bands=bands or ["g", "r", "z"])

    def fn(samples):
        return collator(samples)

    return fn


def _select_columns(batch: Dict[str, torch.Tensor], columns: Iterable[str]) -> Dict[str, torch.Tensor]:
    return {key: batch[key] for key in columns if key in batch}


def build_image_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    center_crop: int = 144,
) -> DataLoader:
    columns = ["image"]
    dataset.set_format(type="torch", columns=["image"])
    collate_fn = _base_collator(center_crop=center_crop)

    def image_only_collate(samples):
        batch = collate_fn(samples)
        return batch["image"]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=image_only_collate,
    )


def build_spectrum_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset.set_format(type="torch", columns=["spectrum"])

    def collate(samples):
        batch = default_collate(samples)
        spectra = batch["spectrum"].float()
        spectra = spectra.view(spectra.size(0), -1)
        return spectra

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )


def build_multimodal_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    center_crop: int = 144,
    include_targetid: bool = True,
    include_redshift: bool = True,
) -> DataLoader:
    columns = ["image", "spectrum"]
    optional_cols = []
    if include_targetid:
        optional_cols.append("targetid")
    if include_redshift:
        optional_cols.append("redshift")

    dataset.set_format(type="torch", columns=[col for col in columns + optional_cols if col in dataset.column_names])

    collate_fn = _base_collator(center_crop=center_crop)

    def multimodal_collate(samples):
        batch = collate_fn(samples)
        spectra = batch["spectrum"].float().view(batch["spectrum"].size(0), -1)
        result = {"image": batch["image"], "spectrum": spectra}
        for col in optional_cols:
            if col in batch:
                result[col] = batch[col]
        return result

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multimodal_collate,
    )
