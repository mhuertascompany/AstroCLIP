"""
Dataset for ZooBOT-based COSMOS-Web CLIP training.

Loads (F277W JPEG stamp, CIGALE SFH) pairs by matching galaxy IDs
between the HDF5 file (for SFH vectors) and the JPEG stamp directory.

Each sample:
    {
        "image": Tensor (3, 224, 224)  — grayscale JPEG replicated to 3 ch,
                                          pixel values in [0, 1] (ToTensor)
        "sfh":   Tensor (N_BINS,)      — log10(normalised SFH), same as CosmosWebDataset
    }

Transform pipeline (matches ZooBOT training / apply_models_to_master.py):
    Grayscale(3ch) → Resize(224, BICUBIC) → CenterCrop / RandomCrop(224) → ToTensor
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import lightning as L
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ── transforms ────────────────────────────────────────────────────────────────

def _inference_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic transform matching ZooBOT inference pipeline."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def _train_transform(image_size: int = 224) -> transforms.Compose:
    """ZooBOT-style transform + light augmentations for training."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(
            int(image_size * 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
    ])


# ── dataset ───────────────────────────────────────────────────────────────────

class CosmosWebZooBotDataset(Dataset):
    """
    Dataset pairing JWST JPEG stamps with CIGALE SFH vectors.

    Parameters
    ----------
    h5_path : str | Path
        HDF5 file from prepare_dataset.py (provides SFH vectors and galaxy IDs).
    stamp_root : str | Path
        Root directory containing per-filter sub-folders (e.g. .../stamps_ilbert).
    filter_name : str
        JWST filter to load (default 'F277W').  Stamps must be at
        ``{stamp_root}/{filter_name}/{filter_name}_{gid}.jpg``.
    split : 'train' | 'val'
        Uses the same contiguous 90/10 split as CosmosWebDataset.
    augment : bool
        If True, apply randomised training transform; otherwise inference transform.
    sfh_noise_std : float
        Gaussian noise std added to log10-SFH in training.
    image_size : int
        Spatial size of the output crops (default: 224).
    """

    def __init__(
        self,
        h5_path:       str | Path,
        stamp_root:    str | Path,
        filter_name:   str   = 'F277W',
        split:         str   = 'train',
        augment:       bool  = True,
        sfh_noise_std: float = 0.0,
        image_size:    int   = 224,
    ) -> None:
        super().__init__()
        self.filter_name   = filter_name
        self.stamp_dir     = Path(stamp_root) / filter_name
        self.sfh_noise_std = sfh_noise_std
        self.transform     = (_train_transform(image_size) if augment
                              else _inference_transform(image_size))

        # Load all IDs and SFHs from HDF5
        with h5py.File(Path(h5_path), 'r') as f:
            n        = int(f.attrs['n_galaxies'])
            all_ids  = f['galaxy_id'][:]   # (N,)
            all_sfhs = f['sfh'][:]         # (N, N_BINS)

        # 90/10 split (same slicing as CosmosWebDataset)
        split_idx = int(0.9 * n)
        idx_range = range(0, split_idx) if split == 'train' else range(split_idx, n)
        all_ids  = all_ids[list(idx_range)]
        all_sfhs = all_sfhs[list(idx_range)]

        # Keep only objects whose stamp file exists
        valid = np.array(
            [self._stamp_path(int(gid)).exists() for gid in all_ids],
            dtype=bool,
        )

        self.galaxy_ids = all_ids[valid]
        self.sfhs       = all_sfhs[valid]

        print(
            f'[ZooBotDataset] {split}: {valid.sum()}/{len(all_ids)} objects '
            f'have {filter_name} stamps'
        )

    def _stamp_path(self, gid: int) -> Path:
        return self.stamp_dir / f'{self.filter_name}_{gid}.jpg'

    def __len__(self) -> int:
        return len(self.galaxy_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        gid   = int(self.galaxy_ids[idx])
        sfh   = torch.tensor(self.sfhs[idx], dtype=torch.float32)
        image = Image.open(self._stamp_path(gid))
        image = self.transform(image)        # (3, 224, 224) in [0, 1]

        if self.sfh_noise_std > 0:
            sfh = sfh + torch.randn_like(sfh) * self.sfh_noise_std

        return {'image': image, 'sfh': sfh}


# ── data module ───────────────────────────────────────────────────────────────

class CosmosWebZooBotDataModule(L.LightningDataModule):
    """
    LightningDataModule for ZooBOT-based CLIP training.

    Parameters
    ----------
    h5_path : str | Path
    stamp_root : str | Path
    filter_name : str
    batch_size : int
    num_workers : int
    augment : bool
    sfh_noise_std : float
    image_size : int
    """

    def __init__(
        self,
        h5_path:       str | Path,
        stamp_root:    str | Path,
        filter_name:   str   = 'F277W',
        batch_size:    int   = 128,
        num_workers:   int   = 4,
        augment:       bool  = True,
        sfh_noise_std: float = 0.0,
        image_size:    int   = 224,
    ) -> None:
        super().__init__()
        self.h5_path       = h5_path
        self.stamp_root    = stamp_root
        self.filter_name   = filter_name
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.augment       = augment
        self.sfh_noise_std = sfh_noise_std
        self.image_size    = image_size

    def setup(self, stage: Optional[str] = None) -> None:
        shared = dict(
            h5_path=self.h5_path,
            stamp_root=self.stamp_root,
            filter_name=self.filter_name,
            sfh_noise_std=self.sfh_noise_std,
            image_size=self.image_size,
        )
        self.train_ds = CosmosWebZooBotDataset(split='train', augment=self.augment, **shared)
        self.val_ds   = CosmosWebZooBotDataset(split='val',   augment=False,        **shared)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
