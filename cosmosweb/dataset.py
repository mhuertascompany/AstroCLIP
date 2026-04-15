"""
PyTorch Dataset and LightningDataModule for the COSMOS-Web image + SFH dataset.

Each sample is a dict:
    {
        "image": Tensor (3, 64, 64)   – per-image z-scored (shape-focused)
        "sfh":   Tensor (N_TIME,)     – log10(normalised shape + eps), raw
    }

Normalisation:
    Images : (arcsinh_flux - image.mean()) / image.std()  computed per image
             This removes absolute brightness so the encoder focuses on
             morphological shape rather than relative flux levels.
    SFHs   : used as-is from the HDF5 (log10 of the sum-normalised shape
             vector).  No further rescaling is applied so the relative
             heights of the SFH bins — the star-formation history shape —
             are preserved exactly.

Augmentations applied to images:
    - Random horizontal / vertical flip
    - Random 90° rotation (physically meaningful for galaxies)
    - Optional Gaussian noise on SFH in log10 space (disabled by default)
"""

from pathlib import Path
from typing import Optional

import h5py
import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


class CosmosWebDataset(Dataset):
    """
    HDF5-backed dataset of COSMOS-Web galaxy stamps + CIGALE SFH vectors.

    Parameters
    ----------
    h5_path : str | Path
        Path to the HDF5 file produced by prepare_dataset.py.
    split : 'train' | 'val'
        Which split to return.  The file is split 90/10 by index order.
    augment : bool
        Whether to apply image augmentations (flips, rotations).
    sfh_noise_std : float
        Standard deviation of Gaussian noise added to SFH in log10 space.
        Set to 0 to disable.
    """

    def __init__(
        self,
        h5_path: str | Path,
        split: str = 'train',
        augment: bool = True,
        sfh_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.h5_path       = Path(h5_path)
        self.augment       = augment
        self.sfh_noise_std = sfh_noise_std

        with h5py.File(self.h5_path, 'r') as f:
            n = f.attrs['n_galaxies']

        # 90 / 10 split on contiguous blocks (reproducible without shuffling)
        split_idx = int(0.9 * n)
        if split == 'train':
            self.indices = list(range(0, split_idx))
        else:
            self.indices = list(range(split_idx, n))

        # h5py file is opened per-worker in __getitem__ to be multiprocessing-safe
        self._h5 = None

    def __len__(self) -> int:
        return len(self.indices)

    def _get_file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        i = self.indices[idx]
        f = self._get_file()

        image = torch.tensor(f['images'][i], dtype=torch.float32)  # (3, H, W)
        sfh   = torch.tensor(f['sfh'][i],    dtype=torch.float32)  # (N_TIME,)

        # ── normalise image (per-image) ──────────────────────────────────────
        # Z-score over all pixels and channels of this single image so that
        # the encoder sees only morphological shape, not absolute brightness.
        mu    = image.mean()
        sigma = image.std()
        image = (image - mu) / (sigma + 1e-8)

        # ── augmentations ────────────────────────────────────────────────────
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = image.flip(-1)
            if torch.rand(1).item() > 0.5:
                image = image.flip(-2)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                image = torch.rot90(image, k, dims=(-2, -1))

        # ── SFH noise (optional, in log10 space) ────────────────────────────
        if self.sfh_noise_std > 0:
            sfh = sfh + torch.randn_like(sfh) * self.sfh_noise_std

        return {'image': image, 'sfh': sfh}

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


class CosmosWebDataModule(L.LightningDataModule):
    """
    LightningDataModule wrapping CosmosWebDataset.

    Parameters
    ----------
    h5_path : str | Path
        Path to the HDF5 file from prepare_dataset.py.
    batch_size : int
    num_workers : int
    augment : bool
    sfh_noise_std : float
    """

    def __init__(
        self,
        h5_path: str | Path,
        batch_size: int   = 256,
        num_workers: int  = 4,
        augment: bool     = True,
        sfh_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.h5_path       = h5_path
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.augment       = augment
        self.sfh_noise_std = sfh_noise_std

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = CosmosWebDataset(
            self.h5_path, split='train',
            augment=self.augment,
            sfh_noise_std=self.sfh_noise_std,
        )
        self.val_ds = CosmosWebDataset(
            self.h5_path, split='val',
            augment=False,
            sfh_noise_std=0.0,
        )

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
