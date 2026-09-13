"""PyTorch dataset pairing Euclid VIS ZooBot stamps with posterior SFHs."""

import os
from pathlib import Path

import h5py
import numpy as np
import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .training_index import build_pair_index


def _inference_transform(image_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def _training_transform(image_size):
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


class EuclidZooBotDataset(Dataset):
    """Lazy HDF5 dataset with posterior sampling for the training split."""

    def __init__(self, sfh_path, stamp_root, rows, galaxy_ids, band='VIS',
                 training=True, sample_posterior=True, image_size=224):
        self.sfh_path = str(Path(sfh_path))
        self.stamp_dir = Path(stamp_root) / band
        self.band = band
        self.rows = np.asarray(rows, dtype=np.int64)
        self.galaxy_ids = np.asarray(galaxy_ids, dtype=np.int64)
        self.training = training
        self.sample_posterior = sample_posterior
        self.transform = (
            _training_transform(image_size) if training
            else _inference_transform(image_size)
        )
        self._h5 = None
        self._h5_pid = None
        if len(self.rows) != len(self.galaxy_ids):
            raise ValueError('rows and galaxy_ids must have equal length.')

    def __len__(self):
        return len(self.rows)

    def _source(self):
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(self.sfh_path, 'r')
            self._h5_pid = pid
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        state['_h5_pid'] = None
        return state

    def __del__(self):
        source = getattr(self, '_h5', None)
        if source is not None:
            source.close()

    def __getitem__(self, index):
        row = int(self.rows[index])
        galaxy_id = int(self.galaxy_ids[index])
        source = self._source()
        realization_index = -1
        if self.training and self.sample_posterior:
            valid = np.flatnonzero(source['sfh_realization_valid'][row])
            if not len(valid):
                raise ValueError(f'No valid SFH realization for galaxy_id={galaxy_id}.')
            draw = int(torch.randint(len(valid), size=()).item())
            realization_index = int(valid[draw])
            sfh = np.asarray(
                source['sfh_realizations'][row, realization_index],
                dtype=np.float32,
            )
        else:
            sfh = np.asarray(source['sfh'][row], dtype=np.float32)
        if not np.all(np.isfinite(sfh)):
            raise ValueError(f'Nonfinite SFH for galaxy_id={galaxy_id}.')

        path = self.stamp_dir / f'{self.band}_{galaxy_id}.jpg'
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert('L'))
        return {
            'image': image_tensor,
            'sfh': torch.from_numpy(sfh.copy()),
            'galaxy_id': torch.tensor(galaxy_id, dtype=torch.int64),
            'sfh_realization': torch.tensor(realization_index, dtype=torch.int64),
        }


class EuclidZooBotDataModule(L.LightningDataModule):
    def __init__(self, sfh_path, stamp_root, band='VIS', batch_size=128,
                 num_workers=8, image_size=224, val_fraction=0.1, seed=42,
                 sample_posterior=True, max_pairs=None):
        super().__init__()
        self.sfh_path = sfh_path
        self.stamp_root = stamp_root
        self.band = band
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_fraction = val_fraction
        self.seed = seed
        self.sample_posterior = sample_posterior
        self.max_pairs = max_pairs
        self.pair_index = None

    def setup(self, stage=None):
        if self.pair_index is None:
            self.pair_index = build_pair_index(
                self.sfh_path, self.stamp_root, self.band,
                self.val_fraction, self.seed, self.max_pairs,
            )
            index = self.pair_index
            print(
                f'[EuclidZooBot] paired {index.n_paired:,}/{index.n_sfh:,}; '
                f'train={len(index.train_rows):,}, val={len(index.val_rows):,}; '
                f'{index.n_bins} bins, {index.n_realizations} realizations',
                flush=True,
            )
        shared = dict(
            sfh_path=self.sfh_path,
            stamp_root=self.stamp_root,
            band=self.band,
            sample_posterior=self.sample_posterior,
            image_size=self.image_size,
        )
        self.train_ds = EuclidZooBotDataset(
            rows=self.pair_index.train_rows,
            galaxy_ids=self.pair_index.train_ids,
            training=True,
            **shared,
        )
        self.val_ds = EuclidZooBotDataset(
            rows=self.pair_index.val_rows,
            galaxy_ids=self.pair_index.val_ids,
            training=False,
            **shared,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
