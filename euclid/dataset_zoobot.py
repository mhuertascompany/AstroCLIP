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
from .vis_fits import (
    VISFitsTransform,
    load_image_stats,
    resolve_fits_directory,
    vis_fits_path,
)


def validate_sfh_override(source_path, override_path):
    """Require a compact SFH override to be exactly row-aligned and normalized."""
    source_path = Path(source_path)
    override_path = Path(override_path)
    with h5py.File(source_path, 'r') as source, h5py.File(override_path, 'r') as override:
        missing = [name for name in ('galaxy_id', 'sfh', 'sfh_time_grid')
                   if name not in override]
        if missing:
            raise ValueError(f'Missing SFH override datasets: {missing}')
        source_ids = np.asarray(source['galaxy_id'])
        override_ids = np.asarray(override['galaxy_id'])
        if not np.array_equal(source_ids, override_ids):
            raise ValueError('SFH override galaxy IDs are not row-aligned with the source.')
        if override['sfh'].shape != source['sfh'].shape:
            raise ValueError('SFH override shape differs from the source SFHs.')
        if not np.allclose(
            np.asarray(override['sfh_time_grid']),
            np.asarray(source['sfh_time_grid']), atol=1e-7, rtol=0,
        ):
            raise ValueError('SFH override uses a different fractional time grid.')
        maximum_error = override.attrs.get(
            'maximum_absolute_stored_integral_error', None,
        )
        tolerance = float(override.attrs.get('integral_tolerance', 2e-6))
        if maximum_error is None:
            raise ValueError(
                'SFH override does not record its stored-integral validation.'
            )
        if float(maximum_error) > tolerance:
            raise ValueError(
                f'SFH override normalization error {float(maximum_error):.3g} '
                f'exceeds tolerance {tolerance:.3g}.'
            )
    return tolerance, float(maximum_error)


def _inference_transform(image_size, num_output_channels=3):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=num_output_channels),
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
                 training=True, sample_posterior=True, image_size=224,
                 image_format='jpg', image_stats=None, asinh_scale=20.0,
                 sfh_override_path=None):
        self.sfh_path = str(Path(sfh_path))
        self.sfh_override_path = (
            str(Path(sfh_override_path)) if sfh_override_path is not None else None
        )
        if self.sfh_override_path is not None and sample_posterior:
            raise ValueError('SFH overrides require sample_posterior=False.')
        self.image_format = image_format
        if image_format == 'jpg':
            self.stamp_dir = Path(stamp_root) / band
        elif image_format == 'fits':
            self.stamp_dir = resolve_fits_directory(stamp_root, band)
        else:
            raise ValueError("image_format must be 'jpg' or 'fits'.")
        self.band = band
        self.rows = np.asarray(rows, dtype=np.int64)
        self.galaxy_ids = np.asarray(galaxy_ids, dtype=np.int64)
        self.training = training
        self.sample_posterior = sample_posterior
        if image_format == 'jpg':
            self.transform = (
                _training_transform(image_size) if training
                else _inference_transform(image_size)
            )
        else:
            if image_stats is None:
                raise ValueError('FITS image loading requires image_stats.')
            p_lo, p_hi = load_image_stats(image_stats, 'euclid-vis')
            self.transform = VISFitsTransform(
                image_size, p_lo, p_hi, asinh_scale, training,
            )
        self._h5 = None
        self._h5_pid = None
        self._sfh_override = None
        self._sfh_override_pid = None
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
        state['_sfh_override'] = None
        state['_sfh_override_pid'] = None
        return state

    def __del__(self):
        source = getattr(self, '_h5', None)
        if source is not None:
            source.close()
        override = getattr(self, '_sfh_override', None)
        if override is not None:
            override.close()

    def _sfh_source(self):
        if self.sfh_override_path is None:
            return self._source()
        pid = os.getpid()
        if self._sfh_override is None or self._sfh_override_pid != pid:
            if self._sfh_override is not None:
                self._sfh_override.close()
            self._sfh_override = h5py.File(self.sfh_override_path, 'r')
            self._sfh_override_pid = pid
        return self._sfh_override

    def __getitem__(self, index):
        row = int(self.rows[index])
        galaxy_id = int(self.galaxy_ids[index])
        source = self._source()
        sfh_source = self._sfh_source()
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
            sfh = np.asarray(sfh_source['sfh'][row], dtype=np.float32)
        if not np.all(np.isfinite(sfh)):
            raise ValueError(f'Nonfinite SFH for galaxy_id={galaxy_id}.')

        # A deterministic reference is used to define SFH-neighbour targets,
        # even when a posterior realization is sampled as encoder input.
        if realization_index >= 0:
            sfh_reference = np.asarray(sfh_source['sfh'][row], dtype=np.float32)
        else:
            sfh_reference = sfh

        if self.sfh_override_path is not None:
            p16 = p84 = sfh
        else:
            p16 = np.asarray(source['sfh_p16'][row], dtype=np.float32)
            p84 = np.asarray(source['sfh_p84'][row], dtype=np.float32)

        if self.image_format == 'jpg':
            path = self.stamp_dir / f'{self.band}_{galaxy_id}.jpg'
            with Image.open(path) as image:
                image_tensor = self.transform(image.convert('L'))
        else:
            path = vis_fits_path(self.stamp_dir, galaxy_id, self.band)
            image_tensor = self.transform(path)
        return {
            'image': image_tensor,
            'sfh': torch.from_numpy(sfh.copy()),
            'sfh_reference': torch.from_numpy(sfh_reference.copy()),
            'sfh_p16': torch.from_numpy(
                p16.copy(),
            ),
            'sfh_p84': torch.from_numpy(
                p84.copy(),
            ),
            'galaxy_id': torch.tensor(galaxy_id, dtype=torch.int64),
            'sfh_realization': torch.tensor(realization_index, dtype=torch.int64),
        }


class EuclidZooBotDataModule(L.LightningDataModule):
    def __init__(self, sfh_path, stamp_root, band='VIS', batch_size=128,
                 num_workers=8, image_size=224, val_fraction=0.1, seed=42,
                 sample_posterior=True, max_pairs=None, max_vis_mag=None,
                 vis_flux_column='flux_detection_total',
                 vis_detection_column='vis_det',
                 require_vis_detection=True, selection_catalog=None,
                 selection_id_column='object_id', image_format='jpg',
                 image_stats=None, asinh_scale=20.0,
                 eligibility_stamp_root=None,
                 exclude_edge_on_axis_ratio_below=None,
                 exclude_edge_on_probability_above=None,
                 sfh_override_path=None):
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
        self.max_vis_mag = max_vis_mag
        self.vis_flux_column = vis_flux_column
        self.vis_detection_column = vis_detection_column
        self.require_vis_detection = require_vis_detection
        self.selection_catalog = selection_catalog
        self.selection_id_column = selection_id_column
        self.image_format = image_format
        self.image_stats = image_stats
        self.asinh_scale = asinh_scale
        self.eligibility_stamp_root = eligibility_stamp_root
        self.exclude_edge_on_axis_ratio_below = exclude_edge_on_axis_ratio_below
        self.exclude_edge_on_probability_above = exclude_edge_on_probability_above
        self.sfh_override_path = sfh_override_path
        self.pair_index = None

    def setup(self, stage=None):
        if self.pair_index is None:
            if self.sfh_override_path is not None:
                tolerance, error = validate_sfh_override(
                    self.sfh_path, self.sfh_override_path,
                )
                print(
                    f'[EuclidImageSFH] override={self.sfh_override_path}; '
                    f'max |integral-1|={error:.3g} <= {tolerance:.3g}',
                    flush=True,
                )
            self.pair_index = build_pair_index(
                self.sfh_path, self.stamp_root, self.band,
                self.val_fraction, self.seed, self.max_pairs,
                self.max_vis_mag, self.vis_flux_column,
                self.vis_detection_column, self.require_vis_detection,
                self.selection_catalog, self.selection_id_column,
                self.image_format,
                self.eligibility_stamp_root,
                self.exclude_edge_on_axis_ratio_below,
                self.exclude_edge_on_probability_above,
            )
            index = self.pair_index
            selection = (
                f'; VIS<={self.max_vis_mag:g} AB'
                if self.max_vis_mag is not None else ''
            )
            if self.exclude_edge_on_axis_ratio_below is not None:
                selection += (
                    f'; removed {index.n_edge_on_excluded:,} with '
                    f'b/a<{self.exclude_edge_on_axis_ratio_below:g} and '
                    f'P(edge-on)>{self.exclude_edge_on_probability_above:g}'
                )
            print(
                f'[EuclidImageSFH:{self.image_format}] '
                f'images={index.n_stamp_paired:,}/{index.n_sfh:,}; '
                f'paired after selection={index.n_paired:,}{selection}; '
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
            image_format=self.image_format,
            image_stats=self.image_stats,
            asinh_scale=self.asinh_scale,
            sfh_override_path=self.sfh_override_path,
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
