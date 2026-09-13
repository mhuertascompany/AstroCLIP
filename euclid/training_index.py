"""Validate and match preprocessed Euclid SFHs to ZooBot JPEG stamps."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class EuclidPairIndex:
    train_rows: np.ndarray
    train_ids: np.ndarray
    val_rows: np.ndarray
    val_ids: np.ndarray
    n_bins: int
    n_realizations: int
    n_sfh: int
    n_paired: int


def inspect_sfh_file(path):
    """Validate the CLIP-ready HDF5 schema and return its dimensions."""
    path = Path(path)
    required = (
        'galaxy_id', 'sfh', 'sfh_realizations',
        'sfh_realization_valid', 'sfh_time_grid',
    )
    with h5py.File(path, 'r') as source:
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f'Missing Euclid SFH datasets: {missing}')
        ids = source['galaxy_id'][:]
        sfh_shape = source['sfh'].shape
        realization_shape = source['sfh_realizations'].shape
        valid_shape = source['sfh_realization_valid'].shape
        time_shape = source['sfh_time_grid'].shape
        declared_n = int(source.attrs.get('n_galaxies', len(ids)))

    if ids.ndim != 1 or ids.dtype.kind not in 'iu':
        raise ValueError('galaxy_id must be a one-dimensional integer dataset.')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('galaxy_id contains duplicates.')
    if len(time_shape) != 1 or sfh_shape != (len(ids), time_shape[0]):
        raise ValueError('sfh and sfh_time_grid shapes are inconsistent.')
    if len(realization_shape) != 3 or realization_shape[0] != len(ids):
        raise ValueError('sfh_realizations must have shape (galaxy, realization, bin).')
    if realization_shape[2] != sfh_shape[1]:
        raise ValueError('Median and realization SFHs use different bin counts.')
    if valid_shape != realization_shape[:2]:
        raise ValueError('sfh_realization_valid has the wrong shape.')
    if declared_n != len(ids):
        raise ValueError('n_galaxies does not match galaxy_id.')
    return ids, sfh_shape[1], realization_shape[1]


def build_pair_index(sfh_path, stamp_root, band='VIS', val_fraction=0.1,
                     seed=42, max_pairs=None):
    """Match IDs by filename and make a deterministic random train/val split."""
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must lie strictly between zero and one.')
    if max_pairs is not None and max_pairs < 2:
        raise ValueError('max_pairs must be at least two.')

    ids, n_bins, n_realizations = inspect_sfh_file(sfh_path)
    stamp_dir = Path(stamp_root) / band
    if not stamp_dir.is_dir():
        raise FileNotFoundError(f'Stamp directory not found: {stamp_dir}')
    paired = np.array([
        (stamp_dir / f'{band}_{int(object_id)}.jpg').is_file()
        for object_id in ids
    ], dtype=bool)
    paired_rows = np.flatnonzero(paired)
    if len(paired_rows) < 2:
        raise ValueError(f'Only {len(paired_rows)} SFH/image pairs were found.')

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    is_validation = np.zeros(len(ids), dtype=bool)
    n_val_all = max(1, int(round(val_fraction * len(ids))))
    is_validation[order[:n_val_all]] = True
    train_rows = paired_rows[~is_validation[paired_rows]]
    val_rows = paired_rows[is_validation[paired_rows]]
    if not len(train_rows) or not len(val_rows):
        raise ValueError('Image filtering left an empty train or validation split.')

    if max_pairs is not None and len(paired_rows) > max_pairs:
        n_val = max(1, int(round(max_pairs * val_fraction)))
        n_train = max_pairs - n_val
        train_rows = rng.permutation(train_rows)[:n_train]
        val_rows = rng.permutation(val_rows)[:n_val]
        if len(train_rows) < n_train or len(val_rows) < n_val:
            raise ValueError('max_pairs cannot preserve the requested split after matching.')

    return EuclidPairIndex(
        train_rows=np.asarray(train_rows, dtype=np.int64),
        train_ids=np.asarray(ids[train_rows], dtype=np.int64),
        val_rows=np.asarray(val_rows, dtype=np.int64),
        val_ids=np.asarray(ids[val_rows], dtype=np.int64),
        n_bins=n_bins,
        n_realizations=n_realizations,
        n_sfh=len(ids),
        n_paired=len(paired_rows),
    )
