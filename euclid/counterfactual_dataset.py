"""Matched-class donor dataset for bidirectional Euclid flow matching."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .sfh_conditional_diffusion import log_sfh_to_weights, weights_to_clr


def activity_classes(log_sfh, time_grid, age_myr, epsilon=1e-10,
                     recent_window_myr=100.0):
    """Return Q/transition/SF classes using the canonical doubling thresholds."""
    weights = log_sfh_to_weights(log_sfh, epsilon)
    time_grid = np.asarray(time_grid, dtype=np.float64)
    age_myr = np.asarray(age_myr, dtype=np.float64)
    edges = np.r_[0.0, (time_grid[:-1] + time_grid[1:]) / 2.0, 1.0]
    widths = np.diff(edges)
    upper = np.minimum(recent_window_myr / age_myr, 1.0)
    recent = np.zeros(len(weights), dtype=np.float64)
    for column, (left, right, width) in enumerate(
        zip(edges[:-1], edges[1:], widths)
    ):
        overlap = np.maximum(0.0, np.minimum(right, upper) - left)
        recent += weights[:, column] * overlap / width
    doubling = recent * age_myr / recent_window_myr
    result = np.full(len(weights), -1, dtype=np.int8)
    valid = np.isfinite(doubling) & np.isfinite(age_myr) & (age_myr > 0)
    result[valid & (doubling <= 1 / 20)] = 0
    result[valid & (doubling > 1 / 20) & (doubling < 1 / 3)] = 1
    result[valid & (doubling >= 1 / 3)] = 2
    return result


def matched_donor_pools(train_rows, query_rows, mass, redshift, activity,
                        pool_size=64):
    """Find mass/redshift neighbours within activity class, excluding self."""
    train_rows = np.asarray(train_rows, dtype=np.int64)
    query_rows = np.asarray(query_rows, dtype=np.int64)
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    output = np.full((len(query_rows), pool_size), -1, dtype=np.int64)
    query_position = {int(row): i for i, row in enumerate(query_rows)}
    for label in (0, 1, 2):
        reference = train_rows[activity[train_rows] == label]
        queries = query_rows[activity[query_rows] == label]
        if len(reference) < 2 or not len(queries):
            continue
        center = np.array([
            np.nanmedian(mass[reference]), np.nanmedian(redshift[reference]),
        ])
        scale = np.array([
            np.nanstd(mass[reference]), np.nanstd(redshift[reference]),
        ])
        scale = np.maximum(scale, (0.1, 0.02))
        reference_values = np.column_stack((mass[reference], redshift[reference]))
        query_values = np.column_stack((mass[queries], redshift[queries]))
        tree = cKDTree((reference_values - center) / scale)
        count = min(len(reference), pool_size + 1)
        _, indices = tree.query((query_values - center) / scale, k=count)
        indices = np.atleast_2d(indices)
        if len(queries) == 1:
            indices = indices.reshape(1, -1)
        for row, candidates in zip(queries, reference[indices]):
            candidates = candidates[candidates != row]
            if len(candidates) < pool_size:
                candidates = np.resize(candidates, pool_size)
            output[query_position[int(row)]] = candidates[:pool_size]
    if np.any(output < 0):
        bad = query_rows[np.any(output < 0, axis=1)]
        raise ValueError(f"Could not construct donor pools for {len(bad)} rows")
    if np.any(output == query_rows[:, None]):
        raise RuntimeError("A target appears in its own donor pool")
    return output


def mass_window_donor_pools(train_rows, query_rows, mass, window=0.5,
                            pool_size=64, seed=42):
    """Uniformly sample donor pools inside a log-mass window only.

    Redshift and activity class deliberately do not enter this selection.  The
    sorted-mass index avoids constructing the full query-by-reference distance
    matrix for the 150k sample.
    """
    train_rows = np.asarray(train_rows, dtype=np.int64)
    query_rows = np.asarray(query_rows, dtype=np.int64)
    mass = np.asarray(mass, dtype=np.float64)
    if not np.isfinite(window) or window <= 0:
        raise ValueError("mass window must be finite and positive")
    if pool_size < 2:
        raise ValueError("pool_size must be at least two")
    valid_train = train_rows[np.isfinite(mass[train_rows])]
    order = np.argsort(mass[valid_train], kind="stable")
    sorted_rows = valid_train[order]
    sorted_mass = mass[sorted_rows]
    rng = np.random.default_rng(seed)
    output = np.empty((len(query_rows), pool_size), dtype=np.int64)
    for position, row in enumerate(query_rows):
        value = mass[row]
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite mass for query row {row}")
        left = np.searchsorted(sorted_mass, value - window, side="left")
        right = np.searchsorted(sorted_mass, value + window, side="right")
        candidates = sorted_rows[left:right]
        candidates = candidates[candidates != row]
        if len(candidates) < 2:
            raise ValueError(
                f"Fewer than two mass-window donors for query row {row}"
            )
        replace = len(candidates) < pool_size
        output[position] = rng.choice(
            candidates, size=pool_size, replace=replace,
        )
    if np.any(output == query_rows[:, None]):
        raise RuntimeError("A target appears in its own mass-window donor pool")
    return output


def fit_sfh_transform(log_sfh, train_rows, floor=1e-8):
    clr = weights_to_clr(log_sfh_to_weights(log_sfh[train_rows]), floor)
    mean = clr.mean(0, dtype=np.float64)
    scale = np.maximum(clr.std(0, dtype=np.float64), 1e-3)
    return mean.astype(np.float32), scale.astype(np.float32)


class CounterfactualEuclidDataset(Dataset):
    def __init__(self, path, stamp_root, rows, donor_pools, mass, redshift,
                 activity, sfh_mean, sfh_scale, covariate_mean,
                 covariate_scale, donors_per_item=8, image_size=64,
                 training=True, band="VIS"):
        self.path = str(Path(path))
        self.stamp_root = Path(stamp_root) / band
        self.rows = np.asarray(rows, dtype=np.int64)
        self.donor_pools = np.asarray(donor_pools, dtype=np.int64)
        self.mass = np.asarray(mass, dtype=np.float32)
        self.redshift = np.asarray(redshift, dtype=np.float32)
        self.activity = np.asarray(activity, dtype=np.int8)
        self.sfh_mean = np.asarray(sfh_mean, dtype=np.float32)
        self.sfh_scale = np.asarray(sfh_scale, dtype=np.float32)
        self.covariate_mean = np.asarray(covariate_mean, dtype=np.float32)
        self.covariate_scale = np.asarray(covariate_scale, dtype=np.float32)
        self.donors_per_item = int(donors_per_item)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.band = band
        self._source_handle = None
        self._pid = None
        if self.donor_pools.shape[0] != len(self.rows):
            raise ValueError("One donor pool is required per target row")
        if not 1 <= self.donors_per_item <= self.donor_pools.shape[1]:
            raise ValueError("Invalid donors_per_item")

    def __len__(self):
        return len(self.rows)

    def _source(self):
        pid = os.getpid()
        if self._source_handle is None or self._pid != pid:
            if self._source_handle is not None:
                self._source_handle.close()
            self._source_handle = h5py.File(self.path, "r")
            self._pid = pid
        return self._source_handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_source_handle"] = None
        state["_pid"] = None
        return state

    def _image(self, galaxy_id, augment=False):
        path = self.stamp_root / f"{self.band}_{int(galaxy_id)}.jpg"
        with Image.open(path) as source:
            image = source.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC,
            )
            values = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(values.copy())[None]
        if augment:
            tensor = torch.rot90(tensor, int(torch.randint(4, ())), (-2, -1))
            if torch.rand(()) < 0.5:
                tensor = tensor.flip(-1)
        return tensor

    def _sfh(self, source, row):
        log_sfh = np.asarray(source["sfh"][int(row)], dtype=np.float32)
        clr = weights_to_clr(log_sfh_to_weights(log_sfh))
        return torch.from_numpy(((clr - self.sfh_mean) / self.sfh_scale).copy())

    def __getitem__(self, index):
        source = self._source()
        row = int(self.rows[index])
        pool = self.donor_pools[index]
        if self.training:
            chosen = torch.randperm(len(pool))[:self.donors_per_item].numpy()
            donors = pool[chosen]
        else:
            donors = pool[:self.donors_per_item]
        galaxy_id = int(source["galaxy_id"][row])
        donor_ids = np.asarray(
            [source["galaxy_id"][int(value)] for value in donors],
            dtype=np.int64,
        )
        image = self._image(galaxy_id, augment=self.training)
        donor_images = torch.stack([
            self._image(value, augment=self.training) for value in donor_ids
        ])
        sfh = self._sfh(source, row)
        donor_sfhs = torch.stack([self._sfh(source, value) for value in donors])
        continuous = (
            np.array([self.mass[row], self.redshift[row]], dtype=np.float32)
            - self.covariate_mean
        ) / self.covariate_scale
        one_hot = np.eye(3, dtype=np.float32)[self.activity[row]]
        return {
            "image": image,
            "sfh": sfh,
            "donor_images": donor_images,
            "donor_sfhs": donor_sfhs,
            "covariates": torch.from_numpy(np.r_[continuous, one_hot]),
            "galaxy_id": torch.tensor(galaxy_id, dtype=torch.int64),
            "donor_ids": torch.from_numpy(donor_ids.copy()),
        }
