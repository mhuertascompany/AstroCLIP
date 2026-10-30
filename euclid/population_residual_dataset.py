"""Packed Euclid targets and population covariates for residual-flow training."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PackedPopulationResidualDataset(Dataset):
    """Read paired VIS/SFH targets from the packed memmaps.

    The population condition is deliberately restricted to standardized
    log-mass, standardized redshift, and a three-class one-hot activity label.
    """

    def __init__(self, path, packed_vis, rows, sfh_mean, sfh_scale,
                 covariates, training=True):
        self.path = str(Path(path))
        self.packed_vis = Path(packed_vis)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.sfh_mean = np.asarray(sfh_mean, dtype=np.float32)
        self.sfh_scale = np.asarray(sfh_scale, dtype=np.float32)
        self.covariates = np.asarray(covariates, dtype=np.float32)
        self.training = bool(training)
        self._images = None
        self._sfhs = None
        self._ids = None
        self._pid = None
        metadata = json.loads((self.packed_vis / "metadata.json").read_text())
        self.image_size = int(metadata["image_size"])
        packed_ids = np.load(self.packed_vis / "galaxy_id.npy", mmap_mode="r")
        valid = np.asarray(
            np.load(self.packed_vis / "valid.npy", mmap_mode="r"), dtype=bool,
        )
        with h5py.File(self.path, "r") as source:
            source_ids = np.asarray(source["galaxy_id"], dtype=np.int64)
        if not np.array_equal(packed_ids, source_ids):
            raise ValueError("Packed VIS rows do not match the SFH HDF5 row order")
        if not np.all(valid[self.rows]):
            raise ValueError("Packed VIS data contain invalid requested rows")
        if self.covariates.shape != (len(source_ids), 5):
            raise ValueError("covariates must have shape (all galaxies, 5)")

    def __len__(self):
        return len(self.rows)

    def _open(self):
        pid = os.getpid()
        if self._pid != pid:
            self._images = np.load(
                self.packed_vis / "vis_images.npy", mmap_mode="r",
            )
            self._sfhs = np.load(
                self.packed_vis / "sfh_clr.npy", mmap_mode="r",
            )
            self._ids = np.load(
                self.packed_vis / "galaxy_id.npy", mmap_mode="r",
            )
            self._pid = pid

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in ("_images", "_sfhs", "_ids", "_pid"):
            state[name] = None
        return state

    @staticmethod
    def _augment(image):
        image = torch.rot90(image, int(torch.randint(4, ())), (-2, -1))
        if torch.rand(()) < 0.5:
            image = image.flip(-1)
        return image

    def __getitem__(self, index):
        self._open()
        row = int(self.rows[index])
        image = torch.from_numpy(
            np.asarray(self._images[row], dtype=np.float32).copy(),
        )[None]
        if self.training:
            image = self._augment(image)
        clr = np.asarray(self._sfhs[row], dtype=np.float32)
        sfh = torch.from_numpy(
            ((clr - self.sfh_mean) / self.sfh_scale).copy(),
        )
        return {
            "image": image,
            "sfh": sfh,
            "covariates": torch.from_numpy(self.covariates[row].copy()),
            "galaxy_id": torch.tensor(int(self._ids[row]), dtype=torch.int64),
        }
