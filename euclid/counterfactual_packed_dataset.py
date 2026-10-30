"""Single-neighbour Euclid image/SFH dataset backed by packed VIS memmaps."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class PackedCounterfactualDataset(Dataset):
    def __init__(self, path, packed_vis, rows, donor_pools, sfh_mean, sfh_scale,
                 training=True):
        self.path = str(Path(path))
        self.packed_vis = Path(packed_vis)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.donor_pools = np.asarray(donor_pools, dtype=np.int64)
        self.sfh_mean = np.asarray(sfh_mean, dtype=np.float32)
        self.sfh_scale = np.asarray(sfh_scale, dtype=np.float32)
        self.training = bool(training)
        self._images = None
        self._sfhs = None
        self._ids = None
        self._pid = None
        metadata = json.loads((self.packed_vis / "metadata.json").read_text())
        self.image_size = int(metadata["image_size"])
        if self.donor_pools.shape[0] != len(self.rows):
            raise ValueError("One donor pool is required per target")
        if self.donor_pools.shape[1] < 2:
            raise ValueError("Two candidates are required for independent modality neighbours")
        packed_ids = np.load(self.packed_vis / "galaxy_id.npy", mmap_mode="r")
        valid = np.load(self.packed_vis / "valid.npy", mmap_mode="r")
        with h5py.File(self.path, "r") as source:
            source_ids = np.asarray(source["galaxy_id"], dtype=np.int64)
        if not np.array_equal(packed_ids, source_ids):
            raise ValueError("Packed VIS rows do not match the SFH HDF5 row order")
        relevant = np.unique(np.r_[self.rows, self.donor_pools.ravel()])
        if not np.all(valid[relevant]):
            raise ValueError("Packed VIS data contain invalid requested rows")

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
        state["_images"] = None
        state["_sfhs"] = None
        state["_ids"] = None
        state["_pid"] = None
        return state

    def _sfh(self, row):
        clr = np.asarray(self._sfhs[int(row)], dtype=np.float32)
        return torch.from_numpy(((clr - self.sfh_mean) / self.sfh_scale).copy())

    @staticmethod
    def _augment(image):
        image = torch.rot90(image, int(torch.randint(4, ())), (-2, -1))
        if torch.rand(()) < 0.5:
            image = image.flip(-1)
        return image

    def __getitem__(self, index):
        self._open()
        row = int(self.rows[index])
        pool = self.donor_pools[index]
        if self.training:
            chosen = torch.randperm(len(pool))[:2]
            image_donor, sfh_donor = (int(pool[int(value)]) for value in chosen)
        else:
            image_donor, sfh_donor = int(pool[0]), int(pool[1])
        image = torch.from_numpy(np.asarray(self._images[row], dtype=np.float32).copy())[None]
        donor_image = torch.from_numpy(
            np.asarray(self._images[image_donor], dtype=np.float32).copy(),
        )[None]
        if self.training:
            image = self._augment(image)
            donor_image = self._augment(donor_image)
        return {
            "image": image,
            "sfh": self._sfh(row),
            "donor_image": donor_image,
            "donor_sfh": self._sfh(sfh_donor),
            "galaxy_id": torch.tensor(int(self._ids[row]), dtype=torch.int64),
            "image_donor_id": torch.tensor(int(self._ids[image_donor]), dtype=torch.int64),
            "sfh_donor_id": torch.tensor(int(self._ids[sfh_donor]), dtype=torch.int64),
        }
