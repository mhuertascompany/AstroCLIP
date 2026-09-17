import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from euclid.training_index import build_pair_index, inspect_sfh_file


def write_sfh_file(path, ids, n_realizations=3, n_bins=4,
                   vis_magnitudes=None, vis_detected=None):
    ids = np.asarray(ids, dtype=np.int64)
    with h5py.File(path, 'w') as target:
        target.create_dataset('galaxy_id', data=ids)
        target.create_dataset(
            'sfh', data=np.full((len(ids), n_bins), 1 / n_bins, dtype=np.float32),
        )
        target.create_dataset(
            'sfh_realizations',
            data=np.full(
                (len(ids), n_realizations, n_bins),
                1 / n_bins,
                dtype=np.float32,
            ),
        )
        target.create_dataset(
            'sfh_realization_valid',
            data=np.ones((len(ids), n_realizations), dtype=bool),
        )
        target.create_dataset(
            'sfh_time_grid', data=np.linspace(0, 1, n_bins, dtype=np.float32),
        )
        target.attrs['n_galaxies'] = len(ids)
        if vis_magnitudes is not None:
            vis_magnitudes = np.asarray(vis_magnitudes, dtype=np.float64)
            target.create_dataset(
                'flux_detection_total',
                data=10.0 ** ((23.9 - vis_magnitudes) / 2.5),
            )
            target.create_dataset(
                'vis_det',
                data=(
                    np.ones(len(ids), dtype=np.int16)
                    if vis_detected is None else np.asarray(vis_detected)
                ),
            )


class EuclidTrainingIndexTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.sfh_path = self.root / 'sfh.h5'
        self.ids = np.arange(1000, 1020, dtype=np.int64)
        write_sfh_file(self.sfh_path, self.ids)
        self.stamp_dir = self.root / 'stamps' / 'VIS'
        self.stamp_dir.mkdir(parents=True)
        self.available = self.ids[[0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 14, 17, 19]]
        for object_id in self.available:
            Image.new('L', (8, 8), color=128).save(
                self.stamp_dir / f'VIS_{int(object_id)}.jpg',
            )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_inspect_sfh_file(self):
        ids, n_bins, n_realizations = inspect_sfh_file(self.sfh_path)
        np.testing.assert_array_equal(ids, self.ids)
        self.assertEqual(n_bins, 4)
        self.assertEqual(n_realizations, 3)

    def test_pairing_is_exact_disjoint_and_reproducible(self):
        first = build_pair_index(
            self.sfh_path, self.root / 'stamps', val_fraction=0.25, seed=7,
        )
        second = build_pair_index(
            self.sfh_path, self.root / 'stamps', val_fraction=0.25, seed=7,
        )
        combined = np.concatenate([first.train_ids, first.val_ids])
        self.assertEqual(first.n_sfh, len(self.ids))
        self.assertEqual(first.n_paired, len(self.available))
        self.assertSetEqual(set(combined), set(self.available))
        self.assertFalse(set(first.train_ids) & set(first.val_ids))
        np.testing.assert_array_equal(first.train_rows, second.train_rows)
        np.testing.assert_array_equal(first.val_rows, second.val_rows)

    def test_max_pairs_limits_the_smoke_test(self):
        index = build_pair_index(
            self.sfh_path,
            self.root / 'stamps',
            val_fraction=0.25,
            seed=1,
            max_pairs=8,
        )
        self.assertEqual(len(index.train_rows), 6)
        self.assertEqual(len(index.val_rows), 2)

    def test_duplicate_ids_are_rejected(self):
        duplicate_path = self.root / 'duplicate.h5'
        write_sfh_file(duplicate_path, [1, 1])
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            inspect_sfh_file(duplicate_path)

    def test_vis_magnitude_cut_preserves_base_validation_membership(self):
        bright_path = self.root / 'bright.h5'
        magnitudes = np.linspace(20.0, 23.8, len(self.ids))
        write_sfh_file(bright_path, self.ids, vis_magnitudes=magnitudes)
        unrestricted = build_pair_index(
            bright_path, self.root / 'stamps', val_fraction=0.25, seed=7,
        )
        bright = build_pair_index(
            bright_path, self.root / 'stamps', val_fraction=0.25, seed=7,
            max_vis_mag=22.0,
        )
        expected = {
            int(value) for value, magnitude in zip(self.ids, magnitudes)
            if magnitude <= 22.0 and value in self.available
        }
        self.assertSetEqual(
            set(np.concatenate([bright.train_ids, bright.val_ids])), expected,
        )
        self.assertSetEqual(set(bright.val_ids), set(unrestricted.val_ids) & expected)
        self.assertEqual(bright.n_stamp_paired, len(self.available))

    def test_vis_cut_excludes_nir_detected_objects(self):
        path = self.root / 'detection.h5'
        detected = np.ones(len(self.ids), dtype=np.int16)
        detected[0] = 0
        write_sfh_file(
            path, self.ids, vis_magnitudes=np.full(len(self.ids), 20.0),
            vis_detected=detected,
        )
        index = build_pair_index(
            path, self.root / 'stamps', val_fraction=0.25, seed=7,
            max_vis_mag=22.0,
        )
        selected = set(np.concatenate([index.train_ids, index.val_ids]))
        self.assertNotIn(int(self.ids[0]), selected)


if __name__ == '__main__':
    unittest.main()
