import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch

from euclid.dataset_zoobot import EuclidZooBotDataset


class RandomSFHPairingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.path = root / 'sfh.h5'
        self.ids = np.arange(100, 106, dtype=np.int64)
        self.rows = np.arange(len(self.ids), dtype=np.int64)
        sfh = np.stack([
            np.full(4, index + 1, dtype=np.float32)
            for index in range(len(self.ids))
        ])
        with h5py.File(self.path, 'w') as target:
            target['galaxy_id'] = self.ids
            target['sfh'] = sfh
            target['sfh_p16'] = sfh - 0.1
            target['sfh_p84'] = sfh + 0.1
            target['sfh_realizations'] = sfh[:, None, :]
            target['sfh_realization_valid'] = np.ones((len(sfh), 1), dtype=bool)
            target['sfh_time_grid'] = np.linspace(0, 1, 4, dtype=np.float32)
        self.stamps = root / 'stamps' / 'VIS'
        self.stamps.mkdir(parents=True)
        for galaxy_id in self.ids:
            Image.new('L', (12, 12), color=128).save(
                self.stamps / f'VIS_{int(galaxy_id)}.jpg'
            )

    def tearDown(self):
        self.directory.cleanup()

    def dataset(self, training):
        return EuclidZooBotDataset(
            self.path, self.stamps.parent, self.rows, self.ids,
            training=training, sample_posterior=False, image_size=8,
            random_sfh_pairing=True, pairing_seed=17,
        )

    def test_validation_pairing_is_deterministic_derangement(self):
        first = self.dataset(training=False)
        second = self.dataset(training=False)
        first_donors = [int(first[index]['sfh_donor_id']) for index in range(len(first))]
        second_donors = [int(second[index]['sfh_donor_id']) for index in range(len(second))]
        self.assertEqual(first_donors, second_donors)
        self.assertTrue(all(target != donor for target, donor in zip(self.ids, first_donors)))
        self.assertSetEqual(set(first_donors), set(map(int, self.ids)))

    def test_training_draw_never_uses_target_and_changes(self):
        dataset = self.dataset(training=True)
        torch.manual_seed(5)
        items = [dataset[0] for _ in range(30)]
        donors = [int(item['sfh_donor_id']) for item in items]
        self.assertTrue(all(donor != int(self.ids[0]) for donor in donors))
        self.assertGreater(len(set(donors)), 1)
        for item, donor in zip(items, donors):
            position = int(np.flatnonzero(self.ids == donor)[0])
            np.testing.assert_array_equal(
                item['sfh'].numpy(),
                np.full(4, position + 1, dtype=np.float32),
            )


if __name__ == '__main__':
    unittest.main()
