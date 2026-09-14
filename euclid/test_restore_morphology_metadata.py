import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table

from euclid.restore_morphology_metadata import align_catalog, restore_metadata


class RestoreMorphologyMetadataTests(unittest.TestCase):
    def test_exact_id_join_adds_columns_and_preserves_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'sfh.h5'
            with h5py.File(dataset, 'w') as target:
                target['galaxy_id'] = np.array([30, 10, 20], dtype=np.int64)
                target['sfh'] = np.arange(12, dtype=np.float32).reshape(3, 4)
                target['sersic_sersic_vis_index'] = [1.0, 2.0, 3.0]
            clean = root / 'clean.fits'
            Table({
                'object_id': np.array([10, 20, 30]),
                'sersic_sersic_vis_radius': [1.1, 2.2, 3.3],
                'kron_radius': [4.0, 5.0, 6.0],
            }).write(clean)
            mer = root / 'mer.fits'
            Table({
                'object_id': np.array([20, 30, 10]),
                'kron_radius': [50.0, 60.0, 40.0],
                'segmentation_area': [200.0, 300.0, 100.0],
                'ellipticity': [0.2, 0.3, 0.1],
            }).write(mer)

            added = restore_metadata(dataset, [clean, mer])
            self.assertIn('segmentation_area', added)
            with h5py.File(dataset, 'r') as source:
                np.testing.assert_array_equal(
                    source['sfh'][:], np.arange(12, dtype=np.float32).reshape(3, 4)
                )
                np.testing.assert_allclose(
                    source['sersic_sersic_vis_radius'][:], [3.3, 1.1, 2.2]
                )
                np.testing.assert_allclose(source['kron_radius'][:], [60, 40, 50])
                np.testing.assert_allclose(source['segmentation_area'][:], [300, 100, 200])
                np.testing.assert_allclose(
                    source['sersic_sersic_vis_index'][:], [1, 2, 3]
                )

    def test_incomplete_join_is_rejected_before_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'sfh.h5'
            with h5py.File(dataset, 'w') as target:
                target['galaxy_id'] = [10, 20]
            catalog = root / 'catalog.fits'
            Table({'object_id': [10], 'ellipticity': [0.2]}).write(catalog)
            with self.assertRaisesRegex(ValueError, '1/2'):
                restore_metadata(dataset, [catalog])
            with h5py.File(dataset, 'r') as source:
                self.assertNotIn('ellipticity', source)

    def test_float_object_ids_are_rejected(self):
        table = Table({'object_id': [10.0], 'ellipticity': [0.2]})
        with self.assertRaisesRegex(ValueError, 'floating point'):
            align_catalog(table, np.array([10], dtype=np.int64))


if __name__ == '__main__':
    unittest.main()
