import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from euclid.umap_zoobot_clip import (
    _normalize_rows,
    _sfh_properties,
    load_embeddings,
)


class EuclidUmapDiagnosticsTests(unittest.TestCase):
    def test_normalize_rows(self):
        values = _normalize_rows([[3.0, 4.0], [0.0, 2.0]])
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0)

    def test_load_embeddings_validates_and_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'embeddings.npz'
            np.savez(
                path,
                galaxy_id=np.array([10, 20]),
                h5_row=np.array([0, 1]),
                redshift=np.array([0.5, 1.0]),
                image_embedding=np.array([[2.0, 0.0], [0.0, 3.0]]),
                sfh_embedding=np.array([[4.0, 0.0], [0.0, 5.0]]),
            )
            ids, rows, redshift, image, sfh, preprojection = load_embeddings(path)
            np.testing.assert_array_equal(ids, [10, 20])
            np.testing.assert_array_equal(rows, [0, 1])
            np.testing.assert_allclose(np.linalg.norm(image, axis=1), 1.0)
            np.testing.assert_allclose(np.linalg.norm(sfh, axis=1), 1.0)
            np.testing.assert_allclose(redshift, [0.5, 1.0])
            self.assertIsNone(preprojection)

    def test_sfh_properties_respect_recent_to_old_time_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sfh.h5'
            weights = np.array([
                [0.8, 0.1, 0.05, 0.05],
                [0.05, 0.05, 0.1, 0.8],
            ])
            with h5py.File(path, 'w') as target:
                target.attrs['sfh_log_epsilon'] = 1e-10
                target['sfh'] = np.log10(weights + 1e-10)
                target['sfh_time_grid'] = np.linspace(0, 1, 4)
            with h5py.File(path, 'r') as source:
                props = _sfh_properties(source, np.array([0, 1]))
            self.assertGreater(props['sfh_recent_20'][0], props['sfh_recent_20'][1])
            self.assertLess(props['sfh_mean_lookback'][0], props['sfh_mean_lookback'][1])
            self.assertLess(props['sfh_t50_lookback'][0], props['sfh_t50_lookback'][1])
            self.assertLess(props['sfh_log_old_recent'][0], props['sfh_log_old_recent'][1])


if __name__ == '__main__':
    unittest.main()
