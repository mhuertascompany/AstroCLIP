"""Tests for Euclid SFH preprocessing."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from euclid.preprocess_sfhs import (
    SFH_EPS,
    preprocess_catalog,
    sfh_realizations_to_common_grid,
    sfh_to_common_grid,
)


class CommonGridTests(unittest.TestCase):
    def test_output_is_log_of_unit_normalized_shape(self):
        age = np.array([25.0, 75.0, 125.0, 175.0])
        realizations = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [0.5, 1.0, 1.5, 2.0],
        ])
        grid = np.linspace(0, 1, 7)
        result, universe_age = sfh_to_common_grid(
            age, realizations, redshift=1.0, time_grid=grid,
        )
        weights = np.maximum(10.0 ** result - SFH_EPS, 0.0)
        self.assertEqual(result.shape, (7,))
        self.assertGreater(universe_age, 0)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_amplitude_does_not_change_shape(self):
        age = np.arange(25.0, 5000.0, 50.0)
        realization = np.exp(-age / 1200.0)
        a, _ = sfh_to_common_grid(age, np.stack([realization] * 3), 0.8)
        b, _ = sfh_to_common_grid(age, np.stack([realization * 1e8] * 3), 0.8)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

    def test_each_realization_is_preserved_and_normalized(self):
        age = np.arange(25.0, 5000.0, 50.0)
        realizations = np.stack([
            np.exp(-age / 800.0),
            np.exp(-age / 1600.0),
            1.0 - 0.5 * age / age.max(),
        ])
        result, _ = sfh_realizations_to_common_grid(age, realizations, 0.8)
        weights = np.maximum(10.0 ** result - SFH_EPS, 0.0)
        self.assertEqual(result.shape, realizations.shape)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        self.assertFalse(np.allclose(result[0], result[1]))

    def test_narrow_burst_is_not_lost_between_output_bins(self):
        age = np.arange(25.0, 1025.0, 50.0)
        realizations = np.zeros((2, len(age)))
        realizations[0, 7] = 1.0
        realizations[1, 12] = 1.0
        result, _ = sfh_realizations_to_common_grid(
            age, realizations, redshift=0.01, time_grid=np.linspace(0, 1, 8),
        )
        weights = np.maximum(10.0 ** result - SFH_EPS, 0.0)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        self.assertGreater(np.count_nonzero(weights[0]), 0)
        self.assertGreater(np.count_nonzero(weights[1]), 0)

    def test_unphysical_realization_is_flagged(self):
        age = np.arange(25.0, 13025.0, 50.0)
        realizations = np.zeros((2, len(age)))
        realizations[0, 10] = 1.0
        realizations[1, -1] = 1.0
        result, _, retained = sfh_realizations_to_common_grid(
            age, realizations, redshift=1.0, return_diagnostics=True,
        )
        self.assertTrue(np.isfinite(result[0]).all())
        self.assertTrue(np.isnan(result[1]).all())
        self.assertGreater(retained[0], 0)
        self.assertEqual(retained[1], 0)

    def test_zero_weight_realization_is_flagged(self):
        age = np.arange(25.0, 1025.0, 50.0)
        realizations = np.zeros((2, len(age)))
        realizations[0] = np.exp(-age / 500.0)
        result, _, retained = sfh_realizations_to_common_grid(
            age, realizations, redshift=0.5, return_diagnostics=True,
        )
        self.assertTrue(np.isfinite(result[0]).all())
        self.assertTrue(np.isnan(result[1]).all())
        self.assertGreater(retained[0], 0)
        self.assertEqual(retained[1], 0)


class CatalogTests(unittest.TestCase):
    def test_default_uses_native_time_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.h5'
            output = root / 'output.h5'
            with h5py.File(source, 'w') as f:
                f['age'] = np.arange(25.0, 525.0, 50.0)
                f['object_id'] = np.array([101], dtype=np.int64)
                f['phz_pp_median_redshift'] = [0.5]
                f['sfh'] = np.ones((1, 2, 10), dtype=np.float32)

            preprocess_catalog(source, output)
            with h5py.File(output, 'r') as f:
                self.assertEqual(f['sfh'].shape, (1, 10))
                self.assertEqual(f['sfh_realizations'].shape, (1, 2, 10))
                self.assertEqual(f['sfh_realization_valid'].shape, (1, 2))
                self.assertEqual(f['sfh_retained_mass_fraction'].shape, (1, 2))
                self.assertEqual(f.attrs['sfh_n_bins'], 10)

    def test_catalog_conversion_preserves_ids_and_scalar_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.h5'
            output = root / 'output.h5'
            with h5py.File(source, 'w') as f:
                f['age'] = [25.0, 75.0, 125.0, 175.0]
                f['object_id'] = np.array([101, 202], dtype=np.int64)
                f['phz_pp_median_redshift'] = [0.5, 1.0]
                f['phz_pp_median_stellarmass'] = [9.5, 10.5]
                f['sfh'] = np.array([
                    [[1, 2, 3, 4], [2, 3, 4, 5]],
                    [[4, 3, 2, 1], [5, 4, 3, 2]],
                ], dtype=np.float32)
                f['t50'] = np.ones((2, 2), dtype=np.float32)

            preprocess_catalog(source, output, n_bins=6, batch_size=1)
            with h5py.File(output, 'r') as f:
                self.assertEqual(f['sfh'].shape, (2, 6))
                self.assertEqual(f['sfh_realizations'].shape, (2, 2, 6))
                self.assertEqual(f['sfh_p16'].shape, (2, 6))
                self.assertEqual(f['sfh_p84'].shape, (2, 6))
                np.testing.assert_array_equal(f['object_id'][:], [101, 202])
                np.testing.assert_array_equal(f['galaxy_id'][:], [101, 202])
                np.testing.assert_allclose(
                    f['redshift'][:], f['phz_pp_median_redshift'][:]
                )
                self.assertIn('phz_pp_median_stellarmass', f)
                self.assertNotIn('t50', f)
                weights = np.maximum(10.0 ** f['sfh'][:] - SFH_EPS, 0.0)
                np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
                realization_weights = np.maximum(
                    10.0 ** f['sfh_realizations'][:] - SFH_EPS, 0.0,
                )
                np.testing.assert_allclose(
                    realization_weights.sum(axis=2), 1.0, atol=1e-6,
                )
                self.assertTrue(f['sfh_realization_valid'][:].all())

    def test_catalog_preserves_partial_zero_weight_posterior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.h5'
            output = root / 'output.h5'
            with h5py.File(source, 'w') as f:
                f['age'] = [25.0, 75.0, 125.0, 175.0]
                f['object_id'] = np.array([101], dtype=np.int64)
                f['phz_pp_median_redshift'] = [0.5]
                f['sfh'] = np.array([
                    [[1, 2, 3, 4], [0, 0, 0, 0], [4, 3, 2, 1]],
                ], dtype=np.float32)

            preprocess_catalog(source, output)
            with h5py.File(output, 'r') as f:
                np.testing.assert_array_equal(
                    f['sfh_realization_valid'][0], [True, False, True],
                )
                self.assertTrue(np.isnan(f['sfh_realizations'][0, 1]).all())
                self.assertTrue(np.isfinite(f['sfh'][0]).all())
                self.assertEqual(f.attrs['sfh_invalid_realizations'], 1)

    def test_catalog_rejects_galaxy_with_no_valid_realization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.h5'
            output = root / 'output.h5'
            with h5py.File(source, 'w') as f:
                f['age'] = [25.0, 75.0, 125.0, 175.0]
                f['object_id'] = np.array([101], dtype=np.int64)
                f['phz_pp_median_redshift'] = [0.5]
                f['sfh'] = np.zeros((1, 3, 4), dtype=np.float32)

            with self.assertRaisesRegex(ValueError, 'No valid SFH realizations'):
                preprocess_catalog(source, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
