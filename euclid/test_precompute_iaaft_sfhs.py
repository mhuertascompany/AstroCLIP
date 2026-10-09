import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from euclid.precompute_iaaft_sfhs import build_dataset
from euclid.pretrain_sfh_autoencoder import (
    EuclidSFHAutoencoderDataset,
    inspect_pretraining_file,
)


class PrecomputeIAAFTTests(unittest.TestCase):
    def test_compact_dataset_roundtrips_to_unit_integrals(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source.h5"
            output_path = directory / "iaaft.h5"
            rng = np.random.default_rng(5)
            weights = rng.lognormal(-2.0, 0.8, size=(12, 250))
            weights /= weights.sum(axis=1, keepdims=True)
            epsilon = 1e-10
            with h5py.File(source_path, "w") as source:
                source.create_dataset("galaxy_id", data=np.arange(100, 112))
                source.create_dataset("sfh", data=np.log10(weights + epsilon))
                source.create_dataset("sfh_time_grid", data=np.linspace(0, 1, 250))
                source.attrs["n_galaxies"] = len(weights)
                source.attrs["sfh_log_epsilon"] = epsilon

            report = build_dataset(
                source_path, output_path, workers=1, chunk_size=5,
                candidates=2, max_iterations=100,
            )
            ids, n_bins, n_realizations = inspect_pretraining_file(
                output_path, "median",
            )
            self.assertEqual((len(ids), n_bins, n_realizations), (12, 250, 0))
            with h5py.File(output_path, "r") as result:
                recovered = np.maximum(
                    10.0 ** np.asarray(result["sfh"], dtype=np.float64) - epsilon,
                    0.0,
                )
                np.testing.assert_allclose(recovered.sum(axis=1), 1.0, atol=2e-6)
                self.assertLessEqual(
                    result.attrs["maximum_absolute_stored_integral_error"], 2e-6,
                )
                self.assertEqual(result.attrs["recent_fraction"], 0.1)
            self.assertLessEqual(report["maximum_absolute_stored_integral_error"], 2e-6)
            item = EuclidSFHAutoencoderDataset(
                output_path, np.arange(12), training=True, input_mode="median",
            )[3]
            np.testing.assert_array_equal(item["sfh"], item["target"])
            np.testing.assert_array_equal(item["p16"], item["target"])
            np.testing.assert_array_equal(item["p84"], item["target"])
            self.assertEqual(int(item["realization"]), -1)

    def test_past_preserved_dataset_keeps_old_bins_and_unit_integrals(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source.h5"
            output_path = directory / "past90.h5"
            rng = np.random.default_rng(15)
            weights = rng.lognormal(-2.0, 0.8, size=(8, 250))
            weights /= weights.sum(axis=1, keepdims=True)
            epsilon = 1e-10
            time = np.linspace(0, 1, 250)
            with h5py.File(source_path, "w") as source:
                source.create_dataset("galaxy_id", data=np.arange(8))
                source.create_dataset("sfh", data=np.log10(weights + epsilon))
                source.create_dataset("sfh_time_grid", data=time)
                source.attrs["sfh_log_epsilon"] = epsilon

            report = build_dataset(
                source_path, output_path, workers=1, chunk_size=4,
                candidates=2, max_iterations=100, preserve="past",
            )
            split = int(np.searchsorted(time, 0.1, side="right"))
            with h5py.File(output_path, "r") as result:
                recovered = np.maximum(
                    10.0 ** np.asarray(result["sfh"], dtype=np.float64) - epsilon,
                    0.0,
                )
                np.testing.assert_allclose(recovered.sum(axis=1), 1.0, atol=2e-6)
                np.testing.assert_allclose(
                    recovered[:, split:], weights[:, split:], atol=2e-7, rtol=2e-6,
                )
                self.assertEqual(result.attrs["preserved_segment"], "old")
            self.assertEqual(report["preserved_segment"], "old")
            self.assertLessEqual(report["maximum_old_segment_error"], 2e-6)

    def test_window_preserved_dataset_keeps_requested_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source.h5"
            output_path = directory / "window.h5"
            rng = np.random.default_rng(25)
            weights = rng.lognormal(-2.0, 0.8, size=(8, 250))
            weights /= weights.sum(axis=1, keepdims=True)
            epsilon = 1e-10
            time = np.linspace(0, 1, 250)
            with h5py.File(source_path, "w") as source:
                source.create_dataset("galaxy_id", data=np.arange(8))
                source.create_dataset("sfh", data=np.log10(weights + epsilon))
                source.create_dataset("sfh_time_grid", data=time)
                source.attrs["sfh_log_epsilon"] = epsilon

            report = build_dataset(
                source_path, output_path, workers=1, chunk_size=4,
                candidates=2, max_iterations=100, preserve="window",
                window_start=0.3, window_end=0.4,
            )
            first = int(np.searchsorted(time, 0.3, side="left"))
            last = int(np.searchsorted(time, 0.4, side="left"))
            with h5py.File(output_path, "r") as result:
                recovered = np.maximum(
                    10.0 ** np.asarray(result["sfh"], dtype=np.float64) - epsilon,
                    0.0,
                )
                np.testing.assert_allclose(recovered.sum(axis=1), 1.0, atol=2e-6)
                np.testing.assert_allclose(
                    recovered[:, first:last], weights[:, first:last],
                    atol=2e-7, rtol=2e-6,
                )
                self.assertEqual(result.attrs["preserved_segment"], "window")
                self.assertEqual(result.attrs["window_start"], 0.3)
                self.assertEqual(result.attrs["window_end"], 0.4)
            self.assertEqual(report["preserved_segment"], "window")
            self.assertLessEqual(report["maximum_window_segment_error"], 2e-6)

    def test_random_windows_never_preserve_recent_tenth_and_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source.h5"
            first_path = directory / "random_a.h5"
            second_path = directory / "random_b.h5"
            rng = np.random.default_rng(125)
            weights = rng.lognormal(-2.0, 0.8, size=(10, 250))
            weights /= weights.sum(axis=1, keepdims=True)
            epsilon = 1e-10
            time = np.linspace(0, 1, 250)
            with h5py.File(source_path, "w") as source:
                source.create_dataset("galaxy_id", data=np.arange(10))
                source.create_dataset("sfh", data=np.log10(weights + epsilon))
                source.create_dataset("sfh_time_grid", data=time)
                source.attrs["sfh_log_epsilon"] = epsilon
            kwargs = dict(
                workers=1, chunk_size=4, candidates=2, max_iterations=100,
                preserve="random-window", seed=77,
            )
            report = build_dataset(source_path, first_path, **kwargs)
            build_dataset(source_path, second_path, **kwargs)
            with h5py.File(first_path, "r") as first, h5py.File(second_path, "r") as second:
                starts = np.asarray(first["preserved_window_start"])
                ends = np.asarray(first["preserved_window_end"])
                self.assertTrue(np.all(starts >= 0.1))
                self.assertTrue(np.all(ends <= 1.0))
                np.testing.assert_allclose(ends - starts, 0.1, atol=1e-6)
                old_starts = np.searchsorted(time, ends, side="left")
                self.assertTrue(np.all(len(time) - old_starts >= 4))
                np.testing.assert_array_equal(
                    starts, np.asarray(second["preserved_window_start"]),
                )
                np.testing.assert_array_equal(first["sfh"], second["sfh"])
                self.assertEqual(first.attrs["preserved_segment"], "random_window")
            self.assertGreaterEqual(report["minimum_window_start"], 0.1)
            self.assertLessEqual(report["maximum_window_end"], 1.0)


if __name__ == "__main__":
    unittest.main()
