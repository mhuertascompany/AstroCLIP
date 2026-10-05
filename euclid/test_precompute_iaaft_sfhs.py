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


if __name__ == "__main__":
    unittest.main()
