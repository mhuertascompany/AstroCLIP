import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from euclid.precompute_factorized_null_sfhs import build_dataset


class FactorizedNullDatasetTests(unittest.TestCase):
    def test_compact_override_has_independent_donors_and_unit_integrals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.h5"
            output_path = root / "null.h5"
            rng = np.random.default_rng(3)
            time = np.linspace(0.0, 1.0, 40)
            weights = rng.lognormal(-2.0, 1.0, size=(24, len(time)))
            weights /= weights.sum(axis=1, keepdims=True)
            epsilon = 1e-10
            with h5py.File(source_path, "w") as source:
                source.create_dataset("galaxy_id", data=np.arange(100, 124))
                source.create_dataset("sfh_time_grid", data=time)
                source.create_dataset(
                    "sfh", data=np.log10(weights + epsilon).astype(np.float32),
                )
                source.attrs["sfh_log_epsilon"] = epsilon
            report = build_dataset(
                source_path, output_path, seed=5, transition_bins=4,
                chunk_size=7,
            )
            with h5py.File(output_path, "r") as result:
                recovered = np.maximum(
                    10.0 ** result["sfh"][:].astype(float) - epsilon, 0.0,
                )
                np.testing.assert_allclose(recovered.sum(axis=1), 1.0, atol=2e-6)
                target = np.arange(24)
                donors = [
                    result["fraction_donor_row"][:],
                    result["recent_shape_donor_row"][:],
                    result["past_shape_donor_row"][:],
                ]
                for donor in donors:
                    self.assertTrue(np.all(donor != target))
                self.assertTrue(np.all(donors[0] != donors[1]))
                self.assertTrue(np.all(donors[0] != donors[2]))
                self.assertTrue(np.all(donors[1] != donors[2]))
                self.assertLessEqual(
                    result.attrs["maximum_absolute_stored_integral_error"], 2e-6,
                )
            self.assertTrue(report["all_donors_distinct_from_target_and_each_other"])


if __name__ == "__main__":
    unittest.main()
