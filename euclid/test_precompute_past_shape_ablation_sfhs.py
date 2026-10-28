import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from euclid.precompute_past_shape_ablation_sfhs import build_dataset


class PastShapeAblationDatasetTests(unittest.TestCase):
    def test_recent_segment_is_preserved_and_past_donor_is_different(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.h5"
            output_path = root / "ablation.h5"
            rng = np.random.default_rng(7)
            time = np.linspace(0.0, 1.0, 40)
            weights = rng.lognormal(-2.0, 1.0, size=(24, len(time)))
            weights /= weights.sum(axis=1, keepdims=True)
            epsilon = 1e-10
            with h5py.File(source_path, "w") as source:
                source.create_dataset("galaxy_id", data=np.arange(200, 224))
                source.create_dataset("sfh_time_grid", data=time)
                source.create_dataset(
                    "sfh", data=np.log10(weights + epsilon).astype(np.float32),
                )
                source.attrs["sfh_log_epsilon"] = epsilon
            report = build_dataset(
                source_path, output_path, seed=11, transition_bins=4,
                chunk_size=7,
            )
            split = int(np.searchsorted(time, 0.1, side="right"))
            with h5py.File(output_path, "r") as result:
                recovered = np.maximum(
                    10.0 ** result["sfh"][:].astype(float) - epsilon, 0.0,
                )
                original = np.maximum(
                    10.0 ** np.log10(weights + epsilon) - epsilon, 0.0,
                )
                np.testing.assert_allclose(
                    recovered[:, :split], original[:, :split], atol=2e-6, rtol=0,
                )
                np.testing.assert_allclose(recovered.sum(axis=1), 1.0, atol=2e-6)
                self.assertTrue(np.all(
                    result["past_shape_donor_row"][:] != np.arange(len(weights))
                ))
            self.assertTrue(report["all_past_donors_distinct_from_target"])
            self.assertLessEqual(report["maximum_recent_segment_error"], 2e-6)


if __name__ == "__main__":
    unittest.main()
