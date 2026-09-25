import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


class FullEmbeddingSelectionTest(unittest.TestCase):
    def test_selects_only_rows_with_stamps(self):
        from euclid.export_zoobot_clip_full_embeddings import select_stamp_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "sample.h5"
            with h5py.File(dataset, "w") as target:
                target["galaxy_id"] = np.array([11, 12, 13, 14])
                target["sfh"] = np.zeros((4, 7), dtype=np.float32)
                target["sfh_realizations"] = np.zeros((4, 3, 7), dtype=np.float32)
                target["sfh_realization_valid"] = np.ones((4, 3), dtype=bool)
                target["sfh_time_grid"] = np.linspace(0, 1, 7, dtype=np.float32)
                target.attrs["n_galaxies"] = 4
            stamps = root / "stamps" / "VIS"
            stamps.mkdir(parents=True)
            (stamps / "VIS_12.jpg").touch()
            (stamps / "VIS_14.jpg").touch()
            rows, ids, n_bins, n_realizations = select_stamp_rows(
                dataset, root / "stamps",
            )
            np.testing.assert_array_equal(rows, [1, 3])
            np.testing.assert_array_equal(ids, [12, 14])
            self.assertEqual(n_bins, 7)
            self.assertEqual(n_realizations, 3)


if __name__ == "__main__":
    unittest.main()
