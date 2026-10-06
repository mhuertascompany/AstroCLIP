import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from euclid.render_progenitor_candidate_images import load_candidates


class ProgenitorCandidateImageTests(unittest.TestCase):
    def test_uses_exact_ranked_candidate_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.csv"
            pd.DataFrame({
                "descendant_id": [10, 10, 10, 10, 10, 10],
                "stage": [1, 1, 1, 2, 2, 2],
                "rank": [1, 2, 3, 1, 2, 3],
                "galaxy_id": [11, 12, 15, 13, 14, 16],
                "state_lookback_gyr": [1, 1, 1, 2, 2, 2],
                "formed_mass_fraction": [.9, .9, .9, .8, .8, .8],
            }).to_csv(path, index=False)
            ids = np.asarray([10, 11, 12, 13, 14, 15, 16])
            conditions = np.eye(7, dtype=np.float32)
            records, vectors = load_candidates(path, 10, ids, conditions, 2)
            self.assertEqual([row["galaxy_id"] for row in records], [10, 11, 12, 13, 14])
            np.testing.assert_array_equal(vectors, conditions[:5])
            self.assertEqual([row["rank"] for row in records], [0, 1, 2, 1, 2])


if __name__ == "__main__":
    unittest.main()
