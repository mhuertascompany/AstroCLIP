import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from euclid.render_progenitor_morphology_movie import (
    build_track,
    interpolate_track,
    slerp,
)


class ProgenitorMorphologyMovieTests(unittest.TestCase):
    def test_slerp_has_unit_norm_and_endpoints(self):
        left = torch.tensor([[1.0, 0.0]])
        right = torch.tensor([[0.0, 1.0]])
        result = slerp(left.repeat(3, 1), right.repeat(3, 1), [0.0, 0.5, 1.0])
        torch.testing.assert_close(result[0], left[0])
        torch.testing.assert_close(result[-1], right[0])
        torch.testing.assert_close(result.norm(dim=1), torch.ones(3))

    def test_track_centroids_and_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.csv"
            pd.DataFrame({
                "descendant_id": [10, 10, 10, 10],
                "stage": [1, 1, 2, 2], "rank": [1, 2, 1, 2],
                "galaxy_id": [11, 12, 13, 14],
                "state_lookback_gyr": [1.0, 1.0, 2.0, 2.0],
                "formed_mass_fraction": [0.9, 0.9, 0.8, 0.8],
            }).to_csv(path, index=False)
            ids = np.asarray([10, 11, 12, 13, 14])
            condition = np.asarray([
                [1.0, 0.0], [0.8, 0.2], [0.6, 0.4],
                [0.2, 0.8], [0.0, 1.0],
            ], dtype=np.float32)
            condition /= np.linalg.norm(condition, axis=1, keepdims=True)
            descendant, anchors = build_track(path, 10, ids, condition, 2)
            self.assertEqual(descendant, 10)
            self.assertEqual(len(anchors), 3)
            time, fraction, embedding = interpolate_track(anchors, 5, "cpu")
            self.assertEqual(time.tolist(), [2.0, 1.5, 1.0, 0.5, 0.0])
            self.assertAlmostEqual(fraction[-1], 1.0)
            torch.testing.assert_close(embedding.norm(dim=1), torch.ones(5))


if __name__ == "__main__":
    unittest.main()
