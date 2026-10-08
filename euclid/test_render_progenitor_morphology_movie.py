import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import h5py

from euclid.render_progenitor_morphology_movie import (
    build_track,
    interpolate_track,
    load_condition_umap,
    load_descendant_sfh,
    nearest_reference,
    morphology_conditions,
    slerp,
    topk_reference,
    umap_density,
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

    def test_nearest_reference_across_chunks(self):
        reference = np.asarray([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8],
        ], dtype=np.float32)
        query = np.asarray([[0.9, 0.1], [0.55, 0.82]], dtype=np.float32)
        similarity, index = nearest_reference(query, reference, chunk=2)
        np.testing.assert_array_equal(index, [0, 3])
        np.testing.assert_allclose(
            similarity, [query[0] @ reference[0], query[1] @ reference[3]],
        )

    def test_topk_reference_is_sorted(self):
        reference = np.asarray([
            [1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0],
        ], dtype=np.float32)
        similarity, index = topk_reference([[1.0, 0.0]], reference, 3, chunk=2)
        np.testing.assert_array_equal(index, [[0, 1, 2]])
        np.testing.assert_allclose(similarity, [[1.0, 0.8, 0.0]])

    def test_morphology_sampling_is_reproducible(self):
        reference = np.asarray([
            [1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.0, 1.0],
        ], dtype=np.float32)
        query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        first = morphology_conditions(
            query, reference, mode="sample", k=3, temperature=0.5, seed=7,
        )
        second = morphology_conditions(
            query, reference, mode="sample", k=3, temperature=0.5, seed=7,
        )
        np.testing.assert_array_equal(
            first["representative_index"], second["representative_index"],
        )
        np.testing.assert_allclose(np.linalg.norm(first["condition"], axis=1), 1)

    def test_morphology_interpolation_is_normalized_and_not_a_donor(self):
        reference = np.asarray([
            [1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.0, 1.0],
        ], dtype=np.float32)
        result = morphology_conditions(
            [[1.0, 0.0]], reference, mode="interpolate", k=3,
            temperature=0.5,
        )
        np.testing.assert_allclose(np.linalg.norm(result["condition"], axis=1), 1)
        self.assertFalse(any(
            np.allclose(result["condition"][0], donor) for donor in reference
        ))

    def test_track_uses_best_candidate_present_in_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.csv"
            pd.DataFrame({
                "descendant_id": [10, 10], "stage": [1, 1], "rank": [1, 2],
                "galaxy_id": [99, 11], "state_lookback_gyr": [1.0, 1.0],
                "formed_mass_fraction": [0.9, 0.9],
            }).to_csv(path, index=False)
            ids = np.asarray([10, 11])
            condition = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            _, anchors = build_track(path, 10, ids, condition, 1)
            self.assertEqual(anchors[1]["galaxy_ids"], [11])

    def test_loaded_sfh_rate_integrates_to_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sfh.h5"
            with h5py.File(path, "w") as target:
                target["galaxy_id"] = [10]
                target["sfh"] = np.log10([[0.2, 0.3, 0.5]])
                target["sfh_time_grid"] = [0.0, 0.5, 1.0]
                target["sfh_time_norm"] = [2000.0]
                target.attrs["sfh_log_epsilon"] = 0.0
            time, rate, norm = load_descendant_sfh(path, 10)
            np.testing.assert_allclose(time, [0.0, 1.0, 2.0])
            self.assertEqual(norm, 2.0)
            np.testing.assert_allclose(np.sum(rate * [0.5, 1.0, 0.5]), 1.0)

    def test_condition_umap_alignment_and_density(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "condition_umap.npz"
            np.savez(
                path, galaxy_id=np.array([20, 10, 30]),
                xy=np.array([[2, 2], [1, 1], [3, 3]], dtype=np.float32),
                metadata=json.dumps({"condition_cache_sha256": "abc"}),
            )
            xy, metadata = load_condition_umap(path, np.array([10, 20, 30]), "abc")
            np.testing.assert_array_equal(xy, [[1, 1], [2, 2], [3, 3]])
            self.assertEqual(metadata["condition_cache_sha256"], "abc")
            density, extent = umap_density(xy, bins=8)
            self.assertEqual(density.shape, (8, 8))
            self.assertEqual(len(extent), 4)


if __name__ == "__main__":
    unittest.main()
