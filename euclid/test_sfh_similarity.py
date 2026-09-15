"""Tests for SFH Wasserstein distances and soft contrastive targets."""

import unittest

import torch
import torch.nn.functional as F

from cosmosweb.sfh_similarity import (
    pairwise_sfh_wasserstein1,
    soft_cross_entropy,
    wasserstein_soft_targets,
)


def as_log_weights(weights, epsilon=1e-10):
    values = torch.as_tensor(weights, dtype=torch.float32)
    values = values / values.sum(dim=1, keepdim=True)
    return torch.log10(values + epsilon)


class WassersteinDistanceTests(unittest.TestCase):
    def test_distance_respects_temporal_displacement(self):
        log_sfhs = as_log_weights([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ])
        distance = pairwise_sfh_wasserstein1(log_sfhs)
        self.assertAlmostEqual(float(distance[0, 0]), 0.0, places=7)
        self.assertAlmostEqual(float(distance[0, 1]), 0.25, places=6)
        self.assertAlmostEqual(float(distance[0, 2]), 1.0, places=6)
        torch.testing.assert_close(distance, distance.T)

    def test_amplitude_scaling_does_not_change_distance(self):
        shape = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        log_sfhs = as_log_weights(torch.cat([shape, shape * 1e6]))
        distance = pairwise_sfh_wasserstein1(log_sfhs)
        self.assertAlmostEqual(float(distance[0, 1]), 0.0, places=6)


class SoftTargetTests(unittest.TestCase):
    def setUp(self):
        self.log_sfhs = as_log_weights([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])

    def test_targets_mix_exact_pair_and_nearest_neighbor(self):
        targets, _ = wasserstein_soft_targets(
            self.log_sfhs, soft_weight=0.25, n_neighbors=1,
        )
        torch.testing.assert_close(targets.sum(dim=1), torch.ones(4))
        torch.testing.assert_close(targets.diagonal(), torch.full((4,), 0.75))
        self.assertAlmostEqual(float(targets[0, 1]), 0.25, places=6)
        self.assertAlmostEqual(float(targets[3, 2]), 0.25, places=6)
        self.assertEqual(int((targets > 0).sum(dim=1).max()), 2)

    def test_zero_weight_is_exact_info_nce(self):
        targets, _ = wasserstein_soft_targets(
            self.log_sfhs, soft_weight=0.0, n_neighbors=1,
        )
        logits = torch.tensor([
            [2.0, 1.0, 0.0, -1.0],
            [0.0, 3.0, 1.0, -2.0],
            [0.0, 1.0, 4.0, 2.0],
            [-1.0, 0.0, 1.0, 2.0],
        ])
        expected = F.cross_entropy(logits, torch.arange(4))
        torch.testing.assert_close(soft_cross_entropy(logits, targets), expected)

    def test_single_object_falls_back_to_exact_target(self):
        targets, distance = wasserstein_soft_targets(
            self.log_sfhs[:1], soft_weight=0.25, n_neighbors=8,
        )
        torch.testing.assert_close(targets, torch.ones((1, 1)))
        torch.testing.assert_close(distance, torch.zeros((1, 1)))


if __name__ == '__main__':
    unittest.main()
