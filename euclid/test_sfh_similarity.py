"""Tests for SFH Wasserstein distances and soft contrastive targets."""

import unittest

import torch
import torch.nn.functional as F

from cosmosweb.sfh_similarity import (
    adjacency_consistency_loss,
    ae_false_negative_mask,
    masked_cross_entropy,
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


class AEFalseNegativeTests(unittest.TestCase):
    def test_nearest_latents_are_excluded_symmetrically(self):
        latents = torch.tensor([
            [1.0, 0.0],
            [0.99, 0.10],
            [-1.0, 0.0],
            [-0.99, 0.10],
        ])
        valid, distance = ae_false_negative_mask(latents, n_neighbors=1)
        self.assertTrue(torch.all(valid.diagonal()))
        self.assertFalse(bool(valid[0, 1]))
        self.assertFalse(bool(valid[1, 0]))
        self.assertFalse(bool(valid[2, 3]))
        self.assertFalse(bool(valid[3, 2]))
        torch.testing.assert_close(distance, distance.T)

    def test_distance_threshold_can_leave_distant_neighbour_valid(self):
        latents = torch.eye(3)
        valid, _ = ae_false_negative_mask(
            latents, n_neighbors=1, max_cosine_distance=0.1,
        )
        self.assertTrue(torch.all(valid))

    def test_masked_loss_removes_strong_false_negative(self):
        logits = torch.tensor([[2.0, 8.0], [7.0, 2.0]])
        exact = F.cross_entropy(logits, torch.arange(2))
        valid = torch.eye(2, dtype=torch.bool)
        filtered = masked_cross_entropy(logits, valid)
        self.assertGreater(float(exact), 5.0)
        self.assertAlmostEqual(float(filtered), 0.0, places=7)


class AdjacencyConsistencyTests(unittest.TestCase):
    def test_loss_is_zero_when_cross_modal_adjacency_matches_ae(self):
        latents = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [-1.0, 0.0],
        ])
        normalized = F.normalize(latents, dim=1)
        scale = torch.tensor(4.0)
        logits = scale * (normalized @ normalized.T)
        loss = adjacency_consistency_loss(logits, latents, scale)
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_mismatched_adjacency_has_positive_loss_and_gradients(self):
        latents = torch.eye(3)
        logits = torch.tensor([
            [1.0, 4.0, 0.0],
            [0.0, 1.0, 4.0],
            [4.0, 0.0, 1.0],
        ], requires_grad=True)
        loss = adjacency_consistency_loss(logits, latents, 2.0)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
