import importlib.util
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class CounterfactualDatasetTests(unittest.TestCase):
    def test_activity_classes_follow_doubling_thresholds(self):
        from euclid.counterfactual_dataset import activity_classes

        time = np.linspace(0.0, 1.0, 4, dtype=np.float32)
        weights = np.array([
            [0.001, 0.3, 0.3, 0.399],
            [0.020, 0.3, 0.3, 0.380],
            [0.100, 0.3, 0.3, 0.300],
        ], dtype=np.float32)
        log_sfh = np.log10(weights + 1e-10)
        classes = activity_classes(log_sfh, time, np.full(3, 10_000.0))
        np.testing.assert_array_equal(classes, [0, 1, 2])

    def test_donor_pools_match_class_and_exclude_target(self):
        from euclid.counterfactual_dataset import matched_donor_pools

        train = np.arange(12, dtype=np.int64)
        query = np.array([1, 5, 9, 12], dtype=np.int64)
        mass = np.linspace(9.0, 11.0, 13)
        redshift = np.linspace(0.2, 1.0, 13)
        activity = np.array([0] * 4 + [1] * 4 + [2] * 5, dtype=np.int8)
        pools = matched_donor_pools(
            train, query, mass, redshift, activity, pool_size=3,
        )
        self.assertEqual(pools.shape, (4, 3))
        for target, donors in zip(query, pools):
            self.assertNotIn(target, donors)
            self.assertTrue(np.all(activity[donors] == activity[target]))
            self.assertTrue(np.all(np.isin(donors, train)))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class CounterfactualFlowModelTests(unittest.TestCase):
    def test_bidirectional_raw_space_loss_is_finite_and_differentiable(self):
        import torch

        from euclid.counterfactual_flow import EuclidCounterfactualFlow

        model = EuclidCounterfactualFlow(
            n_sfh_bins=32, common_dim=16, prior_dim=16,
            condition_dim=32, image_encoder_base=8, image_flow_base=8,
            sfh_d_model=32, sfh_layers=1, sfh_heads=4,
            common_loss_weight=0.05, common_dropout=0.0,
        )
        batch = {
            "image": torch.randn(4, 1, 32, 32),
            "sfh": torch.randn(4, 32),
            "donor_images": torch.randn(4, 3, 1, 32, 32),
            "donor_sfhs": torch.randn(4, 3, 32),
            "covariates": torch.randn(4, 5),
        }
        loss, metrics = model.loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics), {
            "image_flow_loss", "sfh_flow_loss", "common_loss",
        })
        loss.backward()
        self.assertIsNotNone(
            next(model.image_common_encoder.parameters()).grad,
        )
        self.assertIsNotNone(
            next(model.sfh_common_encoder.parameters()).grad,
        )
        self.assertIsNotNone(
            next(model.image_prior_encoder.parameters()).grad,
        )
        self.assertIsNotNone(
            next(model.sfh_prior_encoder.parameters()).grad,
        )

    def test_common_controls_preserve_shapes(self):
        import torch

        from euclid.counterfactual_flow import EuclidCounterfactualFlow

        model = EuclidCounterfactualFlow(
            n_sfh_bins=16, common_dim=8, prior_dim=8,
            condition_dim=16, image_encoder_base=8, image_flow_base=8,
            sfh_d_model=16, sfh_layers=1, sfh_heads=4,
        )
        inputs = (
            torch.randn(3, 1, 32, 32), torch.randn(3, 16),
            torch.randn(3, 2, 1, 32, 32), torch.randn(3, 2, 16),
            torch.randn(3, 5),
        )
        for mode in ("paired", "shuffled", "prior_only"):
            image_condition, sfh_condition, image_common, sfh_common = (
                model.encode_conditions(*inputs, common_mode=mode)
            )
            self.assertEqual(image_condition.shape, (3, 16))
            self.assertEqual(sfh_condition.shape, (3, 16))
            self.assertEqual(image_common.shape, (3, 8))
            self.assertEqual(sfh_common.shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
