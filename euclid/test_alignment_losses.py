import importlib.util
import unittest
from unittest import mock


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is absent')
class AlignmentLossTests(unittest.TestCase):
    def test_cwcl_targets_are_continuous_normalized_and_fixed(self):
        import torch

        from cosmosweb.alignment_losses import cwcl_targets

        reference = torch.tensor([
            [1.0, 0.0],
            [0.8, 0.6],
            [-1.0, 0.0],
        ], requires_grad=True)
        targets = cwcl_targets(reference, similarity_temperature=0.2)
        torch.testing.assert_close(targets.sum(1), torch.ones(3))
        self.assertGreater(targets[0, 1].item(), targets[0, 2].item())
        self.assertFalse(targets.requires_grad)

    def test_cwcl_matches_manual_weighted_cross_entropy(self):
        import torch
        import torch.nn.functional as F

        from cosmosweb.alignment_losses import cwcl_loss, cwcl_targets

        image = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        sfh = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = cwcl_loss(image, sfh, reference, logit_scale=2.0)
        targets = cwcl_targets(reference)
        logits = 2.0 * image @ sfh.T
        weighted = -(targets * F.log_softmax(logits, dim=1)).sum(1).mean()
        exact = F.cross_entropy(logits.T, torch.arange(2))
        torch.testing.assert_close(result.loss, 0.5 * (weighted + exact))
        result.loss.backward()
        self.assertIsNotNone(image.grad)
        self.assertIsNotNone(sfh.grad)

    def test_cyclip_cyclic_terms_vanish_for_identical_modalities(self):
        import torch

        from cosmosweb.alignment_losses import cyclip_loss

        image = torch.randn(5, 8, requires_grad=True)
        result = cyclip_loss(image, image, logit_scale=3.0)
        torch.testing.assert_close(
            result.inmodal_cyclic_loss, torch.tensor(0.0), atol=1e-7, rtol=0,
        )
        torch.testing.assert_close(
            result.crossmodal_cyclic_loss, torch.tensor(0.0), atol=1e-7, rtol=0,
        )
        result.loss.backward()
        self.assertIsNotNone(image.grad)

    def test_cyclip_matches_reference_batch_scaling(self):
        import torch
        import torch.nn.functional as F

        from cosmosweb.alignment_losses import cyclip_loss

        image = F.normalize(torch.randn(4, 6), dim=-1)
        sfh = F.normalize(torch.randn(4, 6), dim=-1)
        result = cyclip_loss(
            image, sfh, logit_scale=1.0,
            clip_weight=0.0, inmodal_weight=1.0, crossmodal_weight=1.0,
        )
        cross = image @ sfh.T
        expected_inmodal = 4 * F.mse_loss(image @ image.T, sfh @ sfh.T)
        expected_crossmodal = 4 * F.mse_loss(cross, cross.T)
        torch.testing.assert_close(result.inmodal_cyclic_loss, expected_inmodal)
        torch.testing.assert_close(result.crossmodal_cyclic_loss, expected_crossmodal)
        torch.testing.assert_close(result.loss, expected_inmodal + expected_crossmodal)


@unittest.skipUnless(
    all(importlib.util.find_spec(package) is not None
        for package in ('torch', 'lightning', 'timm', 'zoobot')),
    'CLIP runtime dependencies are absent',
)
class AlignmentModelIntegrationTests(unittest.TestCase):
    @staticmethod
    def _make_model(objective):
        import torch.nn as nn

        import cosmosweb.model_zoobot as clip_module

        class FakeImageEncoder(nn.Module):
            def __init__(self, ckpt_path=None, model_name=None, embed_dim=8,
                         unfreeze_blocks=0, **kwargs):
                super().__init__()
                self.backbone = nn.Identity()
                self.projection = nn.Linear(embed_dim, embed_dim)

            def forward(self, images):
                return self.projection(self.encode_backbone(images))

            def encode_backbone(self, images):
                return images.flatten(1)[:, :8]

        with mock.patch.object(clip_module, 'ZooBotImageEncoder', FakeImageEncoder):
            from euclid.model_zoobot import EuclidZooBotCLIP
            return EuclidZooBotCLIP(
                zoobot_model_name='fake', embed_dim=8, sfh_input_dim=12,
                sfh_encoder_type='mlp', sfh_projection_type='mlp',
                sfh_projection_hidden_dim=16, freeze_sfh_encoder=True,
                queue_size=0, alignment_objective=objective,
            )

    def test_cwcl_training_step_is_differentiable(self):
        import torch

        model = self._make_model('cwcl')
        sfh = torch.randn(5, 12)
        loss = model.training_step({
            'image': torch.randn(5, 2, 2, 2),
            'sfh': sfh,
            'sfh_reference': sfh,
        }, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.image_encoder.projection.weight.grad)
        self.assertIsNotNone(next(model.sfh_projection.parameters()).grad)
        self.assertIsNone(next(model.sfh_encoder.parameters()).grad)

    def test_cyclip_training_step_is_differentiable(self):
        import torch

        model = self._make_model('cyclip')
        loss = model.training_step({
            'image': torch.randn(5, 2, 2, 2),
            'sfh': torch.randn(5, 12),
        }, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.image_encoder.projection.weight.grad)
        self.assertIsNotNone(next(model.sfh_projection.parameters()).grad)

    def test_cwcl_rejects_a_moving_reference_encoder(self):
        with self.assertRaisesRegex(ValueError, 'freeze_sfh_encoder'):
            import cosmosweb.model_zoobot as clip_module
            with mock.patch.object(
                clip_module, 'ZooBotImageEncoder', mock.MagicMock,
            ):
                from euclid.model_zoobot import EuclidZooBotCLIP
                EuclidZooBotCLIP(
                    zoobot_model_name='fake', embed_dim=8, sfh_input_dim=12,
                    queue_size=0, alignment_objective='cwcl',
                    freeze_sfh_encoder=False,
                )


if __name__ == '__main__':
    unittest.main()
