import importlib.util
import unittest
from unittest import mock


RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ('torch', 'lightning', 'timm', 'zoobot')
)


@unittest.skipUnless(RUNTIME_AVAILABLE, 'CLIP runtime dependencies are absent')
class SharedPrivateAlignmentTests(unittest.TestCase):
    @staticmethod
    def _make_model():
        import torch.nn as nn

        import cosmosweb.model_zoobot as clip_module

        class FakeImageEncoder(nn.Module):
            def __init__(self, ckpt_path=None, model_name=None, embed_dim=8,
                         unfreeze_blocks=0, **kwargs):
                super().__init__()
                self.backbone = nn.Identity()
                self.backbone_dim = 10
                self.projection = nn.Linear(self.backbone_dim, embed_dim)

            def forward(self, images):
                return self.projection(self.encode_backbone(images))

            def encode_backbone(self, images):
                return images.flatten(1)[:, :self.backbone_dim]

        with mock.patch.object(
            clip_module, 'ZooBotImageEncoder', FakeImageEncoder,
        ):
            from euclid.model_zoobot import EuclidZooBotCLIP
            return EuclidZooBotCLIP(
                zoobot_model_name='fake', embed_dim=8, sfh_input_dim=12,
                sfh_encoder_type='mlp', sfh_projection_type='mlp',
                sfh_projection_hidden_dim=16, freeze_sfh_encoder=True,
                queue_size=0, alignment_objective='shared_private',
                shared_dim=3, private_dim=4,
                shared_private_hidden_dim=12,
            )

    def test_common_projector_is_tied_and_public_embeddings_are_common(self):
        import torch

        model = self._make_model()
        model.eval()
        images = torch.randn(6, 1, 2, 5)
        sfhs = torch.randn(6, 12)
        image_common, image_private = model.encode_image_shared_private(images)
        sfh_common, sfh_private = model.encode_sfh_shared_private(sfhs)

        self.assertEqual(image_common.shape, (6, 3))
        self.assertEqual(sfh_common.shape, (6, 3))
        self.assertEqual(image_private.shape, (6, 4))
        self.assertEqual(sfh_private.shape, (6, 4))
        torch.testing.assert_close(model.encode_image(images), image_common)
        torch.testing.assert_close(model.encode_sfh(sfhs), sfh_common)

        common_modules = [
            name for name, _ in model.named_modules()
            if name == 'common_projector'
        ]
        self.assertEqual(common_modules, ['common_projector'])

    def test_training_objective_is_finite_and_updates_all_trainable_branches(self):
        import torch

        model = self._make_model()
        loss, metrics = model._shared_private_step(
            torch.randn(8, 1, 2, 5), torch.randn(8, 12),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics) >= {
            'pair_loss', 'variance_loss', 'covariance_loss',
            'orthogonality_loss', 'reconstruction_loss', 'paired_cosine',
        }, True)
        loss.backward()

        self.assertIsNotNone(
            next(model.common_projector.parameters()).grad,
        )
        self.assertIsNotNone(
            next(model.image_private_projector.parameters()).grad,
        )
        self.assertIsNotNone(
            next(model.sfh_private_projector.parameters()).grad,
        )
        self.assertIsNotNone(
            next(model.image_common_decoder.parameters()).grad,
        )
        self.assertIsNotNone(
            next(model.sfh_private_decoder.parameters()).grad,
        )
        self.assertIsNone(next(model.sfh_encoder.parameters()).grad)

    def test_variance_penalty_detects_collapsed_representation(self):
        import torch

        model = self._make_model()
        collapsed = torch.zeros(16, 3)
        spread = torch.randn(1024, 3) * 2
        collapsed_loss = model._representation_variance_loss(collapsed, 1.0)
        spread_loss = model._representation_variance_loss(spread, 1.0)
        self.assertGreater(collapsed_loss.item(), 0.9)
        self.assertLess(spread_loss.item(), 0.05)

    def test_effective_rank_disables_autocast_for_eigendecomposition(self):
        import torch

        model = self._make_model()
        embedding = torch.randn(32, 8, dtype=torch.bfloat16)
        observed_dtypes = []
        original_eigvalsh = torch.linalg.eigvalsh

        def record_dtype(matrix, *args, **kwargs):
            observed_dtypes.append(matrix.dtype)
            return original_eigvalsh(matrix, *args, **kwargs)

        with mock.patch.object(torch.linalg, 'eigvalsh', record_dtype):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                rank = model._effective_rank(embedding)

        self.assertTrue(torch.isfinite(rank))
        self.assertEqual(observed_dtypes, [torch.float32])


if __name__ == '__main__':
    unittest.main()
