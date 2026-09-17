import importlib.util
import unittest
from unittest import mock


RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ('torch', 'lightning', 'timm', 'zoobot')
)


@unittest.skipUnless(RUNTIME_AVAILABLE, 'CLIP runtime dependencies are absent')
class SFHProjectionTests(unittest.TestCase):
    def test_linear_projection_starts_as_identity_and_receives_gradients(self):
        import torch

        from cosmosweb.model_zoobot import make_sfh_projection

        projection = make_sfh_projection('linear', 8)
        latent = torch.randn(4, 8)
        torch.testing.assert_close(projection(latent), latent)
        projection(latent).square().sum().backward()
        self.assertIsNotNone(projection.weight.grad)

    def test_identity_projection_has_no_parameters(self):
        import torch

        from cosmosweb.model_zoobot import make_sfh_projection

        projection = make_sfh_projection('identity', 8)
        latent = torch.randn(2, 8)
        torch.testing.assert_close(projection(latent), latent)
        self.assertEqual(sum(p.numel() for p in projection.parameters()), 0)

    def test_frozen_autoencoder_is_separate_from_trainable_projection(self):
        import torch
        import torch.nn as nn

        import cosmosweb.model_zoobot as clip_module

        class FakeImageEncoder(nn.Module):
            def __init__(self, ckpt_path=None, model_name=None, embed_dim=8,
                         unfreeze_blocks=0):
                super().__init__()
                self.backbone = nn.Identity()
                self.projection = nn.Linear(embed_dim, embed_dim)

            def forward(self, images):
                return self.projection(images.flatten(1)[:, :8])

        with mock.patch.object(
            clip_module, 'ZooBotImageEncoder', FakeImageEncoder,
        ):
            model = clip_module.CosmosWebZooBotCLIP(
                zoobot_model_name='fake', embed_dim=8, sfh_input_dim=12,
                sfh_encoder_type='transformer', sfh_d_model=16,
                sfh_n_heads=4, sfh_n_layers=1, sfh_decoder_layers=1,
                sfh_reconstruction_weight=0.1,
                sfh_projection_type='linear', freeze_sfh_encoder=True,
                freeze_sfh_decoder=True, queue_size=0,
            )

        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model.sfh_encoder.parameters()
        ))
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model.sfh_decoder.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in model.sfh_projection.parameters()
        ))
        sfh = torch.randn(3, 12)
        latent = model.encode_sfh_latent(sfh)
        torch.testing.assert_close(model.encode_sfh(sfh), latent)
        model.encode_sfh(sfh).square().sum().backward()
        self.assertIsNone(next(model.sfh_encoder.parameters()).grad)
        self.assertIsNotNone(next(model.sfh_projection.parameters()).grad)

        model.zero_grad(set_to_none=True)
        mass = torch.softmax(torch.randn(4, 12), dim=1)
        log_sfh = torch.log10(mass + 1e-10)
        batch = {
            'image': torch.randn(4, 3, 2, 2),
            'sfh': log_sfh,
            'sfh_reference': log_sfh,
            'sfh_p16': log_sfh,
            'sfh_p84': log_sfh,
        }
        loss = model.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(next(model.sfh_encoder.parameters()).grad)
        self.assertIsNone(next(model.sfh_decoder.parameters()).grad)
        self.assertIsNotNone(next(model.sfh_projection.parameters()).grad)


if __name__ == '__main__':
    unittest.main()
