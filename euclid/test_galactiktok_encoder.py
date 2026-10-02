import unittest
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from cosmosweb.galactiktok_encoder import GalactikTokImageEncoder, SpatialTokenPooler


class SpatialTokenPoolerTest(unittest.TestCase):
    def test_shape_and_gradient(self):
        pooler = SpatialTokenPooler(
            token_dim=8,
            n_tokens=144,
            output_dim=32,
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        latent = torch.randn(3, 144 * 8, requires_grad=True)
        output = pooler(latent)
        self.assertEqual(output.shape, (3, 32))
        output.square().mean().backward()
        self.assertIsNotNone(pooler.cls_token.grad)

    def test_rejects_wrong_latent_width(self):
        pooler = SpatialTokenPooler(8, 16, 32, 64, 4, 1)
        with self.assertRaisesRegex(ValueError, 'Expected 128'):
            pooler(torch.randn(2, 127))


class GalactikTokImageEncoderTest(unittest.TestCase):
    def test_frozen_tokenizer_and_trainable_pooler(self):
        class FakeImage:
            def __init__(self, flux, bands):
                self.flux = flux
                self.bands = bands

        class FakeTokenizer(nn.Module):
            canonical_bands = ['euclid-vis']
            image_size = 16
            patch_size = 8
            bottleneck_dim = 4

            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(8 * 8, 4)
                self.post_tok_proj = nn.Linear(4, 8)
                self.decoder_blocks = nn.Linear(8, 8)
                self.decoder_norm = nn.LayerNorm(8)
                self.patch_debed_weight = nn.Parameter(torch.randn(1, 8, 64))
                self.patch_debed_bias = nn.Parameter(torch.zeros(1, 64))
                self.mask_token = nn.Parameter(torch.zeros(4))

            @classmethod
            def from_pretrained(cls, path):
                return cls()

            def encode(self, image, mask_fraction=0.0):
                patches = image.flux.unfold(2, 8, 8).unfold(3, 8, 8)
                patches = patches.contiguous().reshape(image.flux.size(0), 4, 64)
                return types.SimpleNamespace(tokens=self.encoder(patches))

        package = types.ModuleType('galactiktok')
        package.ImageTransformerTokenizer = FakeTokenizer
        models = types.ModuleType('galactiktok.models')
        image_module = types.ModuleType('galactiktok.models.image_transformer')
        image_module.Image = FakeImage
        modules = {
            'galactiktok': package,
            'galactiktok.models': models,
            'galactiktok.models.image_transformer': image_module,
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, modules,
        ):
            encoder = GalactikTokImageEncoder(
                Path(directory), embed_dim=12, pool_hidden_dim=16,
                pool_heads=4, pool_layers=1, dropout=0.0,
            )
            output = encoder(torch.randn(2, 3, 24, 24))
            self.assertEqual(output.shape, (2, 12))
            self.assertFalse(any(p.requires_grad for p in encoder.backbone.parameters()))
            self.assertTrue(any(p.requires_grad for p in encoder.projection.parameters()))
            self.assertFalse(hasattr(encoder.backbone, 'decoder_blocks'))


if __name__ == '__main__':
    unittest.main()
