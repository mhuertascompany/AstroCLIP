import importlib.util
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is not installed')
class SFHAutoencoderTests(unittest.TestCase):
    def test_decoder_is_normalized_and_backpropagates(self):
        import torch

        from cosmosweb.sfh_autoencoder import FixedGridSFHDecoder

        decoder = FixedGridSFHDecoder(
            n_bins=12, embed_dim=8, d_model=16, n_heads=4, n_layers=1,
            dropout=0,
        )
        latent = torch.randn(3, 8, requires_grad=True)
        output = decoder(latent)
        self.assertEqual(tuple(output.shape), (3, 12))
        self.assertTrue(torch.all(output >= 0))
        torch.testing.assert_close(output.sum(dim=1), torch.ones(3))
        output.square().sum().backward()
        self.assertIsNotNone(latent.grad)

    def test_contiguous_mask_has_requested_width(self):
        import torch

        from cosmosweb.sfh_autoencoder import mask_contiguous_bins

        values = torch.arange(20).reshape(2, 10).float()
        masked, mask = mask_contiguous_bins(values, 0.3, -10.0)
        torch.testing.assert_close(mask.sum(dim=1), torch.tensor([3, 3]))
        self.assertTrue(torch.all(masked[mask] == -10.0))
        for row in mask:
            positions = torch.nonzero(row, as_tuple=False).flatten()
            self.assertTrue(torch.all(torch.diff(positions) == 1))

    def test_reconstruction_loss_prefers_the_target(self):
        import torch

        from cosmosweb.sfh_autoencoder import sfh_reconstruction_loss

        target_mass = torch.tensor([[0.05, 0.15, 0.30, 0.50]])
        target_log = torch.log10(target_mass + 1e-10)
        exact, _ = sfh_reconstruction_loss(target_mass, target_log)
        reversed_loss, _ = sfh_reconstruction_loss(target_mass.flip(1), target_log)
        self.assertLess(float(exact), float(reversed_loss))

    def test_lightning_checkpoint_loader(self):
        import torch

        from cosmosweb.sfh_autoencoder import (
            FixedGridSFHDecoder,
            load_sfh_autoencoder_checkpoint,
        )
        from cosmosweb.sfh_transformer import FixedGridSFHTransformerEncoder

        encoder = FixedGridSFHTransformerEncoder(
            n_bins=8, d_model=16, n_heads=4, n_layers=1, embed_dim=8,
        )
        decoder = FixedGridSFHDecoder(
            n_bins=8, embed_dim=8, d_model=16, n_heads=4, n_layers=1,
        )
        state = {
            **{f'encoder.{key}': value.clone() for key, value in encoder.state_dict().items()},
            **{f'decoder.{key}': value.clone() for key, value in decoder.state_dict().items()},
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'model.ckpt'
            torch.save({'state_dict': state}, checkpoint)
            target_encoder = FixedGridSFHTransformerEncoder(
                n_bins=8, d_model=16, n_heads=4, n_layers=1, embed_dim=8,
            )
            target_decoder = FixedGridSFHDecoder(
                n_bins=8, embed_dim=8, d_model=16, n_heads=4, n_layers=1,
            )
            load_sfh_autoencoder_checkpoint(
                target_encoder, checkpoint, target_decoder,
            )
            for expected, actual in zip(encoder.parameters(), target_encoder.parameters()):
                torch.testing.assert_close(expected, actual)


if __name__ == '__main__':
    unittest.main()
