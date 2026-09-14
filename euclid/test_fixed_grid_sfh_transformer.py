import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is not installed')
class FixedGridSFHTransformerTests(unittest.TestCase):
    def test_output_shape_and_gradient(self):
        import torch

        from cosmosweb.sfh_transformer import FixedGridSFHTransformerEncoder

        model = FixedGridSFHTransformerEncoder(
            n_bins=10,
            d_model=16,
            n_heads=4,
            n_layers=2,
            embed_dim=8,
            dropout=0,
        )
        sfh = torch.randn(3, 10, requires_grad=True)
        output = model(sfh)
        self.assertEqual(tuple(output.shape), (3, 8))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(sfh.grad)
        self.assertTrue(torch.isfinite(sfh.grad).all())

    def test_wrong_bin_count_is_rejected(self):
        import torch

        from cosmosweb.sfh_transformer import FixedGridSFHTransformerEncoder

        model = FixedGridSFHTransformerEncoder(
            n_bins=10, d_model=16, n_heads=4, n_layers=1, embed_dim=8,
        )
        with self.assertRaisesRegex(ValueError, 'Expected SFH shape'):
            model(torch.randn(2, 9))


if __name__ == '__main__':
    unittest.main()
