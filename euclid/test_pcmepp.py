import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is absent')
class PCMEPPTests(unittest.TestCase):
    def test_closed_form_distance_matches_manual_expectation(self):
        import torch

        from cosmosweb.pcme import pairwise_closed_form_sampled_distance

        mean_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mean_b = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        variance_a = torch.tensor([[0.5, 0.25], [0.1, 0.2]])
        variance_b = torch.tensor([[0.25, 0.5], [0.3, 0.4]])
        distance = pairwise_closed_form_sampled_distance(
            mean_a, variance_a.log(), mean_b, variance_b.log(),
        )
        expected = torch.empty(2, 2)
        for row in range(2):
            for column in range(2):
                expected[row, column] = (
                    (mean_a[row] - mean_b[column]).square().sum()
                    + variance_a[row].sum()
                    + variance_b[column].sum()
                )
        torch.testing.assert_close(distance, expected)

    def test_pseudo_positives_use_paired_logit_as_anchor_threshold(self):
        import torch

        from cosmosweb.pcme import pseudo_positive_targets

        logits = torch.tensor([
            [2.0, 3.0, 1.0],
            [0.0, 1.0, 1.0],
            [4.0, 2.0, 3.0],
        ])
        expected = torch.tensor([
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ])
        torch.testing.assert_close(pseudo_positive_targets(logits), expected)

    def test_complete_objective_is_finite_and_differentiable(self):
        import torch

        from cosmosweb.pcme import pcmepp_loss

        image_mean = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
        sfh_mean = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
        image_logvar = torch.full((5, 8), -2.0, requires_grad=True)
        sfh_logvar = torch.full((5, 8), -2.0, requires_grad=True)
        scale = torch.tensor(5.0, requires_grad=True)
        bias = torch.tensor(5.0, requires_grad=True)
        result = pcmepp_loss(
            image_mean, image_logvar, sfh_mean, sfh_logvar,
            scale=scale, bias=bias,
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreaterEqual(
            result.image_to_sfh_targets.diagonal().min().item(), 1.0,
        )
        result.loss.backward()
        for tensor in (image_logvar, sfh_logvar, scale, bias):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_vib_is_zero_for_standard_normal(self):
        import torch

        from cosmosweb.pcme import gaussian_vib_loss

        loss = gaussian_vib_loss(torch.zeros(4, 6), torch.zeros(4, 6))
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_vib_averages_batch_and_latent_dimensions_like_reference(self):
        import torch

        from cosmosweb.pcme import gaussian_vib_loss

        # Every element contributes 0.5, and the reference implementation
        # averages over batch and latent dimensions.
        loss = gaussian_vib_loss(torch.ones(4, 6), torch.zeros(4, 6))
        torch.testing.assert_close(loss, torch.tensor(0.5))


if __name__ == '__main__':
    unittest.main()
