"""Focused tests for the population plus cross-modal residual construction."""

import torch

from euclid.population_residual_flow import (
    PopulationFlow,
    PopulationVelocity,
    ResidualVelocity,
)


def _batch(batch_size=4, bins=16, size=32):
    return {
        "image": torch.randn(batch_size, 1, size, size),
        "sfh": torch.randn(batch_size, bins),
        "covariates": torch.randn(batch_size, 5),
    }


def _model(bins=16):
    population = PopulationVelocity(
        n_sfh_bins=bins, image_base=8, sfh_d_model=16,
        sfh_heads=4, sfh_layers=1,
    )
    return ResidualVelocity(
        population, n_sfh_bins=bins, embedding_dim=8,
        image_encoder_base=8, image_flow_base=8,
        sfh_d_model=16, sfh_heads=4, sfh_layers=1,
    )


def test_zero_embedding_is_exact_population_flow():
    model = _model()
    batch = _batch()
    time = torch.rand(len(batch["image"]))
    for anchor in ("image", "sfh"):
        state = torch.randn_like(batch[anchor])
        expected = model.population.predict(
            anchor, state, time, batch["covariates"],
        )
        actual = model.predict(batch, anchor, state, time, mode="zero")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_population_is_frozen_but_residual_receives_gradients():
    model = _model()
    batch = _batch()
    loss, metrics = model.loss(batch, "image")
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert all(parameter.grad is None for parameter in model.population.parameters())
    assert any(
        parameter.grad is not None
        for parameter in model.image_residual.parameters()
    )


def test_output_shapes_for_both_modalities():
    model = _model()
    batch = _batch()
    time = torch.rand(len(batch["image"]))
    for anchor in ("image", "sfh"):
        state = torch.randn_like(batch[anchor])
        assert model.predict(batch, anchor, state, time).shape == state.shape


def test_population_controls_reuse_identical_target_time_and_noise():
    batch = _batch()
    model = PopulationFlow(
        n_sfh_bins=16, image_base=8, sfh_d_model=16,
        sfh_heads=4, sfh_layers=1,
    )
    time = torch.rand(len(batch["sfh"]))
    noise = torch.randn_like(batch["sfh"])
    losses = [model.loss(batch, "sfh", time, noise, mode=mode)
              for mode in ("paired", "shuffled", "zero")]
    assert all(torch.isfinite(value) for value in losses)
    assert not torch.equal(losses[0], losses[1])
