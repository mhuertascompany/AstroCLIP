import h5py
import numpy as np
import torch

from euclid.sfh_conditional_diffusion import (
    ConditionalSFHDiffusionModule,
    clr_to_weights,
    log_sfh_to_weights,
    weights_to_clr,
)
from euclid.train_conditional_sfh_diffusion import load_conditions


def test_clr_round_trip_preserves_normalized_sfh():
    weights = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.1, 0.2, 0.3],
    ], dtype=np.float32)
    recovered = clr_to_weights(weights_to_clr(weights))
    np.testing.assert_allclose(recovered, weights, atol=1e-6)
    np.testing.assert_allclose(recovered.sum(axis=1), 1, atol=1e-7)


def test_stored_log_sfh_is_renormalized():
    weights = np.array([[0.05, 0.15, 0.3, 0.5]], dtype=np.float32)
    stored = np.log10(weights + 1e-10)
    recovered = log_sfh_to_weights(stored)
    np.testing.assert_allclose(recovered, weights, atol=1e-6)
    np.testing.assert_allclose(recovered.sum(axis=1), 1, atol=1e-7)


def test_conditional_sampler_returns_simplex_values():
    module = ConditionalSFHDiffusionModule(
        n_bins=8,
        condition_mean=[10.0, 1.0, 0.0],
        condition_scale=[0.5, 0.5, 1.0],
        clr_mean=np.zeros(8),
        clr_scale=np.ones(8),
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0,
        diffusion_steps=10,
    )
    condition = torch.tensor([[10.2, 0.8, -0.1], [9.8, 1.2, 0.2]])
    weights, clr = module.sample_weights(condition, steps=3, seed=7)
    assert weights.shape == clr.shape == (2, 8)
    assert torch.all(weights >= 0)
    torch.testing.assert_close(weights.sum(1), torch.ones(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(clr.mean(1), torch.zeros(2), atol=1e-6, rtol=0)


def test_load_conditions_uses_phz_catalog_values(tmp_path):
    path = tmp_path / 'catalog.h5'
    with h5py.File(path, 'w') as target:
        target['phz_pp_median_stellarmass'] = [10.0, 10.5, np.nan]
        target['phz_pp_median_redshift'] = [0.5, 1.0, 0.7]
        target['phz_pp_median_sfr'] = [-0.2, 0.4, 0.1]
    conditions, valid = load_conditions(path, sfr_source='phz')
    np.testing.assert_allclose(conditions[:2], [[10.0, 0.5, -0.2], [10.5, 1.0, 0.4]])
    np.testing.assert_array_equal(valid, [True, True, False])


def test_load_conditions_can_derive_sfr100_from_same_sfh(tmp_path):
    path = tmp_path / 'catalog.h5'
    weights = np.array([[0.2, 0.3, 0.5]], dtype=np.float32)
    with h5py.File(path, 'w') as target:
        target['phz_pp_median_stellarmass'] = [10.0]
        target['phz_pp_median_redshift'] = [0.5]
        target['sfh'] = np.log10(weights + 1e-10)
        target['sfh_time_grid'] = [0.0, 0.5, 1.0]
        target['sfh_time_norm'] = [1000.0]
    conditions, valid = load_conditions(path, sfr_source='sfh')
    # The first bin spans fractional time [0,.25]. A 100 Myr window at an
    # age of 1 Gyr covers .1/.25 of its 0.2 mass weight: fraction=0.08.
    expected_log_sfr = 10.0 + np.log10(0.08) - 8.0
    np.testing.assert_allclose(conditions[0, 2], expected_log_sfr, atol=1e-6)
    np.testing.assert_array_equal(valid, [True])
