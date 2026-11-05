import numpy as np

from euclid.plot_image_to_sfh_diagnostics import (
    _diverse_high_probability_queries,
    _fractional_widths,
    _mass_redshift_controls,
    _retrieved_without_pair,
    _sfh_fractional_density,
)


def test_fractional_sfh_density_integrates_to_one():
    time = np.linspace(0.0, 1.0, 8)
    weights = np.array([[0.05, 0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.05]])
    log_weights = np.log10(weights + 1e-10)
    density = _sfh_fractional_density(log_weights, time, 1e-10)
    np.testing.assert_allclose(
        np.sum(density * _fractional_widths(time), axis=1), 1.0,
    )


def test_retrieval_excludes_true_pair():
    top = np.array([[0, 3, 2, 1], [1, 0, 2, 3]])
    np.testing.assert_array_equal(
        _retrieved_without_pair(top, query=0, count=3), [3, 2, 1],
    )


def test_mass_redshift_controls_exclude_query_and_retrieved():
    mass = np.array([10.0, 10.02, 10.04, 10.07, 10.4])
    redshift = np.array([0.5, 0.51, 0.49, 0.52, 0.8])
    selected, _ = _mass_redshift_controls(
        0, np.array([1]), mass, redshift, count=2,
    )
    assert 0 not in selected
    assert 1 not in selected
    np.testing.assert_array_equal(selected, [2, 3])


def test_query_selection_spans_mass_bins():
    mass = np.linspace(9.0, 11.0, 12)
    redshift = np.linspace(0.2, 1.0, 12)
    score = np.linspace(0.8, 0.99, 12)
    selected = _diverse_high_probability_queries(
        score, mass, redshift, np.ones(12, dtype=bool), count=3,
    )
    assert len(np.unique(selected)) == 3
    assert np.ptp(mass[selected]) > 1.0
