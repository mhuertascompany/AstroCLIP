import numpy as np

from euclid.plot_individual_jepa_morphology import (
    _histogram_edges,
    _matched_control_indices,
    _scatter68,
    _scatter_statistics,
    _sfh_density,
)


def test_physical_controls_exclude_query_and_retrieved_and_match_nearest():
    mass = np.array([10.0, 10.01, 10.02, 10.03, 10.4])
    redshift = np.array([0.5, 0.501, 0.502, 0.503, 0.9])
    sfr = np.array([0.0, 0.01, 0.02, 0.03, 1.0])
    selected, delta = _matched_control_indices(
        query=0, retrieved=np.array([1]), mass=mass, redshift=redshift,
        log_sfr=sfr, count=2,
    )
    np.testing.assert_array_equal(selected, np.array([2, 3]))
    assert delta.shape == (2, 3)


def test_probability_histograms_use_common_physical_range():
    edges = _histogram_edges(
        np.array([0.1, 0.2]), np.array([0.8, 0.9]),
        "zoobot_spiral_probability",
    )
    assert edges[0] == 0.0
    assert edges[-1] == 1.0


def test_sfh_density_integrates_to_input_weight():
    time = np.linspace(0.0, 1.0, 5)
    weights = np.full(5, 0.2)
    epsilon = 1e-10
    density = _sfh_density(np.log10(weights + epsilon), time, 10.0, epsilon)
    edges = np.concatenate(([0.0], 0.5 * (time[:-1] + time[1:]), [1.0]))
    np.testing.assert_allclose(np.sum(density * np.diff(edges) * 10.0), 1.0)


def test_scatter_statistics_identify_broader_retrieved_sample():
    control = np.linspace(-1.0, 1.0, 20)
    retrieved = 2.0 * control
    assert np.isclose(_scatter68(retrieved) / _scatter68(control), 2.0)
    result = _scatter_statistics(
        control, retrieved, np.random.default_rng(1), bootstrap_samples=200,
    )
    assert np.isclose(result["scatter_ratio"], 2.0)
    assert result["scatter_ratio_p16"] < result["scatter_ratio_p84"]
