import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from euclid.progenitor_analogues import (
    bin_edges_from_centres,
    candidate_curve,
    candidate_curves,
    curve_distance,
    descendant_epoch_redshifts,
    descendant_state_curve,
    lookback_at_formed_fraction,
    older_fraction,
    _choose_descendant,
)


class ProgenitorAnalogueMathTest(unittest.TestCase):
    def setUp(self):
        self.centres = np.linspace(0.0, 1.0, 5)
        self.edges = bin_edges_from_centres(self.centres)
        self.uniform = np.ones(5)

    def test_cumulative_endpoints_and_inverse(self):
        self.assertAlmostEqual(float(older_fraction(self.uniform, self.edges, 10, 0)), 1)
        self.assertAlmostEqual(float(older_fraction(self.uniform, self.edges, 10, 10)), 0)
        lookback = lookback_at_formed_fraction(self.uniform, self.edges, 10, 0.5)
        self.assertAlmostEqual(lookback, 5.0, places=6)

    def test_state_curve_is_renormalized_at_compared_epoch(self):
        fraction = 0.6
        state = lookback_at_formed_fraction(self.uniform, self.edges, 10, fraction)
        tau = np.linspace(0, 2, 9)
        curve = descendant_state_curve(
            self.uniform, self.edges, 10, state, fraction, tau
        )
        self.assertAlmostEqual(curve[0], 1.0, places=7)
        self.assertTrue(np.all(np.diff(curve) <= 1e-12))

    def test_identical_epoch_histories_have_zero_distance(self):
        tau = np.linspace(0, 2, 17)
        curve = candidate_curve(self.uniform, self.edges, 10, tau)
        self.assertAlmostEqual(curve_distance(curve, curve.copy()), 0.0)

    def test_vectorized_candidate_curves_match_scalar_version(self):
        tau = np.linspace(0, 2, 17)
        weights = np.vstack([self.uniform, np.arange(1, 6)])
        norms = np.array([10.0, 8.0])
        vectorized = candidate_curves(weights, self.edges, norms, tau)
        for index in range(2):
            scalar = candidate_curve(weights[index], self.edges, norms[index], tau)
            np.testing.assert_allclose(vectorized[index], scalar, atol=1e-12)

    def test_descendant_epoch_redshift_uses_descendant_cosmic_clock(self):
        redshifts = descendant_epoch_redshifts(0.7, np.array([0.0, 1.0, 3.0]))
        self.assertAlmostEqual(redshifts[0], 0.7, places=5)
        self.assertTrue(np.all(np.diff(redshifts) > 0))

    def test_quenched_descendant_cut_is_applied_before_mass_proximity(self):
        data = {
            "ids": np.array([11, 12, 13]),
            "mass": np.array([11.00, 11.03, 10.90]),
            "catalog_log_ssfr": np.array([-10.0, -12.0, -11.7]),
            "physical_parameters_matched": np.ones(3, dtype=bool),
        }
        with TemporaryDirectory() as directory:
            stamps = Path(directory)
            for galaxy_id in data["ids"]:
                (stamps / f"VIS_{galaxy_id}.jpg").touch()
            chosen = _choose_descendant(data, stamps, None, 10.5, 11.0, -11.5)
        self.assertEqual(chosen, 1)

    def test_main_sequence_descendant_cut_uses_delta_ms(self):
        data = {
            "ids": np.array([21, 22, 23]),
            "mass": np.array([11.00, 11.04, 10.90]),
            "catalog_log_ssfr": np.array([-10.0, -10.2, -10.1]),
            "catalog_delta_ms": np.array([0.8, 0.15, -0.2]),
            "physical_parameters_matched": np.ones(3, dtype=bool),
        }
        with TemporaryDirectory() as directory:
            stamps = Path(directory)
            for galaxy_id in data["ids"]:
                (stamps / f"VIS_{galaxy_id}.jpg").touch()
            chosen = _choose_descendant(
                data, stamps, None, 10.5, 11.0, None, 0.3
            )
        self.assertEqual(chosen, 1)


if __name__ == "__main__":
    unittest.main()
