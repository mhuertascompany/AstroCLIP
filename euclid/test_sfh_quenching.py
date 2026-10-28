import unittest

import numpy as np

from euclid.sfh_quenching import sfh_quenching_summaries


class SFHQuenchingSummaryTests(unittest.TestCase):
    def test_constant_history_is_star_forming_and_has_expected_tau(self):
        time = np.linspace(0.0, 1.0, 101)
        edges = np.r_[0.0, (time[:-1] + time[1:]) / 2.0, 1.0]
        weights = np.diff(edges)[None, :]
        result = sfh_quenching_summaries(
            np.log10(weights + 1e-10), time, np.array([10000.0]),
            np.array([10.5]), np.array([0.3]), smoothing_myr=300.0,
        )
        self.assertAlmostEqual(float(result["sfh_doubling_number_100myr"][0]), 1.0, places=5)
        self.assertEqual(float(result["sfh_activity_class"][0]), 2.0)
        self.assertAlmostEqual(float(result["sfh_tau_sf_20_80_gyr"][0]), 6.0, places=4)
        self.assertTrue(np.isnan(result["sfh_quenching_timescale_gyr"][0]))

    def test_declining_history_has_completed_quenching_passage(self):
        time = np.linspace(0.0, 1.0, 251)
        edges = np.r_[0.0, (time[:-1] + time[1:]) / 2.0, 1.0]
        # High SFR at old lookback, a broad transition, then low recent SFR.
        relative_rate = 0.01 + 4.0 / (1.0 + np.exp(-(time - 0.42) / 0.07))
        weights = relative_rate * np.diff(edges)
        weights /= weights.sum()
        result = sfh_quenching_summaries(
            np.log10(weights[None, :] + 1e-10), time, np.array([10000.0]),
            np.array([10.5]), np.array([0.3]), smoothing_myr=300.0,
        )
        self.assertEqual(float(result["sfh_activity_class"][0]), 0.0)
        self.assertGreater(float(result["sfh_quenching_timescale_gyr"][0]), 0.0)
        self.assertGreater(float(result["sfh_quenching_epoch_lookback_gyr"][0]), 0.0)
        self.assertGreater(float(result["sfh_quenching_epoch_redshift"][0]), 0.3)

    def test_invalid_catalog_mass_returns_nan(self):
        time = np.linspace(0.0, 1.0, 20)
        weights = np.full((1, len(time)), 1.0 / len(time))
        result = sfh_quenching_summaries(
            np.log10(weights + 1e-10), time, np.array([10000.0]),
            np.array([np.nan]), np.array([0.3]),
        )
        self.assertTrue(np.isnan(result["sfh_doubling_number_100myr"][0]))


if __name__ == "__main__":
    unittest.main()
