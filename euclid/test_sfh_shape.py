import unittest

import numpy as np

from euclid.sfh_shape import (
    sfh_duration_80,
    sfh_post_peak_decline,
    sfh_recent_activity,
)


class SFHDurationTests(unittest.TestCase):
    def test_recent_rates_and_trend_direction(self):
        time = np.linspace(0, 1, 101)
        edges = np.r_[0., (time[:-1] + time[1:]) / 2, 1.]
        weights = np.tile(np.diff(edges), (4, 1))
        weights[1] = 0
        weights[1, 3] = 1  # recent burst: rising toward observation
        weights[2] = 0
        weights[2, 15] = 1  # previous burst: declining toward observation
        weights[3] = 0
        weights[3, 70] = 1  # no recent activity: undefined trend
        props = sfh_recent_activity(np.log10(weights + 1e-10), time,
                                    age_myr=np.full(4, 10000.))
        np.testing.assert_allclose(props['sfh_recent_birthrate'], [1, 10, 0, 0], atol=1e-7)
        np.testing.assert_allclose(props['sfh_recent_trend'][:3], [0, 1, -1], atol=1e-7)
        self.assertTrue(np.isnan(props['sfh_recent_trend'][3]))
        np.testing.assert_allclose(props['sfh_recent_sfr_per_formed_mass'][:2], [1e-10, 1e-9])
        np.testing.assert_allclose(props['sfh_log_recent_sfr_per_formed_mass'], [-10, -9, -15, -15])
        invalid = sfh_recent_activity(np.full((1, 101), np.nan), time,
                                      age_myr=np.array([10000.]))
        self.assertTrue(np.isnan(invalid['sfh_log_recent_sfr_per_formed_mass'][0]))

    def test_burst_extended_and_invalid_histories(self):
        time = np.linspace(0, 1, 101)
        weights = np.zeros((5, 101))
        weights[0, 50] = 1
        weights[1] = 1 / 101
        weights[2, [25, 75]] = 0.5
        weights[3, 50] = np.nan
        values = sfh_duration_80(np.log10(weights + 1e-10), time)
        np.testing.assert_allclose(values[:3], [0, 0.8, 0.5])
        self.assertTrue(np.all(np.isnan(values[3:])))

    def test_fixed_100myr_rate_and_physical_ages(self):
        time = np.array([.05, .15, .5])
        epsilon = 1e-10
        weights = np.array([[.2, .3, .5]])
        result = sfh_recent_activity(
            np.log10(weights + epsilon), time, epsilon, np.array([1000.]),
        )
        self.assertAlmostEqual(
            float(result['sfh_recent_mass_fraction_100myr'][0]), .2, places=6,
        )
        self.assertAlmostEqual(
            float(result['sfh_log_sfr_per_stellar_mass_100myr_r0'][0]),
            np.log10(.2 / 1e8), places=6,
        )
        self.assertAlmostEqual(float(result['sfh_peak_age_gyr'][0]), .5, places=6)
        self.assertAlmostEqual(
            float(result['sfh_mass_weighted_age_gyr'][0]), .305, places=6,
        )

    def test_smoothed_single_peak_decline_slope(self):
        time = np.linspace(0, 1, 251)
        age_myr = np.full(4, 10000.0)
        single = np.exp(-0.5 * ((time - 0.55) / 0.10) ** 2)
        noisy_single = single * (1 + 0.12 * np.sin(2 * np.pi * time / 0.025))
        rising = np.exp(-4 * time)  # maximum at observation
        double = (
            np.exp(-0.5 * ((time - 0.30) / 0.055) ** 2)
            + .9 * np.exp(-0.5 * ((time - 0.72) / 0.055) ** 2)
        )
        weights = np.vstack([single, noisy_single, rising, double])
        result = sfh_post_peak_decline(
            np.log10(weights + 1e-10), time, age_myr,
            smoothing_myr=300.0,
        )
        slopes = result['sfh_post_peak_decline_slope_dex_gyr']
        rise_slopes = result['sfh_pre_peak_rise_slope_dex_gyr']
        selected = result['sfh_single_peak_declining']
        self.assertTrue(np.all(np.isfinite(slopes[:2])))
        self.assertTrue(np.all(slopes[:2] > 0))
        self.assertAlmostEqual(float(slopes[0]), float(slopes[1]), delta=.08)
        self.assertTrue(np.all(np.isfinite(rise_slopes[:2])))
        self.assertTrue(np.all(rise_slopes[:2] > 0))
        self.assertAlmostEqual(float(rise_slopes[0]), float(rise_slopes[1]), delta=.08)
        self.assertTrue(np.all(np.isnan(slopes[2:])))
        self.assertTrue(np.all(np.isnan(rise_slopes[2:])))
        np.testing.assert_array_equal(selected, [1, 1, 0, 0])


if __name__ == '__main__':
    unittest.main()
