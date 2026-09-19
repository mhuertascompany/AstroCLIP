import unittest

import numpy as np

from euclid.sfh_shape import sfh_duration_80, sfh_recent_activity


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


if __name__ == '__main__':
    unittest.main()
