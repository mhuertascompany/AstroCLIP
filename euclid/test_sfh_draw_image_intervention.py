import unittest

import numpy as np

from euclid.sfh_draw_image_intervention import (
    cosine_distance_rows,
    recent_log_sfr,
    select_diverse_draws,
)


class SFHDrawImageInterventionTests(unittest.TestCase):
    def test_recent_log_sfr_matches_known_recent_fraction(self):
        time = np.linspace(0, 1, 250)
        age_myr = 1000.0
        weights = np.ones((2, 250))
        weights /= weights.sum(axis=1, keepdims=True)
        result = recent_log_sfr(weights, 10.0, age_myr, time)
        np.testing.assert_allclose(result, np.full(2, 1.0), atol=0.02)

    def test_diverse_selection_honors_sfr_filter(self):
        time = np.linspace(0, 1, 20)
        draws = np.eye(6, 20)
        sfr = np.asarray([0.01, 0.04, 0.09, 0.8, 1.0, 1.2])
        chosen, relaxed = select_diverse_draws(
            draws, sfr, 0.0, time, 1000.0, count=3, tolerance=0.1,
        )
        self.assertFalse(relaxed)
        self.assertEqual(len(chosen), 3)
        self.assertTrue(np.all(np.abs(sfr[chosen]) <= 0.1))

    def test_cosine_distance(self):
        distance = cosine_distance_rows(np.asarray([[1.0, 0.0], [0.0, 1.0]]))
        np.testing.assert_allclose(distance, [[0, 1], [1, 0]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
