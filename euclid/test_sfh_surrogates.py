import unittest

import numpy as np

from euclid.sfh_surrogates import iaaft_surrogate, recent_preserved_iaaft


class IAAFTSurrogateTests(unittest.TestCase):
    def test_iaaft_preserves_exact_values_and_approximately_preserves_spectrum(self):
        rng = np.random.default_rng(12)
        time = np.linspace(0, 1, 225)
        values = (
            0.1
            + np.exp(-0.5 * ((time - 0.28) / 0.09) ** 2)
            + 0.4 * np.exp(-0.5 * ((time - 0.76) / 0.05) ** 2)
        )
        surrogate, diagnostic = iaaft_surrogate(values, rng)
        np.testing.assert_allclose(np.sort(surrogate), np.sort(values), atol=0, rtol=0)
        self.assertLess(diagnostic["spectral_error"], 0.08)
        self.assertFalse(np.allclose(surrogate, values))

    def test_recent_segment_and_old_integral_are_preserved(self):
        rng = np.random.default_rng(9)
        time = np.linspace(0, 1, 250)
        values = (
            0.02
            + np.exp(-0.5 * ((time - 0.35) / 0.08) ** 2)
            + 0.5 * np.exp(-0.5 * ((time - 0.78) / 0.07) ** 2)
        )
        values /= values.sum()
        surrogate, diagnostic = recent_preserved_iaaft(values, time, rng)
        split = diagnostic["old_start"]
        np.testing.assert_array_equal(surrogate[:split], values[:split])
        self.assertAlmostEqual(float(surrogate[split:].sum()), float(values[split:].sum()), places=14)
        self.assertAlmostEqual(float(surrogate.sum()), float(values.sum()), places=14)
        self.assertGreaterEqual(float(surrogate.min()), 0.0)
        self.assertAlmostEqual(float(surrogate[split]), float(values[split]), places=14)

    def test_many_surrogates_are_unit_normalized(self):
        rng = np.random.default_rng(123)
        time = np.linspace(0, 1, 250)
        for _ in range(100):
            values = rng.lognormal(mean=-2.0, sigma=1.0, size=len(time))
            values /= values.sum()
            surrogate, diagnostic = recent_preserved_iaaft(
                values, time, rng, max_iterations=100,
            )
            self.assertAlmostEqual(float(surrogate.sum()), 1.0, places=13)
            self.assertAlmostEqual(diagnostic["old_integral_error"], 0.0, places=13)
            np.testing.assert_array_equal(
                surrogate[:diagnostic["old_start"]],
                values[:diagnostic["old_start"]],
            )


if __name__ == "__main__":
    unittest.main()
