import unittest
import numpy as np
from euclid.cosmic_sfh import COSMOLOGY, cosmic_sfh_weights


class CosmicSFHTests(unittest.TestCase):
    def test_normalization_and_rebinning(self):
        coarse = (np.arange(10) + .5) / 10
        fine = (np.arange(100) + .5) / 100
        for z in (0., .5, 2., 5.):
            a = cosmic_sfh_weights(coarse, z)
            b = cosmic_sfh_weights(fine, z)
            self.assertTrue(np.all(a >= 0))
            self.assertAlmostEqual(a.sum(), 1.)
            np.testing.assert_allclose(a, b.reshape(10, 10).sum(axis=1), atol=2e-6)
            np.testing.assert_allclose(a, cosmic_sfh_weights(coarse, z, COSMOLOGY.age(z).value*1000))

    def test_peak_and_invalid_redshift(self):
        time = np.linspace(0, 1, 250)
        peak_z = 2.9 * (2.7 / (5.6 - 2.7))**(1/5.6) - 1
        for z in (0., .5, 1.):
            reference = cosmic_sfh_weights(time, z)
            expected = 1 - COSMOLOGY.age(peak_z).value / COSMOLOGY.age(z).value
            self.assertLess(abs(time[np.argmax(reference)] - expected), .006)
        self.assertTrue(np.isnan(cosmic_sfh_weights(time, np.nan)).all())


if __name__ == '__main__':
    unittest.main()
