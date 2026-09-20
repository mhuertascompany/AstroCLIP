import unittest
import numpy as np
from euclid.main_sequence_sfh import main_sequence_track


class MainSequenceTests(unittest.TestCase):
    def test_mass_conservation_and_normalization(self):
        for z in (.2, 1., 3.):
            for mass in (9., 10., 11.5):
                track = main_sequence_track(np.linspace(0, 1, 250), z, mass)
                self.assertAlmostEqual(np.nansum(track['weights']), 1.)
                np.testing.assert_allclose(track['modeled_formed_mass']*.6,
                                           track['final_mass']-track['seed_mass'])
                self.assertTrue(np.all(track['weights'][np.isfinite(track['weights'])] >= 0))

    def test_rebinning_and_mass_offset(self):
        coarse = main_sequence_track((np.arange(10)+.5)/10, .5, 10.)
        fine = main_sequence_track((np.arange(100)+.5)/100, .5, 10.)
        np.testing.assert_allclose(np.nan_to_num(coarse['weights']),
                                   np.nansum(fine['weights'].reshape(10, 10), axis=1), atol=1e-9)
        shifted = main_sequence_track(np.linspace(0, 1, 20), .5, 10., mass_offset=.03)
        direct = main_sequence_track(np.linspace(0, 1, 20), .5, 10.03)
        np.testing.assert_allclose(shifted['weights'], direct['weights'])
        self.assertTrue(np.isnan(main_sequence_track(np.linspace(0, 1, 20), .5, np.nan)['weights']).all())


if __name__ == '__main__':
    unittest.main()
