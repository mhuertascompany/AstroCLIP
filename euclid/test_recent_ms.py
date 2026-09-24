import unittest
import numpy as np
from euclid.recent_ms import recent_offset, matched_mask

class RecentMSTests(unittest.TestCase):
    def test_partial_bin_and_ratio_of_integrals(self):
        track = dict(duration_years=np.array([1., 3.]), sfr=np.array([4., 0.]), ms_sfr=np.array([1., 2.]))
        self.assertAlmostEqual(recent_offset(track, 2.), np.log10(4./3.))
        self.assertTrue(np.isnan(recent_offset(track, 5.)))
    def test_zero_and_missing(self):
        track = dict(duration_years=np.array([1., 3.]), sfr=np.array([0., np.nan]), ms_sfr=np.array([1., np.nan]))
        self.assertEqual(recent_offset(track, 1.), -4.)
        self.assertTrue(np.isnan(recent_offset(track, 2.)))
    def test_intersection_and_missing_values(self):
        result = matched_mask([10.,10.,10.,np.nan], [.5,.5,.9,.5], [0.,1.,0.,0.],
                              (9.9,10.1), (.4,.6), (-.1,.1))
        np.testing.assert_array_equal(result, [True,False,False,False])

if __name__ == '__main__':
    unittest.main()
