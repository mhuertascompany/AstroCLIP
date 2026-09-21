import unittest
import numpy as np
from euclid.ms_deviation import integrated_deviations

class DeviationTests(unittest.TestCase):
    def track(self,b,g):
        return dict(observed_weights=np.array(b),weights=np.array(g),
                    duration_years=np.array([1.,3.]),supported=np.array([True,False]))
    def test_equal_and_opposite_do_not_cancel(self):
        p,m,c=integrated_deviations(self.track([3.,1.],[1.,3.]))
        self.assertEqual(p,.25);self.assertEqual(m,.25);self.assertEqual(c,.25)
        self.assertEqual(integrated_deviations(self.track([1.,3.],[1.,3.]))[:2],(0.,0.))
    def test_zero_and_undefined(self):
        p,m,c=integrated_deviations(self.track([0.,0.],[2.,np.nan]))
        self.assertEqual((p,m,c),(0.,1.,1.))
        self.assertTrue(np.isnan(integrated_deviations({})[0]))
    def test_scale_invariance(self):
        np.testing.assert_allclose(integrated_deviations(self.track([3.,1.],[1.,3.])),
                                   integrated_deviations(self.track([30.,10.],[10.,30.])))
if __name__=='__main__': unittest.main()
