import unittest
import numpy as np
from euclid.ms_deviation import integrated_deviations, history_summary, accumulated_expansion_time

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
class HistorySummaryTests(unittest.TestCase):
    def test_time_weighting_and_floor(self):
        track = dict(delta_ms=np.array([1., -np.inf, np.nan]),
                     duration_years=np.array([1e9, 3e9, 2e9]), observation_age_years=10e9)
        times = accumulated_expansion_time(10e9, np.array([.5e9, 2.5e9]))
        np.testing.assert_allclose(history_summary(track), [1., -3., *times, -2., .75])
    def test_absent_excursion_and_ties(self):
        track = dict(delta_ms=np.array([1., 1.]), duration_years=np.array([1e9, 1e9]), observation_age_years=10e9)
        result = history_summary(track)
        self.assertEqual(result[1], 0.)
        self.assertTrue(np.isnan(result[3]))
        self.assertAlmostEqual(result[2], float(accumulated_expansion_time(10e9, .5e9)))
    def test_expansion_clock_against_redshift(self):
        from euclid.cosmic_sfh import COSMOLOGY
        age = COSMOLOGY.age(.5).to_value('yr')
        past = COSMOLOGY.age(np.array([.5, 1., 3., 10.])).to_value('yr')
        expected = np.log(np.array([1.5, 2., 4., 11.])/1.5)
        np.testing.assert_allclose(accumulated_expansion_time(age, age-past), expected, atol=1e-7)
        self.assertTrue(np.isnan(accumulated_expansion_time(age, age)))

    def test_invalid(self):
        self.assertTrue(np.isnan(history_summary({})).all())
        with self.assertRaises(ValueError):
            history_summary({}, floor=0.)

if __name__=='__main__': unittest.main()
