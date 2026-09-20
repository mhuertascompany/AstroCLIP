import unittest
import numpy as np
from euclid.rejuvenation import diagnose


class RejuvenationTests(unittest.TestCase):
    def test_histories_and_scale(self):
        t=(np.arange(200)+.5)/200
        rejuvenated=np.where(t<.05,4.,np.where(t<.15,.01,2.))
        histories=np.array([rejuvenated,np.ones(200),1+t,2-t,
                            np.where(t<.05,10.,0.)])
        histories/=histories.sum(axis=1,keepdims=True)
        result=diagnose(np.log10(histories+1e-10),t)
        np.testing.assert_equal(result['candidate'],[1,0,0,0,0])
        self.assertGreaterEqual(result['lull_start'][0],.05)
        np.testing.assert_equal(diagnose(np.log10(histories*7+1e-10),t)['candidate'],result['candidate'])
        self.assertTrue(np.isnan(diagnose(np.full((1,200),np.nan),t)['candidate'][0]))

    def test_older_activity_required_and_thresholds(self):
        t=(np.arange(200)+.5)/200
        rising=np.exp(-t*10);rising/=rising.sum()
        self.assertEqual(diagnose(np.log10(rising[None]+1e-10),t)['candidate'][0],0)
        with self.assertRaises(ValueError):
            diagnose(np.zeros((1,200)),t,recent_width=.9,lull_width=.1)


if __name__=='__main__': unittest.main()
