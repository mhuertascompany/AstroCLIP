import unittest
import numpy as np
from euclid.sfh_migration import migration,formation_times


class MigrationTests(unittest.TestCase):
    def test_constant_sfr_and_units(self):
        t=(np.arange(100)+.5)/100;w=np.full((1,100),.01)
        r=migration(np.log10(w+1e-10),t)
        np.testing.assert_allclose(r['angle'],0,atol=1e-10)
        np.testing.assert_allclose(r['delta_log_mass'],-np.log10(.9))
        np.testing.assert_allclose(r['speed_fraction'],r['displacement']/.1)
        times=formation_times(np.log10(w+1e-10),t)
        np.testing.assert_allclose(times['T50'],.5)
        np.testing.assert_allclose(times['T90'],.1)

    def test_direction_normalization_and_missing(self):
        t=(np.arange(100)+.5)/100
        w=np.array([np.exp(-5*t),np.exp(5*t)]);w/=w.sum(axis=1,keepdims=True)
        r=migration(np.log10(w+1e-10),t)
        self.assertGreater(r['angle'][0],0);self.assertLess(r['angle'][1],0)
        scaled=migration(np.log10(w*10+1e-10),t)
        np.testing.assert_allclose(r['angle'],scaled['angle'])
        w[0,:10]=0
        r=migration(np.log10(w+1e-10),t)
        self.assertTrue(np.isnan(r['angle'][0]));self.assertEqual(r['valid'][0],0)
        r=migration(np.log10(w+1e-10),t,lag=.99,window=.02)
        self.assertFalse(r['valid'].any())


if __name__=='__main__':unittest.main()
