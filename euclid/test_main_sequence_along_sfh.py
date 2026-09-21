import unittest
import numpy as np
from euclid.main_sequence_sfh import main_sequence_along_sfh


class InferredMassTests(unittest.TestCase):
    def test_mass_and_shared_normalization(self):
        t=(np.arange(100)+.5)/100
        w=np.exp(-((t-.65)/.2)**2); w/=w.sum()
        r=main_sequence_along_sfh(t,np.log10(w+1e-10),.4,10.2)
        np.testing.assert_allclose(r['mass_at_edges'][0],10**10.2)
        self.assertEqual(r['mass_at_edges'][-1],0.)
        self.assertTrue(np.all(np.diff(r['mass_at_edges'])<=0))
        np.testing.assert_allclose(np.sum(r['sfr']*r['duration_years'])*.6,10**10.2)
        np.testing.assert_allclose(r['observed_weights']/r['weights'],r['sfr']/r['ms_sfr'])
        np.testing.assert_allclose(r['delta_ms'],np.log10(r['sfr']/r['ms_sfr']))
        self.assertFalse(np.isclose(np.nansum(r['weights']),1.))

    def test_empirical_shift_preserves_observed_history(self):
        t=(np.arange(100)+.5)/100
        w=np.ones(100)/100
        base=main_sequence_along_sfh(t,np.log10(w+1e-10),.4,10.2,return_fraction=0.)
        shifted=main_sequence_along_sfh(t,np.log10(w+1e-10),.4,10.2,
                                       return_fraction=0.,ms_sfr_offset=-.93)
        for key in ('sfr','observed_weights','mass_at_edges','supported'):
            np.testing.assert_array_equal(base[key],shifted[key])
        np.testing.assert_allclose(shifted['weights'],base['weights']*10**(-.93))
        np.testing.assert_allclose(shifted['ms_sfr'],base['ms_sfr']*10**(-.93))
        np.testing.assert_allclose(shifted['delta_ms'],base['delta_ms']+.93)

    def test_no_mass_and_quenched_bins(self):
        t=(np.arange(20)+.5)/20
        w=np.zeros(20); w[10:15]=.2
        r=main_sequence_along_sfh(t,np.log10(w+1e-10),.5,10.)
        self.assertTrue(np.isnan(r['weights'][15:]).all())
        self.assertTrue(np.isneginf(r['delta_ms'][:10]).all())
        self.assertTrue(np.all(r['ms_sfr'][:10]>0))
        self.assertTrue(np.isnan(main_sequence_along_sfh(t,np.log10(w+1e-10),.5,np.nan)['weights']).all())


if __name__=='__main__':
    unittest.main()
