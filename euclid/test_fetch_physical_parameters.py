"""Offline checks for exact-ID PHZ joins and FITS round trips."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.table import Table, MaskedColumn
from euclid.fetch_physical_parameters import COLUMNS, align_results, physical_query


class PhysicalJoinTests(unittest.TestCase):
    def test_order_missing_vectors_and_integer_precision(self):
        ids = np.array([2758212946672935686, 2758212946672935687,
                        2758212946672935688], dtype=np.int64)
        sources = Table({'object_id': ids})
        results = Table({'sample_row': [2, 0], 'object_id': ids[[2, 0]]})
        for name in COLUMNS:
            if '_68_' in name:
                results[name] = [[-2., -1.], [0., 1.]]
            elif name in ('phys_param_flags', 'quality_flag', 'galaxyclass', 'sfhtype', 'imf'):
                results[name] = np.array([7, 3], dtype=np.int64)
            else:
                results[name] = MaskedColumn([-1., 0.], mask=[True, False])
        output = align_results(sources, results)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'physical.fits'
            output.write(path)
            restored = Table.read(path)
        np.testing.assert_array_equal(restored['object_id'], ids)
        np.testing.assert_array_equal(restored['physical_parameters_matched'], [True, False, True])
        np.testing.assert_array_equal(restored['phz_pp_68_sfr'][0], [0., 1.])
        self.assertTrue(np.all(restored['phz_pp_68_sfr'].mask[1]))
        self.assertTrue(restored['phz_pp_median_sfr'].mask[2])
        self.assertEqual(restored['phys_param_flags'].dtype.kind, 'i')
        self.assertEqual(restored['phys_param_flags'][0], 3)
        results['sample_row'] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            align_results(sources, results)
        results['sample_row'] = [2, 0]
        results['object_id'][0] = ids[1]
        with self.assertRaisesRegex(ValueError, 'IDs'):
            align_results(sources, results)
        self.assertIn('catalogue.phz_physical_parameters_deep', physical_query())


if __name__ == '__main__':
    unittest.main()
