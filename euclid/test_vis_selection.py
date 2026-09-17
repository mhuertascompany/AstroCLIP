import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from euclid.vis_selection import (
    bright_row_mask,
    flux_ujy_to_ab_magnitude,
    summarize_sample,
)


class VisSelectionTests(unittest.TestCase):
    def test_microjy_conversion(self):
        flux = 10.0 ** ((23.9 - np.array([20.5, 21.0, 22.0])) / 2.5)
        np.testing.assert_allclose(
            flux_ujy_to_ab_magnitude(flux), [20.5, 21.0, 22.0], atol=1e-10,
        )
        self.assertTrue(np.isnan(flux_ujy_to_ab_magnitude([0.0, -1.0])).all())

    def test_summary_and_cut_require_vis_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'sample.h5'
            magnitude = np.array([20.0, 21.0, 22.0, 23.0])
            with h5py.File(path, 'w') as target:
                target['galaxy_id'] = np.arange(4, dtype=np.int64)
                target['redshift'] = np.linspace(0.1, 0.4, 4)
                target['flux_detection_total'] = 10 ** ((23.9 - magnitude) / 2.5)
                target['vis_det'] = [1, 0, 1, 1]
            mask, measured = bright_row_mask(path, 22.0)
            np.testing.assert_array_equal(mask, [True, False, True, False])
            self.assertTrue(np.isnan(measured[1]))
            report, _ = summarize_sample(path, [21.0, 22.0])
            self.assertEqual(report['counts_at_or_brighter_than']['21'], 1)
            self.assertEqual(report['counts_at_or_brighter_than']['22'], 2)


if __name__ == '__main__':
    unittest.main()
