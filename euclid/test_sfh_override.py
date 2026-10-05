import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from euclid.dataset_zoobot import validate_sfh_override


class SFHOverrideTests(unittest.TestCase):
    def _write(self, path, ids, sfh, time, maximum_error=1e-7):
        with h5py.File(path, 'w') as target:
            target.create_dataset('galaxy_id', data=ids)
            target.create_dataset('sfh', data=sfh)
            target.create_dataset('sfh_time_grid', data=time)
            target.attrs['integral_tolerance'] = 2e-6
            target.attrs['maximum_absolute_stored_integral_error'] = maximum_error

    def test_row_aligned_normalized_override_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'source.h5'
            override = directory / 'override.h5'
            ids = np.array([11, 12, 13])
            sfh = np.zeros((3, 5), dtype=np.float32)
            time = np.linspace(0, 1, 5, dtype=np.float32)
            self._write(source, ids, sfh, time)
            self._write(override, ids, sfh, time)
            tolerance, error = validate_sfh_override(source, override)
            self.assertEqual(tolerance, 2e-6)
            self.assertEqual(error, 1e-7)

    def test_misaligned_or_failed_normalization_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'source.h5'
            override = directory / 'override.h5'
            ids = np.array([11, 12, 13])
            sfh = np.zeros((3, 5), dtype=np.float32)
            time = np.linspace(0, 1, 5, dtype=np.float32)
            self._write(source, ids, sfh, time)
            self._write(override, ids[::-1], sfh, time)
            with self.assertRaisesRegex(ValueError, 'row-aligned'):
                validate_sfh_override(source, override)
            override.unlink()
            self._write(override, ids, sfh, time, maximum_error=3e-6)
            with self.assertRaisesRegex(ValueError, 'exceeds tolerance'):
                validate_sfh_override(source, override)


if __name__ == '__main__':
    unittest.main()
