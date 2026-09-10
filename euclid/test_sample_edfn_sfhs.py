"""Regression checks for sample output-directory handling."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from astropy.table import Table

from euclid.sample_edfn_sfhs import check_output_directory, match_and_sample


class SampleDirectoryTests(unittest.TestCase):
    def test_sample_into_existing_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.h5'
            ids = np.array([2688124142666475971, 2688124142666475972], dtype=np.int64)
            with h5py.File(source, 'w') as f:
                f['object_id'] = ids
                f['age'] = np.arange(3.)
                f['sfh'] = np.arange(12.).reshape(2, 2, 3)
                for key in ('phz_pp_median_redshift', 'phz_pp_median_stellarmass',
                            'sersic_sersic_vis_index'):
                    f[key] = [1., 2.]
            output = root / 'precreated'
            output.mkdir()
            selected = match_and_sample(Table({'object_id': ids}), [source], output, n=2)
            self.assertEqual(len(selected), 2)
            with h5py.File(output / 'sfh_000.h5', 'r') as f:
                np.testing.assert_array_equal(f['object_id'][:], ids)
                np.testing.assert_array_equal(f['sfh'][:], np.arange(12.).reshape(2, 2, 3))
            self.assertTrue((output / 'catalog.fits').is_file())
            self.assertTrue((output / 'object_ids.csv').is_file())
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaisesRegex(FileExistsError, 'new or empty'):
                match_and_sample(Table({'object_id': ids}), [source], output, n=2)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_new_directory_is_allowed_without_creating_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'new'
            check_output_directory(output)
            self.assertFalse(output.exists())

    def test_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'file'
            output.write_text('preserve this')
            with self.assertRaisesRegex(FileExistsError, 'new or empty'):
                check_output_directory(output)
            self.assertEqual(output.read_text(), 'preserve this')


if __name__ == '__main__':
    unittest.main()
