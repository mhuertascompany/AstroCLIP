"""Regression checks for sample output-directory handling."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from astropy.table import Table

from euclid.sample_edfn_sfhs import check_output_directory, match_and_sample


class SampleDirectoryTests(unittest.TestCase):
    @staticmethod
    def _write_source(path, ids):
        ids = np.asarray(ids, dtype=np.int64)
        with h5py.File(path, 'w') as target:
            target['object_id'] = ids
            target['age'] = np.arange(3.)
            target['sfh'] = np.arange(len(ids) * 6.).reshape(len(ids), 2, 3)
            target['phz_pp_median_redshift'] = np.linspace(0.1, 1.0, len(ids))
            target['phz_pp_median_stellarmass'] = np.linspace(9.0, 10.0, len(ids))
            target['sersic_sersic_vis_index'] = np.linspace(1.0, 2.0, len(ids))

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
            selected = match_and_sample(Table({
                'object_id': ids,
                'sersic_sersic_vis_radius': [1.5, 2.5],
                'sersic_sersic_vis_axis_ratio': [0.4, 0.8],
            }), [source], output, n=2)
            self.assertEqual(len(selected), 2)
            with h5py.File(output / 'sfh_000.h5', 'r') as f:
                np.testing.assert_array_equal(f['object_id'][:], ids)
                np.testing.assert_array_equal(f['sfh'][:], np.arange(12.).reshape(2, 2, 3))
                np.testing.assert_allclose(f['sersic_sersic_vis_radius'][:], [1.5, 2.5])
                np.testing.assert_allclose(f['sersic_sersic_vis_axis_ratio'][:], [0.4, 0.8])
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

    def test_bright_selection_happens_before_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.h5'
            output = root / 'bright'
            ids = np.array([101, 102, 103, 104], dtype=np.int64)
            self._write_source(source, ids)
            magnitude = np.array([20.0, 22.0, 23.0, 21.0])
            table = Table({
                'object_id': ids,
                'flux_detection_total': 10 ** ((23.9 - magnitude) / 2.5),
                'vis_det': [1, 1, 1, 0],
            })
            selected = match_and_sample(
                table, [source], output, n=2, max_vis_mag=22.5,
            )
            self.assertSetEqual(set(selected['object_id']), {101, 102})
            self.assertEqual(selected.meta['MAXVIS'], 22.5)

    def test_dry_run_reports_sfh_matched_bright_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.h5'
            self._write_source(source, [101, 103])
            magnitude = np.array([20.0, 21.0, 22.0])
            table = Table({
                'object_id': [101, 102, 103],
                'flux_detection_total': 10 ** ((23.9 - magnitude) / 2.5),
                'vis_det': [1, 1, 1],
            })
            report = match_and_sample(
                table, [source], dry_run=True, report_vis_limits=[20.5, 22.0],
            )
            self.assertEqual(
                report['counts_at_or_brighter_than']['20.5'],
                {'catalog': 1, 'with_sfh': 1},
            )
            self.assertEqual(
                report['counts_at_or_brighter_than']['22'],
                {'catalog': 3, 'with_sfh': 2},
            )


if __name__ == '__main__':
    unittest.main()
