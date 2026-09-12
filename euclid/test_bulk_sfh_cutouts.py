"""Offline checks: python -m unittest euclid.test_bulk_sfh_cutouts."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from euclid.bulk_sfh_cutouts import load_sample, make_cutouts, mosaic_query, query_mosaics


class BulkCutoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = np.array([2688124142666475971 + i for i in range(4)], dtype=np.int64)
        self.sample = Table(dict(object_id=self.ids, right_ascension=[10.] * 4,
                                 declination=[20.] * 4, sfh_file=['sfh_000.h5'] * 4,
                                 sfh_row=[3, 2, 1, 0]))
        self.path = self.root / 'catalog.fits'
        self.sample.write(self.path)
        self.sample, self.sources = load_sample(self.path)
        self.output = self.root / 'output'
        self.output.mkdir()

    def mosaic(self, name, center_pixel):
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [center_pixel + 1] * 2
        wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
        wcs.wcs.crval = [10, 20]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        header = wcs.to_header()
        header['BUNIT'] = 'adu'
        data = np.arange(30 * 30, dtype=np.float32).reshape(30, 30)
        path = self.root / name
        fits.PrimaryHDU(data, header=header).writeto(path)
        return path, data, wcs

    def row(self, index, path, distance=0., band='VIS'):
        return dict(sample_row=index, object_id=self.ids[index],
                    datalabs_path=str(path.parent), file_name=path.name,
                    instrument_name='VIS' if band == 'VIS' else 'NISP', filter_name=band,
                    tile_index=123, processing_mode='DEEP', release_name='TEST',
                    reference_observation_date_time='2026-01-01', dist_cent=distance)

    def test_pixels_wcs_ids_fallback_and_resume(self):
        edge, _, _ = self.mosaic('edge.fits', 0)
        good, original, original_wcs = self.mosaic('good.fits', 15)
        matches = Table(rows=[self.row(0, edge), self.row(0, good, 1),
                              self.row(1, good), self.row(3, self.root / 'missing.fits')])
        records = make_cutouts(self.sample, self.sources, matches, self.output)
        self.assertEqual([r['status'] for r in records], ['written', 'written', 'no_mosaic', 'failed'])
        self.assertEqual(records[0]['sfh_row'], '3')
        with fits.open(self.output / records[0]['cutout_file']) as hdul:
            image, header = hdul[0].data, hdul[0].header
            self.assertEqual(image.shape, (10, 10))
            self.assertEqual(header['OBJID'], str(self.ids[0]))
            self.assertEqual(header['BUNIT'], 'adu')
            # Verify new WCS maps output pixels to the original mosaic pixels.
            sky = WCS(header).pixel_to_world(3, 4)
            x, y = original_wcs.world_to_pixel(sky)
            self.assertEqual(image[4, 3], original[round(float(y)), round(float(x))])
            x, y = WCS(header).world_to_pixel_values(10, 20)
            self.assertTrue(0 <= x < 10 and 0 <= y < 10)
        second = make_cutouts(self.sample, self.sources, matches, self.output)
        self.assertEqual([r['status'] for r in second[:2]], ['existing', 'existing'])
        self.assertEqual(second[0]['mosaic_file'], str(good))
        content = (self.output / 'manifest.csv').read_text()
        self.assertIn(str(self.ids[0]), content)
        self.assertIn('no_mosaic', content)

    def test_multiband_and_wrong_id_rejection(self):
        good, _, _ = self.mosaic('good.fits', 15)
        matches = Table(rows=[self.row(0, good, band='NIR_H'), self.row(0, good)])
        records = make_cutouts(self.sample[:1], self.sources[:1], matches,
                               self.output, bands=['VIS', 'NIR-H'])
        self.assertEqual([r['status'] for r in records], ['written', 'written'])
        self.assertNotEqual(records[0]['cutout_file'], records[1]['cutout_file'])
        matches['object_id'][0] += 1
        with self.assertRaisesRegex(ValueError, 'IDs do not match'):
            make_cutouts(self.sample, self.sources, matches, self.output)

    def test_upload_batching_and_cache(self):
        owner = self

        class Client:
            calls = 0

            def launch_job_async(self, query, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return None
                upload = Table.read(kwargs['upload_resource'], format='votable')
                indices = list(upload['sample_row'])
                for row in upload:
                    owner.assertEqual(row['object_id'], owner.ids[row['sample_row']])
                result = Table(rows=[owner.row(i, owner.root / 'mosaic.fits') for i in indices])

                class Job:
                    def get_results(self):
                        return result

                return Job()

        client = Client()
        query = mosaic_query(['VIS', 'NIR-H'], 10)
        self.assertIn('dr1.mosaic_product', query)
        self.assertIn('TAP_UPLOAD.sfh_sample', query)
        self.assertIn("processing_mode = 'DEEP'", query)
        result = query_mosaics(client, self.sources, self.output, query,
                               batch_size=2, query_retries=2, retry_delay=0)
        self.assertEqual(client.calls, 3)
        self.assertEqual(len(result), 4)
        cached = query_mosaics(None, self.sources, self.output, query, batch_size=2)
        np.testing.assert_array_equal(cached['object_id'], self.ids)

    def test_reject_float_ids_and_invalid_coordinates(self):
        table = self.sample.copy()
        table['object_id'] = np.array(self.ids, dtype=float)
        bad = self.root / 'bad.fits'
        table.write(bad)
        with self.assertRaisesRegex(ValueError, 'object_id'):
            load_sample(bad)
        table = self.sample.copy()
        table['declination'][0] = 91.
        table.write(bad, overwrite=True)
        with self.assertRaisesRegex(ValueError, 'Declination'):
            load_sample(bad)


if __name__ == '__main__':
    unittest.main()
