"""Tests for bulk Euclid ZooBot stamp preparation."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from PIL import Image

from euclid import morphology_utils
from euclid.prepare_zoobot_cutouts import (
    load_source_radii,
    prepare_cutout_catalog,
    prepare_zoobot_stamp,
)


HAS_SKIMAGE = importlib.util.find_spec('skimage') is not None


class RadiusTests(unittest.TestCase):
    def test_estimated_radius_matches_reference_utility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'catalog.fits'
            table = Table({
                'object_id': np.array([101, 202], dtype=np.int64),
                'segmentation_area': [1200.0, 600.0],
                'kron_radius': [3.2, 2.1],
                'ellipticity': [0.4, 0.2],
            })
            table.write(path)
            radii, method = load_source_radii(path)
            source = {
                'LOG_SEGMENTATION_AREA': np.log10(1200.0),
                'LOG_KRON_RADIUS': np.log10(3.2),
                'ELLIPTICITY': 0.4,
            }
            self.assertAlmostEqual(
                radii[101], morphology_utils.estimate_source_r_max(source)
            )
            self.assertIn('estimated', method)

    def test_existing_rmax_can_be_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'catalog.fits'
            Table({
                'OBJECT_ID': np.array([101], dtype=np.int64),
                'R_MAX': [17.5],
            }).write(path)
            radii, method = load_source_radii(path, r_max_column='r_max')
            self.assertEqual(radii[101], 17.5)
            self.assertEqual(method, 'catalog:R_MAX')

    def test_crop_larger_than_input_is_rejected_before_resize(self):
        with self.assertRaisesRegex(ValueError, 'regenerate a larger FITS cutout'):
            prepare_zoobot_stamp(
                np.ones((20, 20)), x_center=9.5, y_center=9.5, r_max=11
            )


@unittest.skipUnless(HAS_SKIMAGE, 'scikit-image is required for exact pipeline resize')
class CatalogTests(unittest.TestCase):
    def test_reference_conversion_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cutout_root = root / 'cutouts_run'
            fits_dir = cutout_root / 'cutouts' / 'VIS'
            fits_dir.mkdir(parents=True)

            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [31, 31]
            wcs.wcs.cdelt = [-0.1 / 3600, 0.1 / 3600]
            wcs.wcs.crval = [10, 20]
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            yy, xx = np.indices((60, 60))
            image = np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / 20).astype(np.float32)
            fits.PrimaryHDU(image, header=wcs.to_header()).writeto(fits_dir / '101.fits')

            with (cutout_root / 'manifest.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=['object_id', 'ra', 'dec', 'band', 'cutout_file', 'status'],
                )
                writer.writeheader()
                writer.writerow({
                    'object_id': 101, 'ra': 10, 'dec': 20, 'band': 'VIS',
                    'cutout_file': 'cutouts/VIS/101.fits', 'status': 'written',
                })
            catalog = root / 'catalog.fits'
            Table({
                'object_id': np.array([101], dtype=np.int64),
                'segmentation_area': [100.0],
                'kron_radius': [2.0],
                'ellipticity': [0.2],
            }).write(catalog)

            output = root / 'stamps'
            records = prepare_cutout_catalog(
                cutout_root, catalog, output, image_size=24, workers=1,
            )
            self.assertEqual(records[0]['status'], 'written')
            self.assertGreater(float(records[0]['r_max_pixels']), 0)
            stamp_path = output / 'VIS' / 'VIS_101.jpg'
            with Image.open(stamp_path) as stamp:
                self.assertEqual(stamp.mode, 'L')
                self.assertEqual(stamp.size, (24, 24))
            records = prepare_cutout_catalog(
                cutout_root, catalog, output, image_size=24, workers=1, resume=True,
            )
            self.assertEqual(records[0]['status'], 'existing')


if __name__ == '__main__':
    unittest.main()
