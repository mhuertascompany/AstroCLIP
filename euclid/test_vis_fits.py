import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from euclid.vis_fits import (
    VISFitsTransform,
    load_image_stats,
    normalize_vis_flux,
    resolve_fits_directory,
    vis_fits_path,
    write_image_stats,
)


class VISFitsTest(unittest.TestCase):
    def test_resolve_normalize_and_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fits_dir = root / 'cutouts' / 'VIS'
            fits_dir.mkdir(parents=True)
            path = fits_dir / '42.fits'
            fits.PrimaryHDU(np.arange(100, dtype=np.float32).reshape(10, 10)).writeto(path)
            stats = root / 'image_stats.json'
            write_image_stats(stats, 0.0, 99.0)

            self.assertEqual(resolve_fits_directory(root), fits_dir)
            self.assertEqual(vis_fits_path(fits_dir, 42), path)
            self.assertEqual(load_image_stats(stats), (0.0, 99.0))
            normalized = normalize_vis_flux(np.array([0.0, 99.0]), 0.0, 99.0)
            np.testing.assert_allclose(normalized, [0.0, 1.0], atol=1e-6)
            tensor = VISFitsTransform(8, 0.0, 99.0, training=False)(path)
            self.assertEqual(tensor.shape, (1, 8, 8))
            self.assertTrue(np.isfinite(tensor.numpy()).all())


if __name__ == '__main__':
    unittest.main()
