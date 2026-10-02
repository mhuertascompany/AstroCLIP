"""Report dimensions, sampling, units, and flux ranges of Euclid VIS FITS cutouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .vis_fits import read_vis_fits, resolve_fits_directory


def inspect(root: Path, limit: int = 1000):
    directory = resolve_fits_directory(root, 'VIS')
    paths = sorted(directory.glob('*.fits')) + sorted(directory.glob('*.fits.gz'))
    if not paths:
        raise ValueError(f'No FITS files found in {directory}.')
    paths = paths[:min(limit, len(paths))]
    shapes, units, cut_sizes, scales, low, high = [], [], [], [], [], []
    for path in paths:
        data, header = read_vis_fits(path, header=True)
        shapes.append(tuple(data.shape))
        units.append(str(header.get('BUNIT', '')).strip() or '<missing>')
        cut_sizes.append(str(header.get('CUTSIZE', '<missing>')))
        low.append(float(np.percentile(data, 1)))
        high.append(float(np.percentile(data, 99)))
        try:
            scale = proj_plane_pixel_scales(WCS(header).celestial) * 3600.0
            scales.append(float(np.mean(scale)))
        except Exception:
            pass
    result = {
        'directory': str(directory),
        'n_files_total': len(list(directory.glob('*.fits'))) + len(list(directory.glob('*.fits.gz'))),
        'n_inspected': len(paths),
        'shape_counts': {f'{h}x{w}': n for (h, w), n in Counter(shapes).items()},
        'bunit_counts': dict(Counter(units)),
        'cutsize_counts': dict(Counter(cut_sizes)),
        'pixel_scale_arcsec_median': float(np.median(scales)) if scales else None,
        'per_image_p1_median': float(np.median(low)),
        'per_image_p99_median': float(np.median(high)),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive.')
    inspect(args.fits_root, args.limit)


if __name__ == '__main__':
    main()
