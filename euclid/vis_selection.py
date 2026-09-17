"""Select bright VIS-detected Euclid objects from a CLIP-ready HDF5 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


VIS_AB_ZEROPOINT_UJY = 23.9


def flux_ujy_to_ab_magnitude(flux):
    """Convert positive microJy fluxes to AB magnitude; invalid values are NaN."""
    flux = np.asarray(flux, dtype=np.float64)
    magnitude = np.full(flux.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(flux) & (flux > 0)
    magnitude[valid] = VIS_AB_ZEROPOINT_UJY - 2.5 * np.log10(flux[valid])
    return magnitude


def _dataset_name(source, requested):
    names = {name.lower(): name for name in source.keys()}
    name = names.get(requested.lower())
    if name is None:
        raise ValueError(
            f'Cannot find HDF5 dataset {requested!r}; available: {sorted(source.keys())}'
        )
    return name


def read_vis_magnitude(source, flux_column='flux_detection_total',
                       detection_column='vis_det', require_vis_detection=True):
    """Read a scalar microJy flux column and return row-aligned VIS AB magnitudes."""
    flux_name = _dataset_name(source, flux_column)
    flux = np.asarray(source[flux_name][:], dtype=np.float64)
    if flux.ndim != 1:
        raise ValueError(f'{flux_name} must be a scalar row-aligned dataset.')
    magnitude = flux_ujy_to_ab_magnitude(flux)
    detected = np.ones(len(flux), dtype=bool)
    detection_name = None
    if require_vis_detection:
        detection_name = _dataset_name(source, detection_column)
        detection = np.asarray(source[detection_name][:])
        if detection.shape != flux.shape:
            raise ValueError(f'{detection_name} is not aligned with {flux_name}.')
        detected = np.isfinite(detection) & (detection == 1)
        magnitude[~detected] = np.nan
    return magnitude, flux, detected, flux_name, detection_name


def bright_row_mask(path, max_vis_mag, flux_column='flux_detection_total',
                    detection_column='vis_det', require_vis_detection=True):
    if not np.isfinite(max_vis_mag):
        raise ValueError('max_vis_mag must be finite.')
    with h5py.File(path, 'r') as source:
        magnitude, _, _, _, _ = read_vis_magnitude(
            source, flux_column, detection_column, require_vis_detection,
        )
    return np.isfinite(magnitude) & (magnitude <= max_vis_mag), magnitude


def summarize_sample(dataset, limits, stamp_root=None, band='VIS',
                     flux_column='flux_detection_total',
                     detection_column='vis_det', require_vis_detection=True):
    dataset = Path(dataset)
    with h5py.File(dataset, 'r') as source:
        ids = np.asarray(source['galaxy_id'][:], dtype=np.int64)
        magnitude, flux, detected, flux_name, detection_name = read_vis_magnitude(
            source, flux_column, detection_column, require_vis_detection,
        )
        redshift = (
            np.asarray(source['redshift'][:], dtype=np.float32)
            if 'redshift' in source else np.full(len(ids), np.nan, dtype=np.float32)
        )
    stamps = np.ones(len(ids), dtype=bool)
    if stamp_root is not None:
        stamp_dir = Path(stamp_root) / band
        if not stamp_dir.is_dir():
            raise FileNotFoundError(stamp_dir)
        stamps = np.array([
            (stamp_dir / f'{band}_{int(galaxy_id)}.jpg').is_file()
            for galaxy_id in ids
        ], dtype=bool)
    usable = stamps & np.isfinite(magnitude)
    counts = {
        f'{float(limit):g}': int(np.count_nonzero(usable & (magnitude <= limit)))
        for limit in limits
    }
    report = {
        'dataset': str(dataset),
        'n_objects': int(len(ids)),
        'n_with_stamps': int(np.count_nonzero(stamps)),
        'n_vis_detected_with_positive_flux_and_stamp': int(np.count_nonzero(usable)),
        'flux_column': flux_name,
        'flux_unit': 'microJy',
        'ab_zeropoint': VIS_AB_ZEROPOINT_UJY,
        'detection_column': detection_name,
        'require_vis_detection': bool(require_vis_detection),
        'counts_at_or_brighter_than': counts,
    }
    arrays = {
        'galaxy_id': ids,
        'h5_row': np.arange(len(ids), dtype=np.int64),
        'vis_magnitude': magnitude.astype(np.float32),
        'vis_flux_ujy': flux.astype(np.float32),
        'redshift': redshift,
        'has_stamp': stamps,
        'usable': usable,
        'vis_detected': detected,
    }
    return report, arrays


def write_selection(path, arrays, max_vis_mag, overwrite=False):
    from astropy.table import Table

    selected = arrays['usable'] & (arrays['vis_magnitude'] <= max_vis_mag)
    table = Table({
        key: value[selected] for key, value in arrays.items()
        if key not in {'usable', 'vis_detected'}
    })
    table.meta['max_vis_mag'] = float(max_vis_mag)
    table.meta['magnitude_system'] = 'AB'
    table.meta['flux_unit'] = 'microJy'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, overwrite=overwrite)
    return len(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--stamp-root', type=Path)
    parser.add_argument('--band', default='VIS')
    parser.add_argument('--limits', type=float, nargs='+',
                        default=(20.5, 21.0, 21.5, 22.0))
    parser.add_argument('--flux-column', default='flux_detection_total')
    parser.add_argument('--detection-column', default='vis_det')
    parser.add_argument('--require-vis-detection', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--output', type=Path,
                        help='Optional FITS selection for --max-vis-mag.')
    parser.add_argument('--max-vis-mag', type=float)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output is not None and args.max_vis_mag is None:
        parser.error('--output requires --max-vis-mag.')
    report, arrays = summarize_sample(
        args.dataset, args.limits, args.stamp_root, args.band,
        args.flux_column, args.detection_column, args.require_vis_detection,
    )
    if args.output is not None:
        report['selection_output'] = str(args.output)
        report['selection_count'] = write_selection(
            args.output, arrays, args.max_vis_mag, args.overwrite,
        )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
