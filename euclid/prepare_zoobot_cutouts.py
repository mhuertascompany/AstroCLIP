"""Prepare size-normalized Euclid VIS JPEG stamps for the ZooBot encoder.

This applies the image preparation in ``run_zoobot_within_pipeline.py`` to
the fixed-angular-size FITS cutouts made by ``bulk_sfh_cutouts.py``. The
source radius is reconstructed from final-catalog columns with
``morphology_utils.estimate_source_r_max``, then each image is cropped,
stretched, clipped, and resized as in the pipeline.
"""

import argparse
import csv
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.units import UnitsWarning
from astropy.wcs import WCS
from PIL import Image

from . import morphology_utils


SUCCESSFUL_CUTOUT_STATUSES = {'written', 'existing'}
RADIUS_COLUMNS = ('SEGMENTATION_AREA', 'KRON_RADIUS', 'ELLIPTICITY')


def _column_name(table, requested, aliases):
    names = {name.lower(): name for name in table.colnames}
    candidates = [requested] if requested else aliases
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    raise ValueError(
        f'Cannot find column {requested or aliases}; available: {table.colnames}'
    )


def _text(value):
    return value.decode().strip() if isinstance(value, bytes) else str(value).strip()


def load_source_radii(catalog, id_column=None, r_max_column=None,
                      minimum_r_max=10.0):
    """Return object_id -> R_MAX using the Euclid morphology utility formula."""
    # Euclid uses TUNIT='NA' for dimensionless identifier columns. Astropy
    # warns because that token is not a FITS unit, but the column data are fine.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnitsWarning)
        table = Table.read(catalog)
    if not np.isfinite(minimum_r_max) or minimum_r_max <= 0:
        raise ValueError('minimum_r_max must be finite and positive.')
    id_name = _column_name(table, id_column, ['object_id'])
    ids = table[id_name]
    if ids.dtype.kind not in 'iuSU' or np.any(np.ma.getmaskarray(ids)):
        raise ValueError('object_id must contain unmasked integers or integer strings.')
    exact_ids = np.array([int(_text(value)) for value in ids], dtype=np.int64)
    if len(np.unique(exact_ids)) != len(exact_ids):
        raise ValueError('Morphology catalog contains duplicate object IDs.')

    if r_max_column:
        radius_name = _column_name(table, r_max_column, [])
        radii = np.asarray(table[radius_name], dtype=float)
        method = f'catalog:{radius_name}'
    else:
        available = {name.lower() for name in table.colnames}
        missing = [name for name in RADIUS_COLUMNS if name.lower() not in available]
        if missing:
            raise ValueError(
                f'Morphology catalog is missing {missing}. Run '
                '`python -m euclid.fetch_mer_morphology` on ESA Datalabs and '
                'use its output as --catalog.'
            )
        names = {key: _column_name(table, None, [key]) for key in RADIUS_COLUMNS}
        area = np.asarray(table[names['SEGMENTATION_AREA']], dtype=float)
        kron = np.asarray(table[names['KRON_RADIUS']], dtype=float)
        ellipticity = np.asarray(table[names['ELLIPTICITY']], dtype=float)
        valid = (
            np.isfinite(area) & (area > 0)
            & np.isfinite(kron) & (kron > 0)
            & np.isfinite(ellipticity)
        )
        radii = np.full(len(table), np.nan, dtype=float)
        log_r_max = (
            -0.35048866
            + np.log10(area[valid]) * 0.506900163942416
            + ellipticity[valid] * 0.2405883433225405
            + np.log10(kron[valid]) * 0.11148176647655159
        )
        radii[valid] = np.maximum(minimum_r_max, 10.0 ** log_r_max + 2.0)
        method = 'estimated_from_SEGMENTATION_AREA_KRON_RADIUS_ELLIPTICITY'

    valid = np.isfinite(radii) & (radii > 0)
    result = {
        int(object_id): float(radius)
        for object_id, radius, keep in zip(exact_ids, radii, valid)
        if keep
    }
    print(
        f'Loaded morphology radii for {len(result):,}/{len(table):,} sources '
        f'({method}); {len(table) - len(result):,} invalid rows.',
        flush=True,
    )
    return result, method


def prepare_zoobot_stamp(data, x_center, y_center, r_max, image_size=224):
    """Apply the Euclid pipeline crop, stretch, and resize to one VIS image."""
    data = np.asarray(data)
    if data.ndim != 2 or min(data.shape) < 2:
        raise ValueError(f'Expected a 2D cutout with at least two pixels: {data.shape}')
    if image_size < 2:
        raise ValueError('image_size must be at least 2.')
    values = np.array([x_center, y_center, r_max], dtype=float)
    if not np.all(np.isfinite(values)) or r_max <= 0:
        raise ValueError('Source center and R_MAX must be finite, with R_MAX positive.')
    if not np.all(np.isfinite(data)):
        raise ValueError('Cutout contains nonfinite pixels.')

    height, width = data.shape
    if (x_center - r_max < 0 or y_center - r_max < 0
            or x_center + r_max >= width or y_center + r_max >= height):
        raise ValueError(
            f'R_MAX={r_max:.3f} px does not fit around ({x_center:.3f}, '
            f'{y_center:.3f}) in cutout shape {data.shape}; regenerate a larger FITS cutout.'
        )

    source = {'X_CENTER': x_center, 'Y_CENTER': y_center, 'R_MAX': r_max}
    cutout = morphology_utils.make_vis_only_cutout_from_tiles(source, data)

    try:
        from skimage.transform import resize
    except ImportError as exc:
        raise ImportError(
            'scikit-image is required to match run_zoobot_within_pipeline.py.'
        ) from exc
    resized = resize(
        cutout,
        output_shape=(image_size, image_size),
        order=3,
        anti_aliasing=True,
    )
    return np.rint(np.clip(resized, 0.0, 1.0) * 255).astype(np.uint8)


def _read_fits_image(path):
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                return np.array(hdu.data, copy=True), hdu.header.copy()
    raise ValueError('No two-dimensional image HDU found.')


def _valid_existing_jpeg(path, image_size):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.mode == 'L' and image.size == (image_size, image_size)
    except Exception:
        return False


def _convert_one(job):
    object_id, source, destination, ra, dec, r_max, image_size, resume = job
    source = Path(source)
    destination = Path(destination)
    record = {
        'object_id': str(object_id),
        'source_file': str(source),
        'stamp_file': str(destination),
        'r_max_pixels': f'{r_max:.8g}',
        'x_center_pixels': '',
        'y_center_pixels': '',
        'status': '',
        'error': '',
    }
    temporary = None
    try:
        if destination.exists():
            if resume and _valid_existing_jpeg(destination, image_size):
                record['status'] = 'existing'
                return record
            raise FileExistsError('Destination already exists or is not a valid stamp.')
        data, header = _read_fits_image(source)
        image_wcs = WCS(header).celestial
        x_center, y_center = image_wcs.world_to_pixel(
            SkyCoord(float(ra), float(dec), unit='deg')
        )
        x_center, y_center = float(x_center), float(y_center)
        record['x_center_pixels'] = f'{x_center:.8g}'
        record['y_center_pixels'] = f'{y_center:.8g}'
        stamp = prepare_zoobot_stamp(
            data, x_center, y_center, r_max, image_size=image_size
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + f'.{os.getpid()}.tmp')
        Image.fromarray(stamp, mode='L').save(
            temporary, format='JPEG', quality=95, subsampling=0,
        )
        os.replace(temporary, destination)
        record['status'] = 'written'
    except Exception as exc:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        record['status'] = 'failed'
        record['error'] = str(exc)
    return record


def _write_manifest(records, output):
    path = Path(output) / 'manifest.csv'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def prepare_cutout_catalog(cutout_root, catalog, output, band='VIS',
                           image_size=224, workers=8, limit=None, resume=False,
                           id_column=None, r_max_column=None,
                           minimum_r_max=10.0):
    """Convert successful cutouts using size metadata from the sample catalog."""
    cutout_root = Path(cutout_root)
    output = Path(output)
    manifest_path = cutout_root / 'manifest.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Cutout manifest not found: {manifest_path}')
    if output.exists() and not resume:
        raise FileExistsError('Output exists. Use a new directory or --resume.')
    if workers < 1:
        raise ValueError('workers must be positive.')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive.')

    radii, radius_method = load_source_radii(
        catalog, id_column, r_max_column, minimum_r_max,
    )
    with manifest_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'object_id', 'ra', 'dec', 'band', 'cutout_file', 'status'}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f'Cutout manifest is missing columns: {sorted(missing)}')

    selected = [
        row for row in rows
        if row['band'].upper() == band.upper()
        and row['status'].lower() in SUCCESSFUL_CUTOUT_STATUSES
    ]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f'No successful {band} cutouts found in {manifest_path}.')
    ids = [int(row['object_id']) for row in selected]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate object IDs found for band {band}.')

    output.mkdir(parents=True, exist_ok=True)
    stamp_dir = output / band.upper()
    jobs, invalid_records = [], []
    for object_id, row in zip(ids, selected):
        if object_id not in radii:
            invalid_records.append({
                'object_id': str(object_id),
                'source_file': str(cutout_root / row['cutout_file']),
                'stamp_file': '',
                'r_max_pixels': '',
                'x_center_pixels': '',
                'y_center_pixels': '',
                'status': 'invalid_morphology',
                'error': f'Missing or nonfinite radius input for {radius_method}.',
            })
            continue
        source = cutout_root / row['cutout_file']
        destination = stamp_dir / f'{band.upper()}_{object_id}.jpg'
        jobs.append((
            object_id, source, destination, float(row['ra']), float(row['dec']),
            radii[object_id], image_size, resume,
        ))
    if not jobs:
        raise ValueError(
            f'None of the selected cutouts has a valid morphology radius in {catalog}.'
        )

    records = list(invalid_records)
    if workers == 1:
        iterator = map(_convert_one, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(_convert_one, jobs, chunksize=32)
    try:
        for index, record in enumerate(iterator, 1):
            records.append(record)
            if index % 1000 == 0 or index == len(jobs):
                print(f'Prepared {index:,}/{len(jobs):,}', flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    _write_manifest(records, output)
    counts = {
        status: sum(record['status'] == status for record in records)
        for status in ('written', 'existing', 'invalid_morphology', 'failed')
    }
    print(f'Radius method: {radius_method}')
    print(counts)
    print(f'ZooBot stamps: {stamp_dir}')
    print(f'Conversion manifest: {output / "manifest.csv"}')
    if counts['failed']:
        raise RuntimeError(
            f'{counts["failed"]} cutouts failed conversion; inspect the manifest. '
            'R_MAX failures require larger original FITS cutouts.'
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutout-root', type=Path, required=True,
                        help='Transferred Datalabs output with manifest.csv and cutouts/.')
    parser.add_argument('--catalog', type=Path, required=True,
                        help='The matching sample catalog.fits with morphology columns.')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output root for per-band ZooBot JPEG stamps.')
    parser.add_argument('--band', default='VIS')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--id-column')
    parser.add_argument('--r-max-column',
                        help='Use an existing pixel R_MAX column instead of estimating it.')
    parser.add_argument('--minimum-r-max', type=float, default=10.0,
                        help='Minimum crop half-width in pixels (default: 10).')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    prepare_cutout_catalog(
        args.cutout_root, args.catalog, args.output, args.band, args.image_size,
        args.workers, args.limit, args.resume, args.id_column, args.r_max_column,
        args.minimum_r_max,
    )


if __name__ == '__main__':
    main()
