"""Make SFH-sample mosaic cutouts on ESA Datalabs, following Cutouts_v3.ipynb.

Uses IDR dr1.mosaic_product footprint queries and Astropy Cutout2D. Mosaics
are read from the mounted data volume; no Cutana or n_utils dependency.
Example (EUCLID-TOOLS environment, repository root):
    python -m euclid.bulk_sfh_cutouts --sample edfn_10k/catalog.fits \
        --output /media/user/edfn_10k_cutouts --size-arcsec 10 --bands VIS
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table, vstack
from astropy.wcs import WCS


BANDS = ('VIS', 'NIR-Y', 'NIR-J', 'NIR-H')
SFH_COLUMNS = ('sfh_file', 'sfh_row', 'sfh_source_file', 'sfh_source_row',
               'field_catalog_row')


def text(value):
    return value.decode().strip() if isinstance(value, bytes) else str(value).strip()


def column_name(table, explicit, aliases):
    names = {name.lower(): name for name in table.colnames}
    for name in ([explicit] if explicit else aliases):
        if name.lower() in names:
            return names[name.lower()]
    raise ValueError(f'Cannot find column {explicit or aliases}; available: {table.colnames}')


def load_sample(path, id_column=None, ra_column=None, dec_column=None, limit=None):
    table = Table.read(path)
    if limit is not None:
        table = table[:limit]
    if not len(table):
        raise ValueError('The sample is empty.')
    id_name = column_name(table, id_column, ['object_id'])
    ids = table[id_name]
    if ids.dtype.kind not in 'iuSU' or np.any(np.ma.getmaskarray(ids)):
        raise ValueError('object_id must contain unmasked integers or integer strings.')
    # Never cast IDs through float: these identifiers exceed float64 precision.
    exact_ids = np.array([int(text(value)) for value in ids], dtype=np.int64)
    if np.any(exact_ids < 0) or len(np.unique(exact_ids)) != len(ids):
        raise ValueError('object_id must be nonnegative and unique.')
    sources = Table({'sample_row': np.arange(len(table)), 'object_id': exact_ids})
    for key, explicit, aliases in (
        ('ra', ra_column, ['right_ascension', 'ra', 'ra_deg']),
        ('dec', dec_column, ['declination', 'dec', 'dec_deg']),
    ):
        col = table[column_name(table, explicit, aliases)]
        if np.any(np.ma.getmaskarray(col)):
            raise ValueError(f'Masked {key} coordinates in sample.')
        values = col.quantity.to_value(u.deg) if col.unit else np.asarray(col, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f'Nonfinite {key} coordinates in sample.')
        sources[key] = np.asarray(values, dtype=np.float64)
    if np.any((sources['ra'] < 0) | (sources['ra'] >= 360)):
        raise ValueError('RA must be in [0, 360) degrees.')
    if np.any(np.abs(sources['dec']) > 90):
        raise ValueError('Declination must be in [-90, 90] degrees.')
    return table, sources


def mosaic_query(bands, size_arcsec, processing_mode='DEEP', release_name=None):
    if not set(bands).issubset(BANDS) or processing_mode not in ('DEEP', 'WIDE'):
        raise ValueError('Unsupported band or processing mode.')
    # Circumscribed circle finds candidates; Cutout2D verifies full coverage.
    radius = size_arcsec / np.sqrt(2) / 3600
    instruments = []
    if 'VIS' in bands:
        instruments.append("mosaic.instrument_name = 'VIS'")
    nir = [band for band in bands if band != 'VIS']
    if nir:
        filters = ', '.join("'" + b + "'" for band in nir
                            for b in (band, band.replace('-', '_')))
        instruments.append(f"(mosaic.instrument_name = 'NISP' AND mosaic.filter_name IN ({filters}))")
    release = '' if release_name is None else (
        "\nAND mosaic.release_name = '" + release_name.replace("'", "''") + "'"
    )
    return f"""SELECT src.sample_row, src.object_id,
       mosaic.datalabs_path, mosaic.file_name, mosaic.instrument_name,
       mosaic.filter_name, mosaic.tile_index, mosaic.processing_mode,
       mosaic.release_name, mosaic.purpose,
       mosaic.reference_observation_date_time,
       DISTANCE(src.ra, src.dec, mosaic.ra, mosaic.dec) AS dist_cent
FROM TAP_UPLOAD.sfh_sample AS src
JOIN dr1.mosaic_product AS mosaic
  ON INTERSECTS(CIRCLE(src.ra, src.dec, {radius:.12g}), mosaic.fov)=1
WHERE ({' OR '.join(instruments)})
AND mosaic.processing_mode = '{processing_mode}'
AND mosaic.datalabs_path IS NOT NULL{release}
ORDER BY src.sample_row, dist_cent
"""


def query_mosaics(client, sources, output, query, batch_size=1000):
    """Cache each completed TAP upload/query so interrupted lookups can resume."""
    cache = Path(output) / 'queries'
    cache.mkdir(exist_ok=True)
    (cache / 'mosaics.sql').write_text(query)
    parts = []
    for start in range(0, len(sources), batch_size):
        stop = min(start + batch_size, len(sources))
        path = cache / f'mosaics_{start:06d}_{stop:06d}.ecsv'
        if path.exists():
            result = Table.read(path)
        else:
            if client is None:
                raise ValueError('Archive client required for uncached queries.')
            with tempfile.TemporaryDirectory() as tmp:
                upload = Path(tmp) / 'sample.vot'
                sources[start:stop].write(upload, format='votable')
                job = client.launch_job_async(
                    query, upload_resource=str(upload), upload_table_name='sfh_sample',
                    output_format='votable', verbose=False,
                )
                result = job.get_results()
                if result is None:
                    raise RuntimeError(f'No TAP result for rows {start}:{stop}.')
            temp = path.with_suffix('.tmp')
            result.write(temp, format='ascii.ecsv', overwrite=True)
            temp.replace(path)
        print(f'Mosaic lookup {stop:,}/{len(sources):,}: {len(result):,} matches', flush=True)
        parts.append(result)
    nonempty = [part for part in parts if len(part)]
    result = vstack(nonempty, metadata_conflicts='silent') if nonempty else parts[0]
    result.write(Path(output) / 'mosaic_matches.ecsv', overwrite=True)
    return result


def band_name(row):
    return 'VIS' if text(row['instrument_name']).upper() == 'VIS' else text(row['filter_name']).upper().replace('_', '-')


def mosaic_path(row):
    return Path(text(row['datalabs_path'])) / text(row['file_name'])


def write_manifest(records, output):
    path = Path(output) / 'manifest.csv'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def make_cutouts(sample, sources, matches, output, bands=('VIS',), size_arcsec=10):
    """Try nearest-center mosaics first; group each round by file for bulk I/O."""
    output = Path(output)
    candidates = defaultdict(list)
    seen = set()
    for row in matches:
        index = int(row['sample_row'])
        if not 0 <= index < len(sources) or int(row['object_id']) != int(sources['object_id'][index]):
            raise ValueError('Mosaic query IDs do not match the selected sample.')
        band = band_name(row)
        if band not in bands:
            continue
        key = (index, band)
        signature = (key, str(mosaic_path(row)))
        if signature not in seen:
            seen.add(signature)
            candidates[key].append(row)
    for rows in candidates.values():
        rows.sort(key=lambda row: (float(row['dist_cent']), str(mosaic_path(row))))

    records, record_by_key = [], {}
    for i, src in enumerate(sources):
        for band in bands:
            relative = Path('cutouts') / band / f'{int(src["object_id"])}.fits'
            record = dict(object_id=str(int(src['object_id'])), sample_row=i,
                          ra=float(src['ra']), dec=float(src['dec']), band=band,
                          cutout_file=str(relative), status='pending' if candidates[i, band] else 'no_mosaic',
                          n_candidates=len(candidates[i, band]), mosaic_file='', tile_index='',
                          release_name='', reference_observation_date_time='',
                          size_arcsec=size_arcsec, height=0, width=0, error='')
            record.update({key: text(sample[key][i]) if key in sample.colnames else ''
                           for key in SFH_COLUMNS})
            destination = output / relative
            if destination.exists():
                # A resumed run must not silently accept an unrelated or truncated image.
                with fits.open(destination) as hdul:
                    hdr = hdul[0].header
                    if (text(hdr.get('OBJID')) != record['object_id'] or hdr.get('BAND') != band
                            or hdr.get('CUTSIZE') != size_arcsec
                            or hdul[0].data is None or hdul[0].data.ndim != 2):
                        raise ValueError(f'Existing cutout does not match this request: {destination}')
                    record.update(status='existing', mosaic_file=hdr.get('MOSFILE', ''),
                                  height=hdul[0].data.shape[0], width=hdul[0].data.shape[1],
                                  tile_index=hdr.get('TILEID', ''), release_name=hdr.get('RELEASE', ''),
                                  reference_observation_date_time=hdr.get('REFDATE', ''))
            records.append(record)
            record_by_key[i, band] = record
    write_manifest(records, output)

    rounds = max((len(rows) for rows in candidates.values()), default=0)
    for rank in range(rounds):
        grouped = defaultdict(list)
        for key, rows in candidates.items():
            if rank < len(rows) and record_by_key[key]['status'] not in ('written', 'existing'):
                grouped[mosaic_path(rows[rank])].append((key, rows[rank]))
        for path, jobs in sorted(grouped.items()):
            print(f'Opening {path.name}: {len(jobs)} cutouts (candidate {rank + 1})', flush=True)
            try:
                with fits.open(path, memmap=True) as hdul:
                    data, header = hdul[0].data, hdul[0].header
                    if data is None or data.ndim != 2:
                        raise ValueError('Expected a 2D science image in the primary HDU.')
                    image_wcs = WCS(header).celestial
                    for (i, band), row in jobs:
                        record = record_by_key[i, band]
                        try:
                            src = sources[i]
                            cutout = Cutout2D(
                                data, SkyCoord(src['ra'], src['dec'], unit='deg'),
                                size_arcsec * u.arcsec, wcs=image_wcs, mode='strict', copy=True,
                            )
                            if not np.all(np.isfinite(cutout.data)):
                                raise ValueError('Cutout contains nonfinite pixels.')
                            # Build a fresh WCS header so stale mosaic CD/PC cards cannot survive.
                            hdr = cutout.wcs.to_header(relax=True)
                            for key in ('BUNIT', 'MAGZERO', 'FILTER', 'TELESCOP', 'INSTRUME'):
                                if key in header:
                                    hdr[key] = header[key]
                            hdr['OBJID'] = record['object_id']
                            hdr['BAND'] = band
                            hdr['CUTSIZE'] = (size_arcsec, 'Requested square side in arcsec')
                            hdr['RA_OBJ'], hdr['DEC_OBJ'] = float(src['ra']), float(src['dec'])
                            hdr['MOSFILE'] = str(path)
                            hdr['TILEID'] = text(row['tile_index'])
                            hdr['RELEASE'] = text(row['release_name'])
                            hdr['REFDATE'] = text(row['reference_observation_date_time'])
                            destination = output / record['cutout_file']
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            temporary = destination.with_suffix('.tmp')
                            fits.PrimaryHDU(cutout.data, header=hdr).writeto(temporary, overwrite=True)
                            temporary.replace(destination)
                            record.update(status='written', mosaic_file=str(path), error='',
                                          height=cutout.data.shape[0], width=cutout.data.shape[1],
                                          tile_index=text(row['tile_index']), release_name=text(row['release_name']),
                                          reference_observation_date_time=text(row['reference_observation_date_time']))
                        except (OSError, ValueError, TypeError) as exc:
                            record.update(status='failed', error=f'{path}: {exc}')
            except (OSError, ValueError, TypeError) as exc:
                for key, _ in jobs:
                    if record_by_key[key]['status'] not in ('written', 'existing'):
                        record_by_key[key].update(status='failed', error=f'{path}: {exc}')
            write_manifest(records, output)
    counts = {status: sum(r['status'] == status for r in records)
              for status in ('written', 'existing', 'no_mosaic', 'failed')}
    print(json.dumps(counts, indent=2))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True, help='SFH sample catalog.fits')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bands', nargs='+', choices=BANDS, default=['VIS'])
    parser.add_argument('--size-arcsec', type=float, default=10)
    parser.add_argument('--processing-mode', choices=['DEEP', 'WIDE'], default='DEEP')
    parser.add_argument('--release-name', help='Optional exact mosaic release_name filter')
    parser.add_argument('--batch-size', type=int, default=1000, help='Sources per archive query')
    parser.add_argument('--id-column')
    parser.add_argument('--ra-column')
    parser.add_argument('--dec-column')
    parser.add_argument('--credentials-file', type=Path, help='Optional Astroquery credentials file')
    parser.add_argument('--limit', type=int, help='Use first N sample rows for a smoke run')
    parser.add_argument('--query-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse cached queries and verified cutouts')
    args = parser.parse_args()
    if (not np.isfinite(args.size_arcsec) or args.size_arcsec <= 0 or args.batch_size < 1
            or (args.limit is not None and args.limit < 1)):
        parser.error('Size, batch size, and limit must be positive.')
    bands = list(dict.fromkeys(args.bands))
    sample, sources = load_sample(args.sample, args.id_column, args.ra_column, args.dec_column, args.limit)
    settings = dict(schema_version=1, sample_sha256=hashlib.sha256(args.sample.read_bytes()).hexdigest(),
                    bands=bands, size_arcsec=args.size_arcsec, processing_mode=args.processing_mode,
                    release_name=args.release_name, batch_size=args.batch_size, limit=args.limit,
                    id_column=args.id_column, ra_column=args.ra_column, dec_column=args.dec_column)
    settings_path = args.output / 'run.json'
    if args.output.exists():
        if not args.resume or not settings_path.exists():
            parser.error('Output exists. Use a new directory, or --resume for the same run.')
        if json.loads(settings_path.read_text()) != settings:
            parser.error('Run settings or sample changed; choose a new output directory.')
    else:
        args.output.mkdir(parents=True)
        settings_path.write_text(json.dumps(settings, indent=2))
    query = mosaic_query(bands, args.size_arcsec, args.processing_mode, args.release_name)
    all_cached = all((args.output / 'queries' / f'mosaics_{start:06d}_{min(start + args.batch_size, len(sources)):06d}.ecsv').exists()
                     for start in range(0, len(sources), args.batch_size))
    client = None
    try:
        if not all_cached:
            from astroquery.esa.euclid.core import EuclidClass
            client = EuclidClass(environment='IDR')
            client.ROW_LIMIT = -1
            client.login(**({'credentials_file': str(args.credentials_file.expanduser())}
                            if args.credentials_file else {}))
        matches = query_mosaics(client, sources, args.output, query, args.batch_size)
    finally:
        if client is not None:
            client.logout()
    if args.query_only:
        print(f'Cached mosaic matches in {args.output}; rerun with --resume to create cutouts.')
        return
    records = make_cutouts(sample, sources, matches, args.output, bands, args.size_arcsec)
    print(f'SFH/image mapping: {args.output / "manifest.csv"}')
    if any(r['status'] not in ('written', 'existing') for r in records):
        raise SystemExit('Some requested cutouts are missing; inspect manifest.csv (rerun with --resume).')


if __name__ == '__main__':
    main()
