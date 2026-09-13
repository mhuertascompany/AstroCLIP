"""Fetch exact MER morphology inputs for an SFH sample on ESA Datalabs.

The clean photo-z catalog used to select the SFH sample omits
SEGMENTATION_AREA and ELLIPTICITY. This module joins selected OBJECT_IDs to
catalogue.mer_catalogue_deep_survey through a TAP upload and writes the three
inputs required by morphology_utils.estimate_source_r_max.
"""

import argparse
import hashlib
import json
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from astropy.units import UnitsWarning


MORPHOLOGY_COLUMNS = ('segmentation_area', 'kron_radius', 'ellipticity')


def _text(value):
    return value.decode().strip() if isinstance(value, bytes) else str(value).strip()


def _column_name(table, name):
    names = {column.lower(): column for column in table.colnames}
    if name.lower() not in names:
        raise ValueError(f'Cannot find {name!r}; available columns: {table.colnames}')
    return names[name.lower()]


def _float_values(column):
    return np.asarray(np.ma.asarray(column, dtype=float).filled(np.nan), dtype=float)


def load_ids(sample, id_column='object_id'):
    """Load exact unique int64 IDs and their original sample row."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnitsWarning)
        table = Table.read(sample)
    name = _column_name(table, id_column)
    values = table[name]
    if values.dtype.kind not in 'iuSU' or np.any(np.ma.getmaskarray(values)):
        raise ValueError('object_id must contain unmasked integers or integer strings.')
    ids = np.array([int(_text(value)) for value in values], dtype=np.int64)
    if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError('object_id must be nonnegative and unique.')
    return Table({'sample_row': np.arange(len(ids), dtype=np.int64), 'object_id': ids})


def morphology_query(table_name='catalogue.mer_catalogue_deep_survey'):
    """ADQL query for the exact inputs used by the Euclid R_MAX regression."""
    return f"""SELECT src.sample_row, src.object_id,
       mer.segmentation_area, mer.kron_radius, mer.ellipticity,
       mer.semimajor_axis
FROM TAP_UPLOAD.sfh_sample AS src
JOIN {table_name} AS mer
  ON mer.object_id = src.object_id
ORDER BY src.sample_row
"""


def query_batches(client, sources, cache_dir, query, batch_size=1000,
                  query_retries=5, retry_delay=5):
    """Run cached TAP-upload joins and return their ordered concatenation."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / 'morphology.sql').write_text(query)
    parts = []
    for start in range(0, len(sources), batch_size):
        stop = min(start + batch_size, len(sources))
        path = cache_dir / f'morphology_{start:06d}_{stop:06d}.ecsv'
        if path.exists():
            result = Table.read(path)
        else:
            if client is None:
                raise ValueError(f'Archive client required for uncached batch {start}:{stop}.')
            with tempfile.TemporaryDirectory() as directory:
                upload = Path(directory) / 'sample.vot'
                sources[start:stop].write(upload, format='votable')
                for attempt in range(1, query_retries + 1):
                    try:
                        job = client.launch_job_async(
                            query,
                            upload_resource=str(upload),
                            upload_table_name='sfh_sample',
                            output_format='votable',
                            verbose=False,
                        )
                        if job is None:
                            raise RuntimeError('Euclid TAP returned no job.')
                        result = job.get_results()
                        if result is None:
                            raise RuntimeError('Euclid TAP job returned no result.')
                        break
                    except Exception as exc:
                        if attempt == query_retries:
                            raise RuntimeError(
                                f'MER morphology query failed for rows {start}:{stop} '
                                f'after {query_retries} attempts.'
                            ) from exc
                        delay = min(retry_delay * 2 ** (attempt - 1), 30)
                        print(
                            f'Morphology query failed for rows {start}:{stop} '
                            f'(attempt {attempt}/{query_retries}): {exc}; '
                            f'retrying in {delay:g} s.',
                            flush=True,
                        )
                        time.sleep(delay)
            temporary = path.with_suffix('.tmp')
            result.write(temporary, format='ascii.ecsv', overwrite=True)
            temporary.replace(path)
        print(f'MER metadata {stop:,}/{len(sources):,}: {len(result):,} matches',
              flush=True)
        parts.append(result)
    nonempty = [part for part in parts if len(part)]
    return vstack(nonempty, metadata_conflicts='silent') if nonempty else Table()


def validate_results(sources, results):
    """Require a complete one-to-one OBJECT_ID join and normalize column names."""
    required = ('sample_row', 'object_id') + MORPHOLOGY_COLUMNS
    names = {name: _column_name(results, name) for name in required}
    rows = np.asarray(results[names['sample_row']], dtype=np.int64)
    ids = np.asarray(results[names['object_id']], dtype=np.int64)
    if len(results) != len(sources):
        matched = set(map(int, ids))
        missing = [int(value) for value in sources['object_id'] if int(value) not in matched]
        preview = ', '.join(map(str, missing[:5]))
        raise ValueError(
            f'MER join returned {len(results):,}/{len(sources):,} objects; '
            f'first missing IDs: {preview}'
        )
    order = np.argsort(rows)
    rows, ids = rows[order], ids[order]
    if not np.array_equal(rows, np.arange(len(sources))):
        raise ValueError('MER join contains duplicate or invalid sample_row values.')
    if not np.array_equal(ids, np.asarray(sources['object_id'], dtype=np.int64)):
        raise ValueError('MER join object IDs do not match the SFH sample.')

    output = Table({'object_id': ids})
    for column in MORPHOLOGY_COLUMNS:
        values = _float_values(results[names[column]])[order]
        output[column] = values
    if any(name.lower() == 'semimajor_axis' for name in results.colnames):
        name = _column_name(results, 'semimajor_axis')
        output['semimajor_axis'] = _float_values(results[name])[order]
    valid = (
        np.isfinite(output['segmentation_area']) & (output['segmentation_area'] > 0)
        & np.isfinite(output['kron_radius']) & (output['kron_radius'] > 0)
        & np.isfinite(output['ellipticity'])
    )
    print(f'Valid R_MAX inputs: {np.count_nonzero(valid):,}/{len(output):,}', flush=True)
    return output


def fetch_morphology(sample, output, client=None, batch_size=1000,
                     query_retries=5, retry_delay=5,
                     table_name='catalogue.mer_catalogue_deep_survey',
                     resume=False):
    """Fetch and validate morphology metadata, using cached batches on restart."""
    sample, output = Path(sample), Path(output)
    if output.exists():
        if resume:
            print(f'Output already exists: {output}')
            return Table.read(output)
        raise FileExistsError(f'Output exists: {output}; use --resume or a new path.')
    sources = load_ids(sample)
    if not len(sources):
        raise ValueError('The SFH sample is empty.')
    if batch_size < 1 or query_retries < 1 or retry_delay < 0:
        raise ValueError('Batch size and retries must be positive; delay cannot be negative.')

    cache_dir = output.parent / f'.{output.stem}_queries'
    settings = {
        'schema_version': 1,
        'sample_sha256': hashlib.sha256(sample.read_bytes()).hexdigest(),
        'table_name': table_name,
        'batch_size': batch_size,
    }
    settings_path = cache_dir / 'run.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError(f'Cached query settings differ: {settings_path}')
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2))

    results = query_batches(
        client, sources, cache_dir, morphology_query(table_name),
        batch_size, query_retries, retry_delay
    )
    metadata = validate_results(sources, results)
    temporary = output.with_suffix(output.suffix + '.tmp')
    metadata.write(temporary, format='fits', overwrite=True)
    temporary.replace(output)
    print(f'Exact MER morphology metadata: {output}')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--query-retries', type=int, default=5)
    parser.add_argument('--retry-delay', type=float, default=5)
    parser.add_argument('--table-name', default='catalogue.mer_catalogue_deep_survey')
    parser.add_argument('--credentials-file', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    sources = load_ids(args.sample)
    cache_dir = args.output.parent / f'.{args.output.stem}_queries'
    all_cached = all(
        (cache_dir / f'morphology_{start:06d}_{min(start + args.batch_size, len(sources)):06d}.ecsv').exists()
        for start in range(0, len(sources), args.batch_size)
    )
    client = None
    try:
        if not all_cached and not args.output.exists():
            from astroquery.esa.euclid.core import EuclidClass
            client = EuclidClass(environment='IDR')
            client.ROW_LIMIT = -1
            login_args = (
                {'credentials_file': str(args.credentials_file.expanduser())}
                if args.credentials_file else {}
            )
            client.login(**login_args)
        fetch_morphology(
            args.sample, args.output, client, args.batch_size,
            args.query_retries, args.retry_delay, args.table_name, args.resume,
        )
    finally:
        if client is not None:
            client.logout()


if __name__ == '__main__':
    main()
