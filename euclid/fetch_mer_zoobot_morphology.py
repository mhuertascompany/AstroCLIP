"""Fetch MER CAS and ZooBot morphology for an exact Euclid object-ID sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.table import Table

from .fetch_mer_morphology import (
    _column_name,
    _float_values,
    load_ids,
    query_batches,
)


MER_ZOOBOT_COLUMNS = (
    'concentration',
    'concentration_err',
    'asymmetry',
    'asymmetry_err',
    'smoothness',
    'smoothness_err',
    'gini',
    'gini_err',
    'moment_20',
    'moment_20_err',
    'smooth_or_featured_smooth',
    'smooth_or_featured_featured_or_disk',
    'smooth_or_featured_artifact_star_zoom',
    'disk_edge_on_yes',
    'disk_edge_on_no',
    'has_spiral_arms_yes',
    'has_spiral_arms_no',
    'bar_strong',
    'bar_weak',
    'bar_no',
    'merging_none',
    'merging_minor_disturbance',
    'merging_major_disturbance',
    'merging_merger',
    'etg_or_ltg',
    't_type',
    'major_merger',
    'major_merger_uncertainty',
)


DEFAULT_MORPHOLOGY_TABLE = 'catalogue.mer_morphology_deep_survey'


def zoobot_query(table_name=DEFAULT_MORPHOLOGY_TABLE):
    columns = ',\n       '.join(f'morph.{name}' for name in MER_ZOOBOT_COLUMNS)
    return f"""SELECT src.sample_row, src.object_id,
       {columns}
FROM TAP_UPLOAD.sfh_sample AS src
JOIN {table_name} AS morph
  ON morph.object_id = src.object_id
ORDER BY src.sample_row
"""


def validate_results(sources, results):
    row_name = _column_name(results, 'sample_row')
    id_name = _column_name(results, 'object_id')
    rows = np.asarray(results[row_name], dtype=np.int64)
    ids = np.asarray(results[id_name], dtype=np.int64)
    order = np.argsort(rows)
    rows = rows[order]
    ids = ids[order]
    if (
        np.any(rows < 0) or np.any(rows >= len(sources))
        or len(np.unique(rows)) != len(rows)
    ):
        raise ValueError('MER morphology join has duplicate or invalid sample rows.')
    expected = np.asarray(sources['object_id'], dtype=np.int64)
    if not np.array_equal(ids, expected[rows]):
        raise ValueError('MER morphology object IDs do not match the input sample.')
    output = Table({'object_id': expected})
    for column in MER_ZOOBOT_COLUMNS:
        name = _column_name(results, column)
        values = np.full(len(sources), np.nan, dtype=np.float32)
        values[rows] = _float_values(results[name])[order]
        output[column] = values
    smooth_columns = (
        'smooth_or_featured_smooth',
        'smooth_or_featured_featured_or_disk',
        'smooth_or_featured_artifact_star_zoom',
    )
    smooth_total = sum(np.asarray(output[name], dtype=float) for name in smooth_columns)
    n_zoobot = int(np.count_nonzero(np.isfinite(smooth_total) & (smooth_total > 0)))
    print(
        f'MER morphology rows: {len(results):,}/{len(output):,}; '
        f'finite ZooBot predictions: {n_zoobot:,}/{len(output):,}',
        flush=True,
    )
    return output


def fetch_zoobot_morphology(sample, output, client=None, batch_size=1000,
                             query_retries=5, retry_delay=5,
                             table_name=DEFAULT_MORPHOLOGY_TABLE, resume=False):
    sample, output = Path(sample), Path(output)
    if output.exists():
        if resume:
            print(f'Output already exists: {output}')
            return Table.read(output)
        raise FileExistsError(f'Output exists: {output}; use --resume or a new path.')
    sources = load_ids(sample)
    if not len(sources):
        raise ValueError('Input sample is empty.')
    cache_dir = output.parent / f'.{output.stem}_queries'
    settings = {
        'schema_version': 1,
        'sample_sha256': hashlib.sha256(sample.read_bytes()).hexdigest(),
        'table_name': table_name,
        'batch_size': batch_size,
        'columns': list(MER_ZOOBOT_COLUMNS),
    }
    settings_path = cache_dir / 'run.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError(f'Cached query settings differ: {settings_path}')
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2))
    results = query_batches(
        client, sources, cache_dir, zoobot_query(table_name), batch_size,
        query_retries, retry_delay,
    )
    table = validate_results(sources, results)
    temporary = output.with_suffix(output.suffix + '.tmp')
    table.write(temporary, format='fits', overwrite=True)
    temporary.replace(output)
    print(f'MER ZooBot morphology: {output}', flush=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--query-retries', type=int, default=5)
    parser.add_argument('--retry-delay', type=float, default=5)
    parser.add_argument('--table-name', default=DEFAULT_MORPHOLOGY_TABLE)
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
        fetch_zoobot_morphology(
            args.sample, args.output, client, args.batch_size,
            args.query_retries, args.retry_delay, args.table_name, args.resume,
        )
    finally:
        if client is not None:
            client.logout()


if __name__ == '__main__':
    main()
