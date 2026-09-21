"""Fetch PHZ physical parameters for an exact ID sample on ESA Datalabs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.table import Table, MaskedColumn

from .fetch_mer_morphology import load_ids, query_batches, _column_name

TABLE = 'catalogue.phz_physical_parameters_deep_survey'
COLUMNS = ('phys_param_flags', 'quality_flag', 'galaxyclass', 'sfhtype', 'imf') + tuple(
    f'phz_pp_{stat}_{quantity}'
    for quantity in ('redshift', 'sfr', 'stellarmass', 'massformed')
    for stat in ('median', 'mode', '68')
)


def physical_query(columns=COLUMNS):
    columns = ', '.join(f'phys.{name}' for name in columns)
    return f'''SELECT src.sample_row, src.object_id, {columns}
FROM TAP_UPLOAD.sfh_sample AS src
JOIN {TABLE} AS phys ON phys.object_id = src.object_id
ORDER BY src.sample_row
'''


def align_results(sources, results, columns=COLUMNS):
    """Preserve sample order, integer flags, vector intervals and missing masks."""
    if not len(results):
        raise ValueError('No physical-parameter matches; check the table and sample IDs.')
    rows_col = results[_column_name(results, 'sample_row')]
    ids_col = results[_column_name(results, 'object_id')]
    if any(c.dtype.kind not in 'iu' or np.any(np.ma.getmaskarray(c))
           for c in (rows_col, ids_col)):
        raise ValueError('Join keys must be unmasked integers.')
    rows = np.asarray(rows_col, dtype=np.int64)
    ids = np.asarray(ids_col, dtype=np.int64)
    if (np.any(rows < 0) or np.any(rows >= len(sources))
            or len(np.unique(rows)) != len(rows)):
        raise ValueError('Duplicate or invalid sample rows in physical-parameter join.')
    if not np.array_equal(ids, np.asarray(sources['object_id'])[rows]):
        raise ValueError('Physical-parameter IDs do not match sample rows.')
    output = Table({'object_id': sources['object_id']})
    matched = np.zeros(len(sources), dtype=bool)
    matched[rows] = True
    output['physical_parameters_matched'] = matched
    for name in columns:
        source = results[_column_name(results, name)]
        shape = (len(sources),) + source.shape[1:]
        column = MaskedColumn(np.zeros(shape, dtype=source.dtype),
                              mask=np.ones(shape, dtype=bool), name=name)
        column[rows] = source
        column.unit = source.unit
        column.description = source.description
        output.add_column(column)
    output.meta['source_table'] = TABLE
    output.meta['sfr_convention'] = 'log10(Msun/yr); values copied without conversion'
    output.meta['mass_convention'] = 'log10(Msun); values copied without conversion'
    print(f'Physical parameters: {matched.sum():,}/{len(sources):,} matched', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--credentials-file', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--columns', nargs='+', choices=COLUMNS, default=list(COLUMNS),
                        help='Physical fields to retrieve; object_id is always included.')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.output.exists() and not args.resume:
        parser.error('Output exists; use --resume or a new output path')
    sources = load_ids(args.sample)
    if not len(sources):
        parser.error('Empty sample')
    cache = args.output.parent / f'.{args.output.stem}_queries'
    settings = dict(schema_version=1, table=TABLE, columns=list(args.columns),
                    ids_sha256=hashlib.sha256(np.asarray(sources['object_id'],
                        dtype='<i8').tobytes()).hexdigest(), batch_size=args.batch_size)
    manifest = cache / 'run.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != settings:
            parser.error('Cached sample or settings differ; use a new output path')
    else:
        if args.output.exists():
            parser.error('Cannot verify existing output without its query manifest')
        cache.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(settings, indent=2))
    if args.output.exists():
        print(f'Output already exists: {args.output}')
        return
    all_cached = all((cache / f'morphology_{start:06d}_{min(start + args.batch_size, len(sources)):06d}.ecsv').exists()
                     for start in range(0, len(sources), args.batch_size))
    client = None
    try:
        if not all_cached:
            from astroquery.esa.euclid.core import EuclidClass
            client = EuclidClass(environment='IDR')
            client.ROW_LIMIT = -1
            kwargs = ({'credentials_file': str(args.credentials_file.expanduser())}
                      if args.credentials_file else {})
            client.login(**kwargs)
        results = query_batches(client, sources, cache, physical_query(args.columns), args.batch_size)
        output = align_results(sources, results, args.columns)
        temp = args.output.with_suffix(args.output.suffix + '.tmp')
        output.write(temp, format='fits', overwrite=True)
        temp.replace(args.output)
        print(f'Saved {args.output}')
    finally:
        if client is not None:
            client.logout()


if __name__ == '__main__':
    main()
