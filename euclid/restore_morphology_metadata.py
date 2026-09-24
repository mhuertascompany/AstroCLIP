"""Join missing Euclid morphology columns into an existing CLIP HDF5 file.

The current 100k SFH product was preprocessed before the clean-catalog and MER
morphology FITS tables were merged into the SFH shards.  This command performs
an exact OBJECT_ID join and adds only scalar metadata datasets; SFHs, row order,
embeddings, and train/validation splits remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table
from astropy.units import UnitsWarning

from .fetch_mer_zoobot_morphology import MER_ZOOBOT_COLUMNS


DEFAULT_COLUMNS = (
    'sersic_sersic_vis_axis_ratio',
    'sersic_sersic_vis_axis_ratio_err',
    'sersic_sersic_vis_radius',
    'sersic_sersic_vis_radius_err',
    'sersic_sersic_vis_index',
    'sersic_sersic_vis_index_err',
    'fwhm',
    'kron_radius',
    'kron_radius_err',
    'semimajor_axis',
    'point_like_flag',
    'point_like_prob',
    'segmentation_area',
    'ellipticity',
) + MER_ZOOBOT_COLUMNS


def _column_name(table, requested):
    names = {name.lower(): name for name in table.colnames}
    return names.get(requested.lower())


def _integer_ids(column, label):
    values = np.ma.asarray(column)
    if np.any(np.ma.getmaskarray(values)):
        raise ValueError(f'{label} contains masked object IDs.')
    if values.dtype.kind == 'f':
        raise ValueError(f'{label} object IDs are floating point, not exact integers.')
    try:
        ids = np.array([
            int(value.decode().strip() if isinstance(value, bytes) else value)
            for value in values
        ], dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{label} object IDs are not exact int64 values.') from error
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'{label} contains duplicate object IDs.')
    return ids


def _float_column(column):
    values = np.ma.asarray(column, dtype=np.float64)
    return np.asarray(values.filled(np.nan), dtype=np.float32)


def align_catalog(table, target_ids, columns=DEFAULT_COLUMNS,
                  id_column='object_id', allow_missing=False, label='catalog'):
    """Return requested numeric columns aligned exactly to target HDF5 IDs."""
    id_name = _column_name(table, id_column)
    if id_name is None:
        raise ValueError(f'{label} has no {id_column!r}; available: {table.colnames}')
    source_ids = _integer_ids(table[id_name], label)
    order = np.argsort(source_ids)
    sorted_ids = source_ids[order]
    locations = np.searchsorted(sorted_ids, target_ids)
    matched = locations < len(sorted_ids)
    matched[matched] &= sorted_ids[locations[matched]] == target_ids[matched]
    if not np.all(matched) and not allow_missing:
        missing = target_ids[~matched]
        preview = ', '.join(map(str, missing[:5]))
        raise ValueError(
            f'{label} matches {np.count_nonzero(matched):,}/{len(target_ids):,} '
            f'HDF5 objects; first missing IDs: {preview}'
        )

    aligned = {}
    source_names = {}
    for requested in columns:
        name = _column_name(table, requested)
        if name is None:
            continue
        values = _float_column(table[name])
        if values.ndim != 1 or len(values) != len(source_ids):
            raise ValueError(f'{label} column {name!r} is not scalar and row-aligned.')
        output = np.full(len(target_ids), np.nan, dtype=np.float32)
        output[matched] = values[order[locations[matched]]]
        aligned[requested.lower()] = output
        source_names[requested.lower()] = name
    return aligned, source_names, int(np.count_nonzero(matched))


def read_catalog(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnitsWarning)
        return Table.read(path)


def restore_metadata(dataset, catalogs, columns=DEFAULT_COLUMNS,
                     id_column='object_id', allow_missing=False, replace=False,
                     dry_run=False, preserve_unmatched=False):
    """Add ID-aligned FITS metadata datasets without touching SFH arrays."""
    dataset = Path(dataset)
    catalogs = [Path(path) for path in catalogs]
    if not dataset.is_file():
        raise FileNotFoundError(f'CLIP HDF5 file not found: {dataset}')
    for path in catalogs:
        if not path.is_file():
            raise FileNotFoundError(f'Metadata catalog not found: {path}')
    if not catalogs:
        raise ValueError('At least one metadata catalog is required.')

    with h5py.File(dataset, 'r') as source:
        if 'galaxy_id' not in source:
            raise ValueError(f'{dataset} has no galaxy_id dataset.')
        target_ids = np.asarray(source['galaxy_id'][:], dtype=np.int64)
        if len(np.unique(target_ids)) != len(target_ids):
            raise ValueError('CLIP HDF5 galaxy_id contains duplicates.')
        existing = set(source.keys())

    merged = {}
    provenance = {}
    for catalog_path in catalogs:
        table = read_catalog(catalog_path)
        aligned, source_names, n_matched = align_catalog(
            table, target_ids, columns, id_column, allow_missing,
            label=str(catalog_path),
        )
        print(
            f'{catalog_path}: {n_matched:,}/{len(target_ids):,} IDs; '
            f'columns={sorted(aligned)}',
            flush=True,
        )
        for key, values in aligned.items():
            if key in merged:
                print(f'{key}: using later catalog {catalog_path}', flush=True)
            merged[key] = values
            provenance[key] = {
                'catalog': str(catalog_path.resolve()),
                'source_column': source_names[key],
            }
    if not merged:
        raise ValueError(
            f'None of the requested columns were found: {list(columns)}'
        )

    pending = {
        key: values for key, values in merged.items()
        if replace or key not in existing
    }
    if preserve_unmatched:
        if not replace:
            raise ValueError('--preserve-unmatched requires --replace.')
        with h5py.File(dataset, 'r') as source:
            for key, values in pending.items():
                if key not in source:
                    continue
                previous = np.asarray(source[key][:], dtype=np.float32)
                missing = ~np.isfinite(values)
                values[missing] = previous[missing]
    retained = sorted(set(merged).difference(pending))
    print(f'Existing datasets retained: {retained}', flush=True)
    print(f'Datasets to add: {sorted(pending)}', flush=True)
    if dry_run or not pending:
        return sorted(pending)

    prefix = '__metadata_patch__'
    with h5py.File(dataset, 'r+') as target:
        stale = [key for key in target if key.startswith(prefix)]
        if stale:
            print(f'Removing stale temporary datasets from an interrupted run: {stale}',
                  flush=True)
            for key in stale:
                del target[key]
        created = []
        try:
            for key, values in pending.items():
                temporary = prefix + key
                output = target.create_dataset(
                    temporary,
                    data=values,
                    chunks=True,
                    compression='gzip',
                    compression_opts=1,
                )
                output.attrs['description'] = 'object_id-aligned morphology metadata'
                output.attrs['source_catalog'] = provenance[key]['catalog']
                output.attrs['source_column'] = provenance[key]['source_column']
                created.append(temporary)
            target.flush()
            for key in pending:
                temporary = prefix + key
                if key in target:
                    if not replace:
                        raise FileExistsError(f'Dataset appeared during patch: {key}')
                    del target[key]
                target.move(temporary, key)
            target.attrs['morphology_metadata_sources'] = json.dumps(
                [str(path.resolve()) for path in catalogs]
            )
            target.attrs['morphology_metadata_columns'] = json.dumps(sorted(merged))
            target.attrs['morphology_metadata_updated_utc'] = (
                datetime.now(timezone.utc).isoformat()
            )
            target.flush()
        except Exception:
            for temporary in created:
                if temporary in target:
                    del target[temporary]
            raise

    print(f'Updated morphology metadata in place: {dataset}', flush=True)
    return sorted(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, action='append', required=True,
                        help='FITS metadata catalog; repeatable, with later values winning.')
    parser.add_argument('--columns', nargs='+', default=list(DEFAULT_COLUMNS))
    parser.add_argument('--id-column', default='object_id')
    parser.add_argument('--allow-missing', action='store_true')
    parser.add_argument('--replace', action='store_true',
                        help='Replace existing metadata datasets after staging new values.')
    parser.add_argument('--preserve-unmatched', action='store_true',
                        help='With --replace, retain existing values where the new catalog is missing.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    restore_metadata(
        args.dataset, args.catalog, args.columns, args.id_column,
        args.allow_missing, args.replace, args.dry_run, args.preserve_unmatched,
    )


if __name__ == '__main__':
    main()
