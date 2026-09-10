"""Sample an EDFN clean photometric catalog after matching available SFH IDs.

Run on candide, from the repository root:
    python -m euclid.sample_edfn_sfhs --output /path/to/edfn_10k

Only object IDs are read while scanning SFH files. Selected SFHs are copied in
batches, preserving all realizations and the native age grid in each file.
The selected photometric rows are saved in catalog.fits, with SFH file/row
references, and object_ids.csv provides a lightweight matching manifest.
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

from .sample_sfh_catalog import sample_catalog


def check_output_directory(output):
    """Allow a new or empty directory; never reuse an existing sample."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f'Output must be a new or empty directory: {output}. '
            'Choose another --output path to preserve the existing contents.'
        )


def match_and_sample(table, sfh_files, output, n=10_000, seed=42,
                     id_column='object_id', batch_size=128):
    """Uniformly sample unique field-catalog galaxies that have available SFHs."""
    from astropy.table import Table

    # open_table helpers may return an Astropy table, a FITS record array,
    # or a pandas DataFrame.
    table = Table.from_pandas(table) if hasattr(table, 'to_records') else Table(table)
    output = Path(output)
    check_output_directory(output)
    if id_column not in table.colnames:
        raise ValueError(f'No {id_column!r} column. Available: {table.colnames}')
    if n <= 0 or batch_size <= 0:
        raise ValueError('n and batch_size must be positive.')

    def id_keys(values):
        # Convert only to text, never floating point: Euclid IDs may be large.
        return [v.decode('utf-8').strip() if isinstance(v, bytes) else str(v).strip()
                for v in values]

    ids = table[id_column]
    if np.any(np.ma.getmaskarray(ids)) or ids.dtype.kind == 'f':
        raise ValueError('Object IDs must be unmasked integers or strings, not floats.')
    catalog_ids = id_keys(ids)
    if len(set(catalog_ids)) != len(catalog_ids):
        raise ValueError('Photometric catalog has duplicate object IDs.')
    wanted = set(catalog_ids)
    matches = {}
    sfh_files = sorted({Path(p).resolve() for p in sfh_files})
    if not sfh_files:
        raise ValueError('No HDF5 SFH files found; supply --sfh-files explicitly.')
    for path in sfh_files:
        with h5py.File(path, 'r') as src:
            if 'object_id' not in src or 'sfh' not in src:
                raise ValueError(f'{path} is not an SFH catalog; use --sfh-files.')
            source_ids = src['object_id'][:]
            if source_ids.ndim != 1 or source_ids.dtype.kind == 'f':
                raise ValueError(f'{path}: expected integer or string object_id vector.')
            found = 0
            for row, gid in enumerate(id_keys(source_ids)):
                if gid not in wanted:
                    continue
                if gid in matches:
                    raise ValueError(
                        f'Multiple SFHs for object_id={gid}: {matches[gid][0]} and {path}. '
                        'Use --sfh-files to choose one consistent set of catalogs.'
                    )
                matches[gid] = (path, row)
                found += 1
            print(f'{path.name}: {found:,} matching field-catalog IDs')

    eligible = np.array([i for i, gid in enumerate(catalog_ids) if gid in matches])
    print(f'Field catalog: {len(table):,}; with SFHs: {len(eligible):,}; '
          f'without SFHs: {len(table) - len(eligible):,}')
    if len(eligible) < n:
        raise ValueError(f'Only {len(eligible):,} matched galaxies; requested {n:,}.')
    chosen = np.sort(np.random.default_rng(seed).choice(eligible, n, replace=False))
    selected = table[chosen].copy()
    reserved = {'sfh_file', 'sfh_row', 'sfh_source_file', 'sfh_source_row', 'field_catalog_row'}
    if reserved.intersection(selected.colnames):
        raise ValueError('Input already contains reserved SFH matching columns.')
    selected_ids = [catalog_ids[i] for i in chosen]
    used_files = sorted({matches[gid][0] for gid in selected_ids})
    filenames = {path: f'sfh_{i:03d}.h5' for i, path in enumerate(used_files)}
    selected['field_catalog_row'] = chosen
    selected['sfh_file'] = [filenames[matches[gid][0]] for gid in selected_ids]
    selected['sfh_source_file'] = [str(matches[gid][0]) for gid in selected_ids]
    selected['sfh_source_row'] = [matches[gid][1] for gid in selected_ids]
    selected['sfh_row'] = np.zeros(n, dtype=np.int64)
    selected.meta['SAMPSEED'] = seed
    selected.meta['NFIELD'] = len(table)
    selected.meta['NMATCH'] = len(eligible)

    check_output_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    for path in used_files:
        positions = np.array([i for i, gid in enumerate(selected_ids)
                              if matches[gid][0] == path])
        source_rows = np.array([matches[selected_ids[i]][1] for i in positions])
        indices = sample_catalog(path, output / filenames[path], seed=seed,
                                 batch_size=batch_size, indices=source_rows)
        selected['sfh_row'][positions] = np.searchsorted(indices, source_rows)
    selected.write(output / 'catalog.fits')
    with (output / 'object_ids.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([id_column, 'sfh_file', 'sfh_row', 'sfh_source_file', 'sfh_source_row'])
        for i, gid in enumerate(selected_ids):
            writer.writerow([gid] + [selected[key][i] for key in
                            ('sfh_file', 'sfh_row', 'sfh_source_file', 'sfh_source_row')])
    print(f'Saved {n:,} matched galaxies to {output}/catalog.fits (all catalog columns).')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--utils-dir', type=Path,
                        default=Path('/home/wozny/jobs/These/DR1_science/SFH/utils'))
    parser.add_argument('--field', default='EDFN')
    parser.add_argument('--phot-type', default='2fwhm_aper')
    parser.add_argument('--sfh-dir', type=Path,
                        default=Path('/n17data/wozny/These/science_DR1/SFHs/ready_to_use_sfhs'))
    parser.add_argument('--sfh-files', type=Path, nargs='+',
                        help='Explicit files, overriding the directory scan.')
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory.')
    parser.add_argument('--n', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--id-column', default='object_id')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    sys.path.insert(0, str(args.utils_dir))
    from get_paths import get_file_cat
    from load_table import open_table

    cat_path = get_file_cat(field=args.field, cat_name='clean_photo_phz',
                            phot_type=args.phot_type)
    print(f'Loading {args.field} photometric catalog: {cat_path}')
    table = open_table(cat_path)
    files = args.sfh_files
    if files is None:
        files = [p for p in args.sfh_dir.rglob('*')
                 if p.is_file() and p.suffix.lower() in {'.h5', '.hdf5', '.hdf'}]
    match_and_sample(table, files, args.output, args.n, args.seed,
                     args.id_column, args.batch_size)


if __name__ == '__main__':
    main()
