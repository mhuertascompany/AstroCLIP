"""Sample galaxies from one field-specific SFHCatalog HDF5 file.

Select the EDFN input file explicitly: the schema documented in sfh_catalog.py
contains neither a field label nor sky coordinates. No sky selection is inferred.
All SFH realizations and per-galaxy datasets are retained. The shared ``age``
grid is copied unchanged. A companion CSV contains source_index, object_id,
and all other scalar per-galaxy columns (including coordinates, if present).

Run from the repository root:
    python -m euclid.sample_sfh_catalog --catalog /path/to/EDFN.h5 \
        --output /path/to/edfn_10k.h5 --n 10000 --seed 42
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def sample_catalog(catalog, output, n=10_000, seed=42, batch_size=128, indices=None):
    """Sample without replacement; sort selected rows for efficient HDF5 reads.

    Input must follow SFHCatalog's flat layout: age is a shared time grid,
    and every other non-scalar dataset has galaxy as its first dimension.
    No redshift, mass, morphology, or SFH quality cuts are applied. Explicit
    indices can be supplied by a caller that sampled a matched field catalog.
    """
    catalog, output = Path(catalog), Path(output)
    csv_path = output.with_suffix('.csv')
    if output.suffix.lower() not in {'.h5', '.hdf5'}:
        raise ValueError('Output must end in .h5 or .hdf5.')
    if n <= 0 or batch_size <= 0:
        raise ValueError('n and batch_size must be positive.')
    for path in (output, csv_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')

    with h5py.File(catalog, 'r') as src:
        required = ('object_id', 'sfh', 'age', 'phz_pp_median_redshift',
                    'phz_pp_median_stellarmass', 'sersic_sersic_vis_index')
        for key in required:
            if key not in src:
                raise ValueError(f'Missing required dataset: {key}')
        ids = src['object_id'][:]
        if ids.ndim != 1:
            raise ValueError('object_id must be one-dimensional.')
        total = len(ids)
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            if indices.ndim != 1 or not len(indices):
                raise ValueError('indices must be a nonempty one-dimensional array.')
            if np.any(indices < 0) or np.any(indices >= total):
                raise ValueError('indices are outside the source catalog.')
            if len(np.unique(indices)) != len(indices):
                raise ValueError('indices must not contain duplicates.')
            n = len(indices)
        if n > total:
            raise ValueError(f'Requested {n:,} galaxies, but input has {total:,}.')
        if len(np.unique(ids)) != total:
            raise ValueError('Input has duplicate object_id values; resolve these first.')
        if 'source_index' in src:
            raise ValueError('Input already has source_index; use the original catalog.')

        # Validate the documented layout before writing any output.
        row_keys, shared_keys = [], []
        for key, ds in src.items():
            if not isinstance(ds, h5py.Dataset):
                raise ValueError(f'Expected a flat catalog; found group {key}.')
            if key == 'age' or ds.ndim == 0:
                shared_keys.append(key)
            elif ds.shape[0] == total:
                row_keys.append(key)
            else:
                raise ValueError(f'Unexpected shape for {key}: {ds.shape}.')
        if src['age'].ndim != 1 or src['sfh'].ndim != 3:
            raise ValueError('Expected age=(time,) and sfh=(galaxy, realization, time).')
        if src['sfh'].shape[-1] != len(src['age']):
            raise ValueError('SFH time dimension does not match the age grid.')

        explicit_indices = indices is not None
        if indices is None:
            indices = np.random.default_rng(seed).choice(total, n, replace=False)
        indices = np.sort(indices)
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, 'x') as dst:
            dst.attrs.update(src.attrs)
            if 'n_galaxies' in dst.attrs:
                dst.attrs['n_galaxies'] = n
            dst.attrs['sample_source'] = str(catalog.resolve())
            dst.attrs['sample_seed'] = seed
            dst.attrs['sample_selection'] = 'explicit_indices' if explicit_indices else 'random'
            dst.attrs['sample_size'] = n
            dst.attrs['sample_population_size'] = total
            dst.create_dataset('source_index', data=indices)
            for key in shared_keys:
                src.copy(key, dst)
            for key in row_keys:
                ds = src[key]
                sampled = dst.create_dataset(
                    key, shape=(n,) + ds.shape[1:], dtype=ds.dtype,
                    chunks=True, compression='gzip', compression_opts=1,
                )
                sampled.attrs.update(ds.attrs)
                for start in range(0, n, batch_size):
                    stop = min(start + batch_size, n)
                    sampled[start:stop] = ds[indices[start:stop]]

        scalar_keys = ['object_id'] + sorted(
            key for key in row_keys if key != 'object_id' and src[key].ndim == 1
        )
        with h5py.File(output, 'r') as sampled, csv_path.open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source_index'] + scalar_keys)
            columns = [sampled[key][:] for key in scalar_keys]
            for i, row in enumerate(zip(*columns)):
                writer.writerow([int(indices[i])] + [
                    value.decode('utf-8') if isinstance(value, bytes) else value
                    for value in row
                ])

    print(f'Selected {n:,} / {total:,} galaxies (seed={seed}).')
    print(f'SFH subset: {output}')
    print(f'Image-matching table: {csv_path}')
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True,
                        help='HDF5 input containing only the desired field (EDFN).')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    sample_catalog(args.catalog, args.output, args.n, args.seed, args.batch_size)


if __name__ == '__main__':
    main()
