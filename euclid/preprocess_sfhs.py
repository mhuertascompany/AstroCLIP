"""Convert Euclid posterior SFHs to the common CLIP training representation.

The Euclid catalogs contain 50 posterior realizations on a shared physical
lookback-time grid.  For each galaxy this module:

1. rebins every posterior realization onto a uniform grid in fractional
   lookback time,
   ``t_frac = t_lookback / age_of_universe(z)``;
2. uses cumulative mass fractions so narrow bursts cannot be missed, then
   normalizes every realization to sum to one;
3. stores all realizations plus their median and 16th/84th percentiles as
   ``log10(weight + 1e-10)``.

By default the output grid has the same number of bins as the native Euclid
grid (250 in the DR1 catalogs), retaining its temporal resolution. The output
has the same numerical representation consumed by the COSMOS-Web SFH encoder,
whose input dimension must be set to the chosen number of bins. Scalar catalog
columns are copied so rows can subsequently be joined to cutouts by object ID.
"""

import argparse
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM


SFH_EPS = 1e-10
COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)


def sfh_to_common_grid(age_myr, realizations, redshift, time_grid=None,
                       eps=SFH_EPS):
    """Return the posterior-median SFH on a normalized fractional-time grid.

    Parameters
    ----------
    age_myr : array-like, shape (time,)
        Increasing native lookback-time bin centers in Myr.
    realizations : array-like, shape (realization, time)
        Posterior SFH realizations. Each realization is independently
        normalized after interpolation.
    redshift : float
        Galaxy redshift used to compute the age of the Universe.
    time_grid : array-like
        Common fractional lookback-time grid in [0, 1].
    eps : float
        Positive floor applied before taking log10.

    Returns
    -------
    sfh_log : numpy.ndarray
        ``log10`` normalized SFH weights on ``time_grid``.
    universe_age_myr : float
        Age of the Universe at ``redshift`` in Myr.
    """
    realization_log, universe_age_myr, retained_mass = sfh_realizations_to_common_grid(
        age_myr, realizations, redshift, time_grid, eps, return_diagnostics=True,
    )
    valid = retained_mass > eps
    if not np.any(valid):
        raise ValueError('No SFH realization has mass within the allowed time range.')
    realization_weights = np.maximum(10.0 ** realization_log[valid] - eps, 0.0)
    median = np.median(realization_weights, axis=0)
    median /= median.sum()
    return np.log10(median + eps).astype(np.float32), universe_age_myr


def sfh_realizations_to_common_grid(age_myr, realizations, redshift,
                                    time_grid=None, eps=SFH_EPS,
                                    return_diagnostics=False):
    """Preprocess every posterior realization for one Euclid galaxy.

    Returns an array with shape ``(realization, len(time_grid))`` in log10
    normalized-weight space and the age of the Universe in Myr. Realizations
    with no mass inside the physical time range are filled with NaN. If
    ``return_diagnostics`` is true, also return the fraction of native mass
    inside that range for every realization.
    """
    age_myr = np.asarray(age_myr, dtype=np.float64)
    realizations = np.asarray(realizations, dtype=np.float64)
    if time_grid is None:
        time_grid = np.linspace(0.0, 1.0, len(age_myr), dtype=np.float64)
    else:
        time_grid = np.asarray(time_grid, dtype=np.float64)

    if age_myr.ndim != 1 or len(age_myr) < 2:
        raise ValueError('age must be a one-dimensional grid with at least two bins.')
    if not np.all(np.isfinite(age_myr)) or not np.all(np.diff(age_myr) > 0):
        raise ValueError('age must be finite and strictly increasing.')
    if realizations.ndim != 2 or realizations.shape[1] != len(age_myr):
        raise ValueError('realizations must have shape (realization, len(age)).')
    if not np.all(np.isfinite(realizations)):
        raise ValueError('SFH realizations contain nonfinite values.')
    if np.any(realizations < 0):
        raise ValueError('SFH realizations contain negative values.')
    if not (np.isfinite(redshift) and redshift > 0):
        raise ValueError('redshift must be finite and positive.')
    if (time_grid.ndim != 1 or len(time_grid) < 2 or
            not np.all(np.isfinite(time_grid)) or
            not np.all(np.diff(time_grid) > 0) or
            time_grid[0] < 0 or time_grid[-1] > 1):
        raise ValueError('time_grid must be an increasing one-dimensional grid in [0, 1].')
    if not (np.isfinite(eps) and eps > 0):
        raise ValueError('eps must be finite and positive.')

    native_totals = realizations.sum(axis=1)
    native_valid = native_totals > 0
    normalized_native = np.zeros_like(realizations)
    normalized_native[native_valid] = (
        realizations[native_valid] / native_totals[native_valid, None]
    )

    universe_age_myr = float(COSMOLOGY.age(redshift).to_value('Myr'))

    # SFH entries are integrated mass fractions in bins, rather than point
    # samples. Rebin their cumulative distributions so a narrow, single-bin
    # burst cannot fall between output sample locations and disappear.
    native_edges = np.empty(len(age_myr) + 1, dtype=np.float64)
    native_edges[1:-1] = 0.5 * (age_myr[:-1] + age_myr[1:])
    native_edges[0] = max(0.0, age_myr[0] - 0.5 * (age_myr[1] - age_myr[0]))
    native_edges[-1] = age_myr[-1] + 0.5 * (age_myr[-1] - age_myr[-2])

    fractional_edges = np.empty(len(time_grid) + 1, dtype=np.float64)
    fractional_edges[1:-1] = 0.5 * (time_grid[:-1] + time_grid[1:])
    fractional_edges[0] = 0.0
    fractional_edges[-1] = 1.0
    target_edges = fractional_edges * universe_age_myr

    native_cdf = np.concatenate([
        np.zeros((len(realizations), 1), dtype=np.float64),
        np.cumsum(normalized_native, axis=1),
    ], axis=1)
    target_cdf = np.empty(
        (len(realizations), len(target_edges)), dtype=np.float64,
    )
    below = target_edges <= native_edges[0]
    above = target_edges >= native_edges[-1]
    inside = ~(below | above)
    target_cdf[:, below] = 0.0
    target_cdf[:, above] = native_cdf[:, -1, None]
    if np.any(inside):
        target = target_edges[inside]
        right = np.searchsorted(native_edges, target, side='right')
        left = right - 1
        fraction = (
            (target - native_edges[left]) /
            (native_edges[right] - native_edges[left])
        )
        target_cdf[:, inside] = (
            native_cdf[:, left] * (1.0 - fraction)[None, :]
            + native_cdf[:, right] * fraction[None, :]
        )

    rebinned = np.maximum(np.diff(target_cdf, axis=1), 0.0)
    retained_mass = rebinned.sum(axis=1)
    if np.any(~np.isfinite(retained_mass)):
        raise ValueError('A rebinned SFH has nonfinite total weight.')
    valid = retained_mass > eps
    normalized = np.full_like(rebinned, np.nan)
    normalized[valid] = rebinned[valid] / retained_mass[valid, None]
    result = np.log10(normalized + eps).astype(np.float32)
    if return_diagnostics:
        return result, universe_age_myr, retained_mass.astype(np.float32)
    return result, universe_age_myr


def _validate_input(source):
    required = ('age', 'object_id', 'sfh', 'phz_pp_median_redshift')
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f'Missing required datasets: {missing}')

    age = source['age'][:]
    sfh = source['sfh']
    redshift = source['phz_pp_median_redshift']
    object_id = source['object_id']
    if age.ndim != 1 or len(age) < 2 or not np.all(np.diff(age) > 0):
        raise ValueError('age must be a strictly increasing one-dimensional grid.')
    if sfh.ndim != 3 or sfh.shape[-1] != len(age):
        raise ValueError('sfh must have shape (galaxy, realization, len(age)).')
    if redshift.shape != (sfh.shape[0],) or object_id.shape != (sfh.shape[0],):
        raise ValueError('object_id and redshift must have one value per SFH.')
    if len(np.unique(object_id[:])) != len(object_id):
        raise ValueError('object_id contains duplicates.')
    return age.astype(np.float64), sfh.shape[0]


def preprocess_catalog(input_path, output_path, n_bins=None,
                       batch_size=256, eps=SFH_EPS):
    """Preprocess an Euclid SFH catalog into a CLIP-ready HDF5 file."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Input and output paths must differ.')
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite {output_path}')
    if n_bins is not None and n_bins < 2:
        raise ValueError('n_bins must be at least 2.')
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f'.{output_path.name}.', suffix='.tmp',
            dir=output_path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)

        with h5py.File(input_path, 'r') as source, h5py.File(temporary, 'w') as target:
            age_myr, n_galaxies = _validate_input(source)
            output_n_bins = len(age_myr) if n_bins is None else n_bins
            time_grid = np.linspace(0.0, 1.0, output_n_bins, dtype=np.float64)
            for key, value in source.attrs.items():
                target.attrs[key] = value
            target.attrs['n_galaxies'] = n_galaxies
            target.attrs['source_catalog'] = str(input_path.resolve())
            target.attrs['sfh_preprocessing'] = (
                'realization-wise mass-conserving rebinning; sum_normalized; log10'
            )
            target.attrs['sfh_posterior_reduction'] = 'median of processed realizations'
            target.attrs['sfh_normalization'] = 'sum_to_one_after_interpolation'
            target.attrs['sfh_uncertainty'] = 'all realizations plus p16 and p84'
            target.attrs['sfh_log_epsilon'] = eps
            target.attrs['sfh_n_bins'] = output_n_bins
            target.attrs['sfh_time_coordinate'] = 'lookback_time / age_of_universe_at_redshift'
            target.attrs['cosmology'] = 'FlatLambdaCDM(H0=70 km/s/Mpc, Om0=0.3)'

            target.create_dataset('sfh_time_grid', data=time_grid.astype(np.float32))
            target['sfh_time_grid'].attrs['description'] = 'fractional lookback time'
            target.create_dataset('sfh_native_age_myr', data=age_myr.astype(np.float32))

            output_sfh = target.create_dataset(
                'sfh', shape=(n_galaxies, output_n_bins), dtype='f4',
                chunks=(min(batch_size, max(n_galaxies, 1)), output_n_bins),
                compression='gzip', compression_opts=1,
            )
            output_sfh.attrs['description'] = 'log10(sum-normalized SFH weight + epsilon)'
            n_realizations = source['sfh'].shape[1]
            output_realizations = target.create_dataset(
                'sfh_realizations',
                shape=(n_galaxies, n_realizations, output_n_bins), dtype='f4',
                chunks=(1, n_realizations, output_n_bins),
                compression='gzip', compression_opts=1,
            )
            output_realizations.attrs['description'] = (
                'valid posterior realizations as log10(sum-normalized SFH weight + epsilon); '
                'invalid realizations are NaN'
            )
            output_valid = target.create_dataset(
                'sfh_realization_valid',
                shape=(n_galaxies, n_realizations), dtype='bool',
                chunks=(min(batch_size, max(n_galaxies, 1)), n_realizations),
                compression='gzip', compression_opts=1,
            )
            output_valid.attrs['description'] = (
                'true when the realization has mass within the age of the Universe'
            )
            output_retained = target.create_dataset(
                'sfh_retained_mass_fraction',
                shape=(n_galaxies, n_realizations), dtype='f4',
                chunks=(min(batch_size, max(n_galaxies, 1)), n_realizations),
                compression='gzip', compression_opts=1,
            )
            output_retained.attrs['description'] = (
                'native SFH mass fraction inside the physical time range before renormalization'
            )
            output_p16 = target.create_dataset(
                'sfh_p16', shape=(n_galaxies, output_n_bins), dtype='f4',
                chunks=(min(batch_size, max(n_galaxies, 1)), output_n_bins),
                compression='gzip', compression_opts=1,
            )
            output_p84 = target.create_dataset(
                'sfh_p84', shape=(n_galaxies, output_n_bins), dtype='f4',
                chunks=(min(batch_size, max(n_galaxies, 1)), output_n_bins),
                compression='gzip', compression_opts=1,
            )
            output_p16.attrs['description'] = 'posterior 16th percentile in log10 weight space'
            output_p84.attrs['description'] = 'posterior 84th percentile in log10 weight space'
            output_age = target.create_dataset(
                'sfh_time_norm', shape=(n_galaxies,), dtype='f4',
                chunks=True, compression='gzip', compression_opts=1,
            )
            output_age.attrs['units'] = 'Myr'
            output_age.attrs['description'] = 'age of Universe at galaxy redshift'

            # Preserve all scalar, row-aligned metadata. Higher-dimensional
            # posterior products are intentionally left in the source catalog.
            copied = {}
            for key, dataset in source.items():
                if key in {'age', 'sfh'} or dataset.ndim != 1 or dataset.shape[0] != n_galaxies:
                    continue
                copied[key] = target.create_dataset(
                    key, shape=dataset.shape, dtype=dataset.dtype,
                    chunks=True, compression='gzip', compression_opts=1,
                )
                for attr_key, attr_value in dataset.attrs.items():
                    copied[key].attrs[attr_key] = attr_value

            # COSMOS-Web-style aliases make the product straightforward to use
            # with later paired-dataset code while retaining Euclid names.
            galaxy_id = target.create_dataset(
                'galaxy_id', shape=(n_galaxies,), dtype=source['object_id'].dtype,
                chunks=True, compression='gzip', compression_opts=1,
            )
            redshift_out = target.create_dataset(
                'redshift', shape=(n_galaxies,), dtype='f4', chunks=True,
                compression='gzip', compression_opts=1,
            )

            invalid_realizations = 0
            for start in range(0, n_galaxies, batch_size):
                stop = min(start + batch_size, n_galaxies)
                sfh_batch = source['sfh'][start:stop]
                z_batch = source['phz_pp_median_redshift'][start:stop]
                for offset, (realizations, redshift) in enumerate(zip(sfh_batch, z_batch)):
                    realization_log, universe_age, retained_mass = (
                        sfh_realizations_to_common_grid(
                            age_myr, realizations, float(redshift), time_grid,
                            eps, return_diagnostics=True,
                        )
                    )
                    valid = retained_mass > eps
                    if not np.any(valid):
                        raise ValueError(
                            f'No valid SFH realizations for input row {start + offset}.'
                        )
                    invalid_realizations += int(np.count_nonzero(~valid))
                    linear = np.maximum(10.0 ** realization_log[valid] - eps, 0.0)
                    p16, median, p84 = np.percentile(linear, [16, 50, 84], axis=0)
                    median /= median.sum()
                    row = start + offset
                    output_sfh[row] = np.log10(median + eps)
                    output_realizations[row] = realization_log
                    output_valid[row] = valid
                    output_retained[row] = retained_mass
                    output_p16[row] = np.log10(p16 + eps)
                    output_p84[row] = np.log10(p84 + eps)
                    output_age[row] = universe_age

                for key, dataset in copied.items():
                    dataset[start:stop] = source[key][start:stop]
                galaxy_id[start:stop] = source['object_id'][start:stop]
                redshift_out[start:stop] = z_batch
                print(f'Processed {stop:,}/{n_galaxies:,}', flush=True)

            target.attrs['sfh_invalid_realizations'] = invalid_realizations
            target.attrs['sfh_total_realizations'] = n_galaxies * n_realizations

        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    print(f'CLIP-ready SFHs: {output_path}')
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Sampled Euclid SFH HDF5 catalog.')
    parser.add_argument('--output', type=Path, required=True,
                        help='New CLIP-ready HDF5 file.')
    parser.add_argument(
        '--n-bins', type=int,
        help='Output bins; defaults to the native input resolution (250 for Euclid DR1).',
    )
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=SFH_EPS)
    args = parser.parse_args()
    preprocess_catalog(args.input, args.output, args.n_bins,
                       args.batch_size, args.epsilon)


if __name__ == '__main__':
    main()
