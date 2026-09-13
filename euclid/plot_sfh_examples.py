"""Plot representative SFHs from a Euclid SFH HDF5 catalog."""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def representative_indices(redshift, mass, n_redshift=3, n_mass=3):
    """Select mass quantiles within equal-count redshift bins."""
    valid = np.flatnonzero(np.isfinite(redshift) & np.isfinite(mass))
    if len(valid) < n_redshift * n_mass:
        raise ValueError('Not enough finite redshift and mass measurements.')

    ordered = valid[np.argsort(redshift[valid], kind='stable')]
    redshift_groups = np.array_split(ordered, n_redshift)
    selected = []
    quantiles = np.linspace(0.2, 0.8, n_mass)
    for group in redshift_groups:
        mass_order = group[np.argsort(mass[group], kind='stable')]
        positions = np.rint(quantiles * (len(mass_order) - 1)).astype(int)
        selected.extend(mass_order[positions])
    return np.asarray(selected, dtype=int)


def plot_examples(catalog, output, n_redshift=3, n_mass=3):
    catalog = Path(catalog)
    output = Path(output)
    with h5py.File(catalog, 'r') as source:
        required = ('age', 'object_id', 'sfh', 'phz_pp_median_redshift',
                    'phz_pp_median_stellarmass', 'sersic_sersic_vis_index')
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f'Missing datasets: {missing}')

        age_gyr = source['age'][:].astype(float) / 1000
        if len(age_gyr) < 2 or not np.all(np.diff(age_gyr) > 0):
            raise ValueError('Expected an increasing age grid with at least two bins.')
        bin_width_gyr = float(np.median(np.diff(age_gyr)))
        ids = source['object_id'][:]
        redshift = source['phz_pp_median_redshift'][:].astype(float)
        mass = source['phz_pp_median_stellarmass'][:].astype(float)
        sersic = source['sersic_sersic_vis_index'][:].astype(float)
        indices = representative_indices(redshift, mass, n_redshift, n_mass)
        histories = [source['sfh'][int(index)].astype(float) for index in indices]

    nrows, ncols = n_redshift, n_mass
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 10), sharex=True,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    row_colors = plt.cm.viridis(np.linspace(0.18, 0.82, nrows))

    for panel, (ax, index, sfh) in enumerate(zip(axes.flat, indices, histories)):
        # Each realization is a mass-fraction distribution. Dividing by the
        # bin width expresses it as a normalized formation-rate density.
        rate = sfh / bin_width_gyr
        p16, median, p84 = np.percentile(rate, [16, 50, 84], axis=0)
        color = row_colors[panel // ncols]
        ax.fill_between(age_gyr, p16, p84, color=color, alpha=0.25,
                        linewidth=0)
        ax.plot(age_gyr, median, color=color, linewidth=1.8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_title(
            f'ID {int(ids[index])}\n'
            f'$z={redshift[index]:.3f}$  '
            f'$\\log M_\\star={mass[index]:.2f}$  '
            f'$n={sersic[index]:.2f}$',
            fontsize=10,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel('Lookback time [Gyr]')
    for ax in axes[:, 0]:
        ax.set_ylabel('Normalized star formation [Gyr$^{-1}$]')

    fig.suptitle(
        'Representative Euclid EDFN star-formation histories\n'
        'Columns span stellar mass within three equal-count redshift bins',
        fontsize=15,
    )
    fig.text(
        0.5, 0.002,
        'Line: median; shaded region: 16–84% across 50 SFH realizations. '
        'Each realization integrates to unity.',
        ha='center', fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output}')
    print('Selected HDF5 rows:', ', '.join(map(str, indices)))
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-redshift', type=int, default=3)
    parser.add_argument('--n-mass', type=int, default=3)
    args = parser.parse_args()
    if args.n_redshift < 1 or args.n_mass < 1:
        parser.error('Grid dimensions must be positive.')
    plot_examples(args.catalog, args.output, args.n_redshift, args.n_mass)


if __name__ == '__main__':
    main()
