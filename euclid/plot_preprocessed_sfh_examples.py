"""Compare native Euclid posterior SFHs with their CLIP preprocessing."""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .plot_sfh_examples import representative_indices


def plot_preprocessed_examples(source_path, processed_path, output_path,
                               n_redshift=3, n_mass=3):
    """Plot representative native posterior SFHs and preprocessed curves."""
    source_path = Path(source_path)
    processed_path = Path(processed_path)
    output_path = Path(output_path)

    with h5py.File(source_path, 'r') as source, h5py.File(processed_path, 'r') as processed:
        source_required = ('age', 'object_id', 'sfh')
        processed_required = (
            'object_id', 'sfh', 'sfh_time_grid', 'sfh_time_norm',
            'phz_pp_median_redshift', 'phz_pp_median_stellarmass',
        )
        missing_source = [key for key in source_required if key not in source]
        missing_processed = [key for key in processed_required if key not in processed]
        if missing_source or missing_processed:
            raise ValueError(
                f'Missing source datasets {missing_source}; '
                f'missing processed datasets {missing_processed}'
            )

        source_ids = source['object_id'][:]
        processed_ids = processed['object_id'][:]
        if len(np.unique(source_ids)) != len(source_ids):
            raise ValueError('Source object IDs are not unique.')
        source_rows = {int(object_id): row for row, object_id in enumerate(source_ids)}
        unmatched = [int(object_id) for object_id in processed_ids
                     if int(object_id) not in source_rows]
        if unmatched:
            raise ValueError(f'{len(unmatched)} processed IDs are absent from the source.')

        redshift = processed['phz_pp_median_redshift'][:].astype(float)
        mass = processed['phz_pp_median_stellarmass'][:].astype(float)
        sersic = (processed['sersic_sersic_vis_index'][:].astype(float)
                  if 'sersic_sersic_vis_index' in processed
                  else np.full(len(processed_ids), np.nan))
        selected = representative_indices(redshift, mass, n_redshift, n_mass)
        time_grid = processed['sfh_time_grid'][:].astype(float)
        epsilon = float(processed.attrs.get('sfh_log_epsilon', 1e-10))
        age_myr = source['age'][:].astype(float)

        examples = []
        for row in selected:
            source_row = source_rows[int(processed_ids[row])]
            posterior = source['sfh'][source_row].astype(float)
            native_p16, native_median, native_p84 = np.percentile(
                posterior, [16, 50, 84], axis=0,
            )
            median_total = native_median.sum()
            if not np.isfinite(median_total) or median_total <= 0:
                raise ValueError(f'Invalid native median SFH for row {source_row}.')
            native_p16 /= median_total
            native_median /= median_total
            native_p84 /= median_total
            universe_age_myr = float(processed['sfh_time_norm'][row])
            fractional_native_time = age_myr / universe_age_myr
            native_bin_width = float(np.median(np.diff(age_myr))) / universe_age_myr
            valid = fractional_native_time <= 1.0
            preprocessed = np.maximum(
                10.0 ** processed['sfh'][row].astype(float) - epsilon, 0.0,
            )
            processed_bin_width = float(np.median(np.diff(time_grid)))
            examples.append((
                fractional_native_time[valid], native_p16[valid] / native_bin_width,
                native_median[valid] / native_bin_width,
                native_p84[valid] / native_bin_width,
                preprocessed / processed_bin_width,
            ))

    fig, axes = plt.subplots(
        n_redshift, n_mass, figsize=(13, 10), sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(n_redshift, n_mass)
    colors = plt.cm.viridis(np.linspace(0.18, 0.82, n_redshift))

    for panel, (ax, row, example) in enumerate(zip(axes.flat, selected, examples)):
        native_time, native_p16, native_median, native_p84, preprocessed = example
        color = colors[panel // n_mass]
        ax.fill_between(
            native_time, native_p16, native_p84,
            color='0.55', alpha=0.22, linewidth=0,
            label='Native 16–84%' if panel == 0 else None,
        )
        ax.plot(
            native_time, native_median, color='0.42', linewidth=1.0,
            alpha=0.85, label='Native median' if panel == 0 else None,
        )
        ax.plot(
            time_grid, preprocessed, color=color, linewidth=2.0,
            label='Preprocessed 250-bin SFH' if panel == 0 else None,
        )
        sersic_label = f'{sersic[row]:.2f}' if np.isfinite(sersic[row]) else 'NA'
        ax.set_title(
            f'ID {int(processed_ids[row])}\n'
            f'$z={redshift[row]:.3f}$  '
            f'$\\log M_\\star={mass[row]:.2f}$  '
            f'$n={sersic_label}$',
            fontsize=10,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18, linewidth=0.6)

    for ax in axes[-1, :]:
        ax.set_xlabel('Fractional lookback time')
    for ax in axes[:, 0]:
        ax.set_ylabel('Normalized SFH density')

    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        'Euclid EDFN SFHs after CLIP preprocessing\n'
        'Columns span stellar mass within three equal-count redshift bins',
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')
    print('Selected processed rows:', ', '.join(map(str, selected)))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Original sampled Euclid SFH HDF5 file.')
    parser.add_argument('--processed', type=Path, required=True,
                        help='CLIP-preprocessed SFH HDF5 file.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-redshift', type=int, default=3)
    parser.add_argument('--n-mass', type=int, default=3)
    args = parser.parse_args()
    if args.n_redshift < 1 or args.n_mass < 1:
        parser.error('Grid dimensions must be positive.')
    plot_preprocessed_examples(
        args.source, args.processed, args.output,
        args.n_redshift, args.n_mass,
    )


if __name__ == '__main__':
    main()
