"""Plot conditional diffusion SFH draws against held-out ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def stratified_indices(values, n, seed=42):
    """Random examples distributed approximately uniformly in value rank."""
    values = np.asarray(values, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < n:
        raise ValueError(f'Only {len(valid)} finite objects are available for {n} panels.')
    ordered = valid[np.argsort(values[valid])]
    groups = np.array_split(ordered, n)
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(group) for group in groups], dtype=np.int64)


def make_figure(predictive, output, n_examples=12, draws_to_show=10, seed=42,
                logarithmic=False):
    with h5py.File(predictive, 'r') as source:
        condition = np.asarray(source['condition'], dtype=np.float32)
        time = np.asarray(source['sfh_time_grid'], dtype=np.float32)
        selected = stratified_indices(condition[:, 2], n_examples, seed)
        ids = np.array([source['galaxy_id'][i] for i in selected], dtype=np.int64)
        observed = np.stack([source['observed_sfh'][i] for i in selected])
        draws = np.stack([source['sfh_draws'][i] for i in selected])
        p16 = np.stack([source['predictive_p16_sfh'][i] for i in selected])
        p50 = np.stack([source['predictive_p50_sfh'][i] for i in selected])
        p84 = np.stack([source['predictive_p84_sfh'][i] for i in selected])
        selected_condition = condition[selected]
    rng = np.random.default_rng(seed + 1)
    ncols = 4
    nrows = int(np.ceil(n_examples / ncols))
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(15, 3.5 * nrows), sharex=True, squeeze=False,
    )
    for panel, axis in enumerate(axes.flat[:n_examples]):
        available = draws.shape[1]
        shown = rng.choice(available, min(draws_to_show, available), replace=False)
        for draw_index in shown:
            axis.plot(
                time, draws[panel, draw_index], color='tab:orange', alpha=0.22,
                lw=0.8,
            )
        axis.fill_between(
            time, p16[panel], p84[panel], color='tab:orange', alpha=0.22,
            label='predictive 16–84%' if panel == 0 else None,
        )
        axis.plot(
            time, p50[panel], color='tab:red', lw=1.5,
            label='predictive median' if panel == 0 else None,
        )
        axis.plot(
            time, observed[panel], color='black', lw=2.0,
            label='ground-truth median' if panel == 0 else None,
        )
        if logarithmic:
            axis.set_yscale('log')
            axis.set_ylim(bottom=1e-7)
        mass, redshift, log_sfr = selected_condition[panel]
        axis.set_title(
            f'{int(ids[panel])}\n'
            rf'$\log M_\star={mass:.2f}$, $z={redshift:.2f}$, '
            rf'$\log \mathrm{{SFR}}_{{100}}={log_sfr:.2f}$',
            fontsize=9,
        )
        axis.grid(alpha=0.15)
    for axis in axes.flat[n_examples:]:
        axis.set_visible(False)
    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel('Fractional lookback time')
    for axis in axes[:, 0]:
        axis.set_ylabel('Normalized SFH weight')
    axes.flat[0].legend(frameon=False, fontsize=8)
    scale = 'logarithmic' if logarithmic else 'linear'
    figure.suptitle(
        f'Conditional SFH diffusion: held-out ground truth and generated draws ({scale})',
        y=0.997,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(figure)
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-examples', type=int, default=12)
    parser.add_argument('--draws-to-show', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.predictive.is_file():
        raise FileNotFoundError(args.predictive)
    if min(args.n_examples, args.draws_to_show) < 1:
        raise ValueError('Example and displayed-draw counts must be positive.')
    args.output.mkdir(parents=True, exist_ok=True)
    linear = args.output / 'conditional_sfh_draws_linear.pdf'
    logarithmic = args.output / 'conditional_sfh_draws_log.pdf'
    selected = make_figure(
        args.predictive, linear, args.n_examples, args.draws_to_show, args.seed,
        logarithmic=False,
    )
    make_figure(
        args.predictive, logarithmic, args.n_examples, args.draws_to_show, args.seed,
        logarithmic=True,
    )
    np.savetxt(
        args.output / 'conditional_sfh_draw_example_rows.txt', selected, fmt='%d',
        header='zero-based row in validation_predictive_sfhs.h5',
    )
    print(linear)
    print(logarithmic)


if __name__ == '__main__':
    main()
