"""Compare two contrasting held-out SFHs using identical diffusion noise."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .sfh_shape import sfh_recent_activity
from .prepare_diffusion_conditions import sha256
from .train_pixel_diffusion import PixelDiffusion


def select_pair(activity, seed=42):
    rate = activity['sfh_log_recent_sfr_per_formed_mass']
    trend = activity['sfh_recent_trend']
    quiet = np.flatnonzero(np.isfinite(rate) & (rate < -11.5))
    # Exclude the quiet group: positive trend alone can also describe tiny SFRs.
    rising = np.flatnonzero(np.isfinite(rate) & (rate >= -11.5)
                           & np.isfinite(trend) & (trend > .15))
    if not len(quiet) or not len(rising):
        raise ValueError(f'Insufficient candidates: quiet={len(quiet)}, rising={len(rising)}')
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(quiet), rng.choice(rising)]), [len(quiet), len(rising)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'conditions', 'dataset', 'stamps', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-seeds', type=int, default=8)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.n_seeds < 1:
        parser.error('--n-seeds must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    fingerprint = sha256(args.conditions)
    with np.load(args.conditions, allow_pickle=False) as cache:
        ids, rows = cache['val_ids'], cache['val_rows']
        conditions = cache['val_condition']
    order = np.argsort(rows)
    inverse = np.argsort(order)
    with h5py.File(args.dataset, 'r') as source:
        def read(name):
            return np.asarray(source[name][rows[order]])[inverse]
        if not np.array_equal(read('object_id'), ids):
            raise ValueError('Dataset IDs do not match cached validation rows.')
        sfh = read('sfh')
        time = source['sfh_time_grid'][:]
        epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))
        activity = sfh_recent_activity(sfh, time, epsilon, read('sfh_time_norm'))
    indices, counts = select_pair(activity, args.seed)
    model = PixelDiffusion.load_from_checkpoint(str(args.checkpoint), map_location='cpu')
    if model.hparams.cache_sha256 != fingerprint:
        raise ValueError('Checkpoint condition-cache hash mismatch.')
    model.to(args.device).eval().requires_grad_(False)
    args.output.mkdir(parents=True)
    labels = ['Low recent rate', 'Rising recent SFH']
    weights = np.maximum(10. ** sfh[indices].astype(float) - epsilon, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    from PIL import Image
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    records = []
    for j, index in enumerate(indices):
        record = {'object_id': int(ids[index]), 'dataset_row': int(rows[index]),
                  **{key: float(value[index]) for key, value in activity.items()}}
        records.append(record)
        axes[j, 0].plot(time, weights[j])
        axes[j, 0].set(xlabel='Fractional lookback time', ylabel='Normalized SFH weight',
                       title=f'{labels[j]}: {ids[index]}\nlog recent rate={record["sfh_log_recent_sfr_per_formed_mass"]:.2f}, trend={record["sfh_recent_trend"]:.2f}')
        with Image.open(args.stamps / 'VIS' / f'VIS_{int(ids[index])}.jpg') as image:
            axes[j, 1].imshow(image.convert('L'), cmap='gray', vmin=0, vmax=255)
        axes[j, 1].set_title('Observed stamp (not a reconstruction target)')
        axes[j, 1].axis('off')
    fig.tight_layout()
    fig.savefig(args.output / 'selected_sfhs.png', dpi=150)
    plt.close(fig)
    seeds = [args.seed + k for k in range(args.n_seeds)]
    panels = []
    row_labels = []
    for guidance, pair_index in [(1., 0), (1., 1), (2., 0), (2., 1), (0., 0)]:
        condition = torch.as_tensor(conditions[indices[pair_index]:indices[pair_index]+1],
                                    dtype=torch.float32, device=args.device)
        images = []
        for seed in seeds:
            generated = model.schedule.sample(model.ema, condition, steps=args.steps,
                                              guidance=guidance, seed=seed)
            images.append(((generated[0, 0].cpu().numpy() + 1) / 2).clip(0, 1))
        panels.append(images)
        row_labels.append('Unconditional' if guidance == 0 else f'{labels[pair_index]}\nguidance={guidance:g}')
    fig, axes = plt.subplots(5, args.n_seeds, figsize=(2.3 * args.n_seeds, 11), squeeze=False)
    for row in range(5):
        for col, seed in enumerate(seeds):
            ax = axes[row, col]
            ax.imshow(panels[row][col], cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f'Noise seed {seed}')
            if col == 0:
                ax.set_ylabel(row_labels[row])
    fig.tight_layout()
    fig.savefig(args.output / 'sfh_extremes_comparison.png', dpi=150)
    plt.close(fig)
    metadata = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint),
                'condition_cache_sha256': fingerprint, 'split': 'validation',
                'selection_seed': args.seed, 'candidate_counts': counts, 'objects': records,
                'noise_seeds': seeds, 'sampling_steps': args.steps, 'weights': 'EMA',
                'rate_definition': 'log10[SFR averaged over 0-0.1 fractional lookback time / total formed mass], yr^-1; not surviving-mass sSFR',
                'cuts': ['log recent rate < -11.5', 'recent trend > 0.15 AND log recent rate >= -11.5']}
    (args.output / 'selection.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    print(f'Figures: {args.output}', flush=True)


if __name__ == '__main__':
    main()
