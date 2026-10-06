"""Generate conditional SFH draws and residual histories for held-out galaxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .sfh_conditional_diffusion import log_sfh_to_weights, weights_to_clr
from .train_conditional_sfh_diffusion import (
    BASE_CONDITION_COLUMNS,
    CONDITION_LABELS,
    PHZ_SFR_COLUMN,
    ConditionalSFHDiffusion,
    _read_rows,
    load_conditions,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True,
                        help='conditional_sfh_split.npz produced during training.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition', choices=('train', 'val', 'all'), default='val')
    parser.add_argument('--draws', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of distinct galaxies sampled per GPU batch.')
    parser.add_argument('--sample-steps', type=int, default=100)
    parser.add_argument('--guidance', type=float, default=1.0,
                        help='Use 1 for calibrated conditional sampling.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.checkpoint, args.dataset, args.split):
        if not path.is_file():
            raise FileNotFoundError(path)
    if min(args.draws, args.batch_size, args.sample_steps) < 1 or args.limit < 0:
        raise ValueError('Draws, batch size, and steps must be positive; limit nonnegative.')
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else 'cpu' if args.device == 'auto' else args.device
    )
    model = ConditionalSFHDiffusion.load_from_checkpoint(
        str(args.checkpoint), map_location='cpu',
    ).to(device).eval()
    sfr_source = str(getattr(model.hparams, 'sfr_source', 'phz'))
    with np.load(args.split) as split:
        if args.partition == 'all':
            rows = np.concatenate([split['train_rows'], split['val_rows']]).astype(np.int64)
            ids = np.concatenate([split['train_ids'], split['val_ids']]).astype(np.int64)
        else:
            rows = np.asarray(split[f'{args.partition}_rows'], dtype=np.int64)
            ids = np.asarray(split[f'{args.partition}_ids'], dtype=np.int64)
    if args.limit:
        rows, ids = rows[:args.limit], ids[:args.limit]
    conditions, valid = load_conditions(args.dataset, sfr_source)
    if not np.all(valid[rows]):
        raise ValueError('Sampling split contains invalid physical conditions.')
    with h5py.File(args.dataset, 'r') as source:
        source_ids = _read_rows(source['galaxy_id'], rows).astype(np.int64)
        observed_log = _read_rows(source['sfh'], rows).astype(np.float32)
        time_grid = np.asarray(source['sfh_time_grid'], dtype=np.float32)
        epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))
    if not np.array_equal(source_ids, ids):
        raise ValueError('Sampling rows do not match saved galaxy IDs.')
    observed = log_sfh_to_weights(observed_log, epsilon)
    observed_clr = weights_to_clr(observed)
    raw_condition = conditions[rows]
    n, n_bins = observed.shape
    args.output.parent.mkdir(parents=True, exist_ok=True)
    maximum_error = 0.0
    with h5py.File(args.output, 'w') as target:
        target.create_dataset('galaxy_id', data=ids)
        target.create_dataset('source_h5_row', data=rows)
        target.create_dataset('sfh_time_grid', data=time_grid)
        target.create_dataset('condition', data=raw_condition)
        target.create_dataset('observed_sfh', data=observed, compression='gzip')
        draws_ds = target.create_dataset(
            'sfh_draws', shape=(n, args.draws, n_bins), dtype=np.float32,
            chunks=(1, args.draws, n_bins), compression='gzip', compression_opts=4,
        )
        mean_ds = target.create_dataset('predictive_mean_sfh', shape=(n, n_bins), dtype=np.float32)
        p16_ds = target.create_dataset('predictive_p16_sfh', shape=(n, n_bins), dtype=np.float32)
        p50_ds = target.create_dataset('predictive_p50_sfh', shape=(n, n_bins), dtype=np.float32)
        p84_ds = target.create_dataset('predictive_p84_sfh', shape=(n, n_bins), dtype=np.float32)
        residual_ds = target.create_dataset('residual_sfh', shape=(n, n_bins), dtype=np.float32)
        clr_residual_ds = target.create_dataset(
            'residual_clr', shape=(n, n_bins), dtype=np.float32,
        )
        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            condition = torch.from_numpy(raw_condition[start:stop]).to(device)
            expanded = condition.repeat_interleave(args.draws, dim=0)
            with torch.inference_mode():
                generated, generated_clr = model.model.sample_weights(
                    expanded, steps=args.sample_steps, guidance=args.guidance,
                    seed=args.seed + start * 1000003,
                )
            generated = generated.reshape(stop - start, args.draws, n_bins).cpu().numpy()
            generated_clr = generated_clr.reshape(
                stop - start, args.draws, n_bins,
            ).cpu().numpy()
            sums = generated.sum(axis=-1)
            maximum_error = max(maximum_error, float(np.max(np.abs(sums - 1))))
            mean = generated.mean(axis=1)
            percentiles = np.percentile(generated, [16, 50, 84], axis=1)
            draws_ds[start:stop] = generated
            mean_ds[start:stop] = mean
            p16_ds[start:stop] = percentiles[0]
            p50_ds[start:stop] = percentiles[1]
            p84_ds[start:stop] = percentiles[2]
            residual_ds[start:stop] = observed[start:stop] - mean
            clr_residual_ds[start:stop] = (
                observed_clr[start:stop] - generated_clr.mean(axis=1)
            )
            print(f'Sampled {stop:,}/{n:,} galaxies', flush=True)
        target.attrs['checkpoint'] = str(args.checkpoint.resolve())
        target.attrs['source_dataset'] = str(args.dataset.resolve())
        target.attrs['partition'] = args.partition
        target.attrs['n_draws'] = args.draws
        target.attrs['ddim_steps'] = args.sample_steps
        target.attrs['guidance'] = args.guidance
        target.attrs['seed'] = args.seed
        target.attrs['condition_columns'] = json.dumps(
            (*BASE_CONDITION_COLUMNS, PHZ_SFR_COLUMN)
            if sfr_source == 'phz'
            else (*BASE_CONDITION_COLUMNS, 'SFH-derived SFR100; R=0')
        )
        target.attrs['condition_labels'] = json.dumps(CONDITION_LABELS)
        target.attrs['sfr_source'] = sfr_source
        target.attrs['sfh_normalization'] = 'nonnegative discrete bin weights summing to one'
        target.attrs['residual_definition'] = 'observed normalized SFH - conditional predictive mean'
        target.attrs['clr_residual_definition'] = 'observed CLR - mean generated CLR'
        target.attrs['maximum_absolute_draw_sum_error'] = maximum_error
    report = {
        'output': str(args.output), 'n_galaxies': n, 'n_draws': args.draws,
        'maximum_absolute_draw_sum_error': maximum_error,
        'note': (
            f'These are posterior-predictive draws from p(SFH | catalog mass, '
            f'photo-z, {sfr_source} SFR100), not repeats of the original SED posterior.'
        ),
    }
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
