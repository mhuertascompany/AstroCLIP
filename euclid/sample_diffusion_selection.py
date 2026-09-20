"""Generate VIS stamps for an explorer ID selection, preserving CSV row order."""
import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .prepare_diffusion_conditions import sha256
from .train_pixel_diffusion import PixelDiffusion


def selected_conditions(selection, cache_path):
    with Path(selection).open(newline='') as stream:
        reader = csv.DictReader(stream)
        key = next((k for k in ('galaxy_id', 'object_id') if k in (reader.fieldnames or [])), None)
        if key is None:
            raise ValueError('Selection CSV needs galaxy_id or object_id.')
        # Parse directly as integers: float conversion corrupts Euclid IDs.
        ids = [int(row[key]) for row in reader]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Selection must contain nonempty, unique integer IDs.')
    with np.load(cache_path, allow_pickle=False) as cache:
        lookup = {}
        for split in ('train', 'val'):
            for i, gid in enumerate(cache[split + '_ids']):
                if int(gid) in lookup:
                    raise ValueError('Duplicate ID in condition cache.')
                lookup[int(gid)] = (split, i)
        missing = [gid for gid in ids if gid not in lookup]
        if missing:
            raise ValueError(f'{len(missing)} IDs absent from diffusion condition cache: {missing[:10]}')
        rows, conditions, splits = [], [], []
        for gid in ids:
            split, index = lookup[gid]
            rows.append(int(cache[split + '_rows'][index]))
            conditions.append(cache[split + '_condition'][index])
            splits.append(split)
    conditions = np.asarray(conditions, dtype=np.float32)
    if (conditions.ndim != 2 or not np.isfinite(conditions).all()
            or not np.allclose(np.linalg.norm(conditions, axis=1), 1, atol=1e-4)):
        raise ValueError('Invalid normalized conditions.')
    return np.asarray(ids, dtype=np.int64), np.asarray(rows), conditions, splits


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    ids, rows, conditions, splits = selected_conditions(args.selection, args.conditions)
    order = np.argsort(rows)
    with h5py.File(args.dataset, 'r') as source:
        def read(key):
            return np.asarray(source[key][rows[order]])[np.argsort(order)]
        key = 'object_id' if 'object_id' in source else 'galaxy_id'
        if not np.array_equal(read(key), ids):
            raise ValueError('Cached dataset rows do not match selected IDs.')
        sfh = read('sfh')
        time = source['sfh_time_grid'][:]
        eps = float(source.attrs.get('sfh_log_epsilon', 1e-10))
        posterior = [read(k) if k in source else None for k in ('sfh_p16', 'sfh_p84')]
    real = []
    for gid in ids:
        with Image.open(args.stamps / 'VIS' / f'VIS_{int(gid)}.jpg') as image:
            if image.size != (224, 224):
                raise ValueError(f'Unexpected stamp dimensions for {gid}')
            real.append(np.asarray(image.convert('L')))
    # Freeze provenance even if a running trainer later replaces last.ckpt.
    with tempfile.TemporaryDirectory(prefix='euclid_diffusion_') as temporary:
        snapshot = Path(temporary) / 'model.ckpt'
        shutil.copyfile(args.checkpoint, snapshot)
        checkpoint_digest = sha256(snapshot)
        model = PixelDiffusion.load_from_checkpoint(str(snapshot), map_location='cpu')
    fingerprint = sha256(args.conditions)
    if model.hparams.cache_sha256 != fingerprint:
        raise ValueError('Checkpoint was trained with a different condition cache.')
    model.to(args.device).eval().requires_grad_(False)
    if not 2 <= args.steps <= len(model.schedule.alpha):
        raise ValueError('Sampling steps outside diffusion schedule.')
    args.output.mkdir(parents=True)
    seeds = list(range(args.seed, args.seed + args.n_seeds))
    weights = np.maximum(10.**sfh.astype(float) - eps, 0.)
    weights /= weights.sum(axis=1, keepdims=True)
    manifest = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_digest,
                    selection=str(args.selection), selection_sha256=sha256(args.selection),
                    condition_cache_sha256=fingerprint, noise_seeds=seeds,
                    guidance=args.guidance, steps=args.steps, weights='EMA',
                    note='CSV order preserved. Same initial noise per seed for every galaxy and guidance; generated samples are not reconstructions.',
                    objects=[dict(position=i+1, object_id=int(gid), dataset_row=int(rows[i]), split=splits[i])
                             for i, gid in enumerate(ids)])
    (args.output / 'selection.json').write_text(json.dumps(manifest, indent=2))
    for guidance in args.guidance:
        destination = args.output / f'guidance_{guidance:g}'
        destination.mkdir()
        generated = []
        for i, gid in enumerate(ids):
            condition = torch.as_tensor(conditions[i:i+1], device=args.device)
            images = []
            for seed in seeds:
                # Batch size one deliberately resets the same noise for each SFH.
                result = model.schedule.sample(model.ema, condition, steps=args.steps,
                                               guidance=guidance, seed=seed)
                pixels = ((result[0, 0].cpu().numpy().clip(-1, 1)+1)*127.5).round().astype('uint8')
                Image.fromarray(pixels).save(destination / f'{i+1:03d}_{int(gid)}_seed{seed}.png')
                images.append(pixels)
            generated.append(images)
            print(f'Guidance {guidance:g}: {i+1}/{len(ids)} ({gid})', flush=True)
        for start in range(0, len(ids), 6):
            stop = min(start+6, len(ids))
            fig, axes = plt.subplots(2+len(seeds), stop-start,
                                      figsize=(2.6*(stop-start), 2.2*(2+len(seeds))), squeeze=False)
            for col, i in enumerate(range(start, stop)):
                axes[0,col].imshow(real[i], cmap='gray', vmin=0, vmax=255)
                axes[0,col].set_title(f'{i+1}. {ids[i]}\n{splits[i]}', fontsize=7)
                axes[1,col].plot(time, weights[i], linewidth=1)
                if all(p is not None for p in posterior):
                    axes[1,col].fill_between(time, np.maximum(10.**posterior[0][i]-eps,0),
                                            np.maximum(10.**posterior[1][i]-eps,0), alpha=.2)
                axes[1,col].set(xlim=(0,1), ylim=(0,None), xlabel='Fractional lookback')
                axes[1,col].tick_params(labelsize=7)
                for row, pixels in enumerate(generated[i], 2):
                    axes[row,col].imshow(pixels,cmap='gray',vmin=0,vmax=255)
                for row in [0]+list(range(2,2+len(seeds))):
                    axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            for row,label in enumerate(['Observed','Normalized SFH']+[f'Generated\nseed {seed}' for seed in seeds]):
                axes[row,0].set_ylabel(label,fontsize=9)
            fig.suptitle(f'SFH conditions along selected line — guidance {guidance:g}', fontsize=12)
            fig.tight_layout()
            fig.savefig(destination / f'line_comparison_{start//6+1:02d}.png',dpi=150)
            plt.close(fig)
    print(f'Outputs: {args.output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection','checkpoint','conditions','dataset','stamps','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--n-seeds',type=int,default=4)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--guidance',type=float,nargs='+',default=[1.,2.])
    args = parser.parse_args()
    if args.n_seeds<1 or not all(np.isfinite(g) and g>=0 for g in args.guidance):
        parser.error('Require positive n-seeds and finite nonnegative guidance.')
    if len(set(args.guidance)) != len(args.guidance):
        parser.error('Guidance values must be unique.')
    run(args)


if __name__ == '__main__':
    main()
