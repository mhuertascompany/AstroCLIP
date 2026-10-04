"""Regenerate a valid MAE preview from a trained GalaxyTikTok tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .pretrain_galactiktok_vis import (
    GalaxyTikTokVISPretrainer,
    VISFitsDataset,
)
from .training_index import build_pair_index


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--fits-root', type=Path, required=True)
    parser.add_argument('--image-stats', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--band', default='VIS')
    parser.add_argument('--tokenizer-band', default='euclid-vis')
    parser.add_argument('--asinh-scale', type=float, default=20.0)
    parser.add_argument('--target-gaussian-sigma', type=float, default=0.0)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    from galactiktok import ImageTransformerTokenizer
    from galactiktok.models.image_transformer import Image
    from galactiktok.models.image_transformer.modeling_image_transformer import (
        patchify,
        unpatchify,
    )

    tokenizer = ImageTransformerTokenizer.from_pretrained(
        str(args.tokenizer),
    ).eval()
    index = build_pair_index(
        args.dataset,
        args.fits_root,
        band=args.band,
        val_fraction=args.val_fraction,
        seed=args.seed,
        image_format='fits',
    )
    dataset = VISFitsDataset(
        args.fits_root,
        index.val_ids,
        args.band,
        tokenizer.image_size,
        False,
        args.image_stats,
        args.asinh_scale,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(args.samples, len(dataset)),
        shuffle=False,
        num_workers=0,
    )
    observed, _ = next(iter(loader))
    target = GalaxyTikTokVISPretrainer._gaussian_target(
        observed, args.target_gaussian_sigma,
    )
    torch.manual_seed(args.seed)
    with torch.no_grad():
        tokens = tokenizer.encode(
            Image(flux=observed, bands=[args.tokenizer_band]),
            mask_fraction=tokenizer.encoder_mask_fraction,
        )
        raw_prediction = tokenizer.decode(tokens).flux

    patch_size = (tokenizer.patch_size, tokenizer.patch_size)
    target_patches = patchify(target, patch_size)
    prediction_patches = patchify(raw_prediction, patch_size)
    image_shape = tuple(observed.shape[-2:])
    if tokens.mask_idx is None:
        masked = target
        composed = raw_prediction
    else:
        masked_patches = target_patches.clone()
        composed_patches = target_patches.clone()
        batch_index = torch.arange(observed.size(0)).unsqueeze(-1)
        masked_patches[batch_index, tokens.mask_idx] = 0.0
        composed_patches[batch_index, tokens.mask_idx] = prediction_patches[
            batch_index, tokens.mask_idx
        ]
        masked = unpatchify(
            masked_patches, patch_size, original_size=image_shape,
        )
        composed = unpatchify(
            composed_patches, patch_size, original_size=image_shape,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        torch.cat((observed, target, masked, composed.clamp(0, 1)), dim=0),
        args.output,
        nrow=observed.size(0),
    )
    print(
        f'Wrote {args.output} with rows: observed, target, masked target, '
        'composed MAE prediction.',
        flush=True,
    )


if __name__ == '__main__':
    main()
