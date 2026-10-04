"""Regenerate a valid MAE preview from a trained GalaxyTikTok tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
    parser.add_argument(
        '--mask-ensembles', type=int, default=16,
        help='Minimum masked passes used to predict every patch while hidden.',
    )
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.std() == 0 or right.std() == 0:
        return float('nan')
    return float(np.corrcoef(left, right)[0, 1])


def _low_frequency_fraction(image: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float64)
    image = image - np.median(image)
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(image * window))) ** 2
    yy, xx = np.indices(image.shape)
    radius = np.hypot(
        yy - (image.shape[0] - 1) / 2,
        xx - (image.shape[1] - 1) / 2,
    )
    low = radius <= 0.1 * min(image.shape)
    total = power.sum()
    return float(power[low].sum() / total) if total > 0 else float('nan')


def _residual_diagnostics(observed, target, reconstruction, galaxy_ids):
    observed = observed.detach().cpu().numpy()[:, 0]
    target = target.detach().cpu().numpy()[:, 0]
    reconstruction = reconstruction.detach().cpu().numpy()[:, 0]
    height, width = target.shape[-2:]
    yy, xx = np.indices((height, width))
    radius = np.hypot(yy - (height - 1) / 2, xx - (width - 1) / 2)
    outer = radius >= 0.38 * min(height, width)
    central = radius <= 0.2 * min(height, width)
    rows = []
    for galaxy_id, image, truth, prediction in zip(
            galaxy_ids, observed, target, reconstruction):
        residual = truth - prediction
        background_sigma = _robust_sigma(image[outer])
        residual_sigma = _robust_sigma(residual[outer])
        valid_horizontal = outer[:, 1:] & outer[:, :-1]
        valid_vertical = outer[1:, :] & outer[:-1, :]
        rows.append({
            'galaxy_id': int(galaxy_id),
            'residual_mean': float(residual.mean()),
            'background_sigma_mad': background_sigma,
            'residual_background_sigma_mad': residual_sigma,
            'residual_to_background_sigma': (
                residual_sigma / background_sigma
                if background_sigma > 0 else float('nan')
            ),
            'background_input_residual_correlation': _correlation(
                image[outer], residual[outer],
            ),
            'residual_lag1_horizontal': _correlation(
                residual[:, 1:][valid_horizontal],
                residual[:, :-1][valid_horizontal],
            ),
            'residual_lag1_vertical': _correlation(
                residual[1:, :][valid_vertical],
                residual[:-1, :][valid_vertical],
            ),
            'input_low_frequency_power_fraction': _low_frequency_fraction(image),
            'residual_low_frequency_power_fraction': _low_frequency_fraction(
                residual,
            ),
            'central_residual_mae_to_background_sigma': (
                float(np.mean(np.abs(residual[central]))) / background_sigma
                if background_sigma > 0 else float('nan')
            ),
        })
    numeric_keys = [key for key in rows[0] if key != 'galaxy_id']
    summary = {
        key: float(np.nanmedian([row[key] for row in rows]))
        for key in numeric_keys
    }
    return {'per_galaxy': rows, 'median': summary}


def main():
    args = parse_args()
    if args.samples < 1 or args.mask_ensembles < 1:
        raise ValueError('--samples and --mask-ensembles must be positive.')
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
    observed, galaxy_ids = next(iter(loader))
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
        cross_masked = raw_prediction
        passes = 1
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
        prediction_sum = torch.zeros_like(target_patches)
        prediction_count = torch.zeros(
            target_patches.shape[:2], dtype=target_patches.dtype,
        )
        passes = 0
        maximum_passes = max(args.mask_ensembles, 128)
        with torch.no_grad():
            while passes < args.mask_ensembles or torch.any(
                    prediction_count == 0):
                ensemble_tokens = tokenizer.encode(
                    Image(flux=observed, bands=[args.tokenizer_band]),
                    mask_fraction=tokenizer.encoder_mask_fraction,
                )
                ensemble_prediction = tokenizer.decode(ensemble_tokens).flux
                ensemble_patches = patchify(ensemble_prediction, patch_size)
                for batch in range(observed.size(0)):
                    indices = ensemble_tokens.mask_idx[batch]
                    prediction_sum[batch].index_add_(
                        0, indices, ensemble_patches[batch, indices],
                    )
                    prediction_count[batch].index_add_(
                        0,
                        indices,
                        torch.ones_like(indices, dtype=prediction_count.dtype),
                    )
                passes += 1
                if passes >= maximum_passes:
                    break
        if torch.any(prediction_count == 0):
            missing = int((prediction_count == 0).sum())
            raise RuntimeError(
                f'{missing} patches were never masked after {passes} passes.'
            )
        prediction_mean = prediction_sum / prediction_count.unsqueeze(-1)
        cross_masked = unpatchify(
            prediction_mean, patch_size, original_size=image_shape,
        )
    residual = target - cross_masked
    background_width = max(1, int(round(0.12 * observed.shape[-1])))
    border = torch.cat((
        residual[..., :background_width, :].flatten(2),
        residual[..., -background_width:, :].flatten(2),
        residual[..., :, :background_width].flatten(2),
        residual[..., :, -background_width:].flatten(2),
    ), dim=-1)
    background_sigma = 1.4826 * torch.median(
        torch.abs(border - torch.median(border, dim=-1, keepdim=True).values),
        dim=-1,
    ).values.clamp_min(1e-6)
    residual_display = (
        0.5 + residual / (10.0 * background_sigma[..., None, None])
    ).clamp(0, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        torch.cat((
            observed,
            target,
            masked,
            composed.clamp(0, 1),
            cross_masked.clamp(0, 1),
            residual_display,
        ), dim=0),
        args.output,
        nrow=observed.size(0),
    )
    diagnostics = _residual_diagnostics(
        observed, target, cross_masked, galaxy_ids,
    )
    diagnostics.update({
        'tokenizer': str(args.tokenizer),
        'target_gaussian_sigma': float(args.target_gaussian_sigma),
        'mask_fraction': float(tokenizer.encoder_mask_fraction),
        'cross_mask_passes': int(passes),
        'outer_background_radius_fraction': 0.38,
        'central_radius_fraction': 0.2,
    })
    diagnostics_path = args.output.with_suffix('.json')
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + '\n')
    print(
        f'Wrote {args.output} with rows: observed, target, one masked target, '
        'one composed prediction, complete cross-masked reconstruction, and '
        f'signed residual. Diagnostics: {diagnostics_path}',
        flush=True,
    )


if __name__ == '__main__':
    main()
