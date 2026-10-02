"""Native-flux Euclid VIS FITS loading and fixed global normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from torchvision import transforms


def resolve_fits_directory(root: str | Path, band: str = 'VIS') -> Path:
    root = Path(root)
    candidates = (root / 'cutouts' / band, root / band, root)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f'Cannot find FITS directory below {root}; tried '
        + ', '.join(str(path) for path in candidates)
    )


def vis_fits_path(directory: str | Path, galaxy_id: int,
                  band: str = 'VIS') -> Path:
    directory = Path(directory)
    candidates = (
        directory / f'{int(galaxy_id)}.fits',
        directory / f'{band}_{int(galaxy_id)}.fits',
        directory / f'{int(galaxy_id)}.fits.gz',
        directory / f'{band}_{int(galaxy_id)}.fits.gz',
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def read_vis_fits(path: str | Path, header: bool = False):
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                data = np.asarray(hdu.data, dtype=np.float32).copy()
                if not np.all(np.isfinite(data)):
                    raise ValueError(f'Nonfinite pixels in {path}.')
                return (data, hdu.header.copy()) if header else data
    raise ValueError(f'No two-dimensional image HDU in {path}.')


def write_image_stats(path: str | Path, p_lo: float, p_hi: float,
                      band: str = 'euclid-vis', p_lo_q: float = 1.0,
                      p_hi_q: float = 99.0, n_images: int | None = None) -> None:
    payload = {
        'p_lo_q': float(p_lo_q),
        'p_hi_q': float(p_hi_q),
        'bands': {band: {'p_lo': float(p_lo), 'p_hi': float(p_hi)}},
    }
    if n_images is not None:
        payload['n_images'] = int(n_images)
    Path(path).write_text(json.dumps(payload, indent=2) + '\n')


def load_image_stats(path: str | Path, band: str = 'euclid-vis') -> tuple[float, float]:
    payload = json.loads(Path(path).read_text())
    try:
        values = payload['bands'][band]
        p_lo, p_hi = float(values['p_lo']), float(values['p_hi'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid image statistics for {band!r}: {path}') from exc
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        raise ValueError(f'Invalid percentile range ({p_lo}, {p_hi}) in {path}.')
    return p_lo, p_hi


def estimate_image_stats(fits_directory: str | Path, galaxy_ids,
                         band: str = 'VIS', max_images: int = 4096,
                         max_pixels_per_image: int = 4096, seed: int = 42,
                         p_lo_q: float = 1.0, p_hi_q: float = 99.0):
    ids = np.asarray(galaxy_ids, dtype=np.int64)
    if not len(ids):
        raise ValueError('Cannot estimate image statistics from an empty sample.')
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(ids)[:min(max_images, len(ids))]
    sampled = []
    for galaxy_id in chosen:
        values = read_vis_fits(vis_fits_path(fits_directory, galaxy_id, band)).ravel()
        if len(values) > max_pixels_per_image:
            values = values[rng.choice(
                len(values), size=max_pixels_per_image, replace=False,
            )]
        sampled.append(values)
    pixels = np.concatenate(sampled).astype(np.float32, copy=False)
    p_lo, p_hi = np.percentile(pixels, [p_lo_q, p_hi_q])
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        raise ValueError(f'Invalid estimated percentile range: {p_lo}, {p_hi}.')
    return float(p_lo), float(p_hi), len(chosen), int(len(pixels))


def normalize_vis_flux(data: np.ndarray, p_lo: float, p_hi: float,
                       asinh_scale: float = 20.0) -> np.ndarray:
    if not np.isfinite(asinh_scale) or asinh_scale <= 0:
        raise ValueError('asinh_scale must be finite and positive.')
    scaled = (np.asarray(data, dtype=np.float32) - p_lo) / (p_hi - p_lo)
    return (
        np.arcsinh(asinh_scale * scaled) / np.arcsinh(asinh_scale)
    ).astype(np.float32, copy=False)


class VISFitsTransform:
    """Read native VIS flux, apply fixed scaling, and make a square tensor."""

    def __init__(self, image_size: int, p_lo: float, p_hi: float,
                 asinh_scale: float = 20.0, training: bool = False) -> None:
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)
        self.asinh_scale = float(asinh_scale)
        if training:
            self.transform = transforms.Compose([
                transforms.Resize(
                    int(image_size * 1.1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(image_size),
            ])

    def __call__(self, path: str | Path) -> torch.Tensor:
        data = normalize_vis_flux(
            read_vis_fits(path), self.p_lo, self.p_hi, self.asinh_scale,
        )
        return self.transform(torch.from_numpy(data).unsqueeze(0))


__all__ = [
    'VISFitsTransform', 'estimate_image_stats', 'load_image_stats',
    'normalize_vis_flux', 'read_vis_fits', 'resolve_fits_directory',
    'vis_fits_path', 'write_image_stats',
]
