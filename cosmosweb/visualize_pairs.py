"""
Visualize random (F150W image, SFH) pairs from the cosmosweb HDF5 dataset.

Each page shows a grid of pairs: galaxy stamp on the left, interpolated SFH
on the right.  Output is a multi-page PDF.

Usage:
    python cosmosweb/visualize_pairs.py \\
        --dataset  /n03data/huertas/COSMOS-Web/cosmosweb_clip/cosmosweb_dataset.h5 \\
        --output   /n03data/huertas/COSMOS-Web/cosmosweb_clip/pair_examples.pdf \\
        --n_pages  5 \\
        --seed     42
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# ── layout constants ──────────────────────────────────────────────────────────
PAIRS_PER_ROW  = 3   # image+SFH pairs across the page
ROWS_PER_PAGE  = 4   # rows of pairs per PDF page
PAIRS_PER_PAGE = PAIRS_PER_ROW * ROWS_PER_PAGE   # 12

SFH_EPS    = 1e-10   # must match prepare_dataset.py
SFH_N_BINS = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path,
                   default='/n03data/huertas/COSMOS-Web/cosmosweb_clip/cosmosweb_dataset.h5')
    p.add_argument('--output',  type=Path,
                   default='/n03data/huertas/COSMOS-Web/cosmosweb_clip/pair_examples.pdf')
    p.add_argument('--n_pages', type=int, default=5,
                   help='Number of PDF pages (12 pairs per page)')
    p.add_argument('--seed',    type=int, default=42)
    return p.parse_args()


def load_random_sample(h5_path: Path, n: int, rng: np.random.Generator):
    """Load n random pairs from the HDF5 dataset."""
    with h5py.File(h5_path, 'r') as f:
        n_total  = int(f.attrs['n_galaxies'])
        img_mean = f.attrs['img_mean'].astype(np.float32)   # (3,)
        img_std  = f.attrs['img_std'].astype(np.float32)    # (3,)

        n = min(n, n_total)
        idx = np.sort(rng.choice(n_total, size=n, replace=False))

        images       = f['images'][idx]         # (n, 3, 64, 64)  normalised arcsinh
        sfhs         = f['sfh'][idx]            # (n, N_BINS)     log10(SFR+eps)
        sfh_time_norm = f['sfh_time_norm'][idx] # (n,)            t_universe(z) in Myr
        redshifts    = f['redshift'][idx]       # (n,)
        gal_ids      = f['galaxy_id'][idx]      # (n,)

    return images, sfhs, sfh_time_norm, redshifts, gal_ids, img_mean, img_std


def render_image(ax, img_normalised: np.ndarray, img_mean: float, img_std: float):
    """Display F150W stamp (channel 0) with percentile stretch."""
    # Un-normalise back to arcsinh-flux space for display
    channel = img_normalised[0] * img_std + img_mean   # arcsinh(flux)
    vmin, vmax = np.percentile(channel, [0.5, 99.5])
    ax.imshow(channel, origin='lower', cmap='gray',
              vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])


def render_sfh(ax, sfh_log: np.ndarray, t_universe_myr: float):
    """
    Plot SFH as a step function: SFR (M_sun/yr) vs lookback time (Gyr).

    sfh_log is in log10(SFR + eps) space; the time axis is recovered by
    multiplying the fractional grid [0, 1] by t_universe_myr.
    """
    sfr = 10.0 ** sfh_log - SFH_EPS          # back to physical SFR
    sfr = np.maximum(sfr, SFH_EPS)            # guard against negatives from float precision
    t_frac   = np.linspace(0, 1, SFH_N_BINS)
    time_gyr = t_frac * t_universe_myr / 1e3  # fractional → Gyr

    # Avoid log(0) on the x-axis by starting from the second point
    ax.step(time_gyr[1:], sfr[1:], where='post', color='steelblue', lw=1.2)
    ax.fill_between(time_gyr[1:], sfr[1:], step='post', alpha=0.25, color='steelblue')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(time_gyr[1] * 0.9, time_gyr[-1] * 1.1)
    ax.set_ylim(bottom=max(sfr[sfr > SFH_EPS].min() * 0.1, 1e-4)
                if (sfr > SFH_EPS).any() else 1e-4)
    ax.set_xlabel('Lookback time [Gyr]', fontsize=6)
    ax.set_ylabel(r'SFR [M$_\odot$ yr$^{-1}$]', fontsize=6)
    ax.tick_params(labelsize=5)


def make_page(fig, pairs: list, img_mean: float, img_std: float):
    """Fill a figure with up to PAIRS_PER_PAGE (image, SFH) pairs."""
    n = len(pairs)
    # Each pair occupies 2 columns (image + SFH); leave small gap between pairs
    gs = GridSpec(
        ROWS_PER_PAGE, PAIRS_PER_ROW * 2,
        figure=fig,
        hspace=0.45, wspace=0.15,
    )

    for i, (img, sfh, t_univ, z, gid) in enumerate(pairs):
        row = i // PAIRS_PER_ROW
        col = (i %  PAIRS_PER_ROW) * 2

        ax_img = fig.add_subplot(gs[row, col])
        ax_sfh = fig.add_subplot(gs[row, col + 1])

        render_image(ax_img, img, img_mean, img_std)
        render_sfh(ax_sfh, sfh, t_univ)

        ax_img.set_title(
            f'id={gid}  z={z:.2f}\n'
            f't$_{{univ}}$={t_univ/1e3:.1f} Gyr',
            fontsize=5.5, pad=2,
        )


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rng     = np.random.default_rng(args.seed)
    n_total = args.n_pages * PAIRS_PER_PAGE

    print(f'Loading {n_total} random pairs from {args.dataset}…')
    images, sfhs, sfh_time_norm, redshifts, gal_ids, img_mean, img_std = \
        load_random_sample(args.dataset, n_total, rng)

    print(f'Rendering {args.n_pages} pages → {args.output}')
    with PdfPages(args.output) as pdf:
        for page in range(args.n_pages):
            start = page * PAIRS_PER_PAGE
            end   = min(start + PAIRS_PER_PAGE, len(images))
            if start >= len(images):
                break

            pairs = [
                (images[j], sfhs[j], sfh_time_norm[j], redshifts[j], gal_ids[j])
                for j in range(start, end)
            ]

            fig = plt.figure(figsize=(11, 8.5))   # letter landscape
            make_page(fig, pairs, img_mean[0], img_std[0])
            fig.suptitle(
                f'COSMOS-Web  ·  F150W stamps + CIGALE SFH  '
                f'(page {page + 1}/{args.n_pages})',
                fontsize=9, y=0.98,
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f'  Page {page + 1} done')

    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
