"""
Rising SFH cross-modal retrieval.

For a random sample of galaxies with rising SFH (quenching index < threshold),
find their k nearest neighbours in the joint embedding space and plot:
  - query SFH (9 CIGALE bins)
  - query image
  - k nearest-neighbour images

Usage:
    python -m cosmosweb.rising_sfh_retrieval \\
        --npz   /path/to/cosmosweb_umap_zoobot_v8.npz \\
        --h5    /path/to/cosmosweb_dataset_v6.h5 \\
        --output rising_sfh_retrieval.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SFH_EPS = 1e-10


# ── SFH helpers ───────────────────────────────────────────────────────────────

def compute_quench_index(h5_path: Path, h5_indices: np.ndarray) -> np.ndarray:
    """
    quench_index = f(5-10%) - f(0-5%)
    Negative  → SFR rising  (more SF in last 5% than 5-10% ago)
    Positive  → SFR declining / quenching
    """
    with h5py.File(h5_path, 'r') as f:
        sfh_log = f['sfh'][list(h5_indices)].astype(np.float64)   # (N, 50)
        t_frac  = (f['sfh_time_grid'][:].astype(np.float64)
                   if 'sfh_time_grid' in f
                   else np.linspace(0, 1, sfh_log.shape[1]))

    sfr = np.maximum(10.0 ** sfh_log - SFH_EPS, 0.0)
    sfr /= sfr.sum(axis=1, keepdims=True).clip(min=1e-10)

    cumul_5  = sfr[:, t_frac <= 0.05].sum(axis=1)
    cumul_10 = sfr[:, t_frac <= 0.10].sum(axis=1)
    return (cumul_10 - cumul_5) - cumul_5   # delta(5-10%) - delta(0-5%)


def load_sfh_bins(h5_path: Path, h5_idx: int):
    """Load 9-bin CIGALE SFH for a single galaxy."""
    with h5py.File(h5_path, 'r') as f:
        bins_log  = f['sfh_bins_log'][h5_idx].astype(np.float64)   # (9,)
        times_myr = f['sfh_times_myr'][h5_idx].astype(np.float64)  # (9,)
        t_norm    = float(f['sfh_time_norm'][h5_idx])
    return bins_log, times_myr, t_norm


def load_image(h5_path: Path, h5_idx: int) -> np.ndarray:
    """Load F277W stamp (channel 1) for a single galaxy."""
    with h5py.File(h5_path, 'r') as f:
        img = f['images'][h5_idx, 1]   # F277W channel
    return img.astype(np.float32)


def load_redshift(h5_path: Path, h5_idx: int) -> float:
    with h5py.File(h5_path, 'r') as f:
        return float(f['redshift'][h5_idx])


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_sfh_bars(ax, bins_log: np.ndarray, times_myr: np.ndarray,
                  t_norm: float, quench_index: float, z: float):
    """Bar chart of 9 CIGALE bins on fractional lookback time axis."""
    w = np.maximum(10.0 ** bins_log - SFH_EPS, SFH_EPS)
    w = w / w.sum()
    t_centres = times_myr / t_norm
    edges = np.empty(len(t_centres) + 1)
    edges[0]    = 0.0
    edges[1:-1] = 0.5 * (t_centres[:-1] + t_centres[1:])
    edges[-1]   = 1.0
    widths = np.diff(edges)
    ax.bar(edges[:-1], w, width=widths, align='edge',
           color='steelblue', alpha=0.8, edgecolor='steelblue', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fractional lookback time', fontsize=6)
    ax.set_ylabel('SFH weight', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.set_title(f'z={z:.2f}  QI={quench_index:.3f}', fontsize=6, pad=2)


def plot_stamp(ax, img: np.ndarray, label: str = '', qi: float | None = None):
    vmin, vmax = np.percentile(img, [0.5, 99.5])
    ax.imshow(img, origin='lower', cmap='gray',
              vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    title = label
    if qi is not None:
        title += f'\nQI={qi:.3f}'
    if title:
        ax.set_title(title, fontsize=5, pad=1)


# ── retrieval ─────────────────────────────────────────────────────────────────

def nearest_neighbours(joint_emb: np.ndarray, query_idx: int,
                       k: int, exclude_self: bool = True) -> np.ndarray:
    """Cosine similarity (embeddings assumed L2-normalised)."""
    sims = joint_emb @ joint_emb[query_idx]
    if exclude_self:
        sims[query_idx] = -np.inf
    return np.argsort(sims)[::-1][:k]


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',         type=Path, required=True)
    p.add_argument('--h5',          type=Path, required=True)
    p.add_argument('--output',      type=Path, default=Path('rising_sfh_retrieval.pdf'))
    p.add_argument('--n_queries',   type=int,  default=20,
                   help='Number of random rising-SFH query galaxies')
    p.add_argument('--k',           type=int,  default=9,
                   help='Number of nearest neighbours to retrieve')
    p.add_argument('--qi_threshold', type=float, default=-0.03,
                   help='Quenching index threshold (< this = rising SFH)')
    p.add_argument('--seed',        type=int,  default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # ── load embeddings ───────────────────────────────────────────────────────
    npz        = np.load(args.npz, allow_pickle=True)
    joint_emb  = npz['img_emb'].astype(np.float32) + npz['sfh_emb'].astype(np.float32)
    norms      = np.linalg.norm(joint_emb, axis=1, keepdims=True).clip(min=1e-10)
    joint_emb  = joint_emb / norms                          # (N, 256) L2-normalised
    h5_indices = npz['h5_indices'].astype(int)
    galaxy_ids = npz['galaxy_ids']
    N          = len(h5_indices)

    # ── compute quenching index for all galaxies ──────────────────────────────
    print(f'Computing quenching index for {N} galaxies…')
    qi = compute_quench_index(args.h5, h5_indices)

    # ── select rising-SFH query galaxies ─────────────────────────────────────
    rising_mask = qi < args.qi_threshold
    rising_idx  = np.where(rising_mask)[0]   # indices into the npz/embedding arrays
    print(f'Rising SFH (QI < {args.qi_threshold}): {len(rising_idx)} / {N} galaxies')

    if len(rising_idx) < args.n_queries:
        print(f'Warning: only {len(rising_idx)} rising-SFH galaxies found; '
              f'using all of them.')
        query_idx = rising_idx
    else:
        query_idx = rng.choice(rising_idx, size=args.n_queries, replace=False)

    # ── build PDF ─────────────────────────────────────────────────────────────
    k      = args.k
    ncols  = k + 2          # SFH | query image | k neighbours
    nrows  = 1

    print(f'Writing {len(query_idx)} pages to {args.output} …')
    with PdfPages(args.output) as pdf:
        for qi_pos, emb_idx in enumerate(query_idx):
            h5_idx = h5_indices[emb_idx]
            gid    = int(galaxy_ids[emb_idx])

            # retrieve k nearest neighbours (positions in embedding array)
            nn_emb_idx = nearest_neighbours(joint_emb, emb_idx, k=k)
            nn_h5_idx  = h5_indices[nn_emb_idx]

            # load data
            bins_log, times_myr, t_norm = load_sfh_bins(args.h5, h5_idx)
            query_img = load_image(args.h5, h5_idx)
            query_z   = load_redshift(args.h5, h5_idx)
            query_qi  = float(qi[emb_idx])

            nn_imgs = [load_image(args.h5, idx) for idx in nn_h5_idx]
            nn_qi   = qi[nn_emb_idx]
            nn_z    = [load_redshift(args.h5, idx) for idx in nn_h5_idx]

            # layout: [SFH bar | query image | nn_1 ... nn_k]
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(2.0 * ncols, 2.2 * nrows))
            fig.suptitle(f'Query galaxy {gid}  (#{qi_pos+1}/{len(query_idx)})',
                         fontsize=8, y=1.01)

            # SFH panel
            plot_sfh_bars(axes[0], bins_log, times_myr, t_norm, query_qi, query_z)

            # query image
            plot_stamp(axes[1], query_img, label='query', qi=query_qi)

            # nearest neighbours
            for i, (img, nqi, nz) in enumerate(zip(nn_imgs, nn_qi, nn_z)):
                plot_stamp(axes[2 + i], img,
                           label=f'z={nz:.2f}', qi=float(nqi))

            plt.tight_layout(pad=0.4)
            pdf.savefig(fig, dpi=120, bbox_inches='tight')
            plt.close(fig)

            if (qi_pos + 1) % 5 == 0:
                print(f'  {qi_pos + 1} / {len(query_idx)}')

    print(f'Done → {args.output}')


if __name__ == '__main__':
    main()
