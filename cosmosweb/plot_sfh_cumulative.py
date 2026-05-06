"""
Diagnostic plot comparing the log-differential SFH vs the cumulative SFH,
with galaxies ranked and coloured by redshift.

Two summary panels:
  Left  — log₁₀(norm. SFR + ε) vs fractional lookback time  [current encoder input]
  Right — cumulative fractional SFH (CDF) vs fractional lookback time  [proposed]

Each panel overlays N_GAL curves drawn from evenly-spaced redshift quantiles,
coloured by redshift (viridis).  Median curves per redshift bin are shown thick.

If the cumulative SFH is more redshift-independent, the right panel should show
less colour-stratification than the left panel.

Usage
-----
  python -m cosmosweb.plot_sfh_cumulative \\
      --h5      /path/to/cosmosweb_dataset_v4.h5 \\
      --output  sfh_cumulative_diagnostics.pdf \\
      --n_gal   200 \\
      --n_bins  5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

SFH_EPS = 1e-10


def load_h5(h5_path: Path):
    with h5py.File(h5_path, 'r') as f:
        sfh_log  = f['sfh'][:]           # (N, 50) log10 values
        redshift = f['redshift'][:]      # (N,)
        t_frac   = f['sfh_time_grid'][:] # (50,) ∈ [0, 1]
    return sfh_log, redshift, t_frac


def to_cumulative(sfh_log: np.ndarray) -> np.ndarray:
    """
    Convert log10 differential SFH → cumulative (CDF).

    sfh_log : (N, 50)  log10(norm_sfr + eps)
    returns  : (N, 50)  cumulative fraction ∈ [0, 1]
    """
    sfr_lin = np.maximum(10.0 ** sfh_log - SFH_EPS, 0.0)   # (N, 50), linear
    sfr_sum = sfr_lin.sum(axis=1, keepdims=True)
    sfr_norm = sfr_lin / np.where(sfr_sum > 0, sfr_sum, 1.0)
    return np.cumsum(sfr_norm, axis=1)                       # (N, 50) ∈ [0, 1]


def main():
    parser = argparse.ArgumentParser(
        description='Differential vs cumulative SFH coloured by redshift')
    parser.add_argument('--h5',     required=True, type=Path)
    parser.add_argument('--output', default='sfh_cumulative_diagnostics.pdf', type=Path)
    parser.add_argument('--n_gal',  type=int, default=300,
                        help='Number of galaxies to plot (evenly spaced in redshift)')
    parser.add_argument('--n_bins', type=int, default=5,
                        help='Number of redshift bins for median curves')
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    print(f'Loading {args.h5} …')
    sfh_log, redshift, t_frac = load_h5(args.h5)
    n_tot = len(redshift)
    print(f'  {n_tot} galaxies  z ∈ [{redshift.min():.2f}, {redshift.max():.2f}]')

    # ── select galaxies evenly spaced in redshift quantiles ───────────────────
    sort_idx  = np.argsort(redshift)
    step      = max(1, n_tot // args.n_gal)
    sel       = sort_idx[::step][:args.n_gal]
    sfh_sel   = sfh_log[sel]        # (M, 50)
    z_sel     = redshift[sel]       # (M,)
    cumul_sel = to_cumulative(sfh_sel)  # (M, 50)

    # ── colour map (viridis, normalised to redshift range) ────────────────────
    z_min, z_max = z_sel.min(), z_sel.max()
    norm   = mcolors.Normalize(vmin=z_min, vmax=z_max)
    cmap   = cm.viridis
    colors = cmap(norm(z_sel))      # (M, 4)

    # ── redshift bin edges for median curves ──────────────────────────────────
    bin_edges = np.percentile(z_sel, np.linspace(0, 100, args.n_bins + 1))
    bin_edges[0]  -= 0.01
    bin_edges[-1] += 0.01
    bin_centers   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_colors    = cmap(norm(bin_centers))

    # skip t_frac=0 (always fill value) for plotting
    tf = t_frac[1:]          # (49,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'SFH representations coloured by redshift  (N={len(sel)})',
        fontsize=12, y=1.01
    )

    # ── left panel: log differential SFH ─────────────────────────────────────
    ax = axes[0]
    # draw individual curves (faint)
    for i in np.argsort(z_sel):   # draw low-z first so high-z is on top
        ax.plot(tf, sfh_sel[i, 1:], color=colors[i], alpha=0.15, lw=0.6)

    # draw median per z-bin (thick)
    for b in range(args.n_bins):
        mask = (z_sel >= bin_edges[b]) & (z_sel < bin_edges[b + 1])
        if mask.sum() < 3:
            continue
        med = np.median(sfh_sel[mask][:, 1:], axis=0)
        ax.plot(tf, med, color=bin_colors[b], lw=2.5,
                label=f'z={bin_edges[b]:.1f}–{bin_edges[b+1]:.1f}  (n={mask.sum()})')

    ax.set_xlabel('Fractional lookback time', fontsize=11)
    ax.set_ylabel('log₁₀(norm. SFR + ε)', fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_title('Log-differential SFH  [current encoder input]', fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_facecolor('#f8f8f8')

    # ── right panel: cumulative SFH ───────────────────────────────────────────
    ax = axes[1]
    for i in np.argsort(z_sel):
        ax.plot(tf, cumul_sel[i, 1:], color=colors[i], alpha=0.15, lw=0.6)

    for b in range(args.n_bins):
        mask = (z_sel >= bin_edges[b]) & (z_sel < bin_edges[b + 1])
        if mask.sum() < 3:
            continue
        med = np.median(cumul_sel[mask][:, 1:], axis=0)
        ax.plot(tf, med, color=bin_colors[b], lw=2.5,
                label=f'z={bin_edges[b]:.1f}–{bin_edges[b+1]:.1f}  (n={mask.sum()})')

    ax.set_xlabel('Fractional lookback time', fontsize=11)
    ax.set_ylabel('Cumulative SFR fraction', fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title('Cumulative SFH  [proposed encoder input]', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_facecolor('#f8f8f8')

    # ── shared colorbar ───────────────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label('Redshift z', fontsize=11)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
