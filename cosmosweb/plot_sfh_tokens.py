"""
Visualise the v8 SFH token representation.

For each randomly selected galaxy shows three panels:

  [1] Current fixed grid — 50-bin kind='next' (stored in H5)
  [2] Token representation — 9 (t_frac, log_sfr) points, n_samples random
      draws overlaid; each draw jitters t_frac within the bin's extent
  [3] Token jitter range — for each bin shows the fixed centre (×) and the
      uniform sampling interval (horizontal error bar), coloured by z

A second figure shows the population view: 9-token scatter for all selected
galaxies, coloured by redshift, to check whether z stratification is reduced
compared to the fixed-grid representation.

Usage
-----
  python -m cosmosweb.plot_sfh_tokens \\
      --h5       /path/to/cosmosweb_dataset.h5 \\
      --catalog  /path/to/COSMOSWeb_mastercatalog_v1.fits \\
      --output   sfh_tokens_diagnostics.pdf \\
      --n_gal    20 \\
      --n_samples 8 \\
      --seed     42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors

COSMO      = FlatLambdaCDM(H0=70, Om0=0.3)
SFH_EPS    = 1e-10
GALAXIES_PER_PAGE = 4


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_catalog(catalog_path: Path):
    """Return ids, sfr_bins (N,9), sfh_time_bins (N,9)."""
    print(f'Loading catalog {catalog_path} …')
    with fits.open(catalog_path, memmap=True) as hdul:
        ids  = np.array(hdul[1].data['id'], dtype=np.int64)
        cig  = hdul[4].data
        sfr  = np.array([cig[f'sfh_sfr_bin{i}']  for i in range(1, 10)], dtype=np.float64).T
        time = np.array([cig[f'sfh_time_bin{i}'] for i in range(1, 10)], dtype=np.float64).T
    print(f'  {len(ids)} galaxies in catalog')
    return ids, sfr, time


def _bin_edges(times_myr: np.ndarray, t_universe_myr: float) -> np.ndarray:
    """
    Compute bin edges from 9 bin-centre lookback times.

    edges[0]  = 0  (present)
    edges[i]  = midpoint between bin i-1 and bin i  (i = 1..8)
    edges[9]  = t_universe_myr  (oldest possible)
    """
    n = len(times_myr)
    edges = np.empty(n + 1)
    edges[0] = 0.0
    for i in range(1, n):
        edges[i] = 0.5 * (times_myr[i - 1] + times_myr[i])
    edges[n] = t_universe_myr
    return edges


def _sample_tokens(sfr_bins_log: np.ndarray, times_myr: np.ndarray,
                   t_universe_myr: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample one random t_frac per CIGALE bin.

    Returns (9, 2) array of (t_frac, log_sfr) tokens.
    At train time this is called fresh each step; here we call it n_samples times.
    """
    edges = _bin_edges(times_myr, t_universe_myr)
    t_samples = np.array([
        rng.uniform(edges[i], edges[i + 1])
        for i in range(len(times_myr))
    ])
    t_frac = np.clip(t_samples / t_universe_myr, 0.0, 1.0)
    return np.stack([t_frac, sfr_bins_log], axis=1).astype(np.float32)   # (9, 2)


def _normalise_sfr(sfr_raw: np.ndarray) -> np.ndarray:
    """Normalise 9-bin SFR fractions and take log10."""
    sfr = sfr_raw.copy()
    total = sfr.sum()
    if total > SFH_EPS:
        sfr /= total
    return np.log10(sfr + SFH_EPS).astype(np.float32)


# ── per-galaxy diagnostic row ──────────────────────────────────────────────────

def _plot_row(axes, gid: int, z: float, t_univ: float,
              sfh_grid: np.ndarray, t_frac_grid: np.ndarray,
              sfr_bins_log: np.ndarray, times_myr: np.ndarray,
              n_samples: int, rng: np.random.Generator) -> None:
    """Three-panel row for one galaxy."""
    ax1, ax2, ax3 = axes

    edges    = _bin_edges(times_myr, t_univ)
    t_centre = times_myr / t_univ   # fractional bin centres

    cmap_s   = cm.cool
    sample_colors = cmap_s(np.linspace(0.1, 0.9, n_samples))

    # ── panel 1: current fixed-grid (50 bins, kind='next') ───────────────────
    tf = t_frac_grid[1:]
    ax1.scatter(tf, sfh_grid[1:], s=10, color='tomato', alpha=0.8)
    ax1.set_xlabel('Fractional lookback time')
    ax1.set_ylabel('log₁₀(norm. SFR + ε)')
    ax1.set_xlim(0, 1.05)
    ax1.text(0.02, 0.97, "Fixed 50-bin grid (kind='next')",
             transform=ax1.transAxes, fontsize=8, style='italic',
             va='top', ha='left', color='#333')

    # ── panel 2: n_samples random token draws ────────────────────────────────
    for s in range(n_samples):
        tokens = _sample_tokens(sfr_bins_log, times_myr, t_univ, rng)   # (9, 2)
        ax2.scatter(tokens[:, 0], tokens[:, 1], s=30,
                    color=sample_colors[s], alpha=0.7, zorder=3)
        ax2.plot(tokens[:, 0], tokens[:, 1], '-',
                 color=sample_colors[s], alpha=0.25, lw=0.8)
    # mark bin centres deterministically
    ax2.scatter(t_centre, sfr_bins_log, s=60, marker='x',
                color='k', zorder=5, linewidths=1.5, label='bin centre')
    ax2.set_xlabel('Fractional lookback time')
    ax2.set_ylabel('log₁₀(norm. SFR + ε)')
    ax2.set_xlim(0, 1.05)
    ax2.legend(fontsize=7, loc='lower left')
    ax2.text(0.02, 0.97, f'9-token random sampling ({n_samples} draws)',
             transform=ax2.transAxes, fontsize=8, style='italic',
             va='top', ha='left', color='#333')

    # ── panel 3: jitter range (bin extents in fractional time) ───────────────
    edge_frac = edges / t_univ
    for i in range(len(times_myr)):
        width = edge_frac[i + 1] - edge_frac[i]
        ax3.errorbar(t_centre[i], sfr_bins_log[i],
                     xerr=[[t_centre[i] - edge_frac[i]],
                            [edge_frac[i + 1] - t_centre[i]]],
                     fmt='x', color='steelblue', capsize=3,
                     elinewidth=1.2, markeredgewidth=1.5, ms=7)
    ax3.set_xlabel('Fractional lookback time')
    ax3.set_ylabel('log₁₀(norm. SFR + ε)')
    ax3.set_xlim(0, 1.05)
    ax3.text(0.02, 0.97, 'Bin centres (×) and sampling extents',
             transform=ax3.transAxes, fontsize=8, style='italic',
             va='top', ha='left', color='#333')

    for ax in axes:
        ax.set_facecolor('#fafafa')


# ── population scatter ─────────────────────────────────────────────────────────

def _plot_population(sfr_bins_log_all: np.ndarray, times_myr_all: np.ndarray,
                     t_univ_all: np.ndarray, z_all: np.ndarray,
                     rng: np.random.Generator) -> plt.Figure:
    """
    Two-panel population figure.

    Left  — fixed grid (from H5), 50-bin scatter coloured by z
    Right — one random token draw per galaxy, coloured by z
    """
    z_min, z_max = z_all.min(), z_all.max()
    norm   = mcolors.Normalize(vmin=z_min, vmax=z_max)
    colors = cm.viridis(norm(z_all))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'SFH token representation — population view  (N={len(z_all)})',
        fontsize=12
    )

    ax_fix, ax_tok = axes

    # left: just show the bin-centre points (t_frac, log_sfr) for all galaxies
    order = np.argsort(z_all)
    for i in order:
        t_centre = times_myr_all[i] / t_univ_all[i]
        ax_fix.scatter(t_centre, sfr_bins_log_all[i],
                       s=6, color=colors[i], alpha=0.3)

    ax_fix.set_xlabel('Fractional lookback time (bin centre)')
    ax_fix.set_ylabel('log₁₀(norm. SFR + ε)')
    ax_fix.set_title('Bin centres (deterministic)', fontsize=10)
    ax_fix.set_xlim(0, 1.05)
    ax_fix.set_facecolor('#f8f8f8')

    # right: one random draw per galaxy
    for i in order:
        tokens = _sample_tokens(sfr_bins_log_all[i], times_myr_all[i],
                                t_univ_all[i], rng)
        ax_tok.scatter(tokens[:, 0], tokens[:, 1],
                       s=6, color=colors[i], alpha=0.3)

    ax_tok.set_xlabel('Fractional lookback time (random sample)')
    ax_tok.set_ylabel('log₁₀(norm. SFR + ε)')
    ax_tok.set_title('One random draw per galaxy', fontsize=10)
    ax_tok.set_xlim(0, 1.05)
    ax_tok.set_facecolor('#f8f8f8')

    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02, label='Redshift z')

    plt.tight_layout()
    return fig


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualise v8 SFH token representation')
    parser.add_argument('--h5',       required=True, help='cosmosweb dataset HDF5')
    parser.add_argument('--catalog',  required=True, help='COSMOSWeb_mastercatalog_v1.fits')
    parser.add_argument('--output',   default='sfh_tokens_diagnostics.pdf')
    parser.add_argument('--n_gal',    type=int, default=20)
    parser.add_argument('--n_samples',type=int, default=8,
                        help='Number of random token draws per galaxy (per-galaxy panels)')
    parser.add_argument('--n_pop',    type=int, default=300,
                        help='Number of galaxies in the population scatter panel')
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── load H5 ──────────────────────────────────────────────────────────────
    print(f'Opening {args.h5} …')
    with h5py.File(args.h5, 'r') as f:
        n_tot       = f.attrs['n_galaxies']
        h5_gids     = f['galaxy_id'][:]
        h5_z        = f['redshift'][:]
        h5_sfh      = f['sfh'][:]
        h5_tnorm    = f['sfh_time_norm'][:]
        t_frac_grid = f['sfh_time_grid'][:]
    print(f'  {n_tot} galaxies in H5')

    # ── load catalog ─────────────────────────────────────────────────────────
    cat_ids, cat_sfr, cat_time = _load_catalog(Path(args.catalog))
    id2row = {gid: i for i, gid in enumerate(cat_ids)}

    # ── random selection for per-galaxy panels ────────────────────────────────
    chosen = np.sort(rng.choice(n_tot, size=min(args.n_gal, n_tot), replace=False))

    # ── also load a larger sample for the population panel ────────────────────
    sort_z = np.argsort(h5_z)
    step   = max(1, n_tot // args.n_pop)
    pop_idx = sort_z[::step][:args.n_pop]

    pop_sfr_bins, pop_times, pop_tuniv, pop_z = [], [], [], []
    for idx in pop_idx:
        gid = int(h5_gids[idx])
        row = id2row.get(gid)
        if row is None:
            continue
        sfr_raw  = cat_sfr[row]
        time_raw = cat_time[row]
        t_univ   = float(h5_tnorm[idx])
        z        = float(h5_z[idx])
        if not (np.isfinite(z) and z > 0 and np.all(np.isfinite(sfr_raw))):
            continue
        sort_idx = np.argsort(time_raw)
        times_s  = time_raw[sort_idx]
        sfr_s    = sfr_raw[sort_idx]
        pop_sfr_bins.append(_normalise_sfr(sfr_s))
        pop_times.append(times_s.astype(np.float32))
        pop_tuniv.append(t_univ)
        pop_z.append(z)

    pop_sfr_bins = np.stack(pop_sfr_bins)
    pop_times    = np.stack(pop_times)
    pop_tuniv    = np.array(pop_tuniv)
    pop_z        = np.array(pop_z)

    # ── write PDF ────────────────────────────────────────────────────────────
    print(f'Writing {args.output} …')
    with PdfPages(args.output) as pdf:

        # Page 0: population scatter
        fig_pop = _plot_population(pop_sfr_bins, pop_times, pop_tuniv, pop_z, rng)
        pdf.savefig(fig_pop, bbox_inches='tight')
        plt.close(fig_pop)
        print('  Population page done')

        # Per-galaxy pages
        n_pages = int(np.ceil(len(chosen) / GALAXIES_PER_PAGE))
        for page in range(n_pages):
            page_idx = chosen[page * GALAXIES_PER_PAGE:
                              (page + 1) * GALAXIES_PER_PAGE]
            n_rows   = len(page_idx)

            fig = plt.figure(figsize=(18, 4.5 * n_rows))
            gs  = gridspec.GridSpec(n_rows, 3, figure=fig,
                                    hspace=0.7, wspace=0.35,
                                    top=0.94, bottom=0.06)

            for row, h5_idx in enumerate(page_idx):
                gid   = int(h5_gids[h5_idx])
                z     = float(h5_z[h5_idx])
                t_univ = float(h5_tnorm[h5_idx])

                cat_row = id2row.get(gid)
                if cat_row is None:
                    print(f'  galaxy_id={gid} not in catalog — skipping')
                    continue

                sfr_raw  = cat_sfr[cat_row]
                time_raw = cat_time[cat_row]
                sort_i   = np.argsort(time_raw)
                times_s  = time_raw[sort_i].astype(np.float32)
                sfr_s    = sfr_raw[sort_i]
                sfr_log  = _normalise_sfr(sfr_s)

                axes = [fig.add_subplot(gs[row, col]) for col in range(3)]
                _plot_row(axes, gid, z, t_univ,
                          h5_sfh[h5_idx], t_frac_grid,
                          sfr_log, times_s,
                          n_samples=args.n_samples, rng=rng)

                ax0_pos = axes[0].get_position()
                fig.text(
                    ax0_pos.x0, ax0_pos.y1 + 0.005,
                    f'galaxy_id={gid}   z={z:.3f}   t_univ={t_univ:.0f} Myr',
                    fontsize=9, fontweight='bold', va='bottom', ha='left',
                )

            fig.suptitle(
                f'SFH token diagnostics  (page {page + 1}/{n_pages})',
                fontsize=11, y=1.01
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f'  Page {page + 1}/{n_pages} done')

    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
