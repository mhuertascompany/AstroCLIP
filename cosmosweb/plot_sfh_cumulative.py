"""
Diagnostic plot comparing three SFH representations coloured by redshift:

  Panel 1 — log₁₀(norm. SFR + ε), kind='next'   [current encoder input]
  Panel 2 — log₁₀(norm. SFR + ε), kind='linear'  [proposed fix]
  Panel 3 — cumulative fractional SFH (CDF)

Galaxies are drawn from evenly-spaced redshift quantiles and coloured by
redshift (viridis).  Median curves per redshift bin are shown thick.

Usage
-----
  python -m cosmosweb.plot_sfh_cumulative \\
      --h5      /path/to/cosmosweb_dataset_v4.h5 \\
      --catalog /path/to/COSMOSWeb_mastercatalog_v1.fits \\
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
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

COSMO      = FlatLambdaCDM(H0=70, Om0=0.3)
SFH_EPS    = 1e-10
SFH_N_BINS = 50
T_FRAC     = np.linspace(0.0, 1.0, SFH_N_BINS)


def load_h5(h5_path: Path):
    with h5py.File(h5_path, 'r') as f:
        sfh_log    = f['sfh'][:]
        redshift   = f['redshift'][:]
        galaxy_ids = f['galaxy_id'][:]
        t_frac     = f['sfh_time_grid'][:]
    return sfh_log, redshift, galaxy_ids, t_frac


def load_catalog(catalog_path: Path):
    """Return id→row index and raw SFR/err/time arrays."""
    print(f'Loading catalog {catalog_path} …')
    with fits.open(catalog_path, memmap=True) as hdul:
        ids  = np.array(hdul[1].data['id'], dtype=np.int64)
        cig  = hdul[4].data
        sfr  = np.array([cig[f'sfh_sfr_bin{i}']     for i in range(1, 10)], dtype=np.float64).T
        err  = np.array([cig[f'sfh_sfr_bin{i}_err'] for i in range(1, 10)], dtype=np.float64).T
        time = np.array([cig[f'sfh_time_bin{i}']    for i in range(1, 10)], dtype=np.float64).T
    id2row = {gid: i for i, gid in enumerate(ids)}
    return id2row, sfr, err, time


def preprocess_linear(sfr_raw, err_raw, lb_time_raw, z):
    """Reproduce prepare_dataset.py but with kind='linear' interpolation."""
    t_universe_myr = float(COSMO.age(z).to('Myr').value)
    phys_grid      = T_FRAC * t_universe_myr

    sort_idx = np.argsort(lb_time_raw)
    lb_time  = lb_time_raw[sort_idx]
    sfr_frac = sfr_raw[sort_idx].copy()

    sfr_total = sfr_frac.sum()
    if sfr_total > SFH_EPS:
        sfr_frac = sfr_frac / sfr_total

    f_lin      = interp1d(lb_time, sfr_frac, kind='linear',
                          bounds_error=False, fill_value=(sfr_frac[0], 0.0))
    sfr_interp = f_lin(phys_grid)
    sfr_interp = np.maximum(sfr_interp, 0.0)

    return np.log10(sfr_interp + SFH_EPS).astype(np.float32)


def to_cumulative(sfh_log: np.ndarray) -> np.ndarray:
    sfr_lin  = np.maximum(10.0 ** sfh_log - SFH_EPS, 0.0)
    sfr_sum  = sfr_lin.sum(axis=1, keepdims=True)
    sfr_norm = sfr_lin / np.where(sfr_sum > 0, sfr_sum, 1.0)
    return np.cumsum(sfr_norm, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5',      required=True, type=Path)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--output',  default='sfh_cumulative_diagnostics.pdf', type=Path)
    parser.add_argument('--n_gal',   type=int, default=300)
    parser.add_argument('--n_bins',  type=int, default=5)
    args = parser.parse_args()

    print(f'Loading {args.h5} …')
    sfh_log, redshift, galaxy_ids, t_frac = load_h5(args.h5)
    id2row, cat_sfr, cat_err, cat_time    = load_catalog(args.catalog)
    n_tot = len(redshift)
    print(f'  {n_tot} galaxies  z ∈ [{redshift.min():.2f}, {redshift.max():.2f}]')

    # ── select galaxies evenly spaced in redshift ─────────────────────────────
    sort_idx = np.argsort(redshift)
    step     = max(1, n_tot // args.n_gal)
    sel      = sort_idx[::step][:args.n_gal]

    sfh_sel   = sfh_log[sel]
    z_sel     = redshift[sel]
    gid_sel   = galaxy_ids[sel]
    cumul_sel = to_cumulative(sfh_sel)

    # ── compute linear-interpolated SFH for selected galaxies ────────────────
    print('Computing linear-interpolated SFHs …')
    sfh_lin_sel = []
    skipped = 0
    for gid, z in zip(gid_sel, z_sel):
        row = id2row.get(int(gid))
        if row is None or not (np.isfinite(z) and z > 0):
            sfh_lin_sel.append(np.full(SFH_N_BINS, np.nan))
            skipped += 1
            continue
        sfh_lin_sel.append(preprocess_linear(cat_sfr[row], cat_err[row], cat_time[row], z))
    sfh_lin_sel = np.stack(sfh_lin_sel)
    if skipped:
        print(f'  {skipped} galaxies not found in catalog — shown as NaN')

    # ── colour setup ──────────────────────────────────────────────────────────
    z_min, z_max = z_sel.min(), z_sel.max()
    norm    = mcolors.Normalize(vmin=z_min, vmax=z_max)
    cmap    = cm.viridis
    colors  = cmap(norm(z_sel))

    bin_edges  = np.percentile(z_sel, np.linspace(0, 100, args.n_bins + 1))
    bin_edges[0]  -= 0.01
    bin_edges[-1] += 0.01
    bin_centers   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_colors    = cmap(norm(bin_centers))

    tf = t_frac[1:]   # skip t_frac=0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'SFH representations coloured by redshift  (N={len(sel)})',
        fontsize=12, y=1.01
    )

    panels = [
        (sfh_sel,     'log₁₀(norm. SFR + ε)',    "Log-differential SFH — kind='next'\n[current encoder input]",  False),
        (sfh_lin_sel, 'log₁₀(norm. SFR + ε)',    "Log-differential SFH — kind='linear'\n[proposed fix]",         False),
        (cumul_sel,   'Cumulative SFR fraction',  'Cumulative SFH',                                                True),
    ]

    for ax, (data, ylabel, title, is_cumul) in zip(axes, panels):
        order = np.argsort(z_sel)
        for i in order:
            row = data[i, 1:]
            if np.any(np.isfinite(row)):
                ax.plot(tf, row, color=colors[i], alpha=0.15, lw=0.6)

        for b in range(args.n_bins):
            mask = (z_sel >= bin_edges[b]) & (z_sel < bin_edges[b + 1])
            if mask.sum() < 3:
                continue
            med = np.nanmedian(data[mask][:, 1:], axis=0)
            ax.plot(tf, med, color=bin_colors[b], lw=2.5,
                    label=f'z={bin_edges[b]:.1f}–{bin_edges[b+1]:.1f} (n={mask.sum()})')

        ax.set_xlabel('Fractional lookback time', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1.02)
        if is_cumul:
            ax.set_ylim(-0.02, 1.05)
            ax.legend(fontsize=8, loc='upper left')
        else:
            ax.legend(fontsize=8, loc='lower left')
        ax.set_facecolor('#f8f8f8')

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
