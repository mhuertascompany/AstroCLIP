"""
Diagnostic PDF: original CIGALE SFH vs current and proposed preprocessed SFH.

For each randomly selected galaxy this script produces a row with three panels:

  [1] Original CIGALE  — 9-bin fractional SFR weights vs lookback time [Myr]
  [2] Current grid (log₁₀) — 50-bin linspace fractional grid (stored in H5)
  [3] Proposed grid (log₁₀) — 50-bin log-spaced fractional grid (recomputed)

Panels 2 and 3 share a fractional lookback-time x-axis so the redistribution
of resolution toward recent times is directly visible.

Usage
-----
  python -m cosmosweb.plot_sfh_diagnostics \\
      --h5       /path/to/cosmosweb_dataset_v2.h5 \\
      --catalog  /path/to/COSMOSWeb_mastercatalog_v1.fits \\
      --output   sfh_diagnostics.pdf \\
      --n_gal    20 \\
      --seed     42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

COSMO   = FlatLambdaCDM(H0=70, Om0=0.3)
SFH_EPS = 1e-10
SFH_N_BINS = 50
GALAXIES_PER_PAGE = 4   # rows per PDF page

# Proposed log-spaced fractional time grid (matches prepare_dataset.py update).
# Bin 0 is fixed at t_frac=0 (always fill_value); bins 1-49 are log-spaced
# from 1e-3 to 1.0, concentrating resolution near the present.
NEW_T_FRAC = np.concatenate([[0.0], np.geomspace(1e-3, 1.0, SFH_N_BINS - 1)])


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_catalog(catalog_path: Path):
    """Return ids (N,), sfr_bins (N,9), err_bins (N,9), time_bins (N,9)."""
    print(f"Opening catalog {catalog_path} …")
    with fits.open(catalog_path, memmap=True) as hdul:
        # HDU1 = photometry (has 'id')
        ids = np.array(hdul[1].data['id'], dtype=np.int64)
        # HDU4 = CIGALE (row-aligned with HDU1)
        cig = hdul[4].data
        sfr  = np.array([cig[f'sfh_sfr_bin{i}']     for i in range(1, 10)], dtype=np.float64).T  # (N,9)
        err  = np.array([cig[f'sfh_sfr_bin{i}_err'] for i in range(1, 10)], dtype=np.float64).T
        time = np.array([cig[f'sfh_time_bin{i}']    for i in range(1, 10)], dtype=np.float64).T
    print(f"  {len(ids)} galaxies in catalog")
    return ids, sfr, err, time


def _preprocess(sfr_raw: np.ndarray, err_raw: np.ndarray, lb_time_raw: np.ndarray,
                z: float, sfh_t_frac: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Reproduce the exact preprocessing from prepare_dataset.py.

    Returns (sfh_log, sfh_lin_norm, t_universe_myr).
      sfh_log      – log10(normalised + eps), length N_TIME  [what model sees]
      sfh_lin_norm – normalised linear shape, length N_TIME  [for comparison]
    """
    t_universe_myr = float(COSMO.age(z).to('Myr').value)
    phys_grid      = sfh_t_frac * t_universe_myr    # (N_TIME,) Myr

    sort_idx  = np.argsort(lb_time_raw)
    lb_time   = lb_time_raw[sort_idx]
    sfr_frac  = sfr_raw[sort_idx]
    err_frac  = err_raw[sort_idx]

    # kind='next': at each grid point return the SFR of the next CIGALE bin ≥ t.
    # fill_value=(sfr_frac[0], 0.0): for t < lb_time[0] (more recent than the
    # youngest CIGALE bin) use sfr_frac[0] — the most recent bin extends back to
    # t=0.  For t > lb_time[-1] use 0 (beyond the oldest bin → no SF).
    f_next     = interp1d(lb_time, sfr_frac, kind='next',
                          bounds_error=False, fill_value=(sfr_frac[0], 0.0))
    sfr_interp = f_next(phys_grid)

    sfr_total = sfr_interp.sum()
    if sfr_total > SFH_EPS:
        sfr_interp = sfr_interp / sfr_total

    sfh_log = np.log10(sfr_interp + SFH_EPS).astype(np.float32)
    return sfh_log, sfr_interp.astype(np.float32), t_universe_myr


# ── plotting ───────────────────────────────────────────────────────────────────

def _plot_row(axes, galaxy_id: int, z: float, t_univ: float,
              lb_time_raw: np.ndarray, sfr_raw: np.ndarray, err_raw: np.ndarray,
              sfh_t_frac_old: np.ndarray, sfh_log_h5: np.ndarray,
              sfh_t_frac_new: np.ndarray, sfh_log_new: np.ndarray) -> None:
    """Fill one 3-panel row for a single galaxy.

    Panel 1 – original CIGALE 9-bin SFH (linear scale, Myr x-axis)
    Panel 2 – current linspace grid stored in H5 (log₁₀, fractional x-axis)
    Panel 3 – proposed log-spaced grid recomputed from catalog (log₁₀, fractional x-axis)
    """
    ax1, ax2, ax3 = axes

    # ── sort original bins ────────────────────────────────────────────────────
    sort_idx   = np.argsort(lb_time_raw)
    lb_sorted  = lb_time_raw[sort_idx]
    sfr_sorted = sfr_raw[sort_idx]
    # err_raw is retained in the signature for future use but not plotted here

    bin_edges = np.concatenate([[0.0],
                                (lb_sorted[:-1] + lb_sorted[1:]) / 2,
                                [lb_sorted[-1] * 1.5]])

    def _panel_label(ax, txt):
        ax.text(0.02, 0.97, txt, transform=ax.transAxes,
                fontsize=8, style='italic', va='top', ha='left', color='#333333')

    def _log_stairs(ax, sfh_log, t_frac_grid, **kw):
        """Plot log10 SFH as stairs on a fractional x-axis, skipping bins 0 and 49."""
        tf   = t_frac_grid[1:49]           # (48,) fractional — skip boundary bins
        vals = sfh_log[1:49]               # (48,)
        edges = np.concatenate([[0.0], tf])  # (49,)
        ax.stairs(vals, edges, baseline=None, **kw)

    # ── panel 1: original 9-bin SFH (linear) ─────────────────────────────────
    ax1.stairs(sfr_sorted, bin_edges, fill=True, color='steelblue', alpha=0.6)
    ax1.stairs(sfr_sorted, bin_edges, color='steelblue', lw=1.5)
    ax1.set_xlabel('Lookback time [Myr]')
    ax1.set_ylabel('Fractional SFR weight')
    ax1.set_xlim(0, t_univ * 1.05)
    ax1.set_ylim(bottom=0)
    ax1.axvline(t_univ, color='k', lw=0.8, ls='--', alpha=0.4)
    _panel_label(ax1, 'Original CIGALE (9 bins)')

    # ── panel 2: current linspace grid (from H5, log₁₀) ──────────────────────
    _log_stairs(ax2, sfh_log_h5, sfh_t_frac_old,
                color='tomato', lw=1.4)
    ax2.set_xlabel('Fractional lookback time')
    ax2.set_ylabel('log₁₀(norm. SFR + ε)')
    ax2.set_xlim(0, 1.05)
    _panel_label(ax2, 'Current — linspace grid (bins 1–48, from H5)')

    # ── panel 3: proposed log-spaced grid (recomputed, log₁₀) ────────────────
    _log_stairs(ax3, sfh_log_new, sfh_t_frac_new,
                color='mediumseagreen', lw=1.4)
    # overlay the CIGALE bin boundaries as faint vertical lines for reference
    for lb in lb_sorted:
        tf_lb = lb / t_univ
        if 0 < tf_lb < 1:
            ax3.axvline(tf_lb, color='grey', lw=0.5, ls=':', alpha=0.5)
    ax3.set_xlabel('Fractional lookback time')
    ax3.set_ylabel('log₁₀(norm. SFR + ε)')
    ax3.set_xlim(0, 1.05)
    _panel_label(ax3, 'Proposed — log-spaced grid (bins 1–48, recomputed)')

    for ax in axes:
        ax.set_facecolor('#fafafa')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Diagnostic PDF: original vs preprocessed SFH')
    parser.add_argument('--h5',      required=True, help='cosmosweb_dataset_v2.h5')
    parser.add_argument('--catalog', required=True, help='COSMOSWeb_mastercatalog_v1.fits')
    parser.add_argument('--output',  default='sfh_diagnostics.pdf')
    parser.add_argument('--n_gal',   type=int, default=20, help='Number of galaxies to plot')
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── load H5 ──────────────────────────────────────────────────────────────
    print(f"Opening H5 {args.h5} …")
    with h5py.File(args.h5, 'r') as f:
        n_tot      = f.attrs['n_galaxies']
        h5_gids    = f['galaxy_id'][:]           # (N,) catalog IDs
        h5_z       = f['redshift'][:]            # (N,)
        h5_sfh     = f['sfh'][:]                 # (N, 50) log10 values
        sfh_t_frac = f['sfh_time_grid'][:]       # (50,) ∈ [0,1]
        h5_tnorm   = f['sfh_time_norm'][:]       # (N,) t_universe [Myr]
    print(f"  {n_tot} galaxies in H5")

    # ── load catalog ─────────────────────────────────────────────────────────
    cat_ids, cat_sfr, cat_err, cat_time = _load_catalog(Path(args.catalog))
    id2row = {gid: i for i, gid in enumerate(cat_ids)}

    # ── random selection ──────────────────────────────────────────────────────
    chosen_h5_idx = rng.choice(n_tot, size=min(args.n_gal, n_tot), replace=False)
    chosen_h5_idx = np.sort(chosen_h5_idx)
    print(f"Selected {len(chosen_h5_idx)} galaxies")

    # ── PDF ───────────────────────────────────────────────────────────────────
    print(f"Writing {args.output} …")
    with PdfPages(args.output) as pdf:
        n_pages = int(np.ceil(len(chosen_h5_idx) / GALAXIES_PER_PAGE))

        for page in range(n_pages):
            page_idx = chosen_h5_idx[page * GALAXIES_PER_PAGE :
                                     (page + 1) * GALAXIES_PER_PAGE]
            n_rows   = len(page_idx)

            fig = plt.figure(figsize=(15, 4.5 * n_rows))
            # Extra top margin per row for the row-header text
            gs  = gridspec.GridSpec(n_rows, 3, figure=fig,
                                    hspace=0.7, wspace=0.35,
                                    top=0.94, bottom=0.06)

            for row, h5_idx in enumerate(page_idx):
                gid = int(h5_gids[h5_idx])
                z   = float(h5_z[h5_idx])

                if gid not in id2row:
                    print(f"  galaxy_id={gid} not found in catalog — skipping")
                    continue

                cat_row   = id2row[gid]
                sfr_raw   = cat_sfr[cat_row]     # (9,)
                err_raw   = cat_err[cat_row]      # (9,)
                time_raw  = cat_time[cat_row]     # (9,) lookback Myr
                sfh_log_h5 = h5_sfh[h5_idx]      # (50,)
                t_univ    = float(h5_tnorm[h5_idx])

                # Recompute with the proposed log-spaced grid
                sfh_log_new, _, _ = _preprocess(
                    sfr_raw, err_raw, time_raw, z, NEW_T_FRAC
                )

                axes = [fig.add_subplot(gs[row, col]) for col in range(3)]

                _plot_row(
                    axes, gid, z, t_univ,
                    time_raw, sfr_raw, err_raw,
                    sfh_t_frac, sfh_log_h5,
                    NEW_T_FRAC, sfh_log_new,
                )

                # Row header: placed just above the left panel using figure coords
                ax0_pos = axes[0].get_position()
                fig.text(
                    ax0_pos.x0, ax0_pos.y1 + 0.005,
                    f'galaxy_id={gid}   z={z:.3f}   t_univ={t_univ:.0f} Myr',
                    fontsize=9, fontweight='bold', va='bottom', ha='left',
                    color='#222222',
                )

            fig.suptitle(
                f'SFH preprocessing diagnostics  (page {page + 1}/{n_pages})',
                fontsize=11, y=1.01
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f"  Page {page + 1}/{n_pages} done")

    print(f"Saved → {args.output}")


if __name__ == '__main__':
    main()
