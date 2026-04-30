"""
Compute observed-frame JWST colour gradients from bulge+disk decompositions.

Strategy
--------
Use direct observed-frame JWST colours — no interpolation, no redshift
dependency.  For each galaxy:

    col_bulge  = mag_model_bulge_f115w − mag_model_bulge_f277w
    col_disk   = mag_model_disk_f115w  − mag_model_disk_f277w
    delta_col  = col_bulge − col_disk          ← colour gradient

A positive delta_col means the bulge is bluer than the disk in F115W−F277W.
A second pair (F150W−F444W) is computed as an independent check.

Note: these are observed-frame colours, so they probe different rest-frame
wavelengths at different redshifts.  Once validated, a rest-frame
interpolation scheme can be added on top.

ID matching
-----------
B+D, LePhare, and photom_primary share the same row ordering.
  - id     ← photom_primary catalogue
  - zfinal ← lephare catalogue
  - mags   ← bd catalogue
All read by row position.

Quality cuts
------------
  1. fmf_b+d_chi2 < chi2_max   (default 5.0)
  2. BT in [BT_min, BT_max]    (default [0.05, 0.95])
  3. bulge_radius_deg < disk_radius_deg

Output columns
--------------
  id, z, BT, Re_bulge_deg, Re_disk_deg,
  col_bulge_115_277, col_disk_115_277, delta_col_115_277,
  col_bulge_150_444, col_disk_150_444, delta_col_150_444,
  flag_good

Usage
-----
  python -m cosmosweb.compute_color_gradients \\
      --bd_catalog      /n23data2/cosmosweb/catalogs/DR1/data/catalog/COSMOSWeb_mastercatalog_v1_bulgedisk.fits \\
      --lephare_catalog /n23data2/cosmosweb/catalogs/DR1/data/catalog/COSMOSWeb_mastercatalog_v1_lephare.fits \\
      --photom_catalog  /n23data2/cosmosweb/catalogs/DR1/data/catalog/COSMOSWeb_mastercatalog_v1_photom_primary.fits \\
      --output          /n03data/huertas/COSMOS-Web/cosmosweb_clip/color_gradients.fits \\
      --merge_npz       /n03data/huertas/COSMOS-Web/cosmosweb_clip/cosmosweb_umap_zoobot_v2.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_primary(path: Path) -> Table:
    """Read the first HDU with data."""
    with fits.open(path, memmap=True) as hdul:
        log.info("  HDUs in %s: %s", path.name, [h.name for h in hdul])
        hdu = next(h for h in hdul if h.data is not None)
        return Table(hdu.data)


def _mag_array(cat: Table, col: str) -> np.ndarray:
    arr = np.array(cat[col], dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return arr


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_gradients(
    bd_catalog:      str | Path,
    lephare_catalog: str | Path,
    photom_catalog:  str | Path,
    chi2_max:        float = 5.0,
    BT_min:          float = 0.05,
    BT_max:          float = 0.95,
    delta_max:       float = 2.0,
) -> Table:
    bd_catalog      = Path(bd_catalog)
    lephare_catalog = Path(lephare_catalog)
    photom_catalog  = Path(photom_catalog)

    log.info("Reading B+D catalogue: %s", bd_catalog)
    bd = _read_primary(bd_catalog)
    log.info("  %d rows; %d columns", len(bd), len(bd.colnames))

    log.info("Reading LePhare catalogue: %s", lephare_catalog)
    lp = _read_primary(lephare_catalog)
    log.info("  %d rows; first columns: %s", len(lp), lp.colnames[:10])

    log.info("Reading photometry catalogue (id): %s", photom_catalog)
    ph = _read_primary(photom_catalog)
    log.info("  %d rows; first columns: %s", len(ph), ph.colnames[:10])

    N = min(len(bd), len(lp), len(ph))
    if not (len(bd) == len(lp) == len(ph)):
        log.warning("Row count mismatch: B+D=%d, LePhare=%d, photom=%d — using first %d",
                    len(bd), len(lp), len(ph), N)
    bd = bd[:N]
    lp = lp[:N]
    ph = ph[:N]

    # ── id from photometry catalogue (positional match) ───────────────────────
    id_col = next((c for c in ph.colnames if c.lower() == 'id'), None)
    if id_col is None:
        raise ValueError(f"No 'id' column in photom catalogue. Columns: {ph.colnames[:30]}")
    ids = np.array(ph[id_col], dtype=np.int64)

    # ── redshift from LePhare ─────────────────────────────────────────────────
    z_col = next((c for c in lp.colnames
                  if c.lower() in ('zfinal', 'z_phot', 'z')), None)
    if z_col is None:
        raise ValueError(f"No redshift column in LePhare. Columns: {lp.colnames[:30]}")
    z = np.array(lp[z_col], dtype=np.float32)
    z = np.where(np.isfinite(z) & (z > 0), z, np.nan)
    log.info("Redshift '%s': %d / %d finite", z_col, np.isfinite(z).sum(), N)

    # ── quality columns ───────────────────────────────────────────────────────
    chi2 = _mag_array(bd, 'fmf_b+d_chi2') if 'fmf_b+d_chi2' in bd.colnames \
           else np.zeros(N, dtype=np.float32)

    bt_col = next((c for c in bd.colnames
                   if c.lower() in ('bt_jwst', 'bt_f277w', 'bt_f150w', 'bt')), None)
    BT = _mag_array(bd, bt_col) if bt_col else np.full(N, 0.5, dtype=np.float32)
    if bt_col:
        log.info("B/T column: '%s'", bt_col)
    else:
        log.warning("No B/T column found; B/T quality cut skipped")

    Re_bulge = _mag_array(bd, 'bulge_radius_deg') if 'bulge_radius_deg' in bd.colnames \
               else np.zeros(N, dtype=np.float32)
    Re_disk  = _mag_array(bd, 'disk_radius_deg')  if 'disk_radius_deg'  in bd.colnames \
               else np.ones(N,  dtype=np.float32)

    # ── JWST magnitudes ───────────────────────────────────────────────────────
    def _get(component: str, band: str) -> np.ndarray:
        col = f'mag_model_{component}_{band}'
        if col not in bd.colnames:
            log.warning("Column '%s' not found — filling NaN", col)
            return np.full(N, np.nan, dtype=np.float32)
        return _mag_array(bd, col)

    bulge_f115 = _get('bulge', 'f115w')
    bulge_f150 = _get('bulge', 'f150w')
    bulge_f277 = _get('bulge', 'f277w')
    bulge_f444 = _get('bulge', 'f444w')
    disk_f115  = _get('disk',  'f115w')
    disk_f150  = _get('disk',  'f150w')
    disk_f277  = _get('disk',  'f277w')
    disk_f444  = _get('disk',  'f444w')

    # ── observed-frame colour gradients ───────────────────────────────────────
    # Pair 1: F115W − F277W
    col_bulge_115_277 = bulge_f115 - bulge_f277
    col_disk_115_277  = disk_f115  - disk_f277
    delta_col_115_277 = col_bulge_115_277 - col_disk_115_277

    # Pair 2: F150W − F444W  (independent check)
    col_bulge_150_444 = bulge_f150 - bulge_f444
    col_disk_150_444  = disk_f150  - disk_f444
    delta_col_150_444 = col_bulge_150_444 - col_disk_150_444

    # ── quality flags ─────────────────────────────────────────────────────────
    flag_chi2  = chi2 < chi2_max
    flag_BT    = (BT >= BT_min) & (BT <= BT_max) if bt_col \
                 else np.ones(N, dtype=bool)
    flag_size  = Re_bulge < Re_disk
    flag_mags  = (np.isfinite(bulge_f115) & np.isfinite(bulge_f277) &
                  np.isfinite(disk_f115)  & np.isfinite(disk_f277))
    # Reject unphysical colour gradients from degenerate fits
    flag_range = np.abs(delta_col_115_277) < delta_max
    flag_good  = flag_chi2 & flag_BT & flag_size & flag_mags & flag_range

    log.info("Quality cuts → %d / %d pass (%.1f %%)",
             flag_good.sum(), N, 100.0 * flag_good.sum() / max(N, 1))
    log.info("  chi2 < %.1f          : %d", chi2_max,  flag_chi2.sum())
    log.info("  BT in [%.2f, %.2f]   : %d", BT_min, BT_max, flag_BT.sum())
    log.info("  Re_bulge < Re_disk   : %d", flag_size.sum())
    log.info("  all F115W/F277W finite: %d", flag_mags.sum())
    log.info("  |delta_col| < %.1f   : %d", delta_max, flag_range.sum())

    # ── output table ──────────────────────────────────────────────────────────
    out = Table({
        'id':                 ids,
        'z':                  z,
        'BT':                 BT,
        'Re_bulge_deg':       Re_bulge,
        'Re_disk_deg':        Re_disk,
        'chi2':               chi2,
        'col_bulge_115_277':  col_bulge_115_277,
        'col_disk_115_277':   col_disk_115_277,
        'delta_col_115_277':  delta_col_115_277,
        'col_bulge_150_444':  col_bulge_150_444,
        'col_disk_150_444':   col_disk_150_444,
        'delta_col_150_444':  delta_col_150_444,
        'flag_good':          flag_good,
        # per-band magnitudes for diagnostics
        'bulge_f115w':        bulge_f115,
        'bulge_f150w':        bulge_f150,
        'bulge_f277w':        bulge_f277,
        'bulge_f444w':        bulge_f444,
        'disk_f115w':         disk_f115,
        'disk_f150w':         disk_f150,
        'disk_f277w':         disk_f277,
        'disk_f444w':         disk_f444,
    })

    # per-band total mags for B/T computation (if available)
    for band in ('f115w', 'f150w', 'f277w', 'f444w'):
        col = f'mag_model_bd_total_{band}'
        if col in bd.colnames:
            out[f'total_{band}'] = _mag_array(bd, col)

    return out


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def make_diagnostic_pdf(grad: Table, pdf_path: str | Path) -> None:
    """
    Multi-page PDF with B/T and colour diagnostics.

    Pages
    -----
    1. chi2 distribution (before/after quality cuts)
    2. B/T from the catalogue + per-band B/T derived from fluxes
    3. Observed F115W−F277W colours of bulge and disk, binned by redshift
    4. Observed F150W−F444W colours of bulge and disk, binned by redshift
    5. Colour gradient distributions (delta_col_115_277, delta_col_150_444)
    6. delta_col_115_277 vs redshift (median per z-bin with scatter)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = Path(pdf_path)
    good = np.array(grad['flag_good'], dtype=bool)
    z    = np.array(grad['z'],         dtype=float)

    # z-bins for colour panels
    z_edges = [0.0, 0.5, 1.0, 1.5, 2.5, 5.0]
    z_labels = [f'{z_edges[i]:.1f} < z < {z_edges[i+1]:.1f}'
                for i in range(len(z_edges) - 1)]
    n_zbins  = len(z_labels)
    z_colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_zbins))

    def _hist(ax, vals, mask, label, color, alpha=0.7, bins=60):
        v = vals[mask & np.isfinite(vals)]
        if len(v) == 0:
            return
        p5, p95 = np.nanpercentile(v, [2, 98])
        ax.hist(v, bins=bins, range=(p5, p95),
                histtype='step', color=color, label=label, alpha=alpha, density=True)

    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: chi2 ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        chi2 = np.array(grad['chi2'], dtype=float)
        finite = np.isfinite(chi2)
        p99 = float(np.nanpercentile(chi2[finite], 99))
        ax.hist(chi2[finite], bins=100, range=(0, min(p99, 20)),
                histtype='stepfilled', alpha=0.5, color='steelblue',
                label=f'All ({finite.sum():,})', density=True)
        ax.hist(chi2[good & finite], bins=100, range=(0, min(p99, 20)),
                histtype='step', color='crimson', lw=1.5,
                label=f'Quality cut ({good.sum():,})', density=True)
        ax.set_xlabel('fmf_b+d_chi2')
        ax.set_ylabel('Density')
        ax.set_title('B+D fit chi2 distribution')
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Page 2: B/T distributions ─────────────────────────────────────────
        # Catalogue B/T + per-band B/T derived from flux ratios
        BT_cat = np.array(grad['BT'], dtype=float)
        bands  = ('f115w', 'f150w', 'f277w', 'f444w')
        colors_b = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # Derive per-band B/T = flux_bulge / flux_total
        per_band_BT = {}
        for band in bands:
            if f'total_{band}' in grad.colnames and f'bulge_{band}' in grad.colnames:
                m_tot   = np.array(grad[f'total_{band}'],  dtype=float)
                m_bulge = np.array(grad[f'bulge_{band}'],  dtype=float)
                # BT = 10^(0.4*(m_tot - m_bulge))  [= f_bulge/f_total]
                dm = m_tot - m_bulge
                bt = np.where(np.isfinite(dm), 10 ** (0.4 * dm), np.nan)
                bt = np.where((bt > 0) & (bt < 1), bt, np.nan)
                per_band_BT[band] = bt

        has_per_band = len(per_band_BT) > 0
        n_panels = 1 + len(per_band_BT)
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4),
                                 sharey=False)
        if n_panels == 1:
            axes = [axes]

        ax = axes[0]
        v = BT_cat[good & np.isfinite(BT_cat)]
        ax.hist(v, bins=50, histtype='stepfilled', alpha=0.6,
                color='steelblue', density=True)
        ax.set_xlabel('B/T (catalogue, BT_jwst)')
        ax.set_ylabel('Density')
        ax.set_title('Catalogue B/T')

        for i, (band, bt) in enumerate(per_band_BT.items()):
            ax = axes[i + 1]
            v_all  = bt[np.isfinite(bt)]
            v_good = bt[good & np.isfinite(bt)]
            ax.hist(v_all,  bins=50, histtype='stepfilled', alpha=0.4,
                    color=colors_b[i], density=True, label='all')
            ax.hist(v_good, bins=50, histtype='step', lw=1.5,
                    color=colors_b[i], density=True, label='quality cut')
            ax.set_xlabel(f'B/T derived from {band.upper()}')
            ax.set_title(f'B/T from {band.upper()}')
            ax.legend(fontsize=8)

        fig.suptitle('B/T ratio distributions', fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Pages 3 & 4: colours per z-bin ───────────────────────────────────
        for pair, col_b_key, col_d_key, pair_label in [
            ('115_277', 'col_bulge_115_277', 'col_disk_115_277',  'F115W − F277W'),
            ('150_444', 'col_bulge_150_444', 'col_disk_150_444',  'F150W − F444W'),
        ]:
            col_b = np.array(grad[col_b_key], dtype=float)
            col_d = np.array(grad[col_d_key], dtype=float)

            fig, axes = plt.subplots(1, n_zbins, figsize=(3.5 * n_zbins, 4),
                                     sharey=False)
            for i, (zlo, zhi, zlabel, zc) in enumerate(
                    zip(z_edges[:-1], z_edges[1:], z_labels, z_colors)):
                ax   = axes[i]
                zmask = (z >= zlo) & (z < zhi) & good
                _hist(ax, col_b, zmask, 'Bulge', color=zc,       alpha=0.85)
                _hist(ax, col_d, zmask, 'Disk',  color='0.4',    alpha=0.55)
                ax.set_xlabel(pair_label)
                ax.set_title(f'{zlabel}\n(N={zmask.sum():,})', fontsize=8)
                if i == 0:
                    ax.set_ylabel('Density')
                ax.legend(fontsize=7)

            fig.suptitle(f'Observed-frame {pair_label}: bulge vs disk colour by redshift',
                         fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ── Page 5: gradient distributions ────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, label in [
            (axes[0], 'delta_col_115_277', 'Δ(F115W−F277W)  [bulge − disk]'),
            (axes[1], 'delta_col_150_444', 'Δ(F150W−F444W)  [bulge − disk]'),
        ]:
            vals = np.array(grad[key], dtype=float)
            _hist(ax, vals, good, 'quality cut', color='steelblue', bins=80)
            _hist(ax, vals, np.ones(len(vals), dtype=bool), 'all',
                  color='0.6', bins=80, alpha=0.4)
            ax.axvline(0, color='k', lw=0.8, ls='--')
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            med = float(np.nanmedian(vals[good]))
            ax.set_title(f'median = {med:+.3f} mag')
            ax.legend(fontsize=8)

        fig.suptitle('Colour gradient distributions', fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Page 6: delta_col vs redshift ─────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, key, label in [
            (axes[0], 'delta_col_115_277', 'Δ(F115W−F277W)'),
            (axes[1], 'delta_col_150_444', 'Δ(F150W−F444W)'),
        ]:
            vals  = np.array(grad[key], dtype=float)
            z_mid = np.array([0.25, 0.75, 1.25, 2.0, 3.5])
            medians, p16, p84 = [], [], []
            for zlo, zhi in zip(z_edges[:-1], z_edges[1:]):
                m = good & (z >= zlo) & (z < zhi) & np.isfinite(vals)
                v = vals[m]
                if len(v) > 10:
                    medians.append(float(np.median(v)))
                    p16.append(float(np.percentile(v, 16)))
                    p84.append(float(np.percentile(v, 84)))
                else:
                    medians.append(np.nan); p16.append(np.nan); p84.append(np.nan)

            # scatter (random subset)
            rng  = np.random.default_rng(0)
            idx  = np.where(good & np.isfinite(vals))[0]
            idx  = rng.choice(idx, size=min(5000, len(idx)), replace=False)
            ax.scatter(z[idx], vals[idx], s=1, alpha=0.2, color='0.6', rasterized=True)
            ax.errorbar(z_mid, medians,
                        yerr=[np.array(medians) - np.array(p16),
                              np.array(p84) - np.array(medians)],
                        fmt='o-', color='crimson', lw=1.5, ms=5,
                        label='median ± 1σ')
            ax.axhline(0, color='k', lw=0.8, ls='--')
            ax.set_xlabel('Redshift z')
            ax.set_ylabel(label)
            ax.set_title(f'{label} vs redshift')
            ax.legend(fontsize=8)

        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    log.info("Diagnostic PDF → %s", pdf_path)


# ---------------------------------------------------------------------------
# Merge into existing npz
# ---------------------------------------------------------------------------

def merge_with_npz(grad_table: Table, npz_path: str | Path) -> None:
    """Inject colour gradient columns into an existing UMAP npz (matched by id)."""
    npz_path = Path(npz_path)
    data     = dict(np.load(npz_path, allow_pickle=True))

    npz_ids = data.get('galaxy_id', data.get('galaxy_ids'))
    if npz_ids is None:
        raise KeyError("npz has neither 'galaxy_id' nor 'galaxy_ids'")
    npz_ids = npz_ids.astype(np.int64)
    N_npz   = len(npz_ids)

    grad_id = np.array(grad_table['id'], dtype=np.int64)
    id2idx  = {gid: i for i, gid in enumerate(grad_id)}

    for col in ['delta_col_115_277', 'col_bulge_115_277', 'col_disk_115_277',
                'delta_col_150_444', 'col_bulge_150_444', 'col_disk_150_444',
                'BT']:
        arr = np.full(N_npz, np.nan, dtype=np.float32)
        for j, gid in enumerate(npz_ids):
            if gid in id2idx:
                row = id2idx[gid]
                if grad_table['flag_good'][row]:
                    val = float(grad_table[col][row])
                    arr[j] = val if np.isfinite(val) else np.nan
        data[col] = arr
        log.info("  %-20s : %d / %d finite", col, np.isfinite(arr).sum(), N_npz)

    flag_arr = np.zeros(N_npz, dtype=bool)
    for j, gid in enumerate(npz_ids):
        if gid in id2idx:
            flag_arr[j] = bool(grad_table['flag_good'][id2idx[gid]])
    data['bd_flag_good'] = flag_arr

    np.savez(str(npz_path.with_suffix('')), **data)
    log.info("Saved updated npz → %s", npz_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute rest-frame NUV-r colour gradients from B+D decompositions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--bd_catalog',      required=True,
                   help='Standalone B+D FITS file')
    p.add_argument('--lephare_catalog', required=True,
                   help='Standalone LePhare FITS file (zfinal, same row order as B+D)')
    p.add_argument('--photom_catalog',  required=True,
                   help='Photometry FITS file containing the id column (same row order)')
    p.add_argument('--output',          required=True,
                   help='Output FITS table path')
    p.add_argument('--chi2_max',   type=float, default=5.0,
                   help='Max B+D chi2')
    p.add_argument('--BT_min',     type=float, default=0.05,
                   help='Min B/T')
    p.add_argument('--BT_max',     type=float, default=0.95,
                   help='Max B/T')
    p.add_argument('--delta_max',  type=float, default=2.0,
                   help='Max |delta_col_115_277| in mag; rejects degenerate fits')
    p.add_argument('--merge_npz',       default=None,
                   help='If given, merge results into this npz file')
    p.add_argument('--diagnostic_pdf',  default=None,
                   help='If given, write diagnostic plots to this PDF')
    p.add_argument('--log_level', default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING'])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(message)s',
    )

    grad = compute_gradients(
        bd_catalog=args.bd_catalog,
        lephare_catalog=args.lephare_catalog,
        photom_catalog=args.photom_catalog,
        chi2_max=args.chi2_max,
        BT_min=args.BT_min,
        BT_max=args.BT_max,
        delta_max=args.delta_max,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grad.write(str(out_path), overwrite=True)
    log.info("Wrote %d rows → %s", len(grad), out_path)

    if args.diagnostic_pdf:
        make_diagnostic_pdf(grad, args.diagnostic_pdf)

    if args.merge_npz:
        log.info("Merging into %s", args.merge_npz)
        merge_with_npz(grad, args.merge_npz)


if __name__ == '__main__':
    main()
