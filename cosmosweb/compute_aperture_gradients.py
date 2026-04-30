"""
Compute aperture colour gradients from multi-aperture JWST photometry.

Strategy
--------
For each galaxy, the aperture catalogues contain 5 measurements per band
at circular aperture diameters [0.2, 0.3, 0.5, 0.75, 1.0] arcsec (indices
0–4).  No PSF correction is applied to these fluxes, so absolute colours
carry a PSF bias (F444W PSF FWHM ~0.15" loses more flux in small apertures
than F150W PSF FWHM ~0.05").  However, *relative* colours at the same
aperture between bands are still informative, and the gradient

    Δcol = col(aper_small) − col(aper_large)

isolates how the centre differs from the outskirts.

We compute:
  col_aper{i}_150_444 = mag_aper_f150w[:,i] − mag_aper_f444w[:,i]   i=0..4
  col_aper{i}_115_277 = mag_aper_f115w[:,i] − mag_aper_f277w[:,i]   i=0..4
  col_auto_150_444    = mag_auto_f150w       − mag_auto_f444w        (Kron)
  grad_150_444_1v4    = col_aper1 − col_aper4   (0.3" vs 1.0")  ← primary
  grad_150_444_2v4    = col_aper2 − col_aper4   (0.5" vs 1.0")  ← safer
  grad_115_277_1v4    = col_aper1 − col_aper4   F115W−F277W

Aperture index 0 (0.2") is avoided as primary because it is barely resolved
at F444W.  Index 1 (0.3") is the default small aperture.

Quality cuts (per galaxy):
  - flux_aper > 0 in both bands at both apertures used
  - flux / flux_err > snr_min  in both bands at both apertures
  - resulting colour in [-5, 5] mag (hard sanity limit)

Usage
-----
  python -m cosmosweb.compute_aperture_gradients \\
      --catalog  /path/to/COSMOSWeb_mastercatalog_v1.fits \\
      --npz      /path/to/cosmosweb_umap_zoobot_v2.npz

  # (overwrites npz in place with new columns added)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)

APER_DIAMS  = [0.2, 0.3, 0.5, 0.75, 1.0]   # arcsec, index 0–4
SNR_MIN     = 3.0
COL_MIN     = -5.0
COL_MAX     =  5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cat(catalog_path: Path) -> tuple:
    """Return (ids, mag_aper_150, mag_aper_444, mag_aper_115, mag_aper_277,
               flux_aper_150, flux_err_aper_150, flux_aper_444, flux_err_aper_444,
               mag_auto_150, mag_auto_444)  all as np.ndarray."""
    log.info("Opening %s", catalog_path)
    with fits.open(catalog_path, memmap=True) as hdul:
        log.info("HDUs: %s", [h.name for h in hdul])
        # HDU 1 is photometry
        data = hdul[1].data

        ids = np.array(data['id'], dtype=np.int64)

        def _arr(col):
            return np.array(data[col], dtype=np.float64)

        mag_aper_150  = _arr('mag_aper_f150w')       # (N, 5)
        mag_aper_444  = _arr('mag_aper_f444w')       # (N, 5)
        mag_aper_115  = _arr('mag_aper_f115w')       # (N, 5)
        mag_aper_277  = _arr('mag_aper_f277w')       # (N, 5)
        flux_aper_150 = _arr('flux_aper_f150w')      # (N, 5)
        ferr_aper_150 = _arr('flux_err_aper_f150w')  # (N, 5)
        flux_aper_444 = _arr('flux_aper_f444w')      # (N, 5)
        ferr_aper_444 = _arr('flux_err_aper_f444w')  # (N, 5)
        mag_auto_150  = _arr('mag_auto_f150w')       # (N,)
        mag_auto_444  = _arr('mag_auto_f444w')       # (N,)

        log.info("  %d galaxies", len(ids))

    return (ids, mag_aper_150, mag_aper_444, mag_aper_115, mag_aper_277,
            flux_aper_150, ferr_aper_150, flux_aper_444, ferr_aper_444,
            mag_auto_150, mag_auto_444)


def _quality_mask(flux1, ferr1, flux2, ferr2, snr_min=SNR_MIN):
    """True where both fluxes are positive and above snr_min."""
    ok1 = (flux1 > 0) & (ferr1 > 0) & (flux1 / ferr1 >= snr_min)
    ok2 = (flux2 > 0) & (ferr2 > 0) & (flux2 / ferr2 >= snr_min)
    return ok1 & ok2


def _col(mag1, mag2):
    """Colour = mag1 − mag2, NaN where either is non-finite."""
    c = mag1 - mag2
    c[~np.isfinite(c)] = np.nan
    return c.astype(np.float32)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_aperture_gradients(
    catalog_path: str | Path,
    snr_min:      float = SNR_MIN,
    col_min:      float = COL_MIN,
    col_max:      float = COL_MAX,
) -> dict[str, np.ndarray]:
    """
    Returns a dict of new column arrays (keyed by npz column name).
    All arrays have length N_cat (full catalogue).  Values that fail quality
    cuts are NaN so they are skipped in the explorer's _safe() function.
    """
    catalog_path = Path(catalog_path)
    (ids, mag150, mag444, mag115, mag277,
     fl150, fe150, fl444, fe444,
     mag_auto150, mag_auto444) = _load_cat(catalog_path)

    N = len(ids)
    cols: dict[str, np.ndarray] = {'aper_grad_galaxy_id': ids}

    # ── per-aperture colours ──────────────────────────────────────────────────
    for i, diam in enumerate(APER_DIAMS):
        label = str(diam).replace('.', 'p')   # e.g. "0p3"

        # quality mask for this aperture
        ok_150_444 = _quality_mask(fl150[:, i], fe150[:, i],
                                    fl444[:, i], fe444[:, i], snr_min)

        c = _col(mag150[:, i], mag444[:, i])
        c[~ok_150_444]                      = np.nan
        c[(c < col_min) | (c > col_max)]    = np.nan
        cols[f'col_aper{i}_150_444'] = c
        log.info("col_aper%d_150_444 (%.2f\"): %d finite", i, diam, np.isfinite(c).sum())

        # F115W−F277W
        c2 = _col(mag115[:, i], mag277[:, i])
        c2[~ok_150_444]                     = np.nan   # reuse spatial mask
        c2[(c2 < col_min) | (c2 > col_max)] = np.nan
        cols[f'col_aper{i}_115_277'] = c2

    # ── auto (Kron) colour ────────────────────────────────────────────────────
    c_auto = _col(mag_auto150, mag_auto444)
    c_auto[(c_auto < col_min) | (c_auto > col_max)] = np.nan
    cols['col_auto_150_444'] = c_auto
    log.info("col_auto_150_444: %d finite", np.isfinite(c_auto).sum())

    # ── colour gradients ──────────────────────────────────────────────────────
    # Primary: aper1 (0.3") vs aper4 (1.0")  — avoids undersized F444W core
    for small, large in [(1, 4), (2, 4)]:
        key = f'grad_150_444_{small}v{large}'
        g   = cols[f'col_aper{small}_150_444'] - cols[f'col_aper{large}_150_444']
        g[~np.isfinite(g)] = np.nan
        cols[key] = g
        log.info("%s: %d finite, median=%+.3f",
                 key, np.isfinite(g).sum(),
                 float(np.nanmedian(g)) if np.isfinite(g).any() else np.nan)

    key2 = 'grad_115_277_1v4'
    g2   = cols['col_aper1_115_277'] - cols['col_aper4_115_277']
    g2[~np.isfinite(g2)] = np.nan
    cols[key2] = g2
    log.info("%s: %d finite", key2, np.isfinite(g2).sum())

    return cols


# ---------------------------------------------------------------------------
# Merge into npz
# ---------------------------------------------------------------------------

def merge_into_npz(
    cat_cols:    dict[str, np.ndarray],
    npz_path:    str | Path,
) -> None:
    """
    Match catalogue arrays to the npz galaxy_id and inject columns.
    Unmatched galaxies get NaN.  Overwrites npz in place.
    """
    npz_path = Path(npz_path)
    data     = dict(np.load(npz_path, allow_pickle=True))

    npz_ids = data.get('galaxy_id', data.get('galaxy_ids'))
    if npz_ids is None:
        raise KeyError("npz has neither 'galaxy_id' nor 'galaxy_ids'")
    npz_ids = npz_ids.astype(np.int64)
    N_npz   = len(npz_ids)

    cat_ids  = cat_cols['aper_grad_galaxy_id'].astype(np.int64)
    id2row   = {gid: i for i, gid in enumerate(cat_ids)}

    for col, cat_arr in cat_cols.items():
        if col == 'aper_grad_galaxy_id':
            continue
        out = np.full(N_npz, np.nan, dtype=np.float32)
        for j, gid in enumerate(npz_ids):
            if gid in id2row:
                v = float(cat_arr[id2row[gid]])
                out[j] = v if np.isfinite(v) else np.nan
        data[col] = out
        n_ok = int(np.isfinite(out).sum())
        log.info("  %-30s : %d / %d finite", col, n_ok, N_npz)

    np.savez(str(npz_path.with_suffix('')), **data)
    log.info("Saved → %s", npz_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Compute aperture colour gradients and merge into UMAP npz',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--catalog',  required=True,
                   help='COSMOSWeb_mastercatalog_v1.fits (photometry in HDU 1)')
    p.add_argument('--npz',      required=True,
                   help='UMAP npz file to update in place')
    p.add_argument('--snr_min',  type=float, default=SNR_MIN,
                   help='Min S/N per aperture/band')
    p.add_argument('--col_min',  type=float, default=COL_MIN,
                   help='Min colour [mag] — rejects bad photometry')
    p.add_argument('--col_max',  type=float, default=COL_MAX,
                   help='Max colour [mag]')
    p.add_argument('--log_level', default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING'])
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s %(levelname)s %(message)s')

    cat_cols = compute_aperture_gradients(
        catalog_path=args.catalog,
        snr_min=args.snr_min,
        col_min=args.col_min,
        col_max=args.col_max,
    )
    merge_into_npz(cat_cols, args.npz)


if __name__ == '__main__':
    main()
