"""
Compute rest-frame Balmer-break colour gradients from multi-aperture photometry.

Strategy
--------
The Balmer break sits at ~3646 Å rest-frame.  For each galaxy we select the
observed-frame filter pair whose pivot wavelengths straddle 3646*(1+z):

  z ∈ [1.21, 2.17)  →  HST F814W (8057 Å)  vs  JWST F115W (11543 Å)
  z ∈ [2.17, 3.11)  →  JWST F115W           vs  JWST F150W  (15007 Å)
  z ∈ [3.11, 6.57)  →  JWST F150W           vs  JWST F277W  (27617 Å)
  z outside range   →  NaN

For the chosen pair a positive colour (blue_mag − red_mag) corresponds to a
strong Balmer break (flux drops blueward of the break).

Outputs (per galaxy, NaN where quality cuts fail or z out of range)
-------------------------------------------------------------------
  col_rf_aper{i}       rest-frame colour at aperture i  (i = 0..4)
  grad_rf_1v4          col_rf_aper1 − col_rf_aper4   (0.3" vs 1.0")
  grad_rf_2v4          col_rf_aper2 − col_rf_aper4   (0.5" vs 1.0")
  grad_rf_ann{i}       rest-frame colour in annulus i
  grad_rf_ann_0v4      core annulus − outermost annulus
  grad_rf_ann_0v3      core annulus − 3rd annulus
  rf_pair              integer code for the filter pair used
                         0 → F814W/F115W
                         1 → F115W/F150W
                         2 → F150W/F277W
                        -1 → z out of range

Aperture diameters (same as compute_aperture_gradients.py)
  [0.2", 0.3", 0.5", 0.75", 1.0"]  →  indices 0–4

Usage
-----
  python -m cosmosweb.compute_restframe_gradient \\
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BALMER_BREAK = 3646.0   # Å rest-frame

# Filter pivot wavelengths (Å)
PIVOT = {
    'hst-f814w':  8057.0,
    'f115w':     11543.0,
    'f150w':     15007.0,
    'f277w':     27617.0,
}

# Pairs ordered from lowest to highest redshift
# (blue_key, red_key, z_min, z_max)
# z_min/max defined where BALMER_BREAK*(1+z) falls between the two pivots
PAIRS = [
    ('hst-f814w', 'f115w',
     PIVOT['hst-f814w'] / BALMER_BREAK - 1,   # 1.210
     PIVOT['f115w']     / BALMER_BREAK - 1),   # 2.166
    ('f115w', 'f150w',
     PIVOT['f115w']     / BALMER_BREAK - 1,    # 2.166
     PIVOT['f150w']     / BALMER_BREAK - 1),   # 3.115
    ('f150w', 'f277w',
     PIVOT['f150w']     / BALMER_BREAK - 1,    # 3.115
     PIVOT['f277w']     / BALMER_BREAK - 1),   # 6.575
]

APER_DIAMS     = [0.2, 0.3, 0.5, 0.75, 1.0]   # arcsec, indices 0–4
ANNULUS_LABELS = ['0–0.1"', '0.1–0.15"', '0.15–0.25"', '0.25–0.375"', '0.375–0.5"']

SNR_MIN = 3.0
COL_MIN = -5.0
COL_MAX =  5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cat(catalog_path: Path) -> tuple:
    """Load photometry (HDU1) and photo-z (HDU2).

    Returns
    -------
    ids           (N,)   int64
    fluxes        dict  filter_key → (N, 5) float64
    ferrs         dict  filter_key → (N, 5) float64
    zfinal        (N,)  float64
    """
    log.info("Opening %s", catalog_path)
    with fits.open(catalog_path, memmap=True) as hdul:
        log.info("HDUs: %s", [h.name for h in hdul])

        ph = hdul[1].data
        lp = hdul[2].data

        ids = np.array(ph['id'], dtype=np.int64)
        N   = len(ids)
        log.info("  %d galaxies", N)

        def _arr5(col):
            return np.array(ph[col], dtype=np.float64)   # (N, 5)

        fluxes, ferrs = {}, {}
        for fkey in ('hst-f814w', 'f115w', 'f150w', 'f277w'):
            fluxes[fkey] = _arr5(f'flux_aper_{fkey}')
            ferrs[fkey]  = _arr5(f'flux_err_aper_{fkey}')

        zfinal = np.array(lp['zfinal'], dtype=np.float64)

    return ids, fluxes, ferrs, zfinal


def _quality_mask(f1, fe1, f2, fe2, snr_min=SNR_MIN):
    """True where both fluxes are positive and above snr_min."""
    ok1 = (f1 > 0) & (fe1 > 0) & (f1 / fe1 >= snr_min)
    ok2 = (f2 > 0) & (fe2 > 0) & (f2 / fe2 >= snr_min)
    return ok1 & ok2


def _flux_to_col(f_blue, f_red, col_min=COL_MIN, col_max=COL_MAX):
    """Colour = −2.5 log10(f_blue / f_red); NaN outside [col_min, col_max]."""
    with np.errstate(divide='ignore', invalid='ignore'):
        c = -2.5 * np.log10(f_blue / f_red)
    c = np.where(np.isfinite(c), c, np.nan)
    c = np.where((c >= col_min) & (c <= col_max), c, np.nan)
    return c.astype(np.float32)


def _annular_fluxes(flux_aper, ferr_aper):
    """Cumulative → annular.  f_ann[:,i] = f_aper[:,i] - f_aper[:,i-1]."""
    N, K = flux_aper.shape
    f_ann    = np.empty((N, K), dtype=np.float64)
    ferr_ann = np.empty((N, K), dtype=np.float64)
    f_ann[:, 0]    = flux_aper[:, 0]
    ferr_ann[:, 0] = ferr_aper[:, 0]
    for i in range(1, K):
        f_ann[:, i]    = flux_aper[:, i] - flux_aper[:, i - 1]
        ferr_ann[:, i] = np.sqrt(ferr_aper[:, i]**2 + ferr_aper[:, i - 1]**2)
    return f_ann, ferr_ann


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_restframe_gradient(
    catalog_path: str | Path,
    snr_min:      float = SNR_MIN,
    col_min:      float = COL_MIN,
    col_max:      float = COL_MAX,
) -> dict[str, np.ndarray]:
    """
    Returns a dict of new column arrays keyed by npz column name.
    All arrays have length N.  Values failing quality cuts are NaN.
    """
    catalog_path = Path(catalog_path)
    ids, fluxes, ferrs, zfinal = _load_cat(catalog_path)
    N = len(ids)

    # Output arrays (NaN by default)
    col_rf   = np.full((N, 5), np.nan, dtype=np.float32)   # per aperture
    col_rf_a = np.full((N, 5), np.nan, dtype=np.float32)   # per annulus
    rf_pair  = np.full(N, -1,  dtype=np.int8)

    # Pre-compute annular fluxes for all filter keys
    ann_flux = {k: _annular_fluxes(fluxes[k], ferrs[k]) for k in fluxes}

    for pair_idx, (blue, red, z_lo, z_hi) in enumerate(PAIRS):
        mask = (zfinal >= z_lo) & (zfinal < z_hi) & np.isfinite(zfinal)
        n_in = mask.sum()
        log.info("Pair %d (%s/%s)  z=[%.2f,%.2f)  → %d galaxies",
                 pair_idx, blue, red, z_lo, z_hi, n_in)
        if n_in == 0:
            continue

        rf_pair[mask] = pair_idx

        fb = fluxes[blue][mask]    # (n_in, 5)
        fr = fluxes[red][mask]
        eb = ferrs[blue][mask]
        er = ferrs[red][mask]

        # ── per-aperture colour ───────────────────────────────────────────
        for i in range(5):
            ok = _quality_mask(fb[:, i], eb[:, i], fr[:, i], er[:, i], snr_min)
            c  = _flux_to_col(fb[:, i], fr[:, i], col_min, col_max)
            c[~ok] = np.nan
            col_rf[mask, i] = c

        # ── per-annulus colour ────────────────────────────────────────────
        f_ann_b, fe_ann_b = ann_flux[blue]
        f_ann_r, fe_ann_r = ann_flux[red]
        fab = f_ann_b[mask]
        far = f_ann_r[mask]
        eab = fe_ann_b[mask]
        ear = fe_ann_r[mask]
        for i in range(5):
            ok = _quality_mask(fab[:, i], eab[:, i], far[:, i], ear[:, i], snr_min)
            c  = _flux_to_col(fab[:, i], far[:, i], col_min, col_max)
            c[~ok] = np.nan
            col_rf_a[mask, i] = c

    # ── pack into dict ────────────────────────────────────────────────────
    cols: dict[str, np.ndarray] = {'rf_grad_galaxy_id': ids}
    cols['rf_pair'] = rf_pair.astype(np.float32)   # int8 → float so npz stays uniform

    for i in range(5):
        cols[f'col_rf_aper{i}'] = col_rf[:, i]
        n = int(np.isfinite(col_rf[:, i]).sum())
        log.info("col_rf_aper%d: %d finite", i, n)

    for i in range(5):
        cols[f'col_rf_ann{i}'] = col_rf_a[:, i]

    for small, large in [(1, 4), (2, 4)]:
        key = f'grad_rf_{small}v{large}'
        g   = col_rf[:, small] - col_rf[:, large]
        g   = np.where(np.isfinite(g), g, np.nan)
        cols[key] = g.astype(np.float32)
        log.info("%s: %d finite, median=%+.3f", key,
                 int(np.isfinite(g).sum()),
                 float(np.nanmedian(g)) if np.isfinite(g).any() else np.nan)

    for outer in (4, 3):
        key = f'grad_rf_ann_0v{outer}'
        g   = col_rf_a[:, 0] - col_rf_a[:, outer]
        g   = np.where(np.isfinite(g), g, np.nan)
        cols[key] = g.astype(np.float32)
        log.info("%s: %d finite, median=%+.3f", key,
                 int(np.isfinite(g).sum()),
                 float(np.nanmedian(g)) if np.isfinite(g).any() else np.nan)

    return cols


# ---------------------------------------------------------------------------
# Merge into npz
# ---------------------------------------------------------------------------

def merge_into_npz(cat_cols: dict[str, np.ndarray], npz_path: str | Path) -> None:
    npz_path = Path(npz_path)
    data     = dict(np.load(npz_path, allow_pickle=True))

    npz_ids = data.get('galaxy_id', data.get('galaxy_ids'))
    if npz_ids is None:
        raise KeyError("npz has neither 'galaxy_id' nor 'galaxy_ids'")
    npz_ids = npz_ids.astype(np.int64)
    N_npz   = len(npz_ids)

    cat_ids = cat_cols['rf_grad_galaxy_id'].astype(np.int64)
    id2row  = {gid: i for i, gid in enumerate(cat_ids)}

    for col, cat_arr in cat_cols.items():
        if col == 'rf_grad_galaxy_id':
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
        description='Compute rest-frame Balmer-break colour gradients',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--catalog',   required=True,
                   help='COSMOSWeb_mastercatalog_v1.fits (photometry HDU1, LePhare HDU2)')
    p.add_argument('--npz',       required=True,
                   help='UMAP npz file to update in place')
    p.add_argument('--snr_min',   type=float, default=SNR_MIN)
    p.add_argument('--col_min',   type=float, default=COL_MIN)
    p.add_argument('--col_max',   type=float, default=COL_MAX)
    p.add_argument('--log_level', default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING'])
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s %(levelname)s %(message)s')

    cat_cols = compute_restframe_gradient(
        catalog_path=args.catalog,
        snr_min=args.snr_min,
        col_min=args.col_min,
        col_max=args.col_max,
    )
    merge_into_npz(cat_cols, args.npz)


if __name__ == '__main__':
    main()
