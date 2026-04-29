"""
Compute rest-frame NUV-r colour gradients from bulge+disk decompositions.

Strategy
--------
For each galaxy at redshift z we need:
  - NUV-r of the *bulge* component (rest-frame)
  - NUV-r of the *disk*  component (rest-frame)
  - delta_NUVr = NUVr_bulge − NUVr_disk  (colour gradient proxy)

All three are computed by interpolating the multi-band bulge/disk model
magnitudes onto the observed wavelength that corresponds to the rest-frame
NUV pivot (230 nm) and rest-frame r pivot (620 nm).

Input catalogues
----------------
  COSMOSWeb_mastercatalog_v1_bulgedisk.fits   (standalone, no HDU extension)
    Primary HDU: id, mag_model_bulge_<band>, mag_model_disk_<band>,
                 Re_bulge, Re_disk (deg), BT_jwst, chi2, …

  COSMOSWeb_mastercatalog_v1.fits  (multi-extension master)
    HDU 1  : photometry  (id, ra, dec, …)
    HDU 2  : LePhare     (id, zfinal, mabs_NUV, mabs_r, …)

Band pivot wavelengths [nm]
---------------------------
  cfht-u       380
  sc-ib427     427
  hsc-g        480
  sc-ia484     484
  sc-ib505     505
  sc-ia527     527
  sc-ib574     574
  sc-ia624     624
  hsc-r        620
  sc-ia679     679
  sc-ib709     709
  sc-ia738     738
  sc-ia767     767
  sc-ib827     827
  hst-f814w    814
  hsc-i        770
  sc-nb711     711
  sc-nb816     816
  hsc-z        890
  hsc-y        970
  uvista-y    1020
  uvista-j    1250
  F115W       1150
  uvista-h    1650
  F150W       1500
  uvista-ks   2150
  F277W       2770
  F444W       4440
  F770W       7700

Quality cuts
------------
  1. B+D chi2 < chi2_max (default 5.0)
  2. B/T in [BT_min, BT_max]  (default [0.05, 0.95])
  3. Re_bulge < Re_disk  (unphysical if bulge larger than disk)
  4. Both target bands (NUV, r) must be covered by at least 2 catalogue bands
     at the galaxy's redshift

Output
------
  An astropy Table (FITS or ecsv) with columns:
    id, z, BT_jwst,
    NUVr_bulge, NUVr_disk, delta_NUVr,
    band_lo_nuv, band_hi_nuv, band_lo_r, band_hi_r,   ← which bands interpolated
    flag_good                                           ← True if all cuts passed

Usage
-----
  python -m cosmosweb.compute_color_gradients \\
      --bd_catalog     /n03data/huertas/python/AstroCLIP/COSMOSWeb_mastercatalog_v1_bulgedisk.fits \\
      --master_catalog /n23data2/cosmosweb-public/DR1/data/COSMOSWeb_mastercatalog_v1.fits \\
      --output         /n03data/huertas/COSMOS-Web/cosmosweb_clip/color_gradients.fits \\
      --chi2_max       5.0 \\
      --BT_min         0.05 \\
      --BT_max         0.95

  # To cross-match and merge with existing npz:
  python -m cosmosweb.compute_color_gradients --merge_npz <path>.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, join

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Band pivot wavelengths [nm], ordered by wavelength
# ---------------------------------------------------------------------------
BANDS: list[tuple[str, float]] = sorted([
    ("cfht-u",      380.0),
    ("sc-ib427",    427.0),
    ("hsc-g",       480.0),
    ("sc-ia484",    484.0),
    ("sc-ib505",    505.0),
    ("sc-ia527",    527.0),
    ("sc-ib574",    574.0),
    ("hsc-r",       620.0),
    ("sc-ia624",    624.0),
    ("sc-ia679",    679.0),
    ("sc-nb711",    711.0),
    ("sc-ib709",    709.0),
    ("hsc-i",       770.0),
    ("sc-ia738",    738.0),
    ("sc-ia767",    767.0),
    ("sc-nb816",    816.0),
    ("hst-f814w",   814.0),
    ("sc-ib827",    827.0),
    ("hsc-z",       890.0),
    ("hsc-y",       970.0),
    ("uvista-y",   1020.0),
    ("F115W",      1150.0),
    ("uvista-j",   1250.0),
    ("F150W",      1500.0),
    ("uvista-h",   1650.0),
    ("uvista-ks",  2150.0),
    ("F277W",      2770.0),
    ("F444W",      4440.0),
    ("F770W",      7700.0),
], key=lambda x: x[1])

BAND_NAMES  = [b[0] for b in BANDS]
BAND_LAMBDA = np.array([b[1] for b in BANDS])  # nm

# Rest-frame pivot wavelengths [nm]
LAMBDA_NUV = 230.0
LAMBDA_R   = 620.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mag_col(component: str, band: str) -> str:
    """Return column name for a bulge/disk magnitude."""
    return f"mag_model_{component}_{band}"


def _interp_mag(
    lam_target: float,
    lam_obs: float,              # = lam_target * (1 + z)
    band_names: list[str],
    band_lambdas: np.ndarray,
    mag_arr: np.ndarray,         # (N, n_bands) – NaN where not available
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each galaxy interpolate the magnitude at `lam_obs` [nm] from the
    two nearest catalogue bands.

    Returns
    -------
    mag_interp : (N,)  float32, NaN if not enough coverage
    band_lo    : (N,)  str  name of the shorter-wavelength bracketing band
    band_hi    : (N,)  str  name of the longer-wavelength bracketing band
    """
    N = mag_arr.shape[0]
    mag_interp = np.full(N, np.nan, dtype=np.float32)
    band_lo    = np.full(N, "", dtype=object)
    band_hi    = np.full(N, "", dtype=object)

    # Find bracketing band indices for this single observed wavelength
    # (same for all galaxies at the same z, but we vectorise over galaxies
    #  because mag availability differs per galaxy)
    idx_hi = np.searchsorted(band_lambdas, lam_obs)   # first band >= lam_obs

    if idx_hi == 0 or idx_hi >= len(band_lambdas):
        # Outside band coverage for all galaxies
        return mag_interp, band_lo, band_hi

    idx_lo = idx_hi - 1
    l_lo   = band_lambdas[idx_lo]
    l_hi   = band_lambdas[idx_hi]
    w      = (lam_obs - l_lo) / (l_hi - l_lo)   # interpolation weight

    m_lo = mag_arr[:, idx_lo]
    m_hi = mag_arr[:, idx_hi]

    ok = np.isfinite(m_lo) & np.isfinite(m_hi)
    mag_interp[ok] = (1.0 - w) * m_lo[ok] + w * m_hi[ok]
    band_lo[ok]    = band_names[idx_lo]
    band_hi[ok]    = band_names[idx_hi]

    return mag_interp, band_lo, band_hi


def _per_galaxy_interp(
    z_arr: np.ndarray,
    lam_rest: float,
    band_names: list[str],
    band_lambdas: np.ndarray,
    mag_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-galaxy interpolation: each galaxy has its own observed wavelength
    lam_obs_i = lam_rest * (1 + z_i).

    Returns mag_interp (N,), band_lo (N,), band_hi (N,).
    """
    N = len(z_arr)
    mag_interp = np.full(N, np.nan, dtype=np.float32)
    band_lo    = np.full(N, "", dtype=object)
    band_hi    = np.full(N, "", dtype=object)

    lam_obs_all = lam_rest * (1.0 + z_arr)

    for i in range(N):
        lam_obs = lam_obs_all[i]
        idx_hi  = np.searchsorted(band_lambdas, lam_obs)
        if idx_hi == 0 or idx_hi >= len(band_lambdas):
            continue
        idx_lo = idx_hi - 1
        l_lo   = band_lambdas[idx_lo]
        l_hi   = band_lambdas[idx_hi]
        m_lo   = mag_arr[i, idx_lo]
        m_hi   = mag_arr[i, idx_hi]
        if not (np.isfinite(m_lo) and np.isfinite(m_hi)):
            continue
        w              = (lam_obs - l_lo) / (l_hi - l_lo)
        mag_interp[i]  = (1.0 - w) * m_lo + w * m_hi
        band_lo[i]     = band_names[idx_lo]
        band_hi[i]     = band_names[idx_hi]

    return mag_interp, band_lo, band_hi


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_gradients(
    bd_catalog:     str | Path,
    master_catalog: str | Path,
    chi2_max:       float = 5.0,
    BT_min:         float = 0.05,
    BT_max:         float = 0.95,
) -> Table:
    """
    Read the B+D catalogue and master catalogue, apply quality cuts, compute
    colour gradients.

    Parameters
    ----------
    bd_catalog     : standalone B+D FITS file (primary HDU), e.g.
                     COSMOSWeb_mastercatalog_v1_bulgedisk.fits
    master_catalog : multi-extension master catalogue; LePhare is in HDU 2
                     (for redshifts and optional absolute magnitudes)

    Returns an astropy Table with one row per *catalogue* galaxy (all B+D
    sources), with NaN for those that failed quality cuts.
    """
    bd_catalog     = Path(bd_catalog)
    master_catalog = Path(master_catalog)

    log.info("Opening B+D catalogue: %s", bd_catalog)
    with fits.open(bd_catalog, memmap=True) as hdul:
        log.info("HDUs: %s", [h.name for h in hdul])
        # Primary extension may be 0 (empty) or 1; pick the first with data
        bd_hdu = next(h for h in hdul if h.data is not None)
        bd     = Table(bd_hdu.data)
        log.info("B+D catalogue: %d rows, first columns: %s",
                 len(bd), bd.colnames[:20])

    log.info("Opening master catalogue (LePhare): %s", master_catalog)
    with fits.open(master_catalog, memmap=True) as hdul:
        log.info("HDUs: %s", [h.name for h in hdul])
        # ── LePhare photo-z (HDU 2) ──────────────────────────────────────────
        lp = Table(hdul[2].data)
        log.info("LePhare catalogue: %d rows", len(lp))

    # Identify id column (may be 'id' or 'ID')
    id_col_bd = next(c for c in bd.colnames if c.lower() == 'id')
    id_col_lp = next(c for c in lp.colnames if c.lower() == 'id')

    # Keep only LePhare columns we need
    lp_keep = [id_col_lp, 'zfinal']
    for c in ['mabs_NUV', 'mabs_r']:
        if c in lp.colnames:
            lp_keep.append(c)
    lp_sub  = lp[lp_keep]
    lp_sub.rename_column(id_col_lp, 'id')

    bd.rename_column(id_col_bd, 'id')

    # Cross-match B+D × LePhare on id
    cat = join(bd, lp_sub, keys='id', join_type='left')
    N   = len(cat)
    log.info("After join: %d rows", N)

    ids = np.array(cat['id'])
    z   = np.array(cat['zfinal'], dtype=np.float32)
    z   = np.where(np.isfinite(z) & (z > 0), z, np.nan)

    # ── B/T and size ────────────────────────────────────────────────────────
    # B/T column: prefer BT_jwst (from JWST bands); it may be per-band or
    # a single value – detect the column name.
    bt_col = None
    for c in ['BT_jwst', 'BT_F150W', 'BT_F277W', 'BT']:
        if c in cat.colnames:
            bt_col = c
            break
    if bt_col is None:
        log.warning("No B/T column found; skipping B/T quality cuts")
        BT = np.full(N, 0.5, dtype=np.float32)
    else:
        BT = np.array(cat[bt_col], dtype=np.float32)

    # Re columns (degrees)
    re_bulge_col = next((c for c in cat.colnames
                         if 'Re_bulge' in c or 're_bulge' in c.lower()), None)
    re_disk_col  = next((c for c in cat.colnames
                         if 'Re_disk'  in c or 're_disk'  in c.lower()), None)

    if re_bulge_col and re_disk_col:
        Re_bulge = np.array(cat[re_bulge_col], dtype=np.float32)
        Re_disk  = np.array(cat[re_disk_col],  dtype=np.float32)
    else:
        log.warning("Re_bulge/Re_disk columns not found; skipping size cut")
        Re_bulge = np.zeros(N, dtype=np.float32)
        Re_disk  = np.ones(N,  dtype=np.float32)

    # chi2 column
    chi2_col = next((c for c in cat.colnames
                     if 'chi2' in c.lower() and ('bd' in c.lower()
                                                  or 'b+d' in c.lower()
                                                  or 'fmf' in c.lower()
                                                  or 'fit' in c.lower())), None)
    if chi2_col is None:
        # Fall back to any chi2 column
        chi2_col = next((c for c in cat.colnames if 'chi2' in c.lower()), None)

    if chi2_col:
        chi2 = np.array(cat[chi2_col], dtype=np.float32)
        log.info("Using chi2 column: %s", chi2_col)
    else:
        log.warning("No chi2 column found; skipping chi2 cut")
        chi2 = np.zeros(N, dtype=np.float32)

    # ── Build mag arrays for each component ─────────────────────────────────
    # Only keep bands that exist in the catalogue
    avail_bands  = []
    avail_lambda = []
    for bname, blam in BANDS:
        bulge_c = _mag_col("bulge", bname)
        disk_c  = _mag_col("disk",  bname)
        if bulge_c in cat.colnames and disk_c in cat.colnames:
            avail_bands.append(bname)
            avail_lambda.append(blam)

    avail_lambda = np.array(avail_lambda)
    n_bands      = len(avail_bands)
    log.info("Available B+D bands (%d): %s", n_bands, avail_bands)

    mag_bulge = np.full((N, n_bands), np.nan, dtype=np.float32)
    mag_disk  = np.full((N, n_bands), np.nan, dtype=np.float32)

    for j, bname in enumerate(avail_bands):
        bc = _mag_col("bulge", bname)
        dc = _mag_col("disk",  bname)
        mag_bulge[:, j] = np.where(
            np.isfinite(np.array(cat[bc], dtype=np.float32)),
            np.array(cat[bc], dtype=np.float32), np.nan)
        mag_disk[:, j]  = np.where(
            np.isfinite(np.array(cat[dc], dtype=np.float32)),
            np.array(cat[dc], dtype=np.float32), np.nan)

    # ── Per-galaxy interpolation ─────────────────────────────────────────────
    log.info("Interpolating NUV magnitudes …")
    m_bulge_nuv, blo_nuv, bhi_nuv = _per_galaxy_interp(
        z, LAMBDA_NUV, avail_bands, avail_lambda, mag_bulge)
    m_disk_nuv, _, _ = _per_galaxy_interp(
        z, LAMBDA_NUV, avail_bands, avail_lambda, mag_disk)

    log.info("Interpolating r magnitudes …")
    m_bulge_r, blo_r, bhi_r = _per_galaxy_interp(
        z, LAMBDA_R, avail_bands, avail_lambda, mag_bulge)
    m_disk_r, _, _ = _per_galaxy_interp(
        z, LAMBDA_R, avail_bands, avail_lambda, mag_disk)

    # ── Colour gradients ────────────────────────────────────────────────────
    NUVr_bulge = m_bulge_nuv - m_bulge_r
    NUVr_disk  = m_disk_nuv  - m_disk_r
    delta_NUVr = NUVr_bulge  - NUVr_disk

    # ── Quality flags ────────────────────────────────────────────────────────
    flag_chi2   = chi2     < chi2_max
    flag_BT     = (BT >= BT_min) & (BT <= BT_max)
    flag_size   = Re_bulge < Re_disk          # exclude Re_bulge >= Re_disk
    flag_z      = np.isfinite(z) & (z > 0)
    flag_interp = (np.isfinite(NUVr_bulge) &
                   np.isfinite(NUVr_disk))

    if chi2_col is None:
        flag_chi2[:] = True
    if re_bulge_col is None:
        flag_size[:] = True

    flag_good = flag_chi2 & flag_BT & flag_size & flag_z & flag_interp

    n_good = flag_good.sum()
    log.info("Quality cuts: %d / %d galaxies pass (%.1f %%)",
             n_good, N, 100.0 * n_good / max(N, 1))
    log.info("  chi2 < %.1f : %d", chi2_max, flag_chi2.sum())
    log.info("  BT in [%.2f, %.2f] : %d", BT_min, BT_max, flag_BT.sum())
    log.info("  Re_bulge < Re_disk : %d", flag_size.sum())
    log.info("  valid redshift : %d", flag_z.sum())
    log.info("  both colours finite : %d", flag_interp.sum())

    # ── Build output table ───────────────────────────────────────────────────
    out = Table({
        'id':           ids,
        'z':            z,
        'BT':           BT,
        'Re_bulge_deg': Re_bulge,
        'Re_disk_deg':  Re_disk,
        'NUVr_bulge':   NUVr_bulge,
        'NUVr_disk':    NUVr_disk,
        'delta_NUVr':   delta_NUVr,
        'band_lo_nuv':  blo_nuv.astype(str),
        'band_hi_nuv':  bhi_nuv.astype(str),
        'band_lo_r':    blo_r.astype(str),
        'band_hi_r':    bhi_r.astype(str),
        'flag_good':    flag_good,
    })

    if 'mabs_NUV' in cat.colnames and 'mabs_r' in cat.colnames:
        out['NUVr_total_lephare'] = (
            np.array(cat['mabs_NUV'], dtype=np.float32) -
            np.array(cat['mabs_r'],   dtype=np.float32))

    return out


# ---------------------------------------------------------------------------
# Merge into existing npz
# ---------------------------------------------------------------------------

def merge_with_npz(grad_table: Table, npz_path: str | Path) -> None:
    """
    Add delta_NUVr, NUVr_bulge, NUVr_disk to an existing UMAP npz file.
    Matches on galaxy 'id' column; fills NaN for unmatched galaxies.
    """
    npz_path = Path(npz_path)
    data = dict(np.load(npz_path, allow_pickle=True))

    npz_ids = data['galaxy_id'].astype(np.int64)
    N_npz   = len(npz_ids)

    grad_id  = np.array(grad_table['id'], dtype=np.int64)
    id2idx   = {gid: i for i, gid in enumerate(grad_id)}

    for col in ['delta_NUVr', 'NUVr_bulge', 'NUVr_disk', 'BT']:
        arr = np.full(N_npz, np.nan, dtype=np.float32)
        for j, gid in enumerate(npz_ids):
            if gid in id2idx:
                row = id2idx[gid]
                if grad_table['flag_good'][row]:
                    val = grad_table[col][row]
                    arr[j] = float(val) if np.isfinite(float(val)) else np.nan
        data[col] = arr
        log.info("Added '%s': %d / %d finite values",
                 col, np.isfinite(arr).sum(), N_npz)

    # Also add flag_good
    flag_arr = np.zeros(N_npz, dtype=bool)
    for j, gid in enumerate(npz_ids):
        if gid in id2idx:
            flag_arr[j] = bool(grad_table['flag_good'][id2idx[gid]])
    data['bd_flag_good'] = flag_arr

    np.savez(npz_path.with_suffix('') , **data)
    log.info("Saved updated npz to %s", npz_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute rest-frame NUV-r colour gradients from B+D catalogue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--bd_catalog',     required=True,
                   help='Standalone B+D FITS file (e.g. COSMOSWeb_mastercatalog_v1_bulgedisk.fits)')
    p.add_argument('--master_catalog', required=True,
                   help='Multi-extension master catalogue (LePhare redshifts in HDU 2)')
    p.add_argument('--output',         required=True,
                   help='Output FITS table path')
    p.add_argument('--chi2_max',   type=float, default=5.0,
                   help='Max B+D chi2 for quality cut')
    p.add_argument('--BT_min',     type=float, default=0.05,
                   help='Min B/T for quality cut')
    p.add_argument('--BT_max',     type=float, default=0.95,
                   help='Max B/T for quality cut')
    p.add_argument('--merge_npz',  default=None,
                   help='If given, merge gradients into this npz file')
    p.add_argument('--log_level',  default='INFO',
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
        master_catalog=args.master_catalog,
        chi2_max=args.chi2_max,
        BT_min=args.BT_min,
        BT_max=args.BT_max,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grad.write(str(out_path), overwrite=True)
    log.info("Wrote %d rows to %s", len(grad), out_path)

    if args.merge_npz:
        merge_with_npz(grad, args.merge_npz)


if __name__ == '__main__':
    main()
