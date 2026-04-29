"""
Compute rest-frame NUV-r colour gradients from bulge+disk decompositions.

Strategy
--------
For each galaxy at redshift z we interpolate apparent magnitudes across the
available multi-band grid to the observed wavelengths that correspond to
rest-frame NUV (230 nm) and rest-frame r (620 nm):

    λ_obs = λ_rest × (1 + z)

Then:
    NUVr_bulge  = m_bulge(λ_NUV_obs) − m_bulge(λ_r_obs)
    NUVr_disk   = m_disk(λ_NUV_obs)  − m_disk(λ_r_obs)
    delta_NUVr  = NUVr_bulge − NUVr_disk

Because we subtract the two components *at the same observed wavelengths*,
the K-correction and any common photometric zero-point offset cancel exactly.
Apparent magnitudes are therefore correct inputs — no absolute magnitudes needed.

ID matching
-----------
The B+D catalogue has no id column; rows are in the same order as the
LePhare catalogue.  We read id / zfinal from LePhare by row position.

Input catalogues
----------------
  COSMOSWeb_mastercatalog_v1_bulgedisk.fits  (standalone, primary HDU)
    Columns include:
      ra_detec_bd, dec_detec_bd
      bulge_radius_deg, disk_radius_deg
      fmf_b+d_chi2
      BT_jwst  (or similar)
      mag_model_bulge_{band}, mag_model_disk_{band}
        where {band} ∈ {cfht-u, hsc-g, hsc-r, …, hst-f814w, f115w, …, f444w}

  COSMOSWeb_mastercatalog_v1_lephare.fits  (standalone LePhare catalogue)
    Columns: id, zfinal, mabs_NUV, mabs_r, …
    Same row ordering as the B+D catalogue.

Quality cuts
------------
  1. fmf_b+d_chi2 < chi2_max        (default 5.0)
  2. BT in [BT_min, BT_max]          (default [0.05, 0.95])
  3. bulge_radius_deg < disk_radius_deg
  4. Both rest-frame interpolations finite

Output columns
--------------
  id, z, BT, Re_bulge_deg, Re_disk_deg,
  NUVr_bulge, NUVr_disk, delta_NUVr,
  band_lo_nuv, band_hi_nuv, band_lo_r, band_hi_r,
  flag_good

Usage
-----
  python -m cosmosweb.compute_color_gradients \\
      --bd_catalog      /n23data2/cosmosweb/catalogs/DR1/data/catalog/COSMOSWeb_mastercatalog_v1_bulgedisk.fits \\
      --lephare_catalog /n23data2/cosmosweb/catalogs/DR1/data/catalog/COSMOSWeb_mastercatalog_v1_lephare.fits \\
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
# Known band pivot wavelengths [nm]  (catalog suffix → wavelength)
# ---------------------------------------------------------------------------
_BAND_LAMBDA: dict[str, float] = {
    'cfht-u':      380.0,
    'sc-ib427':    427.0,
    'hsc-g':       480.0,
    'sc-ia484':    484.0,
    'sc-ib505':    505.0,
    'sc-ia527':    527.0,
    'sc-ib574':    574.0,
    'hsc-r':       620.0,
    'sc-ia624':    624.0,
    'sc-ia679':    679.0,
    'sc-nb711':    711.0,
    'sc-ib709':    709.0,
    'hsc-i':       770.0,
    'sc-ia738':    738.0,
    'sc-ia767':    767.0,
    'sc-nb816':    816.0,
    'hst-f814w':   814.0,
    'sc-ib827':    827.0,
    'hsc-z':       890.0,
    'hsc-y':       970.0,
    'uvista-y':   1020.0,
    'f115w':      1150.0,
    'uvista-j':   1250.0,
    'f150w':      1500.0,
    'uvista-h':   1650.0,
    'uvista-ks':  2150.0,
    'f277w':      2770.0,
    'f444w':      4440.0,
    'f770w':      7700.0,
}

# Rest-frame pivot wavelengths [nm]
LAMBDA_NUV = 230.0
LAMBDA_R   = 620.0


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


def _detect_bands(colnames: list[str], component: str) -> list[tuple[str, float]]:
    """
    Find all mag_model_{component}_{band} columns that have a known wavelength.
    Returns list of (band_suffix, wavelength_nm) sorted by wavelength.
    """
    prefix = f'mag_model_{component}_'
    found  = []
    for c in colnames:
        if c.startswith(prefix):
            suffix = c[len(prefix):]
            if suffix in _BAND_LAMBDA:
                found.append((suffix, _BAND_LAMBDA[suffix]))
            else:
                log.debug("  Unknown band suffix '%s' — skipped", suffix)
    found.sort(key=lambda x: x[1])
    return found


def _interp_per_galaxy(
    z_arr:       np.ndarray,       # (N,)
    lam_rest:    float,            # nm
    band_names:  list[str],
    band_lambda: np.ndarray,       # (B,) sorted
    mag_mat:     np.ndarray,       # (N, B)
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    For each galaxy interpolate the apparent magnitude at λ_rest*(1+z).
    Returns (mag_interp (N,), band_lo_list, band_hi_list).
    """
    N = len(z_arr)
    mag_out  = np.full(N, np.nan, dtype=np.float32)
    blo_list = [''] * N
    bhi_list = [''] * N

    for i in range(N):
        if not np.isfinite(z_arr[i]):
            continue
        lam_obs = lam_rest * (1.0 + z_arr[i])
        idx_hi  = int(np.searchsorted(band_lambda, lam_obs))
        if idx_hi == 0 or idx_hi >= len(band_lambda):
            continue                          # outside coverage
        idx_lo = idx_hi - 1
        m_lo   = mag_mat[i, idx_lo]
        m_hi   = mag_mat[i, idx_hi]
        if not (np.isfinite(m_lo) and np.isfinite(m_hi)):
            continue
        w          = (lam_obs - band_lambda[idx_lo]) / (band_lambda[idx_hi] - band_lambda[idx_lo])
        mag_out[i] = (1.0 - w) * m_lo + w * m_hi
        blo_list[i] = band_names[idx_lo]
        bhi_list[i] = band_names[idx_hi]

    return mag_out, blo_list, bhi_list


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_gradients(
    bd_catalog:      str | Path,
    lephare_catalog: str | Path,
    chi2_max:        float = 5.0,
    BT_min:          float = 0.05,
    BT_max:          float = 0.95,
) -> Table:
    bd_catalog      = Path(bd_catalog)
    lephare_catalog = Path(lephare_catalog)

    log.info("Reading B+D catalogue: %s", bd_catalog)
    bd = _read_primary(bd_catalog)
    log.info("  %d rows; %d columns", len(bd), len(bd.colnames))

    log.info("Reading LePhare catalogue: %s", lephare_catalog)
    lp = _read_primary(lephare_catalog)
    log.info("  %d rows; first columns: %s", len(lp), lp.colnames[:10])

    N = min(len(bd), len(lp))
    if len(bd) != len(lp):
        log.warning("Row count mismatch: B+D=%d, LePhare=%d — using first %d",
                    len(bd), len(lp), N)
    bd = bd[:N]
    lp = lp[:N]

    # ── id and redshift from LePhare (positional match) ──────────────────────
    id_col = next((c for c in lp.colnames if c.lower() == 'id'), None)
    if id_col is None:
        raise ValueError(f"No 'id' column in LePhare. Columns: {lp.colnames[:30]}")
    ids = np.array(lp[id_col], dtype=np.int64)

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

    # ── auto-detect available bands ───────────────────────────────────────────
    bulge_bands = _detect_bands(bd.colnames, 'bulge')
    disk_bands  = _detect_bands(bd.colnames, 'disk')
    # Keep only bands present in both
    bulge_names = {b[0] for b in bulge_bands}
    disk_names  = {b[0] for b in disk_bands}
    common      = bulge_names & disk_names
    bands       = [(b, lam) for b, lam in bulge_bands if b in common]
    log.info("Common B+D bands (%d): %s", len(bands), [b[0] for b in bands])

    if len(bands) < 2:
        raise RuntimeError("Fewer than 2 bands in common between bulge and disk — cannot interpolate")

    band_names  = [b[0] for b in bands]
    band_lambda = np.array([b[1] for b in bands])

    # Build magnitude matrices (N, B)
    mag_bulge = np.column_stack([_mag_array(bd, f'mag_model_bulge_{b}') for b in band_names])
    mag_disk  = np.column_stack([_mag_array(bd, f'mag_model_disk_{b}')  for b in band_names])

    log.info("Interpolating rest-frame NUV (%.0f nm) …", LAMBDA_NUV)
    m_bulge_nuv, blo_nuv, bhi_nuv = _interp_per_galaxy(
        z, LAMBDA_NUV, band_names, band_lambda, mag_bulge)
    m_disk_nuv, _, _ = _interp_per_galaxy(
        z, LAMBDA_NUV, band_names, band_lambda, mag_disk)

    log.info("Interpolating rest-frame r (%.0f nm) …", LAMBDA_R)
    m_bulge_r, blo_r, bhi_r = _interp_per_galaxy(
        z, LAMBDA_R, band_names, band_lambda, mag_bulge)
    m_disk_r, _, _ = _interp_per_galaxy(
        z, LAMBDA_R, band_names, band_lambda, mag_disk)

    # ── colour gradients ──────────────────────────────────────────────────────
    # K-correction cancels: delta_NUVr = (m_bulge_NUV - m_bulge_r)
    #                                  - (m_disk_NUV  - m_disk_r)
    NUVr_bulge = m_bulge_nuv - m_bulge_r
    NUVr_disk  = m_disk_nuv  - m_disk_r
    delta_NUVr = NUVr_bulge  - NUVr_disk

    # ── quality flags ─────────────────────────────────────────────────────────
    flag_chi2   = chi2 < chi2_max
    flag_BT     = (BT >= BT_min) & (BT <= BT_max) if bt_col \
                  else np.ones(N, dtype=bool)
    flag_size   = Re_bulge < Re_disk
    flag_z      = np.isfinite(z) & (z > 0)
    flag_interp = np.isfinite(NUVr_bulge) & np.isfinite(NUVr_disk)
    flag_good   = flag_chi2 & flag_BT & flag_size & flag_z & flag_interp

    log.info("Quality cuts → %d / %d pass (%.1f %%)",
             flag_good.sum(), N, 100.0 * flag_good.sum() / max(N, 1))
    log.info("  chi2 < %.1f          : %d", chi2_max,  flag_chi2.sum())
    log.info("  BT in [%.2f, %.2f]   : %d", BT_min, BT_max, flag_BT.sum())
    log.info("  Re_bulge < Re_disk   : %d", flag_size.sum())
    log.info("  valid z              : %d", flag_z.sum())
    log.info("  both colours finite  : %d", flag_interp.sum())

    # ── output table ──────────────────────────────────────────────────────────
    out = Table({
        'id':           ids,
        'z':            z,
        'BT':           BT,
        'Re_bulge_deg': Re_bulge,
        'Re_disk_deg':  Re_disk,
        'NUVr_bulge':   NUVr_bulge,
        'NUVr_disk':    NUVr_disk,
        'delta_NUVr':   delta_NUVr,
        'band_lo_nuv':  np.array(blo_nuv, dtype='U32'),
        'band_hi_nuv':  np.array(bhi_nuv, dtype='U32'),
        'band_lo_r':    np.array(blo_r,   dtype='U32'),
        'band_hi_r':    np.array(bhi_r,   dtype='U32'),
        'flag_good':    flag_good,
    })
    return out


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

    for col in ['delta_NUVr', 'NUVr_bulge', 'NUVr_disk', 'BT']:
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
                   help='Standalone LePhare FITS file (id + zfinal, same row order as B+D)')
    p.add_argument('--output',          required=True,
                   help='Output FITS table path')
    p.add_argument('--chi2_max',  type=float, default=5.0)
    p.add_argument('--BT_min',    type=float, default=0.05)
    p.add_argument('--BT_max',    type=float, default=0.95)
    p.add_argument('--merge_npz', default=None,
                   help='If given, merge results into this npz file')
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
        chi2_max=args.chi2_max,
        BT_min=args.BT_min,
        BT_max=args.BT_max,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grad.write(str(out_path), overwrite=True)
    log.info("Wrote %d rows → %s", len(grad), out_path)

    if args.merge_npz:
        log.info("Merging into %s", args.merge_npz)
        merge_with_npz(grad, args.merge_npz)


if __name__ == '__main__':
    main()
