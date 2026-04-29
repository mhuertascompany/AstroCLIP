"""
Compute JWST colour gradients from bulge+disk decompositions.

Strategy
--------
The B+D catalogue contains per-component magnitudes in four JWST bands:
  F115W (1150 nm), F150W (1500 nm), F277W (2770 nm), F444W (4440 nm)

We compute the observed-frame colour gradient:
  col_bulge  = mag_model_bulge_f115w − mag_model_bulge_f277w
  col_disk   = mag_model_disk_f115w  − mag_model_disk_f277w
  delta_col  = col_bulge − col_disk

A positive delta_col means the bulge is bluer in F115W−F277W than the disk
(relative to the disk colour), which at z~0.5–3 corresponds to roughly
rest-frame optical–NIR.  We also save F150W−F444W as a second pair for
consistency checks.

ID matching
-----------
The B+D catalogue has no id column; rows are in the same order as the
LePhare catalogue (and master photometry).  We read the LePhare id/zfinal
by row position and assign them directly.

Input catalogues
----------------
  COSMOSWeb_mastercatalog_v1_bulgedisk.fits  (standalone, no HDU extension)
    Primary HDU: ra_detec_bd, dec_detec_bd, bulge_radius_deg, disk_radius_deg,
                 fmf_b+d_chi2, mag_model_bulge_f{115,150,277,444}w,
                 mag_model_disk_f{115,150,277,444}w, BT_jwst, …

  COSMOSWeb_mastercatalog_v1_lephare.fits  (standalone LePhare catalogue)
    Primary HDU: id, zfinal, mabs_NUV, mabs_r, …
    (Same row ordering as the B+D catalogue)

Quality cuts
------------
  1. fmf_b+d_chi2 < chi2_max  (default 5.0)
  2. BT in [BT_min, BT_max]   (default [0.05, 0.95])
  3. bulge_radius_deg < disk_radius_deg  (exclude unphysical Re_bulge ≥ Re_disk)
  4. All four component magnitudes finite

Output
------
  FITS table with columns:
    id, z, BT,
    col_bulge_115_277, col_disk_115_277, delta_col_115_277,
    col_bulge_150_444, col_disk_150_444, delta_col_150_444,
    flag_good

Usage
-----
  python -m cosmosweb.compute_color_gradients \\
      --bd_catalog     /n23data2/cosmosweb/catalogs/DR1/data/catalog/COSMOSWeb_mastercatalog_v1_bulgedisk.fits \\
      --lephare_catalog /n23data2/cosmosweb/catalogs/DR1/data/catalog/COSMOSWeb_mastercatalog_v1_lephare.fits \\
      --output         /n03data/huertas/COSMOS-Web/cosmosweb_clip/color_gradients.fits \\
      --merge_npz      /n03data/huertas/COSMOS-Web/cosmosweb_clip/cosmosweb_umap_zoobot_v2.npz
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
    """Read the first HDU with data from a FITS file."""
    with fits.open(path, memmap=True) as hdul:
        log.info("  HDUs in %s: %s", path.name, [h.name for h in hdul])
        hdu = next(h for h in hdul if h.data is not None)
        t   = Table(hdu.data)
    return t


def _mag(cat: Table, col: str) -> np.ndarray:
    """Return magnitude column as float32, with non-finite values → NaN."""
    arr = np.array(cat[col], dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return arr


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
    """
    Apply quality cuts and compute per-component JWST colour gradients.

    Row ordering in bd_catalog matches lephare_catalog — id is taken from
    LePhare by position.

    Returns an astropy Table (one row per B+D source).
    """
    bd_catalog      = Path(bd_catalog)
    lephare_catalog = Path(lephare_catalog)

    log.info("Reading B+D catalogue: %s", bd_catalog)
    bd = _read_primary(bd_catalog)
    log.info("  %d rows; first columns: %s", len(bd), bd.colnames[:15])

    log.info("Reading LePhare catalogue: %s", lephare_catalog)
    lp = _read_primary(lephare_catalog)
    log.info("  %d rows; first columns: %s", len(lp), lp.colnames[:15])

    N_bd = len(bd)
    N_lp = len(lp)
    if N_bd != N_lp:
        log.warning(
            "Row count mismatch: B+D=%d, LePhare=%d — truncating to min",
            N_bd, N_lp)
    N = min(N_bd, N_lp)
    bd = bd[:N]
    lp = lp[:N]

    # ── id and redshift from LePhare ─────────────────────────────────────────
    id_col = next((c for c in lp.colnames if c.lower() == 'id'), None)
    if id_col is None:
        raise ValueError(
            f"No 'id' column in LePhare catalogue. Columns: {lp.colnames[:30]}")
    ids = np.array(lp[id_col], dtype=np.int64)

    z_col = next((c for c in lp.colnames
                  if c.lower() in ('zfinal', 'z_phot', 'z')), None)
    if z_col is None:
        raise ValueError(
            f"No redshift column in LePhare catalogue. Columns: {lp.colnames[:30]}")
    z = np.array(lp[z_col], dtype=np.float32)
    z = np.where(np.isfinite(z) & (z > 0), z, np.nan)
    log.info("Redshift column: '%s', finite z: %d / %d",
             z_col, np.isfinite(z).sum(), N)

    # ── quality columns from B+D ──────────────────────────────────────────────
    chi2 = _mag(bd, 'fmf_b+d_chi2') if 'fmf_b+d_chi2' in bd.colnames else np.zeros(N)

    bt_col = next((c for c in bd.colnames
                   if c.lower() in ('bt_jwst', 'bt_f277w', 'bt_f150w', 'bt')), None)
    BT = _mag(bd, bt_col) if bt_col else np.full(N, 0.5, dtype=np.float32)
    if bt_col:
        log.info("B/T column: '%s'", bt_col)
    else:
        log.warning("No B/T column found; B/T quality cut skipped")

    Re_bulge = _mag(bd, 'bulge_radius_deg') if 'bulge_radius_deg' in bd.colnames \
               else np.zeros(N, dtype=np.float32)
    Re_disk  = _mag(bd, 'disk_radius_deg')  if 'disk_radius_deg'  in bd.colnames \
               else np.ones(N, dtype=np.float32)

    # ── per-component magnitudes ──────────────────────────────────────────────
    def _get(component: str, band: str) -> np.ndarray:
        col = f'mag_model_{component}_{band}'
        if col not in bd.colnames:
            log.warning("Column '%s' not found", col)
            return np.full(N, np.nan, dtype=np.float32)
        return _mag(bd, col)

    bulge_f115 = _get('bulge', 'f115w')
    bulge_f150 = _get('bulge', 'f150w')
    bulge_f277 = _get('bulge', 'f277w')
    bulge_f444 = _get('bulge', 'f444w')
    disk_f115  = _get('disk',  'f115w')
    disk_f150  = _get('disk',  'f150w')
    disk_f277  = _get('disk',  'f277w')
    disk_f444  = _get('disk',  'f444w')

    # ── colour gradients ──────────────────────────────────────────────────────
    # Pair 1: F115W − F277W  (traces optical vs NIR at z~0.5–3)
    col_bulge_115_277 = bulge_f115 - bulge_f277
    col_disk_115_277  = disk_f115  - disk_f277
    delta_col_115_277 = col_bulge_115_277 - col_disk_115_277

    # Pair 2: F150W − F444W  (independent check, same rest-frame range)
    col_bulge_150_444 = bulge_f150 - bulge_f444
    col_disk_150_444  = disk_f150  - disk_f444
    delta_col_150_444 = col_bulge_150_444 - col_disk_150_444

    # ── quality flags ─────────────────────────────────────────────────────────
    flag_chi2   = (chi2 < chi2_max) if bt_col else np.ones(N, dtype=bool)
    flag_BT     = (BT >= BT_min) & (BT <= BT_max) if bt_col \
                  else np.ones(N, dtype=bool)
    flag_size   = Re_bulge < Re_disk
    flag_z      = np.isfinite(z) & (z > 0)
    flag_mags   = (np.isfinite(bulge_f115) & np.isfinite(bulge_f277) &
                   np.isfinite(disk_f115)  & np.isfinite(disk_f277))

    flag_good = flag_chi2 & flag_BT & flag_size & flag_z & flag_mags

    log.info("Quality cuts → %d / %d pass (%.1f %%)",
             flag_good.sum(), N, 100.0 * flag_good.sum() / max(N, 1))
    log.info("  chi2 < %.1f          : %d", chi2_max,  flag_chi2.sum())
    log.info("  BT in [%.2f, %.2f]   : %d", BT_min, BT_max, flag_BT.sum())
    log.info("  Re_bulge < Re_disk   : %d", flag_size.sum())
    log.info("  valid z              : %d", flag_z.sum())
    log.info("  all mags finite      : %d", flag_mags.sum())

    # ── output table ──────────────────────────────────────────────────────────
    out = Table({
        'id':                 ids,
        'z':                  z,
        'BT':                 BT,
        'Re_bulge_deg':       Re_bulge,
        'Re_disk_deg':        Re_disk,
        'col_bulge_115_277':  col_bulge_115_277,
        'col_disk_115_277':   col_disk_115_277,
        'delta_col_115_277':  delta_col_115_277,
        'col_bulge_150_444':  col_bulge_150_444,
        'col_disk_150_444':   col_disk_150_444,
        'delta_col_150_444':  delta_col_150_444,
        'flag_good':          flag_good,
    })
    return out


# ---------------------------------------------------------------------------
# Merge into existing npz
# ---------------------------------------------------------------------------

def merge_with_npz(grad_table: Table, npz_path: str | Path) -> None:
    """
    Inject colour gradient columns into an existing UMAP npz file.
    Matches on galaxy id; fills NaN for unmatched or quality-cut-failed rows.
    """
    npz_path = Path(npz_path)
    data     = dict(np.load(npz_path, allow_pickle=True))

    # npz may store ids under 'galaxy_id' or 'galaxy_ids'
    npz_ids = data.get('galaxy_id', data.get('galaxy_ids'))
    if npz_ids is None:
        raise KeyError("npz has neither 'galaxy_id' nor 'galaxy_ids'")
    npz_ids = npz_ids.astype(np.int64)
    N_npz   = len(npz_ids)

    grad_id = np.array(grad_table['id'], dtype=np.int64)
    id2idx  = {gid: i for i, gid in enumerate(grad_id)}

    new_cols = ['delta_col_115_277', 'col_bulge_115_277', 'col_disk_115_277',
                'delta_col_150_444', 'col_bulge_150_444', 'col_disk_150_444',
                'BT']
    for col in new_cols:
        arr = np.full(N_npz, np.nan, dtype=np.float32)
        for j, gid in enumerate(npz_ids):
            if gid in id2idx:
                row = id2idx[gid]
                if grad_table['flag_good'][row]:
                    val = float(grad_table[col][row])
                    arr[j] = val if np.isfinite(val) else np.nan
        data[col] = arr
        log.info("  %-28s : %d / %d finite", col, np.isfinite(arr).sum(), N_npz)

    flag_arr = np.zeros(N_npz, dtype=bool)
    for j, gid in enumerate(npz_ids):
        if gid in id2idx:
            flag_arr[j] = bool(grad_table['flag_good'][id2idx[gid]])
    data['bd_flag_good'] = flag_arr

    np.savez(str(npz_path.with_suffix('')), **data)
    log.info("Saved updated npz → %s", npz_path)


# ---------------------------------------------------------------------------
# explore.py colour keys produced by this script
# ---------------------------------------------------------------------------
# After merging, the npz will contain:
#   delta_col_115_277  → 'ΔF115W−F277W (bulge − disk)'
#   col_bulge_115_277  → 'F115W−F277W bulge'
#   col_disk_115_277   → 'F115W−F277W disk'
#   delta_col_150_444  → 'ΔF150W−F444W (bulge − disk)'
#   BT                 → 'B/T ratio'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute JWST colour gradients from B+D decompositions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--bd_catalog',      required=True,
                   help='Standalone B+D FITS file')
    p.add_argument('--lephare_catalog', required=True,
                   help='Standalone LePhare FITS file (for id and zfinal)')
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
        log.info("Merging into npz: %s", args.merge_npz)
        merge_with_npz(grad, args.merge_npz)


if __name__ == '__main__':
    main()
