"""
Prepare the paired COSMOS-Web image + CIGALE SFH dataset.

For each galaxy selected from the visual morphology catalog:
  1. Cut a 64x64 stamp in F150W, F277W, F444W from the NIRCam tile mosaics
  2. Interpolate the CIGALE SFH to a common log-spaced lookback-time grid
  3. Store everything in an HDF5 file ready for training

Output HDF5 layout:
    /images         (N, 3, 64, 64)  float32  arcsinh-stretched flux
    /sfh            (N, N_TIME)     float32  log10(SFR + EPS) on common grid
    /sfh_time_grid  (N_TIME,)       float32  lookback time in Myr
    /galaxy_id      (N,)            int64
    /ra             (N,)            float64
    /dec            (N,)            float64
    /redshift       (N,)            float32
    attrs: img_mean, img_std  (3,) per-channel normalization stats

Usage:
    python prepare_dataset.py \\
        --morpho_db  /n07data/ilbert/COSMOS-Web/photoz_MASTER_v3.1.0/MORPHO/visualmorpho_COSMOSWeb_v7.db \\
        --master_cat /n23data2/cosmosweb-public/DR1/data/COSMOSWeb_mastercatalog_v1.fits \\
        --img_dir    /n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/ \\
        --output     cosmosweb_dataset.h5

Notes:
  - The master catalog (COSMOS2025 DR1) is a multi-extension FITS file.
    Photometry is in HDU 1, LePhare photo-z in HDU 2, CIGALE SFH in HDU 4.
    CIGALE SFH columns: sfh_sfr_bin1..9, sfh_sfr_bin{i}_err, sfh_time_bin1..9,
    sfh_integrated.  Galaxy ID column: 'id'.
  - Images must be the v0.8 30-mas NIRCam mosaics from Shuntov et al.
"""

import sys
import sqlite3
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cigale'))
from SFHandle.sfh import SFH

# ── constants ─────────────────────────────────────────────────────────────────

FILTERS     = ['f150w', 'f277w', 'f444w']
STAMP_SIZE  = 64
SFH_N_BINS  = 50
SFH_EPS     = 1e-10  # avoids log10(0) for quiescent galaxies
SFH_TIME_GRID = np.logspace(np.log10(10), np.log10(14_000), SFH_N_BINS)  # Myr

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)


# ── catalog helpers ────────────────────────────────────────────────────────────

def load_morpho_db(db_path: str) -> pd.DataFrame:
    """Read the visual morphology SQLite DB; return galaxies passing basic QC."""
    con = sqlite3.connect(db_path)
    # Read native column names directly from the SQLite schema
    cursor = con.execute('SELECT * FROM morphology LIMIT 0')
    db_col_names = [desc[0] for desc in cursor.description]
    df  = pd.read_sql_query('SELECT * FROM morphology', con)
    con.close()

    # The DB already carries column names; only rename if they are positional
    # integers (older DB versions that lack a schema).  Otherwise keep as-is.
    if df.columns[0] == 0 or str(df.columns[0]).isdigit():
        known = [
            'id',
            'FAKE', 'PHOTOMETRY_OFF', 'SERSIC_OFF', 'SUBCOMPONENT', 'BLENDED',
            'TOO_FAINT', 'TOO_SMALL', 'UNCERTAIN', 'BRIGHT_FOREGROUND',
            'ELL_REGULAR', 'ELL_INTER', 'ELL_DISTURB',
            'S0_REGULAR', 'S0_INTER', 'S0_DISTURB',
            'EDISK_REGULAR', 'EDISK_INTER', 'EDISK_DISTURB',
            'LDISK_REGULAR', 'LDISK_INTER', 'LDISK_DISTURB',
            'EDGE_ON', 'ASYMETRY', 'ARMS', 'BAR', 'LSB_DISK',
            'CLUMP', 'MANY_CLUMPS', 'IS_A_CLUMP', 'CHAIN', 'COMPACT',
            'IRR', 'POINT_LIKE', 'POWERLAW',
            'MINOR_MERGER', 'MINOR_CLOSE', 'MINOR_PAIR',
            'MAJOR_MERGER', 'MAJOR_CLOSE', 'MAJOR_PAIR',
            'IS_SMALL_COMPANION', 'CONSISTENT_Z', 'DRY', 'REMNANT',
            'LENS', 'GROUPE', 'INFO', 'ADDI', 'VERSION',
        ]
        # Pad with generic names for any extra columns beyond the known list
        padded = known + [f'_COL{i}' for i in range(len(known), len(df.columns))]
        df.columns = padded
    else:
        log.info(f'Morpho DB native columns: {list(df.columns)}')

    log.info(f'Morpho DB: {len(df)} total columns={len(df.columns)}')

    qc_cols = ['FAKE', 'TOO_FAINT', 'TOO_SMALL', 'BRIGHT_FOREGROUND', 'POINT_LIKE']
    missing = [c for c in qc_cols if c not in df.columns]
    if missing:
        log.warning(f'QC columns not found (skipping): {missing}')
        qc_cols = [c for c in qc_cols if c in df.columns]

    if qc_cols:
        keep = (df[qc_cols] == 0).all(axis=1)
    else:
        keep = pd.Series(True, index=df.index)

    log.info(f'  {keep.sum()} / {len(df)} pass QC cuts')
    return df[keep].copy()


def _fits_to_df(fits_path: str, hdu_index: int = 1, cols: list | None = None) -> pd.DataFrame:
    """Read a single HDU from a FITS file into a DataFrame, keeping only scalar columns."""
    with fits.open(fits_path, memmap=True) as hdu:
        tbl = Table(hdu[hdu_index].data)
    scalar_cols = [c for c in tbl.colnames if len(tbl[c].shape) <= 1]
    if cols is not None:
        scalar_cols = [c for c in cols if c in scalar_cols]
    return tbl[scalar_cols].to_pandas()


def load_photom(photom_path: str) -> pd.DataFrame:
    """Load photometry + quality flags from the primary photometry FITS file."""
    want = ['id', 'tile', 'ra', 'dec', 'radius_sersic',
            'flag_star', 'flag_blend', 'warn_flag']
    df = _fits_to_df(photom_path, cols=want)
    log.info(f'Photom catalog: {len(df)} rows')
    return df


def load_lephare(lephare_path: str) -> pd.DataFrame:
    """Load LePhare photo-z (row-aligned with photom, no id column)."""
    want = ['zfinal', 'zpdf_l68', 'zpdf_u68']
    df = _fits_to_df(lephare_path, cols=want)
    log.info(f'LePhare catalog: {len(df)} rows')
    return df


def load_cigale(cigale_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load CIGALE SFH data and auto-detect the column naming convention.

    COSMOS2025 DR1 may use either:
      - 'sfh_sfr_bin{i}' / 'sfh_time_bin{i}' / 'sfh_integrated'   (DR1 style)
      - 'bayes.sfh.sfr_bin{i}' / 'bayes.sfh.time_bin{i}' / 'bayes.sfh.integrated'
        (raw CIGALE output style)

    Returns the DataFrame and a dict with the resolved column-name lists.
    """
    df = _fits_to_df(cigale_path)
    cols = set(df.columns)
    log.info(f'CIGALE catalog: {len(df)} rows, {len(cols)} columns')

    # ── detect naming convention ──────────────────────────────────────────────
    if f'sfh_sfr_bin1' in cols:
        sfr_cols  = [f'sfh_sfr_bin{i}'     for i in range(1, 10)]
        err_cols  = [f'sfh_sfr_bin{i}_err' for i in range(1, 10)]
        time_cols = [f'sfh_time_bin{i}'    for i in range(1, 10)]
        int_col   = 'sfh_integrated'
        log.info('  Using DR1-style column names  (sfh_sfr_bin{i})')
    elif 'bayes.sfh.sfr_bin1' in cols:
        sfr_cols  = [f'bayes.sfh.sfr_bin{i}'     for i in range(1, 10)]
        err_cols  = [f'bayes.sfh.sfr_bin{i}_err' for i in range(1, 10)]
        time_cols = [f'bayes.sfh.time_bin{i}'    for i in range(1, 10)]
        int_col   = 'bayes.sfh.integrated'
        log.info('  Using raw-CIGALE column names  (bayes.sfh.sfr_bin{i})')
    else:
        # Print a sample so the user can identify the correct names
        sfr_like = [c for c in cols if 'sfr' in c.lower() and 'bin' in c.lower()]
        log.error(
            f'Cannot detect SFH column naming.  '
            f'Columns containing "sfr" and "bin": {sorted(sfr_like)[:20]}\n'
            f'First 40 column names: {sorted(cols)[:40]}'
        )
        raise ValueError('Unknown CIGALE column naming convention; see log above.')

    missing = [c for c in sfr_cols + err_cols + time_cols + [int_col] if c not in cols]
    if missing:
        raise ValueError(f'Expected CIGALE columns not found: {missing}')

    col_map = dict(sfr=sfr_cols, err=err_cols, time=time_cols, integrated=int_col)
    return df, col_map


# ── SFH helpers ────────────────────────────────────────────────────────────────

# Populated at runtime after load_cigale() resolves the naming convention
_COL_MAP: dict = {}


def sfh_to_common_grid(row: pd.Series) -> np.ndarray | None:
    """
    Extract and interpolate one galaxy's CIGALE SFH to SFH_TIME_GRID.

    Returns log10(SFR + SFH_EPS) array of shape (SFH_N_BINS,), or None if
    the row does not contain valid SFH data.  Requires _COL_MAP to be set.
    """
    int_col = _COL_MAP['integrated']
    integrated = float(row.get(int_col, np.nan))
    if not (np.isfinite(integrated) and integrated > 0):
        return None

    lb_time = np.array([row[c] for c in _COL_MAP['time']], dtype=np.float64)
    sfr     = np.array([row[c] for c in _COL_MAP['sfr']],  dtype=np.float64) * integrated
    err     = np.array([row[c] for c in _COL_MAP['err']],  dtype=np.float64) * integrated

    if not (np.all(np.isfinite(lb_time)) and np.all(np.isfinite(sfr))):
        return None

    sfh_obj       = SFH(lb_time, sfr, err)
    sfr_interp, _ = sfh_obj.interpolate_sfh(
        SFH_TIME_GRID, kind='next', bounds_error=False, fill_value=0.0
    )
    return np.log10(sfr_interp + SFH_EPS).astype(np.float32)


# ── image helpers ──────────────────────────────────────────────────────────────

def _img_path(img_dir: str, filt: str, tile: str) -> Path:
    return (Path(img_dir) /
            f'mosaic_nircam_{filt}_COSMOS-Web_30mas_{tile}_v0_8_sci.fits')


def cut_stamp(
    data: np.ndarray, wcs: WCS, ra: float, dec: float, size: int
) -> np.ndarray | None:
    """
    Cut a (size × size) stamp centred on (ra, dec).

    Applies arcsinh stretch and replaces non-finite values with 0.
    Returns float32 array or None if the cutout falls outside the image.
    """
    try:
        pix = wcs.wcs_world2pix([[ra, dec]], 0)[0]
        if not np.all(np.isfinite(pix)):
            return None
        cut   = Cutout2D(data, pix, size=size, wcs=wcs, mode='strict')
        stamp = np.array(cut.data, dtype=np.float32)          # copy out of memmap
        if stamp.dtype.byteorder == '>':                       # big-endian FITS
            stamp = stamp.byteswap().newbyteorder()
        stamp = np.arcsinh(stamp)
        stamp = np.nan_to_num(stamp, nan=0.0, posinf=0.0, neginf=0.0)
        return stamp
    except Exception:
        return None


# ── main ───────────────────────────────────────────────────────────────────────

_CAT_DIR = '/n23data2/cosmosweb/catalogs/DR1/data/catalog'


def main() -> None:
    global _COL_MAP

    parser = argparse.ArgumentParser(description='Build CosmosWeb image+SFH HDF5 dataset')
    parser.add_argument('--morpho_db',
                        default='/n07data/ilbert/COSMOS-Web/photoz_MASTER_v3.1.0/MORPHO/visualmorpho_COSMOSWeb_v7.db',
                        help='Path to visualmorpho_COSMOSWeb_v7.db')
    parser.add_argument('--photom_cat',
                        default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_photom_primary.fits',
                        help='Photometry FITS file (primary)')
    parser.add_argument('--lephare_cat',
                        default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_lephare.fits',
                        help='LePhare photo-z FITS file')
    parser.add_argument('--cigale_cat',
                        default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_cigale.fits',
                        help='CIGALE SFH FITS file')
    parser.add_argument('--img_dir',
                        default='/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/',
                        help='Directory containing NIRCam tile mosaics')
    parser.add_argument('--output',     default='cosmosweb_dataset.h5')
    parser.add_argument('--stamp_size', type=int, default=STAMP_SIZE)
    args = parser.parse_args()

    # ── load & merge catalogs ────────────────────────────────────────────────
    log.info('Loading visual morphology DB…')
    df_morpho   = load_morpho_db(args.morpho_db)
    morpho_ids  = set(df_morpho['id'])

    # The three FITS catalogs are row-aligned (same 784016 rows, same order).
    # Only the photom file carries 'id'; lephare and cigale have no id column.
    # → concat by position, then filter.
    log.info('Loading photometry catalog…')
    df_phot   = load_photom(args.photom_cat)

    log.info('Loading LePhare catalog…')
    df_lp     = load_lephare(args.lephare_cat)

    log.info('Loading CIGALE catalog…')
    df_cigale, _COL_MAP = load_cigale(args.cigale_cat)

    # Positional concat — all three have the same number of rows
    df_all = pd.concat(
        [df_phot, df_lp, df_cigale],
        axis=1,
    ).reset_index(drop=True)
    log.info(f'Combined catalog: {len(df_all)} rows')

    # Quality cuts
    keep = (
        (df_all['flag_star'] == 0) &
        (df_all['warn_flag'] == 0) &
        df_all['ra'].notna() &
        df_all['tile'].notna()
    )
    df_all = df_all[keep].copy()
    log.info(f'After photometric quality cuts: {len(df_all)}')

    # Morpho cross-match (id is from the photom file)
    df_merged = df_all[df_all['id'].isin(morpho_ids)].copy()
    log.info(f'After morpho cross-match: {len(df_merged)} galaxies')

    # Drop rows where any SFH column is null
    sfh_cols = _COL_MAP['sfr'] + _COL_MAP['time'] + [_COL_MAP['integrated']]
    has_sfh  = df_merged[sfh_cols].notna().all(axis=1)
    df_merged = df_merged[has_sfh].copy()
    log.info(f'After SFH completeness filter: {len(df_merged)} galaxies')

    if len(df_merged) == 0:
        log.error('No galaxies remain; check catalog paths and column names.')
        return

    # ── create HDF5 ─────────────────────────────────────────────────────────
    sz = args.stamp_size
    with h5py.File(args.output, 'w') as f:
        n_max   = len(df_merged)
        kw      = dict(maxshape=(None,), chunks=True)
        img_kw  = dict(maxshape=(None, 3, sz, sz), chunks=(64, 3, sz, sz))

        ds_img  = f.create_dataset('images',    shape=(n_max, 3, sz, sz), dtype='f4', **img_kw)
        ds_sfh  = f.create_dataset('sfh',       shape=(n_max, SFH_N_BINS), dtype='f4',
                                   maxshape=(None, SFH_N_BINS), chunks=(256, SFH_N_BINS))
        ds_id   = f.create_dataset('galaxy_id', shape=(n_max,), dtype='i8', **kw)
        ds_ra   = f.create_dataset('ra',        shape=(n_max,), dtype='f8', **kw)
        ds_dec  = f.create_dataset('dec',       shape=(n_max,), dtype='f8', **kw)
        ds_z    = f.create_dataset('redshift',  shape=(n_max,), dtype='f4', **kw)
        f.create_dataset('sfh_time_grid', data=SFH_TIME_GRID.astype(np.float32))

        f.attrs['filters']    = ','.join(FILTERS)
        f.attrs['stamp_size'] = sz

        written = 0

        # ── tile loop ────────────────────────────────────────────────────────
        for tile_name, tile_df in df_merged.groupby('tile'):
            log.info(f'Tile {tile_name}: {len(tile_df)} candidates…')

            paths = {filt: _img_path(args.img_dir, filt, tile_name)
                     for filt in FILTERS}
            missing = [str(p) for p in paths.values() if not p.exists()]
            if missing:
                log.warning(f'  Skipping tile {tile_name}: missing {missing}')
                continue

            # Open all 3 FITS files; keep memmap open for the whole tile
            hduls  = {filt: fits.open(paths[filt], memmap=True) for filt in FILTERS}
            try:
                arrays = {filt: hduls[filt][1].data             for filt in FILTERS}
                wcss   = {filt: WCS(hduls[filt][1].header, naxis=2)
                          for filt in FILTERS}

                tile_ok = 0
                for _, row in tile_df.iterrows():
                    # ── SFH ──────────────────────────────────────────────────
                    sfh_vec = sfh_to_common_grid(row)
                    if sfh_vec is None:
                        continue

                    # ── stamps ───────────────────────────────────────────────
                    stamps, ok = [], True
                    for filt in FILTERS:
                        stamp = cut_stamp(
                            arrays[filt], wcss[filt],
                            row['ra'], row['dec'], sz
                        )
                        if stamp is None:
                            ok = False
                            break
                        stamps.append(stamp)
                    if not ok:
                        continue

                    ds_img[written]  = np.stack(stamps, axis=0)    # (3, sz, sz)
                    ds_sfh[written]  = sfh_vec
                    ds_id[written]   = int(row['id'])
                    ds_ra[written]   = float(row['ra'])
                    ds_dec[written]  = float(row['dec'])
                    ds_z[written]    = float(row.get('zfinal', -1.0))
                    written += 1
                    tile_ok += 1

                log.info(f'  → {tile_ok} galaxies written')

            finally:
                for hdul in hduls.values():
                    hdul.close()

        # ── truncate datasets to actual size ──────────────────────────────
        for ds in [ds_img, ds_sfh, ds_id, ds_ra, ds_dec, ds_z]:
            ds.resize(written, axis=0)

        f.attrs['n_galaxies'] = written
        log.info(f'Wrote {written} galaxies total')

        # ── compute per-channel normalization stats ────────────────────────
        if written > 0:
            log.info('Computing per-channel image statistics…')
            imgs      = f['images'][:]               # (N, 3, sz, sz)
            img_mean  = imgs.mean(axis=(0, 2, 3))    # (3,)
            img_std   = imgs.std (axis=(0, 2, 3))    # (3,)
            img_std[img_std == 0] = 1.0
            f.attrs['img_mean'] = img_mean
            f.attrs['img_std']  = img_std
            log.info(f'  mean={img_mean}  std={img_std}')

    log.info(f'Dataset saved to {args.output}')


if __name__ == '__main__':
    main()
