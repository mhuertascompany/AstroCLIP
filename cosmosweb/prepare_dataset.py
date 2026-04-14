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
    df  = pd.read_sql_query('SELECT * FROM morphology', con)
    con.close()

    col_names = [
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
    df.columns = col_names[:len(df.columns)]

    keep = (
        (df['FAKE']             == 0) &
        (df['TOO_FAINT']        == 0) &
        (df['TOO_SMALL']        == 0) &
        (df['BRIGHT_FOREGROUND']== 0) &
        (df['POINT_LIKE']       == 0)
    )
    log.info(f'Morpho DB: {len(df)} total, {keep.sum()} pass QC cuts')
    return df[keep].copy()


def load_master_catalog(fits_path: str) -> pd.DataFrame:
    """
    Load position, tile, photo-z, and CIGALE SFH from the COSMOS2025 master catalog.

    HDU layout (COSMOSWeb_mastercatalog_v1.fits):
        HDU 1 – photometry  (id, tile, ra, dec, flags, …)
        HDU 2 – LePhare     (zfinal, …)
        HDU 4 – CIGALE      (sfh_sfr_bin1..9, sfh_time_bin1..9, sfh_integrated, …)
    """
    photom_cols = ['id', 'tile', 'ra', 'dec', 'radius_sersic',
                   'flag_star', 'flag_blend', 'warn_flag']
    lephare_cols = ['zfinal', 'zpdf_l68', 'zpdf_u68']
    cigale_cols  = (
        [f'sfh_sfr_bin{i}'     for i in range(1, 10)] +
        [f'sfh_sfr_bin{i}_err' for i in range(1, 10)] +
        [f'sfh_time_bin{i}'    for i in range(1, 10)] +
        ['sfh_integrated', 'mass', 'sfr_100myr', 'sfr_inst']
    )

    def _read(hdu_data, cols):
        tbl   = Table(hdu_data)
        valid = [c for c in cols if c in tbl.colnames and len(tbl[c].shape) <= 1]
        return tbl[valid].to_pandas()

    with fits.open(fits_path, memmap=True) as hdu:
        phot    = _read(hdu[1].data, photom_cols)
        lp      = _read(hdu[2].data, lephare_cols)
        cigale  = _read(hdu[4].data, cigale_cols)

    df = pd.concat(
        [phot.reset_index(drop=True),
         lp.reset_index(drop=True),
         cigale.reset_index(drop=True)],
        axis=1,
    )

    # Basic photometric quality cuts
    keep = (
        (df['flag_star']  == 0) &
        (df['warn_flag']  == 0) &
        (df['ra'].notna()) &
        (df['tile'].notna())
    )
    log.info(f'Master catalog: {len(df)} total, {keep.sum()} pass photometric cuts')
    return df[keep].copy()


# ── SFH helpers ────────────────────────────────────────────────────────────────

_SFR_COLS  = [f'sfh_sfr_bin{i}'     for i in range(1, 10)]
_TIME_COLS = [f'sfh_time_bin{i}'    for i in range(1, 10)]
_ERR_COLS  = [f'sfh_sfr_bin{i}_err' for i in range(1, 10)]
_INT_COL   = 'sfh_integrated'


def sfh_to_common_grid(row: pd.Series) -> np.ndarray | None:
    """
    Extract and interpolate one galaxy's CIGALE SFH to SFH_TIME_GRID.

    Returns log10(SFR + SFH_EPS) array of shape (SFH_N_BINS,), or None if the
    row does not contain valid SFH data.
    """
    required = _SFR_COLS + _TIME_COLS + [_INT_COL]
    if not all(c in row.index for c in required):
        return None

    integrated = float(row[_INT_COL])
    if not (np.isfinite(integrated) and integrated > 0):
        return None

    lb_time = np.array([row[c] for c in _TIME_COLS], dtype=np.float64)
    sfr     = np.array([row[c] for c in _SFR_COLS],  dtype=np.float64) * integrated
    err     = np.array([row[c] for c in _ERR_COLS],  dtype=np.float64) * integrated

    if not (np.all(np.isfinite(lb_time)) and np.all(np.isfinite(sfr))):
        return None

    sfh_obj           = SFH(lb_time, sfr, err)
    sfr_interp, _     = sfh_obj.interpolate_sfh(
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

def main() -> None:
    parser = argparse.ArgumentParser(description='Build CosmosWeb image+SFH HDF5 dataset')
    parser.add_argument('--morpho_db',  required=True,
                        help='Path to visualmorpho_COSMOSWeb_v7.db')
    parser.add_argument('--master_cat',
                        default='/n23data2/cosmosweb-public/DR1/data/COSMOSWeb_mastercatalog_v1.fits',
                        help='Path to COSMOSWeb_mastercatalog_v1.fits (COSMOS2025 DR1)')
    parser.add_argument('--img_dir',
                        default='/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/',
                        help='Directory containing NIRCam tile mosaics')
    parser.add_argument('--output',     default='cosmosweb_dataset.h5')
    parser.add_argument('--stamp_size', type=int, default=STAMP_SIZE)
    args = parser.parse_args()

    # ── load & merge catalogs ────────────────────────────────────────────────
    log.info('Loading visual morphology DB…')
    df_morpho = load_morpho_db(args.morpho_db)

    log.info('Loading master catalog (photometry + LePhare + CIGALE SFH)…')
    df_merged = load_master_catalog(args.master_cat)
    df_merged = df_merged[df_merged['id'].isin(set(df_morpho['id']))].copy()
    log.info(f'After morpho cross-match: {len(df_merged)} galaxies')

    # Drop rows with missing SFH data
    sfh_required = _SFR_COLS + _TIME_COLS + [_INT_COL]
    has_sfh = df_merged[sfh_required].notna().all(axis=1)
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
