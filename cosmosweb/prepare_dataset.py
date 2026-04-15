"""
Prepare the paired COSMOS-Web image + CIGALE SFH dataset.

For each galaxy selected from the visual morphology catalog:
  1. Cut a 64x64 stamp in F150W, F277W, F444W from the NIRCam tile mosaics
  2. Interpolate the CIGALE SFH onto a redshift-normalized fractional time grid
     (Option A: t_frac = t_lookback / t_universe(z), so all SFHs share the
      same [0, 1] axis regardless of redshift)
  3. Store everything in an HDF5 file ready for training

Output HDF5 layout:
    /images         (N, 3, 64, 64)  float32  arcsinh-stretched flux
    /sfh            (N, N_TIME)     float32  log10(SFR + EPS) on fractional grid
    /sfh_time_grid  (N_TIME,)       float32  fractional lookback time ∈ [0, 1]
    /sfh_time_norm  (N,)            float32  age of universe at galaxy z [Myr]
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
from astropy.cosmology import FlatLambdaCDM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cigale'))
from SFHandle.sfh import SFH

# ── constants ─────────────────────────────────────────────────────────────────

FILTERS    = ['f150w', 'f277w', 'f444w']
STAMP_SIZE = 64
SFH_N_BINS = 50
SFH_EPS    = 1e-10   # avoids log10(0) for quiescent galaxies

# Fractional lookback-time grid ∈ [0, 1].
# Multiplied by t_universe(z) [Myr] per galaxy to get the physical grid.
SFH_T_FRAC = np.linspace(0, 1, SFH_N_BINS)

# Flat ΛCDM cosmology for computing t_universe(z)
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)


# ── catalog helpers ────────────────────────────────────────────────────────────

def load_morpho_catalog(fits_path: str) -> pd.DataFrame:
    """
    Load the morphology selection catalog (FITS).

    Expects a column named 'id' (or 'ID') matching galaxy IDs in the
    photometry catalog.  All rows are used as the base selection —
    apply any desired morphological cuts before calling this function,
    or extend the body below.
    """
    df = _fits_to_df(fits_path)
    log.info(f'Morpho catalog: {len(df)} rows, columns: {list(df.columns)}')

    # Normalise id column name to lowercase
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)

    if 'id' not in df.columns:
        raise ValueError(
            f"No 'id' column found in {fits_path}.  "
            f"Available columns: {list(df.columns)}"
        )
    return df


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


def sfh_to_common_grid(row: pd.Series, z: float) -> tuple[np.ndarray, float] | None:
    """
    Extract and interpolate one galaxy's CIGALE SFH to the fractional time grid.

    Option A (z-dependent normalization): the physical time axis is
    ``SFH_T_FRAC * t_universe(z)`` so that all SFHs live on the same
    normalized [0, 1] lookback-time axis independent of redshift.

    The CIGALE columns sfh_sfr_bin{i} are fractional weights (0–1, summing to
    ~1), NOT absolute SFR values.  We do NOT multiply by sfh_integrated here:
    that multiplication would inject log10(M_star) as an additive offset into
    every SFH, creating a spurious mass → redshift signal in the embedding.
    Instead we normalise the interpolated shape so that it sums to 1, making
    the encoder see only the temporal distribution of star formation.

    Returns (sfh_log, t_universe_myr) where
      sfh_log        – float32 array of shape (SFH_N_BINS,), log10(frac + EPS)
      t_universe_myr – age of universe at z in Myr (stored per galaxy in HDF5)

    Returns None if the row does not contain valid SFH data or z is invalid.
    Requires _COL_MAP to be set.
    """
    if not (np.isfinite(z) and z > 0):
        return None

    # Use the fractional SFH weights directly — do NOT multiply by sfh_integrated.
    # sfh_sfr_bin{i} are dimensionless fractions; sfh_integrated is stellar mass
    # (M_sun).  The product would give stellar mass per bin, adding log10(M_star)
    # as an amplitude offset that correlates with redshift.
    lb_time = np.array([row[c] for c in _COL_MAP['time']], dtype=np.float64)
    sfr_frac = np.array([row[c] for c in _COL_MAP['sfr']],  dtype=np.float64)
    err_frac = np.array([row[c] for c in _COL_MAP['err']],  dtype=np.float64)

    if not (np.all(np.isfinite(lb_time)) and np.all(np.isfinite(sfr_frac))):
        return None

    # Sort bins by ascending lookback time (interp1d requires monotonic x).
    # CIGALE bins can be stored oldest-first (descending lb_time); if so,
    # fill_value=0 would zero out the RECENT end instead of the ancient end.
    sort_idx = np.argsort(lb_time)
    lb_time  = lb_time[sort_idx]
    sfr_frac = sfr_frac[sort_idx]
    err_frac = err_frac[sort_idx]

    # Age of universe at this galaxy's redshift → physical time grid [0, t_u] Myr
    t_universe_myr = float(COSMO.age(z).to('Myr').value)
    phys_grid      = SFH_T_FRAC * t_universe_myr   # (SFH_N_BINS,) in Myr

    sfh_obj       = SFH(lb_time, sfr_frac, err_frac)
    sfr_interp, _ = sfh_obj.interpolate_sfh(
        phys_grid, kind='next', bounds_error=False, fill_value=0.0
    )

    # Normalise to unit sum so the encoder sees only the SHAPE of the SFH,
    # not its overall amplitude (which would otherwise encode stellar mass).
    sfr_total = sfr_interp.sum()
    if sfr_total > SFH_EPS:
        sfr_interp = sfr_interp / sfr_total

    sfh_log = np.log10(sfr_interp + SFH_EPS).astype(np.float32)
    return sfh_log, t_universe_myr


# ── image helpers ──────────────────────────────────────────────────────────────

def _img_path(img_dir: str, filt: str, tile: str) -> Path:
    return (Path(img_dir) /
            f'mosaic_nircam_{filt}_COSMOS-Web_30mas_{tile}_v0_8_sci.fits')


def _data_hdu_index(hdul) -> int:
    """Return the index of the first HDU that contains 2D image data."""
    for i, hdu in enumerate(hdul):
        if hdu.data is not None and hasattr(hdu.data, 'ndim') and hdu.data.ndim == 2:
            return i
    raise ValueError(f'No 2D image data found in {hdul.filename()}')


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
    parser.add_argument('--morpho_cat',
                        default='/n03data/huertas/COSMOS-Web/ilbert_finetune/ilbert_visual_zoobot_morphology.fits',
                        help='Morphology selection FITS catalog (must contain an id column)')
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
    log.info('Loading morphology catalog…')
    df_morpho  = load_morpho_catalog(args.morpho_cat)
    morpho_ids = set(df_morpho['id'])

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
    log.info(f'Morpho  id sample (first 5): {list(df_morpho["id"].head())}  dtype={df_morpho["id"].dtype}')
    log.info(f'Photom  id sample (first 5): {list(df_all["id"].head())}  dtype={df_all["id"].dtype}')
    n_overlap = df_all['id'].isin(morpho_ids).sum()
    log.info(f'ID overlap before type coercion: {n_overlap}')

    # Try casting both to the same type if no overlap found
    if n_overlap == 0:
        try:
            morpho_ids_int = set(df_morpho['id'].astype(np.int64))
            n_overlap_int  = df_all['id'].astype(np.int64).isin(morpho_ids_int).sum()
            log.info(f'ID overlap after int64 coercion: {n_overlap_int}')
            if n_overlap_int > 0:
                morpho_ids = morpho_ids_int
                df_all['id'] = df_all['id'].astype(np.int64)
        except (ValueError, OverflowError):
            log.warning('int64 coercion failed; IDs may be strings or floats')

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

        ds_img   = f.create_dataset('images',       shape=(n_max, 3, sz, sz), dtype='f4', **img_kw)
        ds_sfh   = f.create_dataset('sfh',          shape=(n_max, SFH_N_BINS), dtype='f4',
                                    maxshape=(None, SFH_N_BINS), chunks=(256, SFH_N_BINS))
        ds_tnorm = f.create_dataset('sfh_time_norm', shape=(n_max,), dtype='f4', **kw)
        ds_id    = f.create_dataset('galaxy_id',    shape=(n_max,), dtype='i8', **kw)
        ds_ra    = f.create_dataset('ra',           shape=(n_max,), dtype='f8', **kw)
        ds_dec   = f.create_dataset('dec',          shape=(n_max,), dtype='f8', **kw)
        ds_z     = f.create_dataset('redshift',     shape=(n_max,), dtype='f4', **kw)
        # Common fractional grid ∈ [0, 1]; multiply by sfh_time_norm[i] to get Myr
        f.create_dataset('sfh_time_grid', data=SFH_T_FRAC.astype(np.float32))

        f.attrs['filters']    = ','.join(FILTERS)
        f.attrs['stamp_size'] = sz

        written = 0

        # ── tile loop ────────────────────────────────────────────────────────
        tile_counts = df_merged['tile'].value_counts()
        log.info(f'Tile distribution (top 10): {tile_counts.head(10).to_dict()}')

        for tile_name, tile_df in df_merged.groupby('tile'):
            log.info(f'Tile {tile_name!r}: {len(tile_df)} candidates…')

            paths = {filt: _img_path(args.img_dir, filt, tile_name)
                     for filt in FILTERS}
            missing = [str(p) for p in paths.values() if not p.exists()]
            if missing:
                log.warning(f'  Skipping tile {tile_name!r}: missing {missing}')
                continue

            # Open all 3 FITS files; keep memmap open for the whole tile
            hduls  = {filt: fits.open(paths[filt], memmap=True) for filt in FILTERS}
            try:
                ext    = {filt: _data_hdu_index(hduls[filt]) for filt in FILTERS}
                arrays = {filt: hduls[filt][ext[filt]].data             for filt in FILTERS}
                wcss   = {filt: WCS(hduls[filt][ext[filt]].header, naxis=2)
                          for filt in FILTERS}
                log.info(f'  HDU indices: { {f: ext[f] for f in FILTERS} }')

                tile_ok = 0
                for _, row in tile_df.iterrows():
                    z = float(row.get('zfinal', -1.0))

                    # ── SFH ──────────────────────────────────────────────────
                    result = sfh_to_common_grid(row, z)
                    if result is None:
                        continue
                    sfh_vec, t_universe_myr = result

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

                    ds_img[written]   = np.stack(stamps, axis=0)    # (3, sz, sz)
                    ds_sfh[written]   = sfh_vec
                    ds_tnorm[written] = t_universe_myr
                    ds_id[written]    = int(row['id'])
                    ds_ra[written]    = float(row['ra'])
                    ds_dec[written]   = float(row['dec'])
                    ds_z[written]     = z
                    written += 1
                    tile_ok += 1

                log.info(f'  → {tile_ok} galaxies written')

            finally:
                for hdul in hduls.values():
                    hdul.close()

        # ── truncate datasets to actual size ──────────────────────────────
        for ds in [ds_img, ds_sfh, ds_tnorm, ds_id, ds_ra, ds_dec, ds_z]:
            ds.resize(written, axis=0)

        f.attrs['n_galaxies'] = written
        log.info(f'Wrote {written} galaxies total')

        # ── compute per-channel image stats and per-bin SFH stats ────────────
        if written > 0:
            log.info('Computing per-channel image statistics…')
            imgs      = f['images'][:]               # (N, 3, sz, sz)
            img_mean  = imgs.mean(axis=(0, 2, 3))    # (3,)
            img_std   = imgs.std (axis=(0, 2, 3))    # (3,)
            img_std[img_std == 0] = 1.0
            f.attrs['img_mean'] = img_mean.astype(np.float32)
            f.attrs['img_std']  = img_std.astype(np.float32)
            log.info(f'  img mean={img_mean}  std={img_std}')

            log.info('Computing per-bin SFH statistics…')
            sfhs     = f['sfh'][:]                   # (N, SFH_N_BINS)
            sfh_mean = sfhs.mean(axis=0)             # (SFH_N_BINS,)
            sfh_std  = sfhs.std (axis=0)             # (SFH_N_BINS,)
            sfh_std[sfh_std < 1e-6] = 1.0            # constant bins → no-op after z-score
            f.attrs['sfh_mean'] = sfh_mean.astype(np.float32)
            f.attrs['sfh_std']  = sfh_std.astype(np.float32)
            log.info(f'  sfh mean range [{sfh_mean.min():.3f}, {sfh_mean.max():.3f}]  '
                     f'std range [{sfh_std.min():.3f}, {sfh_std.max():.3f}]')

    log.info(f'Dataset saved to {args.output}')


if __name__ == '__main__':
    main()
