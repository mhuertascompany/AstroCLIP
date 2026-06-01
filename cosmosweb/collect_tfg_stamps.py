"""
Collect JPEG stamps and FITS cutouts for all galaxies listed in the
cosmosweb/galaxias_select/*.csv files and deposit them into a tfg_laura/
output directory on candide.

Output layout
-------------
    tfg_laura/
        COSMOS_130567/          ← one folder per CSV (one per student)
            F115W_130567.jpg    ← copied from existing JPEG stamp
            F150W_130567.jpg
            F277W_130567.jpg
            F444W_130567.jpg
            F115W_130567.fits   ← fresh FITS cutout from mosaic
            F150W_130567.fits
            F277W_130567.fits
            F444W_130567.fits
            F115W_316983.jpg    ← next galaxy in the same CSV
            ...
        COSMOS_175080/
            ...

Run (from the repo root):
    python -m cosmosweb.collect_tfg_stamps \\
        --csv_dir  cosmosweb/datos_CIGALE \\
        --output   tfg_stamps \\
        --arcsec   5.0

Or with defaults (reads cosmosweb/datos_CIGALE, writes to tfg_stamps/):
    python -m cosmosweb.collect_tfg_stamps
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import wcs as astropy_wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

# ── server paths ──────────────────────────────────────────────────────────────
STAMP_ROOT  = Path('/n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert')
NIRCAM_ROOT = Path('/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8')

JWST_FILTERS = ['F115W', 'F150W', 'F277W', 'F444W']
ALL_TILES    = [f'A{i}' for i in range(1, 11)] + [f'B{i}' for i in range(1, 11)]


# ── helpers ───────────────────────────────────────────────────────────────────

def mosaic_path(filt: str, tile: str) -> Path:
    return NIRCAM_ROOT / f'mosaic_nircam_{filt.lower()}_COSMOS-Web_30mas_{tile}_v0_8_sci.fits'


def find_tile(ra: float, dec: float) -> str | None:
    """
    Return the tile name whose F277W mosaic covers (ra, dec), or None.
    Reads only the FITS header (+ minimal data via memmap) for speed.
    """
    for tile in ALL_TILES:
        fpath = mosaic_path('F277W', tile)
        if not fpath.exists():
            continue
        try:
            with fits.open(fpath, memmap=True) as hdul:
                hdr  = hdul[0].header
                ny   = hdr['NAXIS2']
                nx   = hdr['NAXIS1']
                w    = WCS(hdr)
            px, py = w.wcs_world2pix(ra, dec, 0)
            if 0 < float(px) < nx and 0 < float(py) < ny:
                return tile
        except Exception as exc:
            log.debug('Tile %s check failed: %s', tile, exc)
            continue
    return None


def make_fits_cutout(mosaic: Path, ra: float, dec: float,
                     arcsec: float, out_path: Path) -> bool:
    """
    Cut a square stamp of side `arcsec` arcseconds centred on (ra, dec)
    from `mosaic` and write it to `out_path` as a FITS file with
    updated WCS.  Returns True on success.
    """
    try:
        with fits.open(mosaic, memmap=True) as hdul:
            data = hdul[0].data
            hdr  = hdul[0].header
            w    = WCS(hdr)

        pix_scale = astropy_wcs.utils.proj_plane_pixel_scales(w)[0]
        pix_scale *= w.wcs.cunit[0].to('arcsec')
        size_pix  = int(round(arcsec / pix_scale))

        px, py = w.wcs_world2pix(ra, dec, 0)
        cutout = Cutout2D(data, (int(px), int(py)),
                          (size_pix, size_pix), wcs=w)

        out_hdr = cutout.wcs.to_header()
        # Preserve useful keywords from the original header
        for key in ('BUNIT', 'TELESCOP', 'INSTRUME', 'FILTER', 'EXPTIME'):
            if key in hdr:
                out_hdr[key] = hdr[key]

        hdu = fits.PrimaryHDU(data=cutout.data.astype(np.float32),
                              header=out_hdr)
        hdu.writeto(out_path, overwrite=True)
        return True

    except Exception as exc:
        log.warning('  FITS cutout failed (%s): %s', mosaic.name, exc)
        return False


# ── per-galaxy processing ─────────────────────────────────────────────────────

def process_galaxy(gid: int, ra: float, dec: float,
                   out_dir: Path, arcsec: float) -> dict:
    """
    Copy JPEGs and cut FITS stamps for one galaxy.
    Returns a summary dict with counts.
    """
    counts = {'jpg_ok': 0, 'jpg_miss': 0, 'fits_ok': 0, 'fits_miss': 0}

    # ── 1. JPEGs: copy from existing stamp directory ──────────────────────────
    for filt in JWST_FILTERS:
        src = STAMP_ROOT / filt / f'{filt}_{gid}.jpg'
        dst = out_dir / f'{filt}_{gid}.jpg'
        if src.exists():
            shutil.copy2(src, dst)
            counts['jpg_ok'] += 1
        else:
            log.debug('  JPEG missing: %s', src)
            counts['jpg_miss'] += 1

    # ── 2. FITS cutouts: find tile, then cut each filter ─────────────────────
    tile = find_tile(ra, dec)
    if tile is None:
        log.warning('  No tile found for id=%d (ra=%.4f dec=%.4f)', gid, ra, dec)
        counts['fits_miss'] += len(JWST_FILTERS)
        return counts

    log.info('  id=%-8d  tile=%s', gid, tile)
    for filt in JWST_FILTERS:
        mpath = mosaic_path(filt, tile)
        dst   = out_dir / f'{filt}_{gid}.fits'
        if not mpath.exists():
            log.debug('  Mosaic missing: %s', mpath)
            counts['fits_miss'] += 1
            continue
        if make_fits_cutout(mpath, ra, dec, arcsec, dst):
            counts['fits_ok'] += 1
        else:
            counts['fits_miss'] += 1

    return counts


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Collect TFG stamps for Laura')
    p.add_argument('--csv_dir', type=Path,
                   default=Path(__file__).parent / 'datos_CIGALE',
                   help='Directory containing the COSMOS_*.csv files')
    p.add_argument('--output',  type=Path,
                   default=Path('/n03data/huertas/COSMOS-Web/tfg_laura'),
                   help='Root output directory (created if absent)')
    p.add_argument('--arcsec',  type=float, default=5.0,
                   help='Half-side of FITS cutout in arcseconds')
    p.add_argument('--filters', nargs='+', default=JWST_FILTERS,
                   help='Filters to process')
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.csv_dir.glob('COSMOS_*.csv'))
    if not csv_files:
        log.error('No COSMOS_*.csv files found in %s', args.csv_dir)
        sys.exit(1)
    log.info('Found %d CSV files in %s', len(csv_files), args.csv_dir)

    total = {'jpg_ok': 0, 'jpg_miss': 0, 'fits_ok': 0, 'fits_miss': 0}

    for csv_path in csv_files:
        student_name = csv_path.stem          # e.g. COSMOS_130567
        out_dir      = args.output / student_name
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)
        log.info('── %s  (%d galaxies)', student_name, len(df))

        for _, row in df.iterrows():
            gid = int(float(row['id']))
            ra  = float(row['ra'])
            dec = float(row['dec'])
            counts = process_galaxy(gid, ra, dec, out_dir, args.arcsec)
            for k in total:
                total[k] += counts[k]

    log.info('Done.')
    log.info('  JPEGs  copied: %d   missing: %d', total['jpg_ok'],  total['jpg_miss'])
    log.info('  FITS cutouts:  %d   failed:  %d', total['fits_ok'], total['fits_miss'])
    log.info('Output → %s', args.output)


if __name__ == '__main__':
    main()
