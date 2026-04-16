"""
UMAP visualization of the CosmosWebCLIP embedding space.

Loads a trained checkpoint, encodes all galaxies (image + SFH branch),
runs UMAP on the joint L2-normalised embedding, and writes a multi-page
PDF of scatter plots color-coded by physical properties:

  Page 1 – Morphological properties
            (Sersic radius, P(Ell), P(S0), P(Early Disk), P(Late Disk), …)
  Page 2 – SED / CIGALE properties
            (redshift, log M★, log SFR, log sSFR, SFH ratio, …)
  Page 3 – Modality comparison
            (image-only / SFH-only / joint UMAP, all colored by redshift)

Usage (from repo root):
    python -m cosmosweb.umap_embeddings \\
        --checkpoint /n03data/huertas/COSMOS-Web/cosmosweb_clip/checkpoints/cosmosweb_clip-epoch=028-val_loss=3.5435.ckpt \\
        --dataset    /n03data/huertas/COSMOS-Web/cosmosweb_clip/cosmosweb_dataset.h5 \\
        --output     /n03data/huertas/COSMOS-Web/cosmosweb_clip/umap_plots.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, LogNorm
import umap

from astropy.io import fits
from astropy.table import Table

from .model import CosmosWebCLIP

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

# ── default paths ─────────────────────────────────────────────────────────────
_CAT_DIR = '/n23data2/cosmosweb/catalogs/DR1/data/catalog'


# ── catalog helpers ────────────────────────────────────────────────────────────

def _fits_to_df(path: str, hdu: int = 1,
                cols: list | None = None) -> pd.DataFrame:
    with fits.open(path, memmap=True) as h:
        tbl = Table(h[hdu].data)
    scalar = [c for c in tbl.colnames if len(tbl[c].shape) <= 1]
    if cols:
        scalar = [c for c in cols if c in scalar]
    return tbl[scalar].to_pandas()


def load_properties(photom_path: str, lephare_path: str,
                    cigale_path: str, morpho_path: str,
                    galaxy_ids: np.ndarray) -> pd.DataFrame:
    """
    Load physical properties from the four catalog files and return a
    DataFrame indexed by galaxy_id, filtered to the galaxies in the
    HDF5 dataset.
    """
    id_set = set(galaxy_ids.astype(np.int64))

    # ── photometry (carries 'id') ────────────────────────────────────────
    log.info('Loading photometry catalog…')
    want_phot = ['id', 'tile', 'ra', 'dec',
                 'sersic',          # Sersic index n
                 'radius_sersic',   # effective radius
                 'axratio_sersic',  # minor/major axis ratio
                 'flag_star', 'flag_blend']
    df_phot = _fits_to_df(photom_path, cols=want_phot)
    log.info(f'  {len(df_phot)} rows; cols={list(df_phot.columns)}')

    # ── LePhare (row-aligned with photom) ────────────────────────────────
    log.info('Loading LePhare catalog…')
    df_lp = _fits_to_df(lephare_path, cols=['zfinal', 'zpdf_l68', 'zpdf_u68'])

    # ── CIGALE (row-aligned with photom) ─────────────────────────────────
    log.info('Loading CIGALE catalog…')
    df_cig = _fits_to_df(cigale_path)
    log.info(f'  CIGALE cols available: {sorted(df_cig.columns)[:40]}')

    # Positional concat — all three have the same rows
    df_all = pd.concat([df_phot, df_lp, df_cig], axis=1)
    # Cast id to int64 for matching
    try:
        df_all['id'] = df_all['id'].astype(np.int64)
    except Exception:
        pass

    df_all = df_all[df_all['id'].isin(id_set)].copy()
    df_all['id'] = df_all['id'].astype(np.int64)   # guarantee int64 before indexing
    df_all.set_index('id', inplace=True)
    log.info(f'  After filtering to dataset galaxies: {len(df_all)}')

    # ── morphology catalog (has its own 'id') ─────────────────────────────
    log.info('Loading morphology catalog…')
    df_morph = _fits_to_df(morpho_path)
    col_map  = {c: c.lower() for c in df_morph.columns}
    df_morph = df_morph.rename(columns=col_map)
    try:
        df_morph['id'] = df_morph['id'].astype(np.int64)
    except Exception:
        pass
    df_morph = df_morph[df_morph['id'].isin(id_set)].copy()
    df_morph.set_index('id', inplace=True)
    family_cols = sorted(c for c in df_morph.columns if 'family' in c)
    log.info(f'  Morpho overlap: {len(df_morph)} rows; family cols: {family_cols}')

    df_merged = df_all.join(df_morph, how='left', rsuffix='_morph')
    return df_merged


def _derive_cigale(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived SED quantities (SFR, sSFR, SFH ratio) to the DataFrame."""
    # Detect naming convention
    if 'sfh_integrated' in df.columns:
        mass_col = 'sfh_integrated'
        sfr1_col = 'sfh_sfr_bin1'
        sfr9_col = 'sfh_sfr_bin9'
    elif 'bayes.sfh.integrated' in df.columns:
        mass_col = 'bayes.sfh.integrated'
        sfr1_col = 'bayes.sfh.sfr_bin1'
        sfr9_col = 'bayes.sfh.sfr_bin9'
    else:
        log.warning('Cannot identify CIGALE mass/SFR columns; skipping derived quantities.')
        return df

    mass = df[mass_col].values.astype(float)
    frac1 = df[sfr1_col].values.astype(float) if sfr1_col in df.columns else np.full(len(df), np.nan)
    frac9 = df[sfr9_col].values.astype(float) if sfr9_col in df.columns else np.full(len(df), np.nan)

    sfr  = frac1 * mass   # M_sun / yr (most recent bin)
    ssfr = np.where(mass > 0, sfr / mass, np.nan)
    # High-z / early SFR fraction: bin9 (oldest) relative to bin1 (newest)
    sfh_ratio = np.where(frac1 > 0, frac9 / np.maximum(frac1, 1e-30), np.nan)

    df = df.copy()
    df['_log_mass'] = np.where(mass > 0, np.log10(mass), np.nan)
    df['_log_sfr']  = np.where(sfr  > 0, np.log10(sfr),  np.nan)
    df['_log_ssfr'] = np.where(ssfr > 0, np.log10(ssfr), np.nan)
    df['_sfh_ratio'] = sfh_ratio   # log ratio plotted below

    return df


# ── embedding extraction ───────────────────────────────────────────────────────

def extract_embeddings(
    checkpoint: Path,
    h5_path: Path,
    batch_size: int = 512,
    device: str = 'cuda',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load model from checkpoint; encode all galaxies.

    Returns
    -------
    img_emb   : (N, D)  L2-normalised image embeddings
    sfh_emb   : (N, D)  L2-normalised SFH embeddings
    galaxy_ids: (N,)    int64
    redshifts : (N,)    float32
    """
    log.info(f'Loading checkpoint {checkpoint}')
    model = CosmosWebCLIP.load_from_checkpoint(str(checkpoint), map_location='cpu')
    model.eval()
    model.to(device)
    log.info(f'  Hyperparameters: {model.hparams}')

    # Read normalisation stats from HDF5
    with h5py.File(h5_path, 'r') as f:
        n_total    = int(f.attrs['n_galaxies'])
        img_mean   = torch.tensor(f.attrs['img_mean'], dtype=torch.float32)
        img_std    = torch.tensor(f.attrs['img_std'],  dtype=torch.float32)
        galaxy_ids = f['galaxy_id'][:]
        redshifts  = f['redshift'][:]

    log.info(f'  Encoding {n_total} galaxies…')

    img_embs, sfh_embs = [], []

    with h5py.File(h5_path, 'r') as f, torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)

            images = torch.tensor(f['images'][start:end], dtype=torch.float32)
            sfhs   = torch.tensor(f['sfh'][start:end],    dtype=torch.float32)

            # Normalise images
            images = (images - img_mean[None, :, None, None]) / img_std[None, :, None, None]

            images = images.to(device)
            sfhs   = sfhs.to(device)

            img_e = F.normalize(model.encode_image(images), dim=-1)
            sfh_e = F.normalize(model.encode_sfh(sfhs),    dim=-1)

            img_embs.append(img_e.cpu().numpy())
            sfh_embs.append(sfh_e.cpu().numpy())

            if (start // batch_size) % 10 == 0:
                log.info(f'  {end}/{n_total}')

    img_emb = np.concatenate(img_embs, axis=0)
    sfh_emb = np.concatenate(sfh_embs, axis=0)
    log.info(f'  Done. Embeddings: img={img_emb.shape}, sfh={sfh_emb.shape}')
    return img_emb, sfh_emb, galaxy_ids, redshifts


# ── UMAP ──────────────────────────────────────────────────────────────────────

def fit_umap(
    emb: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float  = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(emb)


# ── plotting helpers ───────────────────────────────────────────────────────────

COLS_PER_ROW   = 3
ROWS_PER_PAGE  = 3
PANELS_PER_PAGE = COLS_PER_ROW * ROWS_PER_PAGE   # 9


def _scatter(ax, xy, values, label, cmap='viridis',
             log_scale=False, vmin_p=1, vmax_p=99,
             point_size=1.5, alpha=0.6):
    """Single UMAP scatter panel."""
    mask = np.isfinite(values)

    # Apply log transform before re-masking, so zeros/negatives are excluded
    if log_scale:
        with np.errstate(divide='ignore', invalid='ignore'):
            lv = np.where(values > 0, np.log10(values), np.nan)
        mask = np.isfinite(lv)
        v = lv[mask]
    else:
        v = values[mask]

    if mask.sum() < 10:
        ax.text(0.5, 0.5, f'{label}\n(no valid data\n{mask.sum()} points)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=7, color='grey', style='italic')
        ax.set_xticks([]); ax.set_yticks([])
        return

    vmin = np.percentile(v, vmin_p)
    vmax = np.percentile(v, vmax_p)

    # Grey background for the full sample
    ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', s=point_size * 0.5,
               rasterized=True, linewidths=0, zorder=1)
    sc = ax.scatter(xy[mask, 0], xy[mask, 1], c=v,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    s=point_size, alpha=alpha, linewidths=0,
                    rasterized=True, zorder=2)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(label, fontsize=6)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=7, pad=3)
    ax.set_aspect('equal', 'datalim')


def _make_pages(pdf, xy, prop_df, panels, section_title):
    """
    Render as many PDF pages as needed to display all panels (9 per page).
    """
    total = len(panels)
    n_pages = max(1, (total + PANELS_PER_PAGE - 1) // PANELS_PER_PAGE)

    for page_idx in range(n_pages):
        chunk = panels[page_idx * PANELS_PER_PAGE:
                       (page_idx + 1) * PANELS_PER_PAGE]

        suffix = f' ({page_idx + 1}/{n_pages})' if n_pages > 1 else ''
        page_title = section_title + suffix

        fig, axes = plt.subplots(
            ROWS_PER_PAGE, COLS_PER_ROW,
            figsize=(11, 8.5),
            squeeze=False,
        )
        fig.suptitle(page_title, fontsize=10, y=0.99)

        flat = axes.flatten()
        for ax in flat:
            ax.set_visible(False)

        for i, (label, col, log_sc, cmap) in enumerate(chunk):
            ax = flat[i]
            ax.set_visible(True)

            if col not in prop_df.columns:
                ax.text(0.5, 0.5, f'{label}\n(not found)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=8, color='grey')
                ax.set_xticks([]); ax.set_yticks([])
                continue

            values = prop_df[col].values.astype(float)
            _scatter(ax, xy, values, label=label, cmap=cmap, log_scale=log_sc)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)


# ── panel definitions ─────────────────────────────────────────────────────────

# Fixed structural panels (Sersic quantities from the photometry catalog).
# Family morphology panels are discovered dynamically at runtime from the
# columns of ilbert_visual_zoobot_morphology.fits that contain "family".
MORPH_PANELS_FIXED = [
    # (display_label, df_column, log_scale, cmap)
    ('Redshift z',            'zfinal',          False, 'plasma'),
    ('Sersic index n',        'sersic',          False, 'viridis'),
    ('Sersic radius [px]',    'radius_sersic',   False, 'viridis'),
    ('Axis ratio (b/a)',      'axratio_sersic',  False, 'viridis'),
]

SED_PANELS = [
    ('Redshift z',          'zfinal',     False, 'plasma'),
    ('log M★ [M⊙]',        '_log_mass',  False, 'inferno'),
    ('log SFR [M⊙/yr]',    '_log_sfr',   False, 'magma'),
    ('log sSFR [yr⁻¹]',    '_log_ssfr',  False, 'coolwarm'),
    ('log SFR ratio bin9/bin1', '_sfh_ratio', True,  'RdBu_r'),
    ('SFR frac bin1 (recent)',  'sfh_sfr_bin1', False, 'hot'),
    ('SFR frac bin5 (mid)',     'sfh_sfr_bin5', False, 'hot'),
    ('SFR frac bin9 (old)',     'sfh_sfr_bin9', False, 'hot'),
    ('log sfh_integrated [M⊙]', 'sfh_integrated', True, 'viridis'),
]

# Extra CIGALE columns to try (DR1 or raw CIGALE naming)
_EXTRA_CIGALE = [
    ('log Mstar (bayes)',  'bayes.stellar.mass',    True,  'inferno'),
    ('log Age mass-wtd',   'bayes.sfh.age_mass',    True,  'cividis'),
    ('log Age light-wtd',  'bayes.sfh.age_light',   True,  'cividis'),
    ('Av [mag]',           'bayes.attenuation.V',   False, 'Oranges'),
    ('Av (DR1)',           'attenuation_V',          False, 'Oranges'),
    ('log Mstar (DR1)',    'stellar_mass',           True,  'inferno'),
]


# ── extra CIGALE columns for the npz / explorer ───────────────────────────────
#
# Each entry: (candidate column names in catalog, npz key, log10-transform?)
# We try plain names first, then the bayes.* naming convention.
_CIGALE_EXTRA_COLS = [
    (['age_form',             'bayes.sfh.age_form'],              'age_form',             False),
    (['sfr_inst',             'bayes.sfh.sfr_inst'],              'log_sfr_inst',         True),
    (['sfr_100myr',           'bayes.sfh.sfr_100myr'],            'log_sfr_100myr',       True),
    (['sfr_mass_vector_dir',  'bayes.sfh.sfr_mass_vector_dir'],   'sfr_mass_vector_dir',  False),
    (['sfr_mass_vector_norm', 'bayes.sfh.sfr_mass_vector_norm'],  'sfr_mass_vector_norm', False),
]


def _save_cigale_extras(npz_data: dict, prop_aligned: 'pd.DataFrame') -> None:
    """Add extra CIGALE SFH columns to the npz data dict (in-place)."""
    for candidates, key, do_log in _CIGALE_EXTRA_COLS:
        for col in candidates:
            if col not in prop_aligned.columns:
                continue
            vals = prop_aligned[col].values.astype(np.float64)
            if do_log:
                with np.errstate(divide='ignore', invalid='ignore'):
                    vals = np.where(vals > 0, np.log10(vals), np.nan)
            npz_data[key] = vals.astype(np.float32)
            break   # found; no need to try the next candidate name


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='UMAP of CosmosWebCLIP embeddings')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset',    type=Path, required=True)
    p.add_argument('--output',     type=Path,
                   default=Path('/n03data/huertas/COSMOS-Web/cosmosweb_clip/umap_plots.pdf'))
    p.add_argument('--morpho_cat', type=Path,
                   default='/n03data/huertas/COSMOS-Web/ilbert_finetune/ilbert_visual_zoobot_morphology.fits')
    p.add_argument('--photom_cat', type=Path,
                   default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_photom_primary.fits')
    p.add_argument('--lephare_cat', type=Path,
                   default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_lephare.fits')
    p.add_argument('--cigale_cat', type=Path,
                   default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_cigale.fits')
    p.add_argument('--batch_size',   type=int,   default=512)
    p.add_argument('--n_neighbors',  type=int,   default=15)
    p.add_argument('--min_dist',     type=float, default=0.1)
    p.add_argument('--device',       type=str,   default='cuda')
    p.add_argument('--max_galaxies', type=int,   default=0,
                   help='Cap at N galaxies for quick tests (0 = all)')
    p.add_argument('--npz_output',   type=Path,  default=None,
                   help='If set, save UMAP coords + metadata to this .npz '
                        '(download alongside the h5 to run explore.py locally)')
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ── step 1: extract embeddings ───────────────────────────────────────────
    img_emb, sfh_emb, galaxy_ids, redshifts = extract_embeddings(
        args.checkpoint, args.dataset,
        batch_size=args.batch_size,
        device=args.device if torch.cuda.is_available() else 'cpu',
    )

    # h5_indices[i] = row in the HDF5 file that corresponds to embedding i
    h5_indices = np.arange(len(galaxy_ids))

    if args.max_galaxies > 0:
        n = min(args.max_galaxies, len(galaxy_ids))
        idx = np.random.default_rng(0).choice(len(galaxy_ids), n, replace=False)
        img_emb    = img_emb[idx]
        sfh_emb    = sfh_emb[idx]
        galaxy_ids = galaxy_ids[idx]
        redshifts  = redshifts[idx]
        h5_indices = h5_indices[idx]

    # Joint embedding: average of L2-normalised modalities
    joint_emb = F.normalize(
        torch.tensor(img_emb + sfh_emb), dim=-1
    ).numpy()

    # ── step 2: load properties ──────────────────────────────────────────────
    log.info('Loading catalog properties…')
    prop_df = load_properties(
        str(args.photom_cat),
        str(args.lephare_cat),
        str(args.cigale_cat),
        str(args.morpho_cat),
        galaxy_ids,
    )
    prop_df = _derive_cigale(prop_df)
    log.info(f'Properties DataFrame: {len(prop_df)} rows, {len(prop_df.columns)} cols')

    # Align prop_df rows to the embedding order
    gid_int = galaxy_ids.astype(np.int64)
    # Force index to int64 to prevent dtype-mismatch NaN rows after reindex
    prop_df.index = prop_df.index.astype(np.int64)
    prop_aligned = prop_df.reindex(gid_int)

    # Diagnostic: report how many valid (non-NaN) values each key column has
    key_cols = ['zfinal', 'radius_sersic', 'sersic', 'axratio_sersic',
                '_log_mass', '_log_sfr', '_log_ssfr',
                'sfh_integrated', 'sfh_sfr_bin1'] + \
               [c for c in prop_aligned.columns if 'family' in c]
    log.info('Column availability in prop_aligned (non-NaN counts):')
    for c in key_cols:
        if c in prop_aligned.columns:
            n_valid = prop_aligned[c].notna().sum()
            log.info(f'  {c:30s}  {n_valid:6d} / {len(prop_aligned)}')
        else:
            log.info(f'  {c:30s}  NOT FOUND')

    # ── step 3: build UMAP projections ───────────────────────────────────────
    log.info('Fitting UMAP on joint embeddings…')
    xy_joint = fit_umap(joint_emb,
                        n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist)

    log.info('Fitting UMAP on image embeddings…')
    xy_img   = fit_umap(img_emb,
                        n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist)

    log.info('Fitting UMAP on SFH embeddings…')
    xy_sfh   = fit_umap(sfh_emb,
                        n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist)

    # ── step 4: build panel lists ─────────────────────────────────────────
    # Morphology: fixed Sersic columns + all family_* columns from morpho catalog
    family_panels = [
        (col.replace('family_', 'P(').replace('_', ' ').capitalize() + ')',
         col, False, 'RdBu_r')
        for col in sorted(c for c in prop_aligned.columns if 'family' in c)
    ]
    morph_panels = MORPH_PANELS_FIXED + family_panels

    # SED: base + any extra CIGALE columns that exist in the catalog
    sed_panels   = SED_PANELS[:]

    for entry in _EXTRA_CIGALE:
        if entry[1] in prop_aligned.columns:
            sed_panels.append(entry)

    # ── step 5: render PDF ───────────────────────────────────────────────────
    log.info(f'Writing PDF → {args.output}')
    n_gal = len(galaxy_ids)
    ckpt_name = args.checkpoint.name

    with PdfPages(args.output) as pdf:

        # ── pages 1+: morphological properties (joint UMAP) ─────────────────
        _make_pages(
            pdf, xy_joint, prop_aligned, morph_panels,
            f'UMAP (joint, N={n_gal:,})  ·  Morphological properties  ·  {ckpt_name}',
        )

        # ── pages N+: SED / CIGALE properties (joint UMAP) ──────────────────
        _make_pages(
            pdf, xy_joint, prop_aligned, sed_panels,
            f'UMAP (joint, N={n_gal:,})  ·  SED / CIGALE properties  ·  {ckpt_name}',
        )

        # ── page 3: modality comparison ──────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
        fig.suptitle(
            f'Embedding-space comparison  ·  colored by redshift z\n{ckpt_name}',
            fontsize=10, y=1.01,
        )

        z_values = redshifts.astype(float)
        for ax, xy, title in zip(
            axes[0],
            [xy_img, xy_sfh, xy_joint],
            ['Image encoder', 'SFH encoder', 'Joint (avg)'],
        ):
            _scatter(ax, xy, z_values, label='redshift z',
                     cmap='plasma', log_scale=False)
            ax.set_title(title, fontsize=9, pad=4)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

    log.info(f'Saved {args.output}')

    # ── step 6: save companion npz for the interactive explorer ──────────────
    npz_path = args.npz_output or args.output.with_suffix('.npz')
    npz_data = dict(
        xy_joint   = xy_joint.astype(np.float32),
        xy_img     = xy_img.astype(np.float32),
        xy_sfh     = xy_sfh.astype(np.float32),
        galaxy_ids = galaxy_ids,
        h5_indices = h5_indices,
        redshifts  = redshifts.astype(np.float32),
    )
    # Include a small set of physical properties so the explorer can color
    # the UMAP without needing the full catalog files locally
    for col, key in [('zfinal', 'zfinal'), ('radius_sersic', 'radius_sersic'),
                     ('sersic', 'sersic'), ('axratio_sersic', 'axratio_sersic'),
                     ('_log_mass', 'log_mass'), ('_log_sfr', 'log_sfr'),
                     ('_log_ssfr', 'log_ssfr')]:
        if col in prop_aligned.columns:
            npz_data[key] = prop_aligned[col].values.astype(np.float32)
    # Extra CIGALE SFH properties (try both plain and bayes.* naming conventions)
    _save_cigale_extras(npz_data, prop_aligned)
    # Family morphology columns
    for col in sorted(c for c in prop_aligned.columns if 'family' in c):
        npz_data[col] = prop_aligned[col].values.astype(np.float32)

    np.savez_compressed(npz_path, **npz_data)
    log.info(f'Companion npz saved to {npz_path}')
    log.info(f'  → download this + the h5 file to run explore.py locally')


if __name__ == '__main__':
    main()
