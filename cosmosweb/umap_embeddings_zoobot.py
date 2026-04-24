"""
UMAP visualization of the CosmosWebZooBotCLIP embedding space.

Image embeddings are extracted from F277W JPEG stamps (server-side).
SFH embeddings come from the HDF5 dataset.
The output .npz stores h5_indices so explore.py can still use the locally
downloaded HDF5 file for gallery display — no need to download the JPEGs.

Usage (from repo root):
    python -m cosmosweb.umap_embeddings_zoobot \\
        --checkpoint /n03data/huertas/COSMOS-Web/cosmosweb_clip/checkpoints/cosmosweb_zoobot_v1-epoch=029-val_loss=4.0436.ckpt \\
        --dataset    /n03data/huertas/COSMOS-Web/cosmosweb_clip/cosmosweb_dataset_v2.h5 \\
        --stamp_root /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \\
        --filter     F277W \\
        --output     /n03data/huertas/COSMOS-Web/cosmosweb_clip/umap_plots_zoobot_v1.pdf \\
        --npz_output /n03data/huertas/COSMOS-Web/cosmosweb_clip/cosmosweb_umap_zoobot_v1.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import umap
try:
    from hdbscan import HDBSCAN
except ImportError:
    from sklearn.cluster import HDBSCAN

from astropy.io import fits
from astropy.table import Table

from .dataset_zoobot import _inference_transform
from .model_zoobot import CosmosWebZooBotCLIP
from .umap_embeddings import (
    load_properties, _derive_cigale,
    fit_umap, _make_pages, _scatter,
    MORPH_PANELS_FIXED, SED_PANELS, _EXTRA_CIGALE,
    _CAT_DIR, _save_cigale_extras,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)


# ── embedding extraction ───────────────────────────────────────────────────────

def extract_embeddings(
    checkpoint:  Path,
    h5_path:     Path,
    stamp_root:  Path,
    filter_name: str  = 'F277W',
    batch_size:  int  = 128,
    device:      str  = 'cuda',
    image_size:  int  = 224,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load model; encode images from JPEGs and SFHs from HDF5.

    Returns
    -------
    img_emb    : (M, D)  L2-normalised image embeddings
    sfh_emb    : (M, D)  L2-normalised SFH embeddings
    galaxy_ids : (M,)    int64  — IDs of the M galaxies that have stamps
    redshifts  : (M,)    float32
    h5_indices : (M,)    int64  — row index in the HDF5 for each galaxy
    """
    log.info('Loading checkpoint %s', checkpoint)
    model = CosmosWebZooBotCLIP.load_from_checkpoint(
        str(checkpoint), map_location='cpu'
    )
    model.eval()
    model.to(device)
    log.info('  Hyperparameters: %s', model.hparams)

    stamp_dir = Path(stamp_root) / filter_name
    transform = _inference_transform(image_size)

    # Build list of valid (h5_idx, galaxy_id, sfh, redshift) entries
    log.info('Scanning HDF5 for galaxies with %s stamps…', filter_name)
    with h5py.File(h5_path, 'r') as f:
        all_ids  = f['galaxy_id'][:]   # (N,)
        all_sfhs = f['sfh'][:]         # (N, N_BINS)
        all_z    = f['redshift'][:]    # (N,)

    valid_h5idx = []
    valid_ids   = []
    valid_sfhs  = []
    valid_z     = []

    for h5_idx, gid in enumerate(all_ids):
        stamp_path = stamp_dir / f'{filter_name}_{int(gid)}.jpg'
        if stamp_path.exists():
            valid_h5idx.append(h5_idx)
            valid_ids.append(int(gid))
            valid_sfhs.append(all_sfhs[h5_idx])
            valid_z.append(float(all_z[h5_idx]))

    n_total = len(all_ids)
    n_valid = len(valid_ids)
    log.info('  %d / %d HDF5 galaxies have %s stamps', n_valid, n_total, filter_name)

    if n_valid == 0:
        raise RuntimeError(f'No stamps found under {stamp_dir}')

    h5_indices = np.array(valid_h5idx, dtype=np.int64)
    galaxy_ids = np.array(valid_ids,   dtype=np.int64)
    sfhs_arr   = np.stack(valid_sfhs,  axis=0).astype(np.float32)
    redshifts  = np.array(valid_z,     dtype=np.float32)

    # Encode in batches
    img_embs, sfh_embs = [], []

    with torch.no_grad():
        for start in range(0, n_valid, batch_size):
            end = min(start + batch_size, n_valid)

            # Load and transform JPEG stamps
            imgs = []
            for h5_idx in valid_h5idx[start:end]:
                gid  = int(all_ids[h5_idx])
                path = stamp_dir / f'{filter_name}_{gid}.jpg'
                imgs.append(transform(Image.open(path)))
            images = torch.stack(imgs).to(device)          # (B, 3, 224, 224)

            sfhs = torch.tensor(sfhs_arr[start:end]).to(device)

            img_e = F.normalize(model.encode_image(images), dim=-1)
            sfh_e = F.normalize(model.encode_sfh(sfhs),    dim=-1)

            img_embs.append(img_e.cpu().numpy())
            sfh_embs.append(sfh_e.cpu().numpy())

            if (start // batch_size) % 10 == 0:
                log.info('  %d / %d', end, n_valid)

    img_emb = np.concatenate(img_embs, axis=0)
    sfh_emb = np.concatenate(sfh_embs, axis=0)
    log.info('  Done. img=%s  sfh=%s', img_emb.shape, sfh_emb.shape)

    return img_emb, sfh_emb, galaxy_ids, redshifts, h5_indices


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='UMAP of CosmosWebZooBotCLIP embeddings')
    p.add_argument('--checkpoint',  type=Path, required=True)
    p.add_argument('--dataset',     type=Path, required=True,
                   help='HDF5 file from prepare_dataset.py (for SFHs + h5_indices)')
    p.add_argument('--stamp_root',  type=Path,
                   default='/n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert',
                   help='Root dir with per-filter JPEG stamp folders')
    p.add_argument('--filter',      type=str,  default='F277W')
    p.add_argument('--image_size',  type=int,  default=224)
    p.add_argument('--output',      type=Path,
                   default=Path('/n03data/huertas/COSMOS-Web/cosmosweb_clip/umap_plots_zoobot_v1.pdf'))
    p.add_argument('--npz_output',  type=Path, default=None)
    p.add_argument('--morpho_cat',  type=Path,
                   default='/n03data/huertas/COSMOS-Web/ilbert_finetune/ilbert_visual_zoobot_morphology.fits')
    p.add_argument('--photom_cat',  type=Path,
                   default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_photom_primary.fits')
    p.add_argument('--lephare_cat', type=Path,
                   default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_lephare.fits')
    p.add_argument('--cigale_cat',  type=Path,
                   default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_cigale.fits')
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--n_neighbors', type=int,   default=15)
    p.add_argument('--min_dist',    type=float, default=0.1)
    p.add_argument('--device',      type=str,   default='cuda')
    p.add_argument('--max_galaxies', type=int,  default=0,
                   help='Cap at N galaxies for quick tests (0 = all)')
    p.add_argument('--min_cluster_size', type=int, default=200,
                   help='HDBSCAN min_cluster_size (smaller → more clusters)')
    p.add_argument('--min_samples',      type=int, default=50,
                   help='HDBSCAN min_samples (larger → more conservative clusters)')
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    # ── step 1: extract embeddings ───────────────────────────────────────────
    img_emb, sfh_emb, galaxy_ids, redshifts, h5_indices = extract_embeddings(
        checkpoint=args.checkpoint,
        h5_path=args.dataset,
        stamp_root=args.stamp_root,
        filter_name=args.filter,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    if args.max_galaxies > 0:
        n = min(args.max_galaxies, len(galaxy_ids))
        idx = np.random.default_rng(0).choice(len(galaxy_ids), n, replace=False)
        img_emb    = img_emb[idx]
        sfh_emb    = sfh_emb[idx]
        galaxy_ids = galaxy_ids[idx]
        redshifts  = redshifts[idx]
        h5_indices = h5_indices[idx]

    joint_emb = F.normalize(
        torch.tensor(img_emb + sfh_emb), dim=-1
    ).numpy()

    # ── step 2: load catalog properties ──────────────────────────────────────
    log.info('Loading catalog properties…')
    prop_df = load_properties(
        str(args.photom_cat), str(args.lephare_cat),
        str(args.cigale_cat), str(args.morpho_cat),
        galaxy_ids,
    )
    prop_df = _derive_cigale(prop_df)
    prop_df.index = prop_df.index.astype(np.int64)
    prop_aligned = prop_df.reindex(galaxy_ids.astype(np.int64))

    # ── step 3: UMAP projections ──────────────────────────────────────────────
    log.info('Fitting UMAP on joint embeddings…')
    xy_joint = fit_umap(joint_emb,  n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    log.info('Fitting UMAP on image embeddings…')
    xy_img   = fit_umap(img_emb,    n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    log.info('Fitting UMAP on SFH embeddings…')
    xy_sfh   = fit_umap(sfh_emb,    n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    # ── step 3.5: HDBSCAN clustering on 2D UMAP coords ───────────────────────
    # Running on xy_joint (2D) rather than the full 256D embedding for speed.
    # Not ideal (UMAP distorts global distances) but tractable at 137k galaxies.
    log.info('Running HDBSCAN on 2D UMAP (min_cluster_size=%d, min_samples=%d)…',
             args.min_cluster_size, args.min_samples)
    clusterer  = HDBSCAN(min_cluster_size=args.min_cluster_size,
                         min_samples=args.min_samples,
                         metric='euclidean')
    hdb_labels = clusterer.fit_predict(xy_joint).astype(np.int32)
    n_clusters = int(hdb_labels.max()) + 1 if hdb_labels.max() >= 0 else 0
    n_noise    = int((hdb_labels == -1).sum())
    log.info('  %d clusters found, %d noise points (%.1f%%)',
             n_clusters, n_noise, 100.0 * n_noise / len(hdb_labels))

    # ── step 4: save companion npz for explore.py (done BEFORE PDF) ──────────
    npz_path = args.npz_output or args.output.with_suffix('.npz')
    npz_data = dict(
        xy_joint   = xy_joint.astype(np.float32),
        xy_img     = xy_img.astype(np.float32),
        xy_sfh     = xy_sfh.astype(np.float32),
        # Raw L2-normalised embeddings (needed for downstream evaluation)
        img_emb    = img_emb.astype(np.float32),
        sfh_emb    = sfh_emb.astype(np.float32),
        galaxy_ids = galaxy_ids,
        h5_indices = h5_indices,           # ← HDF5 row for each point
        redshifts  = redshifts.astype(np.float32),
    )
    for col, key in [('zfinal', 'zfinal'), ('radius_sersic', 'radius_sersic'),
                     ('sersic', 'sersic'), ('axratio_sersic', 'axratio_sersic'),
                     ('_log_mass', 'log_mass'), ('_log_sfr', 'log_sfr'),
                     ('_log_ssfr', 'log_ssfr')]:
        if col in prop_aligned.columns:
            npz_data[key] = prop_aligned[col].values.astype(np.float32)
    _save_cigale_extras(npz_data, prop_aligned)
    for col in sorted(c for c in prop_aligned.columns if 'family' in c):
        npz_data[col] = prop_aligned[col].values.astype(np.float32)
    for col in ('binary_disturbed', 'binary_not_disturbed'):
        if col in prop_aligned.columns:
            npz_data[col] = prop_aligned[col].values.astype(np.float32)
    npz_data['hdbscan_labels'] = hdb_labels

    np.savez_compressed(npz_path, **npz_data)
    log.info('Companion npz → %s', npz_path)
    log.info('  Download this + the h5 file to run explore.py locally')

    # ── step 5: render PDF ────────────────────────────────────────────────────
    family_panels = [
        (col.replace('family_', 'P(').replace('_', ' ').capitalize() + ')',
         col, False, 'RdBu_r')
        for col in sorted(c for c in prop_aligned.columns if 'family' in c)
    ]
    morph_panels = MORPH_PANELS_FIXED + family_panels
    sed_panels   = SED_PANELS[:]
    for entry in _EXTRA_CIGALE:
        if entry[1] in prop_aligned.columns:
            sed_panels.append(entry)

    n_gal     = len(galaxy_ids)
    ckpt_name = args.checkpoint.name
    log.info('Writing PDF → %s', args.output)

    with PdfPages(args.output) as pdf:
        _make_pages(pdf, xy_joint, prop_aligned, morph_panels,
                    f'UMAP (joint, N={n_gal:,}) · Morphology · {ckpt_name}')
        _make_pages(pdf, xy_joint, prop_aligned, sed_panels,
                    f'UMAP (joint, N={n_gal:,}) · SED/CIGALE · {ckpt_name}')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
        fig.suptitle(f'Embedding comparison · redshift z\n{ckpt_name}', fontsize=10, y=1.01)
        for ax, xy, title in zip(axes[0],
                                 [xy_img, xy_sfh, xy_joint],
                                 ['Image (ZooBOT)', 'SFH encoder', 'Joint (avg)']):
            _scatter(ax, xy, redshifts.astype(float), label='redshift z', cmap='plasma')
            ax.set_title(title, fontsize=9, pad=4)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # HDBSCAN cluster map
        fig, ax = plt.subplots(figsize=(8, 7))
        noise_mask   = hdb_labels == -1
        cluster_mask = hdb_labels >= 0
        ax.scatter(xy_joint[noise_mask, 0], xy_joint[noise_mask, 1],
                   c='#cccccc', s=1, alpha=0.3, linewidths=0, label='noise')
        if cluster_mask.any():
            sc = ax.scatter(xy_joint[cluster_mask, 0], xy_joint[cluster_mask, 1],
                            c=hdb_labels[cluster_mask], cmap='tab20',
                            s=2, alpha=0.7, linewidths=0)
            plt.colorbar(sc, ax=ax, label='Cluster ID')
        ax.set_title(f'HDBSCAN clusters (N={n_clusters}, noise={n_noise:,})\n{ckpt_name}',
                     fontsize=9)
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

    log.info('Saved PDF → %s', args.output)


if __name__ == '__main__':
    main()
