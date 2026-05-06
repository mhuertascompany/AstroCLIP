"""
UMAP visualization of the CosmosWebZooBotCLIP v5 embedding space.

Like umap_embeddings_zoobot.py but each galaxy's image is loaded from the
rest-frame optical filter stamp (F150W for z<1, F277W for 1≤z<3, F444W for
z≥3) instead of always using F277W.  The model is still a single-backbone
CosmosWebZooBotCLIP (family_2.ckpt covers all three filters).

Usage (from repo root):
    python -m cosmosweb.umap_embeddings_zoobot_multifilter \\
        --checkpoint /path/to/cosmosweb_zoobot_v5-epoch=xxx-val_loss=x.xxxx.ckpt \\
        --dataset    /path/to/cosmosweb_dataset_v4.h5 \\
        --stamp_root /path/to/stamps_ilbert \\
        --output     /path/to/umap_plots_zoobot_v5.pdf \\
        --npz_output /path/to/cosmosweb_umap_zoobot_v5.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

try:
    from hdbscan import HDBSCAN
except ImportError:
    from sklearn.cluster import HDBSCAN

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


def _filter_for_z(z: float, z_low: float, z_high: float) -> str:
    if z < z_low:
        return 'F150W'
    if z >= z_high:
        return 'F444W'
    return 'F277W'


def extract_embeddings(
    checkpoint:  Path,
    h5_path:     Path,
    stamp_root:  Path,
    z_low:       float = 1.0,
    z_high:      float = 3.0,
    batch_size:  int   = 128,
    device:      str   = 'cuda',
    image_size:  int   = 224,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load model; encode each image from its per-galaxy rest-frame filter stamp
    and SFHs from HDF5.

    Returns
    -------
    img_emb    : (M, D)
    sfh_emb    : (M, D)
    galaxy_ids : (M,)
    redshifts  : (M,)
    h5_indices : (M,)
    """
    log.info('Loading checkpoint %s', checkpoint)
    model = CosmosWebZooBotCLIP.load_from_checkpoint(
        str(checkpoint), map_location='cpu'
    )
    model.eval()
    model.to(device)

    transform = _inference_transform(image_size)

    log.info('Scanning HDF5 for galaxies with valid per-filter stamps…')
    with h5py.File(h5_path, 'r') as f:
        all_ids  = f['galaxy_id'][:]
        all_sfhs = f['sfh'][:]
        all_z    = f['redshift'][:]

    valid_h5idx = []
    valid_ids   = []
    valid_sfhs  = []
    valid_z     = []
    valid_filts = []

    for h5_idx, (gid, z) in enumerate(zip(all_ids, all_z)):
        filt = _filter_for_z(float(z), z_low, z_high)
        stamp_path = Path(stamp_root) / filt / f'{filt}_{int(gid)}.jpg'
        if stamp_path.exists():
            valid_h5idx.append(h5_idx)
            valid_ids.append(int(gid))
            valid_sfhs.append(all_sfhs[h5_idx])
            valid_z.append(float(z))
            valid_filts.append(filt)

    n_total = len(all_ids)
    n_valid = len(valid_ids)
    counts  = {f: valid_filts.count(f) for f in ['F150W', 'F277W', 'F444W']}
    log.info('  %d / %d HDF5 galaxies have valid stamps — %s', n_valid, n_total, counts)

    if n_valid == 0:
        raise RuntimeError(f'No stamps found under {stamp_root}')

    h5_indices = np.array(valid_h5idx, dtype=np.int64)
    galaxy_ids = np.array(valid_ids,   dtype=np.int64)
    sfhs_arr   = np.stack(valid_sfhs,  axis=0).astype(np.float32)
    redshifts  = np.array(valid_z,     dtype=np.float32)

    img_embs, sfh_embs = [], []

    with torch.no_grad():
        for start in range(0, n_valid, batch_size):
            end = min(start + batch_size, n_valid)

            imgs = []
            for i in range(start, end):
                gid  = valid_ids[i]
                filt = valid_filts[i]
                path = Path(stamp_root) / filt / f'{filt}_{gid}.jpg'
                imgs.append(transform(Image.open(path)))
            images = torch.stack(imgs).to(device)
            sfhs   = torch.tensor(sfhs_arr[start:end]).to(device)

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='UMAP of CosmosWebZooBotCLIP v5 (multi-filter stamps)')
    p.add_argument('--checkpoint',   type=Path, required=True)
    p.add_argument('--dataset',      type=Path, required=True)
    p.add_argument('--stamp_root',   type=Path,
                   default='/n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert')
    p.add_argument('--z_low',        type=float, default=1.0)
    p.add_argument('--z_high',       type=float, default=3.0)
    p.add_argument('--image_size',   type=int,   default=224)
    p.add_argument('--output',       type=Path, required=True)
    p.add_argument('--npz_output',   type=Path, default=None)
    p.add_argument('--morpho_cat',   type=Path,
                   default='/n03data/huertas/COSMOS-Web/ilbert_finetune/ilbert_visual_zoobot_morphology.fits')
    p.add_argument('--photom_cat',   type=Path,
                   default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_photom_primary.fits')
    p.add_argument('--lephare_cat',  type=Path,
                   default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_lephare.fits')
    p.add_argument('--cigale_cat',   type=Path,
                   default=f'{_CAT_DIR}/COSMOSWeb_mastercatalog_v1_cigale.fits')
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--n_neighbors',  type=int,   default=15)
    p.add_argument('--min_dist',     type=float, default=0.1)
    p.add_argument('--device',       type=str,   default='cuda')
    p.add_argument('--max_galaxies', type=int,   default=0)
    p.add_argument('--clustering',   type=str,   default='kmeans',
                   choices=['kmeans', 'hdbscan'])
    p.add_argument('--n_clusters',        type=int, default=12)
    p.add_argument('--min_cluster_size',  type=int, default=200)
    p.add_argument('--min_samples',       type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    img_emb, sfh_emb, galaxy_ids, redshifts, h5_indices = extract_embeddings(
        checkpoint=args.checkpoint,
        h5_path=args.dataset,
        stamp_root=args.stamp_root,
        z_low=args.z_low,
        z_high=args.z_high,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    if args.max_galaxies > 0:
        n   = min(args.max_galaxies, len(galaxy_ids))
        idx = np.random.default_rng(0).choice(len(galaxy_ids), n, replace=False)
        img_emb    = img_emb[idx]
        sfh_emb    = sfh_emb[idx]
        galaxy_ids = galaxy_ids[idx]
        redshifts  = redshifts[idx]
        h5_indices = h5_indices[idx]

    joint_emb = F.normalize(
        torch.tensor(img_emb + sfh_emb), dim=-1
    ).numpy()

    log.info('Loading catalog properties…')
    prop_df = load_properties(
        str(args.photom_cat), str(args.lephare_cat),
        str(args.cigale_cat), str(args.morpho_cat),
        galaxy_ids,
    )
    prop_df = _derive_cigale(prop_df)
    prop_df.index = prop_df.index.astype(np.int64)
    prop_aligned = prop_df.reindex(galaxy_ids.astype(np.int64))

    log.info('Fitting UMAP…')
    xy_joint = fit_umap(joint_emb, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    xy_img   = fit_umap(img_emb,   n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    xy_sfh   = fit_umap(sfh_emb,   n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    if args.clustering == 'kmeans':
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init='auto')
        hdb_labels = km.fit_predict(xy_joint).astype(np.int32)
        n_clusters = args.n_clusters
        n_noise    = 0
    else:
        clusterer  = HDBSCAN(min_cluster_size=args.min_cluster_size,
                             min_samples=args.min_samples)
        hdb_labels = clusterer.fit_predict(xy_joint).astype(np.int32)
        n_clusters = int(hdb_labels.max()) + 1 if hdb_labels.max() >= 0 else 0
        n_noise    = int((hdb_labels == -1).sum())
        log.info('  %d clusters, %d noise points', n_clusters, n_noise)

    npz_path = args.npz_output or args.output.with_suffix('.npz')
    npz_data = dict(
        xy_joint   = xy_joint.astype(np.float32),
        xy_img     = xy_img.astype(np.float32),
        xy_sfh     = xy_sfh.astype(np.float32),
        img_emb    = img_emb.astype(np.float32),
        sfh_emb    = sfh_emb.astype(np.float32),
        galaxy_ids = galaxy_ids,
        h5_indices = h5_indices,
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
        fig.suptitle(f'Embedding comparison · redshift z\n{ckpt_name}',
                     fontsize=10, y=1.01)
        for ax, xy, title in zip(axes[0],
                                 [xy_img, xy_sfh, xy_joint],
                                 ['Image (ZooBOT multi-filter)', 'SFH encoder', 'Joint (avg)']):
            _scatter(ax, xy, redshifts.astype(float), label='redshift z', cmap='plasma')
            ax.set_title(title, fontsize=9, pad=4)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

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
        ax.set_title(
            f'Clusters (N={n_clusters}, noise={n_noise:,})\n{ckpt_name}', fontsize=9)
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

    log.info('Saved PDF → %s', args.output)


if __name__ == '__main__':
    main()
