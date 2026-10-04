"""Compare frozen GalaxyTikTok and ZooBot representations on identical IDs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

from .umap_zoobot_clip import comparison_page, fit_umap, property_specs


log = logging.getLogger(__name__)
_RESERVED = {
    'galaxy_id', 'h5_row', 'image_embedding', 'xy_image',
    'image_preprojection_embedding', 'xy_image_preprojection',
}


def load_zoobot_archive(path: Path):
    with np.load(path) as source:
        required = {'galaxy_id', 'h5_row', 'xy_image'}
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f'Missing ZooBot arrays in {path}: {missing}')
        galaxy_ids = np.asarray(source['galaxy_id'], dtype=np.int64)
        rows = np.asarray(source['h5_row'], dtype=np.int64)
        coordinates = np.asarray(source['xy_image'], dtype=np.float32)
        embedding = (
            np.asarray(source['image_embedding'], dtype=np.float32)
            if 'image_embedding' in source else None
        )
        properties = {
            key: np.asarray(source[key], dtype=np.float32)
            for key in source.files
            if key not in _RESERVED
            and np.asarray(source[key]).ndim == 1
            and len(np.asarray(source[key])) == len(galaxy_ids)
            and np.asarray(source[key]).dtype.kind in 'biuf'
        }
    if rows.shape != galaxy_ids.shape or coordinates.shape != (len(galaxy_ids), 2):
        raise ValueError('ZooBot archive row or coordinate shapes are invalid.')
    if len(np.unique(galaxy_ids)) != len(galaxy_ids):
        raise ValueError('ZooBot archive contains duplicate galaxy IDs.')
    return galaxy_ids, rows, coordinates, embedding, properties


def encode_tokenizer(tokenizer, loader, band: str, device: torch.device):
    from galactiktok.models.image_transformer import Image
    import torch
    import torch.nn.functional as F

    embeddings = []
    encoded_ids = []
    token_shape = None
    use_amp = device.type == 'cuda'
    tokenizer.eval().to(device)
    with torch.inference_mode():
        for batch_index, (images, galaxy_ids) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=use_amp):
                tokens = tokenizer.encode(
                    Image(flux=images, bands=[band]), mask_fraction=0.0,
                ).tokens
            current_shape = tuple(int(value) for value in tokens.shape[1:])
            if token_shape is None:
                token_shape = current_shape
            elif current_shape != token_shape:
                raise RuntimeError(
                    f'Token shape changed from {token_shape} to {current_shape}.'
                )
            flattened = F.normalize(tokens.flatten(1).float(), dim=1)
            embeddings.append(flattened.cpu().numpy())
            encoded_ids.append(np.asarray(galaxy_ids, dtype=np.int64))
            if batch_index == 0 or (batch_index + 1) % 50 == 0:
                log.info(
                    'GalaxyTikTok encoded %,d images',
                    sum(len(values) for values in encoded_ids),
                )
    return (
        np.concatenate(embeddings).astype(np.float32),
        np.concatenate(encoded_ids),
        token_shape,
    )


def compact_archive(path, galaxy_ids, rows, coordinates, properties):
    arrays = {
        'galaxy_id': galaxy_ids,
        'h5_row': rows,
        'xy_image': coordinates,
    }
    arrays.update(properties)
    np.savez_compressed(path, **arrays)


def run(args):
    from galactiktok import ImageTransformerTokenizer
    import torch
    from torch.utils.data import DataLoader

    from .pretrain_galactiktok_vis import VISFitsDataset

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (galaxy_ids, rows, zoo_coordinates, zoo_embedding,
     properties) = load_zoobot_archive(args.zoobot_archive)
    tokenizer = ImageTransformerTokenizer.from_pretrained(str(args.tokenizer))
    dataset = VISFitsDataset(
        args.fits_root,
        galaxy_ids,
        args.band,
        tokenizer.image_size,
        False,
        args.image_stats,
        args.asinh_scale,
    )
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )
    tokenizer_embedding, encoded_ids, token_shape = encode_tokenizer(
        tokenizer, loader, args.tokenizer_band, device,
    )
    if not np.array_equal(encoded_ids, galaxy_ids):
        raise RuntimeError('GalaxyTikTok DataLoader changed the ZooBot ID order.')

    log.info('Fitting GalaxyTikTok UMAP for %,d identical galaxies', len(galaxy_ids))
    tokenizer_coordinates = fit_umap(
        tokenizer_embedding,
        args.n_neighbors,
        args.min_dist,
        args.seed,
    )
    compact_archive(
        args.output_dir / 'zoobot_image_umap_compact.npz',
        galaxy_ids, rows, zoo_coordinates, properties,
    )
    compact_archive(
        args.output_dir / 'galactiktok_image_umap.npz',
        galaxy_ids, rows, tokenizer_coordinates, properties,
    )
    embedding_arrays = {
        'galaxy_id': galaxy_ids,
        'h5_row': rows,
        'galactiktok_embedding': tokenizer_embedding,
    }
    if zoo_embedding is not None:
        embedding_arrays['zoobot_embedding'] = zoo_embedding
    np.savez_compressed(
        args.output_dir / 'frozen_image_embeddings.npz',
        **embedding_arrays,
    )

    specs = property_specs(properties)
    with PdfPages(args.output_dir / 'galactiktok_vs_zoobot_umap.pdf') as pdf:
        comparison_page(
            pdf,
            [
                ('GalaxyTikTok tokenizer', tokenizer_coordinates),
                ('Frozen ZooBot', zoo_coordinates),
            ],
            list(specs.values()),
            run_label='Same galaxies; independent cosine UMAPs',
        )
    manifest = {
        'n_galaxies': int(len(galaxy_ids)),
        'tokenizer': str(args.tokenizer),
        'zoobot_archive': str(args.zoobot_archive),
        'galactiktok_feature': 'flattened spatial bottleneck tokens, L2-normalized',
        'galactiktok_token_shape': token_shape,
        'galactiktok_embedding_dimension': int(tokenizer_embedding.shape[1]),
        'zoobot_embedding_dimension': (
            int(zoo_embedding.shape[1]) if zoo_embedding is not None else None
        ),
        'umap_metric': 'cosine',
        'umap_n_neighbors': args.n_neighbors,
        'umap_min_dist': args.min_dist,
        'seed': args.seed,
        'properties': sorted(properties),
    }
    (args.output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n',
    )
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--fits-root', type=Path, required=True)
    parser.add_argument('--image-stats', type=Path, required=True)
    parser.add_argument('--zoobot-archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--band', default='VIS')
    parser.add_argument('--tokenizer-band', default='euclid-vis')
    parser.add_argument('--asinh-scale', type=float, default=20.0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('Invalid batch size or worker count.')
    run(args)


if __name__ == '__main__':
    main()
