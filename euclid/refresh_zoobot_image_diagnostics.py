"""Refresh ZooBot image UMAP properties without re-encoding the JPEGs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .export_zoobot_image_embeddings import write_diagnostic_products


def refresh_diagnostics(archive_path, dataset_path, output_dir):
    archive_path = Path(archive_path)
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(archive_path) as source:
        required = {
            'galaxy_id', 'h5_row', 'image_embedding', 'xy_image',
        }
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f'Missing arrays in {archive_path}: {missing}')
        galaxy_ids = np.asarray(source['galaxy_id'], dtype=np.int64)
        rows = np.asarray(source['h5_row'], dtype=np.int64)
        embedding = np.asarray(source['image_embedding'], dtype=np.float32)
        coordinates = np.asarray(source['xy_image'], dtype=np.float32)
    n_objects = len(galaxy_ids)
    if rows.shape != (n_objects,):
        raise ValueError('h5_row must have the same one-dimensional shape as galaxy_id.')
    if embedding.ndim != 2 or len(embedding) != n_objects:
        raise ValueError('image_embedding must be a row-aligned two-dimensional array.')
    if coordinates.shape != (n_objects, 2):
        raise ValueError('xy_image must have shape (n_objects, 2).')

    diagnostics = write_diagnostic_products(
        dataset_path, output_dir, galaxy_ids, rows, embedding, coordinates,
    )
    manifest = {
        'source_archive': str(archive_path),
        'dataset': str(dataset_path),
        'n_objects': n_objects,
        'embedding_dimension': int(embedding.shape[1]),
        'umap_recomputed': False,
        **diagnostics,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )
    manifest = refresh_diagnostics(args.archive, args.dataset, args.output_dir)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
