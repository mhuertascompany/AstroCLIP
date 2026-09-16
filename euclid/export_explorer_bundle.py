"""Create a compact local-explorer bundle from Euclid validation products.

The full preprocessed catalog contains all posterior realizations for 100k
objects. The interactive explorer only needs the median and percentile SFHs for
the objects present in one or more UMAP archives, plus their VIS JPEG stamps.
This command writes that subset and copies the requested UMAP archives into a
single directory. Stamps are collected in an uncompressed tar archive because
JPEG data do not benefit materially from another compression pass.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import h5py
import numpy as np


def _read_rows(dataset, rows):
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(dataset[rows[order]])[inverse]


def _load_archive(path):
    with np.load(path) as archive:
        required = {'galaxy_id', 'h5_row', 'xy_image', 'xy_sfh', 'xy_joint'}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f'Missing arrays in {path}: {missing}')
        ids = np.asarray(archive['galaxy_id'], dtype=np.int64)
        rows = np.asarray(archive['h5_row'], dtype=np.int64)
    if ids.ndim != 1 or rows.shape != ids.shape:
        raise ValueError(f'Invalid galaxy_id or h5_row shape in {path}.')
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'Duplicate galaxy IDs in {path}.')
    return ids, rows


def _common_ids(archives):
    first_ids, _ = _load_archive(archives[0])
    common = set(map(int, first_ids))
    for path in archives[1:]:
        ids, _ = _load_archive(path)
        common.intersection_update(map(int, ids))
    selected = np.array(
        [value for value in first_ids if int(value) in common], dtype=np.int64,
    )
    if not len(selected):
        raise ValueError('The UMAP archives have no galaxy IDs in common.')
    return selected


def _unique_archive_name(path, index):
    run = path.parent.parent.name if path.parent.parent.name else path.parent.name
    return f'{index:02d}_{run}_{path.name}'


def _copy_umap_with_embeddings(source_path, destination):
    """Copy a diagnostic archive, adding sibling evaluation embeddings."""
    with np.load(source_path) as archive:
        output = {key: np.asarray(archive[key]) for key in archive.files}
    added = []
    embeddings_path = source_path.parent / 'validation_embeddings.npz'
    if embeddings_path.is_file() and not {
        'image_embedding', 'sfh_embedding',
    }.issubset(output):
        with np.load(embeddings_path) as embeddings:
            required = {'galaxy_id', 'image_embedding', 'sfh_embedding'}
            missing = sorted(required.difference(embeddings.files))
            if missing:
                raise ValueError(f'Missing arrays in {embeddings_path}: {missing}')
            embedding_ids = np.asarray(embeddings['galaxy_id'], dtype=np.int64)
            by_id = {int(value): index for index, value in enumerate(embedding_ids)}
            archive_ids = np.asarray(output['galaxy_id'], dtype=np.int64)
            try:
                positions = np.array(
                    [by_id[int(value)] for value in archive_ids], dtype=np.int64,
                )
            except KeyError as error:
                raise ValueError(
                    f'An ID in {source_path} is absent from {embeddings_path}.'
                ) from error
            image = np.asarray(embeddings['image_embedding'][positions], dtype=np.float32)
            sfh = np.asarray(embeddings['sfh_embedding'][positions], dtype=np.float32)
            output['image_embedding'] = image
            output['sfh_embedding'] = sfh
            joint = image + sfh
            joint /= np.maximum(np.linalg.norm(joint, axis=1, keepdims=True), 1e-12)
            output['joint_embedding'] = joint.astype(np.float32)
            added = ['image_embedding', 'sfh_embedding', 'joint_embedding']
    np.savez_compressed(destination, **output)
    return added


def export_bundle(dataset_path, stamp_root, umap_paths, output_dir, band='VIS'):
    dataset_path = Path(dataset_path)
    stamp_root = Path(stamp_root)
    umap_paths = [Path(path) for path in umap_paths]
    output_dir = Path(output_dir)
    for path in [dataset_path, *umap_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    stamp_dir = stamp_root / band if (stamp_root / band).is_dir() else stamp_root
    if not stamp_dir.is_dir():
        raise FileNotFoundError(f'Stamp directory not found: {stamp_dir}')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    galaxy_ids = _common_ids(umap_paths)
    stamps = [
        stamp_dir / f'{band}_{int(galaxy_id)}.jpg'
        for galaxy_id in galaxy_ids
    ]
    missing_stamps = [
        int(galaxy_id) for galaxy_id, stamp in zip(galaxy_ids, stamps)
        if not stamp.is_file()
    ]
    if missing_stamps:
        raise FileNotFoundError(
            f'{len(missing_stamps)} requested stamps are missing; first IDs: '
            f'{missing_stamps[:10]}'
        )

    first_ids, first_rows = _load_archive(umap_paths[0])
    row_by_id = dict(zip(map(int, first_ids), map(int, first_rows)))
    source_rows = np.array(
        [row_by_id[int(galaxy_id)] for galaxy_id in galaxy_ids], dtype=np.int64,
    )

    compact_path = output_dir / 'euclid_explorer.h5'
    copied_datasets = []
    with h5py.File(dataset_path, 'r') as source, h5py.File(compact_path, 'w') as target:
        source_ids = _read_rows(source['galaxy_id'], source_rows).astype(np.int64)
        if not np.array_equal(source_ids, galaxy_ids):
            raise ValueError('UMAP h5_row values do not match the source galaxy IDs.')
        target.create_dataset('galaxy_id', data=galaxy_ids)
        target.create_dataset('source_h5_row', data=source_rows)
        for name in ('sfh', 'sfh_p16', 'sfh_p84', 'redshift', 'sfh_time_norm'):
            if name not in source:
                continue
            target.create_dataset(
                name, data=_read_rows(source[name], source_rows),
                compression='gzip', compression_opts=4,
            )
            copied_datasets.append(name)
        if 'sfh_time_grid' not in source:
            raise ValueError('Source HDF5 has no sfh_time_grid dataset.')
        target.create_dataset('sfh_time_grid', data=source['sfh_time_grid'][:])
        for name in ('sfh_log_epsilon', 'sfh_time_coordinate', 'sfh_normalization'):
            if name in source.attrs:
                target.attrs[name] = source.attrs[name]
        target.attrs['source_dataset'] = str(dataset_path)
        target.attrs['n_galaxies'] = len(galaxy_ids)

    local_umaps = []
    embedded_archives = {}
    for index, source_path in enumerate(umap_paths):
        destination = output_dir / _unique_archive_name(source_path, index)
        added = _copy_umap_with_embeddings(source_path, destination)
        local_umaps.append(destination.name)
        embedded_archives[destination.name] = added

    tar_path = output_dir / f'{band}_stamps.tar'
    with tarfile.open(tar_path, mode='w') as archive:
        for stamp in stamps:
            archive.add(stamp, arcname=f'{band}/{stamp.name}', recursive=False)

    manifest = {
        'n_galaxies': int(len(galaxy_ids)),
        'band': band,
        'data': compact_path.name,
        'stamp_archive': tar_path.name,
        'umap_archives': local_umaps,
        'arrays_added_from_validation_embeddings': embedded_archives,
        'copied_hdf5_datasets': copied_datasets,
        'source_dataset': str(dataset_path),
        'source_stamp_directory': str(stamp_dir),
        'source_umap_archives': [str(path) for path in umap_paths],
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--stamp-root', type=Path, required=True)
    parser.add_argument('--umap', type=Path, action='append', required=True,
                        help='Euclid diagnostic NPZ; repeat to compare runs.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--band', default='VIS')
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = export_bundle(
        args.dataset, args.stamp_root, args.umap, args.output_dir, args.band,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
