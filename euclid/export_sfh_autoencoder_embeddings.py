"""Export held-out SFH autoencoder embeddings for the local Euclid explorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .pretrain_sfh_autoencoder import EuclidSFHAutoencoder
from .umap_zoobot_clip import (
    _catalog_property,
    _read_rows,
    _sfh_properties,
    fit_umap,
)


CATALOG_SPECS = {
    'log_stellar_mass': (
        ['phz_pp_median_stellarmass'], lambda x: (x > 0) & (x < 20),
    ),
    'sersic_index': (
        ['sersic_sersic_vis_index'], lambda x: (x > 0) & (x < 20),
    ),
    'sersic_radius': (['sersic_sersic_vis_radius'], lambda x: x > 0),
    'axis_ratio': (
        ['sersic_sersic_vis_axis_ratio'], lambda x: (x > 0) & (x <= 1),
    ),
    'fwhm': (['fwhm'], lambda x: x > 0),
    'kron_radius': (['kron_radius'], lambda x: x > 0),
    'semimajor_axis': (['semimajor_axis'], lambda x: x > 0),
    'point_like_probability': (
        ['point_like_prob'], lambda x: (x >= 0) & (x <= 1),
    ),
}


def load_validation_rows(dataset_path, split_path, max_objects=0, seed=42):
    with h5py.File(dataset_path, 'r') as source, np.load(split_path) as split:
        required = {'val_rows', 'val_ids'}
        missing = sorted(required.difference(split.files))
        if missing:
            raise ValueError(f'Missing split arrays: {missing}')
        rows = np.asarray(split['val_rows'], dtype=np.int64)
        galaxy_ids = np.asarray(split['val_ids'], dtype=np.int64)
        if not np.array_equal(_read_rows(source['galaxy_id'], rows), galaxy_ids):
            raise ValueError('Validation IDs do not match their HDF5 rows.')
    if max_objects > 0 and len(rows) > max_objects:
        selected = np.sort(
            np.random.default_rng(seed).choice(len(rows), max_objects, replace=False)
        )
        rows, galaxy_ids = rows[selected], galaxy_ids[selected]
    return rows, galaxy_ids


def encode_sfhs(model, log_sfhs, device, batch_size=512):
    embeddings = []
    reconstructions = []
    use_amp = device.type == 'cuda'
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(log_sfhs), batch_size):
            batch = torch.from_numpy(log_sfhs[start:start + batch_size]).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp,
            ):
                latent = model.encoder(batch)
                reconstruction = model.decoder(latent)
            embeddings.append(latent.float().cpu().numpy())
            reconstructions.append(reconstruction.float().cpu().numpy())
            print(f'Encoded {min(start + batch_size, len(log_sfhs)):,}/'
                  f'{len(log_sfhs):,}', flush=True)
    embedding = np.concatenate(embeddings).astype(np.float32)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError('Encoder produced a nonfinite or zero-norm embedding.')
    return embedding / norms, np.concatenate(reconstructions).astype(np.float32)


def linear_sfhs(log_sfhs, epsilon):
    weights = np.maximum(10.0 ** np.asarray(log_sfhs, dtype=np.float64) - epsilon, 0.0)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), epsilon)


def redshift_probe(embedding, sfh_mass, redshift, seed=42):
    """Compare redshift predictability from the latent and raw SFH shape."""
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import NearestNeighbors
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError('scikit-learn is required for the redshift probe.') from error

    valid = np.isfinite(redshift)
    indices = np.flatnonzero(valid)
    if len(indices) < 20:
        raise ValueError('At least 20 finite redshifts are required for the probe.')
    train, test = train_test_split(indices, test_size=0.25, random_state=seed)

    def score(features):
        regression = make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
        regression.fit(features[train], redshift[train])
        return float(regression.score(features[test], redshift[test]))

    n_neighbors = min(11, len(indices))

    def neighbor_positions(features):
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        neighbors.fit(features[indices])
        candidates = neighbors.kneighbors(
            features[indices], return_distance=False,
        )
        return np.stack([
            row[row != position][:n_neighbors - 1]
            for position, row in enumerate(candidates)
        ])

    latent_neighbor_positions = neighbor_positions(embedding)
    raw_neighbor_positions = neighbor_positions(sfh_mass)
    z = redshift[indices]
    latent_neighbor_delta = np.abs(z[:, None] - z[latent_neighbor_positions])
    raw_neighbor_delta = np.abs(z[:, None] - z[raw_neighbor_positions])
    rng = np.random.default_rng(seed)
    offsets = rng.integers(
        1, len(indices), size=latent_neighbor_positions.shape,
    )
    random_positions = (
        np.arange(len(indices))[:, None] + offsets
    ) % len(indices)
    random_delta = np.abs(z[:, None] - z[random_positions])
    return {
        'n_objects': int(len(indices)),
        'latent_linear_redshift_r2': score(embedding),
        'raw_sfh_linear_redshift_r2': score(sfh_mass),
        'latent_neighbor_mean_abs_delta_z': float(latent_neighbor_delta.mean()),
        'raw_sfh_neighbor_mean_abs_delta_z': float(raw_neighbor_delta.mean()),
        'random_neighbor_mean_abs_delta_z': float(random_delta.mean()),
    }


def shape_probe(embedding, sfh_mass, seed=42, max_objects=5000, k=10):
    """Measure whether latent neighbours have similar input SFH shapes."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as error:
        raise ImportError('scikit-learn is required for the shape probe.') from error

    if len(embedding) < 2:
        raise ValueError('At least two objects are required for the shape probe.')
    rng = np.random.default_rng(seed)
    sample = np.arange(len(embedding))
    if len(sample) > max_objects:
        sample = np.sort(rng.choice(sample, max_objects, replace=False))
    latent = embedding[sample]
    sfh = sfh_mass[sample]
    neighbor_count = min(k, len(sample) - 1)

    def positions(features):
        model = NearestNeighbors(
            n_neighbors=neighbor_count + 1, metric='cosine',
        ).fit(features)
        candidates = model.kneighbors(features, return_distance=False)
        return np.stack([
            row[row != position][:neighbor_count]
            for position, row in enumerate(candidates)
        ])

    latent_neighbors = positions(latent)
    raw_neighbors = positions(sfh)
    sfh_norm = sfh / np.maximum(
        np.linalg.norm(sfh, axis=1, keepdims=True), 1e-12,
    )
    latent_neighbor_cosine = np.sum(
        sfh_norm[:, None, :] * sfh_norm[latent_neighbors], axis=-1,
    )
    offsets = rng.integers(1, len(sample), size=latent_neighbors.shape)
    random_neighbors = (
        np.arange(len(sample))[:, None] + offsets
    ) % len(sample)
    random_cosine = np.sum(
        sfh_norm[:, None, :] * sfh_norm[random_neighbors], axis=-1,
    )
    overlap = np.mean([
        len(set(latent_row).intersection(raw_row)) / neighbor_count
        for latent_row, raw_row in zip(latent_neighbors, raw_neighbors)
    ])
    return {
        'n_objects': int(len(sample)),
        'k': int(neighbor_count),
        'latent_neighbor_raw_sfh_cosine_mean': float(latent_neighbor_cosine.mean()),
        'random_pair_raw_sfh_cosine_mean': float(random_cosine.mean()),
        'latent_raw_neighbor_overlap_at_k': float(overlap),
        'chance_neighbor_overlap_at_k': float(neighbor_count / (len(sample) - 1)),
    }


def write_compact_h5(dataset_path, output_path, rows, galaxy_ids, reconstruction):
    with h5py.File(dataset_path, 'r') as source, h5py.File(output_path, 'w') as target:
        target['galaxy_id'] = galaxy_ids
        target['source_h5_row'] = rows
        target.create_dataset(
            'sfh_reconstruction', data=np.asarray(reconstruction, dtype=np.float32),
            compression='gzip', compression_opts=4,
        )
        for name in ('sfh', 'sfh_p16', 'sfh_p84', 'redshift', 'sfh_time_norm'):
            if name in source:
                target.create_dataset(
                    name, data=_read_rows(source[name], rows),
                    compression='gzip', compression_opts=4,
                )
        target['sfh_time_grid'] = source['sfh_time_grid'][:]
        for name in ('sfh_log_epsilon', 'sfh_time_coordinate', 'sfh_normalization'):
            if name in source.attrs:
                target.attrs[name] = source.attrs[name]
        target.attrs['source_dataset'] = str(dataset_path)
        target.attrs['n_galaxies'] = len(galaxy_ids)


def export_embeddings(checkpoint, dataset, split, output_dir, device='auto',
                      batch_size=512, n_neighbors=15, min_dist=0.1,
                      max_objects=0, seed=42):
    checkpoint, dataset, split = map(Path, (checkpoint, dataset, split))
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    if n_neighbors < 2:
        raise ValueError('n_neighbors must be at least two.')
    if not 0 <= min_dist <= 1:
        raise ValueError('min_dist must lie in [0, 1].')
    output_dir = Path(output_dir)
    for path in (checkpoint, dataset, split):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    if device == 'auto':
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device(device)
    if torch_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')

    rows, galaxy_ids = load_validation_rows(dataset, split, max_objects, seed)
    with h5py.File(dataset, 'r') as source:
        log_sfhs = _read_rows(source['sfh'], rows).astype(np.float32)
        redshift = _read_rows(source['redshift'], rows).astype(np.float32)
        p16 = _read_rows(source['sfh_p16'], rows).astype(np.float32)
        p84 = _read_rows(source['sfh_p84'], rows).astype(np.float32)
        epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))

    model = EuclidSFHAutoencoder.load_from_checkpoint(
        str(checkpoint), map_location='cpu',
    ).to(torch_device)
    embedding, reconstruction = encode_sfhs(
        model, log_sfhs, torch_device, batch_size,
    )
    sfh_mass = linear_sfhs(log_sfhs, epsilon)
    reconstruction_w1 = np.abs(
        np.cumsum(reconstruction, axis=1) - np.cumsum(sfh_mass, axis=1)
    ).mean(axis=1)
    reconstruction_mae = np.abs(reconstruction - sfh_mass).mean(axis=1)
    lower = np.maximum(10.0 ** p16.astype(np.float64) - epsilon, 0.0)
    upper = np.maximum(10.0 ** p84.astype(np.float64) - epsilon, 0.0)

    properties = {
        'redshift': redshift,
        'reconstruction_w1': reconstruction_w1.astype(np.float32),
        'reconstruction_mae': reconstruction_mae.astype(np.float32),
        'sfh_posterior_width': (
            np.maximum(upper - lower, 0.0).mean(axis=1).astype(np.float32)
        ),
    }
    with h5py.File(dataset, 'r') as source:
        for key, (candidates, valid) in CATALOG_SPECS.items():
            values, _ = _catalog_property(source, rows, candidates, valid)
            if values is not None:
                properties[key] = values
        properties.update(_sfh_properties(source, rows))

    xy_sfh = fit_umap(embedding, n_neighbors, min_dist, seed)
    archive_path = output_dir / 'sfh_autoencoder_umap.npz'
    np.savez_compressed(
        archive_path,
        galaxy_id=galaxy_ids,
        h5_row=rows,
        sfh_embedding=embedding,
        xy_sfh=xy_sfh,
        **properties,
    )
    compact_path = output_dir / 'euclid_explorer.h5'
    write_compact_h5(
        dataset, compact_path, rows, galaxy_ids, reconstruction,
    )
    metrics = {
        'checkpoint': str(checkpoint),
        'dataset': str(dataset),
        'split': str(split),
        'n_objects': int(len(rows)),
        'embedding_dimension': int(embedding.shape[1]),
        'reconstruction_w1_mean': float(reconstruction_w1.mean()),
        'reconstruction_w1_median': float(np.median(reconstruction_w1)),
        'shape_probe': shape_probe(embedding, sfh_mass, seed),
        'redshift_probe': redshift_probe(embedding, sfh_mass, redshift, seed),
        'archive': archive_path.name,
        'compact_h5': compact_path.name,
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--max-objects', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    export_embeddings(
        args.checkpoint, args.dataset, args.split, args.output_dir,
        args.device, args.batch_size, args.n_neighbors, args.min_dist,
        args.max_objects, args.seed,
    )


if __name__ == '__main__':
    main()
