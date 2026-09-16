"""Quantitative validation of a trained Euclid image--SFH CLIP checkpoint.

This entry point deliberately produces tables and machine-readable files before
any UMAP or gallery inspection.  It evaluates the exact saved validation split,
so no training objects can silently enter the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

from cosmosweb.model_zoobot import CosmosWebZooBotCLIP

from .dataset_zoobot import EuclidZooBotDataset
from .training_index import inspect_sfh_file


log = logging.getLogger(__name__)


def _read_rows(dataset, rows):
    """Read arbitrary HDF5 rows while satisfying h5py's sorted-index rule."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(dataset[rows[order]])[inverse]


def validate_saved_split(dataset_path, split_path, stamp_root, band='VIS'):
    """Validate IDs, row assignments, train/validation separation, and stamps."""
    all_ids, n_bins, n_realizations = inspect_sfh_file(dataset_path)
    split = np.load(split_path)
    required = {'train_rows', 'train_ids', 'val_rows', 'val_ids'}
    missing = sorted(required.difference(split.files))
    if missing:
        raise ValueError(f'Missing arrays in {split_path}: {missing}')

    train_rows = np.asarray(split['train_rows'], dtype=np.int64)
    train_ids = np.asarray(split['train_ids'], dtype=np.int64)
    val_rows = np.asarray(split['val_rows'], dtype=np.int64)
    val_ids = np.asarray(split['val_ids'], dtype=np.int64)
    if len(train_rows) != len(train_ids) or len(val_rows) != len(val_ids):
        raise ValueError('Saved split row and ID arrays have inconsistent lengths.')
    if not len(val_rows):
        raise ValueError('Saved validation split is empty.')
    if np.any(train_rows < 0) or np.any(val_rows < 0):
        raise ValueError('Saved split contains negative HDF5 rows.')
    if np.any(train_rows >= len(all_ids)) or np.any(val_rows >= len(all_ids)):
        raise ValueError('Saved split contains an HDF5 row outside the dataset.')
    if not np.array_equal(all_ids[train_rows], train_ids):
        raise ValueError('Training IDs do not match their saved HDF5 rows.')
    if not np.array_equal(all_ids[val_rows], val_ids):
        raise ValueError('Validation IDs do not match their saved HDF5 rows.')
    if len(np.unique(train_ids)) != len(train_ids):
        raise ValueError('Training split contains duplicate galaxy IDs.')
    if len(np.unique(val_ids)) != len(val_ids):
        raise ValueError('Validation split contains duplicate galaxy IDs.')
    if np.intersect1d(train_ids, val_ids).size:
        raise ValueError('Training and validation galaxy IDs overlap.')

    stamp_dir = Path(stamp_root) / band
    missing_stamps = [
        int(galaxy_id) for galaxy_id in val_ids
        if not (stamp_dir / f'{band}_{int(galaxy_id)}.jpg').is_file()
    ]
    if missing_stamps:
        preview = ', '.join(map(str, missing_stamps[:5]))
        raise FileNotFoundError(
            f'{len(missing_stamps)} validation stamps are missing; first IDs: {preview}'
        )
    return train_rows, train_ids, val_rows, val_ids, n_bins, n_realizations


def extraction_loader(dataset_path, stamp_root, rows, galaxy_ids, band,
                      image_size, batch_size, num_workers):
    dataset = EuclidZooBotDataset(
        sfh_path=dataset_path,
        stamp_root=stamp_root,
        rows=rows,
        galaxy_ids=galaxy_ids,
        band=band,
        training=False,
        sample_posterior=False,
        image_size=image_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def extract_embeddings(model, loader, device):
    image_embeddings = []
    sfh_embeddings = []
    reconstructions = []
    seen_ids = []
    use_amp = device.type == 'cuda'
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch['image'].to(device, non_blocking=True)
            sfhs = batch['sfh'].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                image_embedding = F.normalize(model.encode_image(images), dim=-1)
                sfh_features = model.encode_sfh(sfhs)
                sfh_embedding = F.normalize(sfh_features, dim=-1)
                if model.sfh_decoder is not None:
                    reconstructions.append(
                        model.sfh_decoder(sfh_features).float().cpu().numpy()
                    )
            image_embeddings.append(image_embedding.float().cpu().numpy())
            sfh_embeddings.append(sfh_embedding.float().cpu().numpy())
            seen_ids.append(batch['galaxy_id'].numpy())
            if batch_index % 20 == 0:
                log.info('Encoded %d validation batches', batch_index + 1)
    reconstruction = (
        np.concatenate(reconstructions) if reconstructions else None
    )
    return (
        np.concatenate(image_embeddings),
        np.concatenate(sfh_embeddings),
        np.concatenate(seen_ids),
        reconstruction,
    )


def summarize_sfh_reconstruction(prediction, target_log, epsilon=1e-10):
    """Report per-object shape errors for a decoder-enabled checkpoint."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.maximum(10.0 ** np.asarray(target_log, dtype=np.float64) - epsilon, 0.0)
    target /= np.maximum(target.sum(axis=1, keepdims=True), epsilon)
    if prediction.shape != target.shape:
        raise ValueError('SFH reconstruction and target shapes differ.')
    w1 = np.abs(np.cumsum(prediction, axis=1) - np.cumsum(target, axis=1)).mean(axis=1)
    mae = np.abs(prediction - target).mean(axis=1)
    return {
        'n_objects': len(prediction),
        'w1_mean': float(w1.mean()),
        'w1_median': float(np.median(w1)),
        'w1_p90': float(np.percentile(w1, 90)),
        'mae_mean': float(mae.mean()),
        'mae_median': float(np.median(mae)),
    }


def retrieval_ranks(query, gallery, chunk_size=512):
    """Return one-indexed exact-match ranks for aligned query/gallery arrays."""
    if query.shape != gallery.shape or query.ndim != 2:
        raise ValueError('Query and gallery must have the same two-dimensional shape.')
    n_objects = len(query)
    ranks = np.empty(n_objects, dtype=np.int32)
    for start in range(0, n_objects, chunk_size):
        stop = min(start + chunk_size, n_objects)
        similarity = query[start:stop] @ gallery.T
        positive = similarity[np.arange(stop - start), np.arange(start, stop)]
        ranks[start:stop] = 1 + np.count_nonzero(
            similarity > positive[:, None], axis=1,
        )
    return ranks


def summarize_ranks(ranks, ks=(1, 5, 10, 50, 100)):
    ranks = np.asarray(ranks)
    n_objects = len(ranks)
    return {
        'n_objects': n_objects,
        'recall': {
            f'R@{k}': float(np.mean(ranks <= k))
            for k in ks if k <= n_objects
        },
        'chance_recall': {
            f'R@{k}': float(k / n_objects)
            for k in ks if k <= n_objects
        },
        'median_rank': float(np.median(ranks)),
        'median_rank_percentile': float(np.median(ranks / n_objects)),
        'mean_reciprocal_rank': float(np.mean(1.0 / ranks)),
    }


def _derangement(n_objects, rng):
    if n_objects < 2:
        raise ValueError('At least two objects are required for a shuffled null.')
    shift = int(rng.integers(1, n_objects))
    return np.roll(np.arange(n_objects), shift)


def shuffled_alignment_test(image_embedding, sfh_embedding, rng,
                            n_permutations=200, permutation_size=5000):
    n_objects = len(image_embedding)
    size = min(n_objects, permutation_size)
    selected = np.sort(rng.choice(n_objects, size=size, replace=False))
    image = image_embedding[selected]
    sfh = sfh_embedding[selected]
    paired = np.sum(image * sfh, axis=1)

    shuffled_index = _derangement(size, rng)
    shuffled = np.sum(image * sfh[shuffled_index], axis=1)
    differences = paired - shuffled
    null_means = np.empty(n_permutations, dtype=np.float64)
    for index in range(n_permutations):
        permutation = _derangement(size, rng)
        null_means[index] = np.mean(np.sum(image * sfh[permutation], axis=1))

    bootstrap_means = np.empty(1000, dtype=np.float64)
    for index in range(len(bootstrap_means)):
        draw = rng.integers(0, size, size=size)
        bootstrap_means[index] = differences[draw].mean()
    observed_mean = float(paired.mean())
    return {
        'n_objects': size,
        'paired_cosine_mean': observed_mean,
        'paired_cosine_std': float(paired.std()),
        'shuffled_cosine_mean': float(shuffled.mean()),
        'shuffled_cosine_std': float(shuffled.std()),
        'paired_minus_shuffled_mean': float(differences.mean()),
        'paired_minus_shuffled_95pct_ci': [
            float(value) for value in np.percentile(bootstrap_means, [2.5, 97.5])
        ],
        'permutation_null_mean': float(null_means.mean()),
        'permutation_null_std': float(null_means.std()),
        'permutation_p_one_sided': float(
            (1 + np.count_nonzero(null_means >= observed_mean)) /
            (n_permutations + 1)
        ),
        'n_permutations': n_permutations,
    }


def embedding_diagnostics(embedding, rng, n_random_pairs=20000):
    """Detect constant dimensions or concentration into very few directions."""
    centered = embedding - embedding.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(embedding) - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = eigenvalues.sum()
    if total > 0:
        probabilities = eigenvalues / total
        nonzero = probabilities > 0
        effective_rank = np.exp(
            -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
        )
        participation_ratio = 1.0 / np.sum(probabilities ** 2)
    else:
        effective_rank = 0.0
        participation_ratio = 0.0

    left = rng.integers(0, len(embedding), size=n_random_pairs)
    right = rng.integers(0, len(embedding), size=n_random_pairs)
    same = left == right
    right[same] = (right[same] + 1) % len(embedding)
    random_cosines = np.sum(embedding[left] * embedding[right], axis=1)
    dimension_std = embedding.std(axis=0)
    return {
        'dimension': int(embedding.shape[1]),
        'mean_dimension_std': float(dimension_std.mean()),
        'minimum_dimension_std': float(dimension_std.min()),
        'dimensions_std_above_1e-3': int(np.count_nonzero(dimension_std > 1e-3)),
        'effective_rank': float(effective_rank),
        'participation_ratio': float(participation_ratio),
        'random_pair_cosine_mean': float(random_cosines.mean()),
        'random_pair_cosine_std': float(random_cosines.std()),
    }


def sfh_shape_neighborhood_test(image_embedding, sfh_embedding, raw_sfh, rng,
                                subset_size=2000, k=10):
    """Test whether images retrieve SFHs with a similar physical shape."""
    size = min(len(raw_sfh), subset_size)
    if size < 2:
        raise ValueError('SFH neighborhood evaluation requires at least two objects.')
    selected = np.sort(rng.choice(len(raw_sfh), size=size, replace=False))
    image = image_embedding[selected]
    sfh_embedding = sfh_embedding[selected]
    linear = np.maximum(10.0 ** raw_sfh[selected] - 1e-10, 0.0)
    linear /= np.maximum(linear.sum(axis=1, keepdims=True), 1e-12)
    unit_linear = linear / np.maximum(
        np.linalg.norm(linear, axis=1, keepdims=True), 1e-12,
    )

    cross_similarity = image @ sfh_embedding.T
    raw_similarity = unit_linear @ unit_linear.T
    cumulative = np.cumsum(linear, axis=1)
    raw_wasserstein = cdist(cumulative, cumulative, metric='cityblock')
    raw_wasserstein /= max(linear.shape[1] - 1, 1)
    np.fill_diagonal(cross_similarity, -np.inf)
    np.fill_diagonal(raw_similarity, -np.inf)
    np.fill_diagonal(raw_wasserstein, np.inf)
    k = min(k, size - 1)
    cross_top = np.argpartition(cross_similarity, -k, axis=1)[:, -k:]
    raw_top = np.argpartition(raw_similarity, -k, axis=1)[:, -k:]
    wasserstein_top = np.argpartition(raw_wasserstein, k, axis=1)[:, :k]
    retrieved_shape_cosine = np.take_along_axis(
        raw_similarity, cross_top, axis=1,
    ).mean(axis=1)
    random_index = _derangement(size, rng)
    random_shape_cosine = np.sum(
        unit_linear * unit_linear[random_index], axis=1,
    )
    retrieved_shape_wasserstein = np.take_along_axis(
        raw_wasserstein, cross_top, axis=1,
    ).mean(axis=1)
    random_shape_wasserstein = raw_wasserstein[np.arange(size), random_index]
    cosine_overlaps = np.array([
        np.intersect1d(cross_top[row], raw_top[row], assume_unique=True).size / k
        for row in range(size)
    ])
    wasserstein_overlaps = np.array([
        np.intersect1d(
            cross_top[row], wasserstein_top[row], assume_unique=True,
        ).size / k
        for row in range(size)
    ])
    return {
        'n_objects': size,
        'k': k,
        'mean_raw_sfh_cosine_of_cross_modal_neighbors': float(
            retrieved_shape_cosine.mean()
        ),
        'mean_raw_sfh_cosine_of_random_neighbors': float(random_shape_cosine.mean()),
        # Retain the original key for compatibility with existing reports.
        'raw_sfh_neighbor_overlap_at_k': float(cosine_overlaps.mean()),
        'raw_sfh_cosine_neighbor_overlap_at_k': float(cosine_overlaps.mean()),
        'mean_raw_sfh_wasserstein_of_cross_modal_neighbors': float(
            retrieved_shape_wasserstein.mean()
        ),
        'mean_raw_sfh_wasserstein_of_random_neighbors': float(
            random_shape_wasserstein.mean()
        ),
        'raw_sfh_wasserstein_neighbor_overlap_at_k': float(
            wasserstein_overlaps.mean()
        ),
        'chance_neighbor_overlap_at_k': float(k / (size - 1)),
    }


def encode_sfh_arrays(model, arrays, device, batch_size):
    output = []
    use_amp = device.type == 'cuda'
    with torch.inference_mode():
        for start in range(0, len(arrays), batch_size):
            values = torch.from_numpy(arrays[start:start + batch_size]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                encoded = F.normalize(model.encode_sfh(values), dim=-1)
            output.append(encoded.float().cpu().numpy())
    return np.concatenate(output)


def posterior_robustness_test(model, dataset_path, val_rows, image_embedding,
                              median_embedding, device, rng, subset_size=1000,
                              n_draws=5, batch_size=256):
    """Measure how posterior SFH draws move embeddings and retrievals."""
    size = min(len(val_rows), subset_size)
    selected = np.sort(rng.choice(len(val_rows), size=size, replace=False))
    rows = val_rows[selected]
    images = image_embedding[selected]
    medians = median_embedding[selected]
    with h5py.File(dataset_path, 'r') as source:
        realizations = _read_rows(source['sfh_realizations'], rows).astype(np.float32)
        valid = _read_rows(source['sfh_realization_valid'], rows).astype(bool)
    if np.any(valid.sum(axis=1) == 0):
        raise ValueError('A posterior test galaxy has no valid realization.')

    median_cosines = []
    image_cosines = []
    rank_summaries = []
    median_ranks = retrieval_ranks(images, medians, chunk_size=512)
    for _ in range(n_draws):
        choices = np.array([
            rng.choice(np.flatnonzero(row_valid)) for row_valid in valid
        ])
        draw = realizations[np.arange(size), choices]
        draw_embedding = encode_sfh_arrays(
            model, draw, device=device, batch_size=batch_size,
        )
        median_cosines.append(np.sum(draw_embedding * medians, axis=1))
        image_cosines.append(np.sum(draw_embedding * images, axis=1))
        ranks = retrieval_ranks(images, draw_embedding, chunk_size=512)
        rank_summaries.append(summarize_ranks(ranks))

    median_cosines = np.concatenate(median_cosines)
    image_cosines = np.concatenate(image_cosines)
    return {
        'n_objects': size,
        'n_draws_per_object': n_draws,
        'median_sfh_retrieval_on_same_subset': summarize_ranks(median_ranks),
        'image_to_median_cosine_mean': float(np.sum(images * medians, axis=1).mean()),
        'posterior_to_median_embedding_cosine_mean': float(median_cosines.mean()),
        'posterior_to_median_embedding_cosine_p05': float(
            np.percentile(median_cosines, 5)
        ),
        'image_to_posterior_cosine_mean': float(image_cosines.mean()),
        'image_to_posterior_cosine_std': float(image_cosines.std()),
        'retrieval_per_draw': rank_summaries,
        'retrieval_mean_recall': {
            key: float(np.mean([item['recall'][key] for item in rank_summaries]))
            for key in rank_summaries[0]['recall']
        },
    }


def redshift_strata(redshifts, ranks_i2s, ranks_s2i, paired_cosine):
    finite = np.isfinite(redshifts)
    if np.count_nonzero(finite) < 20:
        return []
    edges = np.unique(np.quantile(redshifts[finite], [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(edges) < 3:
        return []
    records = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = finite & (redshifts >= lower) & (redshifts <= upper)
        else:
            mask = finite & (redshifts >= lower) & (redshifts < upper)
        record = {
            'z_min': float(lower),
            'z_max': float(upper),
            'n_objects': int(mask.sum()),
            'image_to_sfh_median_global_rank_percentile': float(
                np.median(ranks_i2s[mask] / len(ranks_i2s))
            ),
            'sfh_to_image_median_global_rank_percentile': float(
                np.median(ranks_s2i[mask] / len(ranks_s2i))
            ),
            'paired_cosine_mean': float(paired_cosine[mask].mean()),
        }
        for k in (1, 5, 10):
            record[f'image_to_sfh_R@{k}'] = float(np.mean(ranks_i2s[mask] <= k))
            record[f'sfh_to_image_R@{k}'] = float(np.mean(ranks_s2i[mask] <= k))
        records.append(record)
    return records


def write_per_object(path, galaxy_ids, rows, redshifts, ranks_i2s, ranks_s2i,
                     paired_cosine):
    fieldnames = [
        'galaxy_id', 'h5_row', 'redshift', 'image_to_sfh_rank',
        'sfh_to_image_rank', 'image_to_sfh_rank_percentile',
        'sfh_to_image_rank_percentile', 'paired_cosine',
    ]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, galaxy_id in enumerate(galaxy_ids):
            writer.writerow({
                'galaxy_id': int(galaxy_id),
                'h5_row': int(rows[index]),
                'redshift': float(redshifts[index]),
                'image_to_sfh_rank': int(ranks_i2s[index]),
                'sfh_to_image_rank': int(ranks_s2i[index]),
                'image_to_sfh_rank_percentile': float(ranks_i2s[index] / len(rows)),
                'sfh_to_image_rank_percentile': float(ranks_s2i[index] / len(rows)),
                'paired_cosine': float(paired_cosine[index]),
            })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--stamp-root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True,
                        help='pair_split.npz written by the training run.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--band', default='VIS')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-permutations', type=int, default=200)
    parser.add_argument('--permutation-size', type=int, default=5000)
    parser.add_argument('--shape-subset', type=int, default=2000)
    parser.add_argument('--posterior-subset', type=int, default=1000)
    parser.add_argument('--posterior-draws', type=int, default=5)
    parser.add_argument('--skip-posterior', action='store_true')
    parser.add_argument('--no-save-embeddings', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )
    for path in (args.checkpoint, args.dataset, args.split):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    rng = np.random.default_rng(args.seed)
    if min(args.batch_size, args.chunk_size, args.permutation_size,
           args.shape_subset, args.posterior_subset) < 1:
        raise ValueError('Batch, chunk, permutation, shape, and posterior sizes must be positive.')
    if args.n_permutations < 1 or args.posterior_draws < 1:
        raise ValueError('Permutation and posterior draw counts must be positive.')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (train_rows, train_ids, val_rows, val_ids,
     n_bins, n_realizations) = validate_saved_split(
        args.dataset, args.split, args.stamp_root, args.band,
    )
    log.info('Validated saved split: train=%d, validation=%d',
             len(train_rows), len(val_rows))

    model = CosmosWebZooBotCLIP.load_from_checkpoint(
        str(args.checkpoint), map_location='cpu',
    ).to(device)
    loader = extraction_loader(
        args.dataset, args.stamp_root, val_rows, val_ids, args.band,
        args.image_size, args.batch_size, args.num_workers,
    )
    image_embedding, sfh_embedding, encoded_ids, reconstruction = extract_embeddings(
        model, loader, device,
    )
    if not np.array_equal(encoded_ids, val_ids):
        raise ValueError('Embedding extraction changed validation ID order.')

    ranks_i2s = retrieval_ranks(image_embedding, sfh_embedding, args.chunk_size)
    ranks_s2i = retrieval_ranks(sfh_embedding, image_embedding, args.chunk_size)
    paired_cosine = np.sum(image_embedding * sfh_embedding, axis=1)
    with h5py.File(args.dataset, 'r') as source:
        redshifts = _read_rows(source['redshift'], val_rows).astype(np.float32)
        raw_sfh = _read_rows(source['sfh'], val_rows).astype(np.float32)

    report = {
        'checkpoint': str(args.checkpoint.resolve()),
        'dataset': str(args.dataset.resolve()),
        'split': str(args.split.resolve()),
        'device': str(device),
        'split_integrity': {
            'n_train': len(train_rows),
            'n_validation': len(val_rows),
            'train_validation_overlap': 0,
            'all_validation_ids_match_h5_rows': True,
            'all_validation_stamps_exist': True,
            'n_sfh_bins': n_bins,
            'n_posterior_realizations': n_realizations,
        },
        'image_to_sfh_retrieval': summarize_ranks(ranks_i2s),
        'sfh_to_image_retrieval': summarize_ranks(ranks_s2i),
        'shuffled_alignment': shuffled_alignment_test(
            image_embedding, sfh_embedding, rng,
            n_permutations=args.n_permutations,
            permutation_size=args.permutation_size,
        ),
        'embedding_diagnostics': {
            'image': embedding_diagnostics(image_embedding, rng),
            'sfh': embedding_diagnostics(sfh_embedding, rng),
        },
        'sfh_shape_neighborhood': sfh_shape_neighborhood_test(
            image_embedding, sfh_embedding, raw_sfh, rng,
            subset_size=args.shape_subset,
        ),
        'redshift_quartiles': redshift_strata(
            redshifts, ranks_i2s, ranks_s2i, paired_cosine,
        ),
    }
    if not args.skip_posterior:
        report['posterior_robustness'] = posterior_robustness_test(
            model, args.dataset, val_rows, image_embedding, sfh_embedding,
            device, rng, subset_size=args.posterior_subset,
            n_draws=args.posterior_draws, batch_size=args.batch_size,
        )
    if reconstruction is not None:
        report['sfh_reconstruction'] = summarize_sfh_reconstruction(
            reconstruction, raw_sfh,
            epsilon=float(model.hparams.sfh_log_epsilon),
        )

    report_path = args.output_dir / 'metrics.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    write_per_object(
        args.output_dir / 'per_object.csv', val_ids, val_rows, redshifts,
        ranks_i2s, ranks_s2i, paired_cosine,
    )
    if not args.no_save_embeddings:
        np.savez_compressed(
            args.output_dir / 'validation_embeddings.npz',
            galaxy_id=val_ids,
            h5_row=val_rows,
            redshift=redshifts,
            image_embedding=image_embedding,
            sfh_embedding=sfh_embedding,
        )

    print(json.dumps({
        'image_to_sfh': report['image_to_sfh_retrieval'],
        'sfh_to_image': report['sfh_to_image_retrieval'],
        'shuffled_alignment': report['shuffled_alignment'],
        'sfh_shape_neighborhood': report['sfh_shape_neighborhood'],
        'posterior_robustness': report.get('posterior_robustness'),
        'sfh_reconstruction': report.get('sfh_reconstruction'),
        'output': str(report_path),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
