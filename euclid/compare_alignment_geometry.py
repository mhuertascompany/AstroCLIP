"""Compare local geometry and morphology coherence across alignment runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


MORPHOLOGY_FIELDS = (
    'zoobot_smooth_probability',
    'zoobot_spiral_probability',
    'zoobot_merger_probability',
)


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def nearest_neighbors(values, maximum_k):
    values = normalize(values)
    model = NearestNeighbors(
        n_neighbors=maximum_k + 1, metric='cosine', algorithm='brute', n_jobs=-1,
    ).fit(values)
    neighbors = model.kneighbors(values, return_distance=False)
    # The query itself should be first, but remove it explicitly so ties cannot
    # leak self-matches into the neighbourhood statistics.
    output = np.empty((len(values), maximum_k), dtype=np.int32)
    for row, indices in enumerate(neighbors):
        without_self = indices[indices != row]
        output[row] = without_self[:maximum_k]
    return output


def row_overlap(first, second, k):
    return np.asarray([
        len(set(a[:k]).intersection(b[:k])) / k
        for a, b in zip(first, second)
    ], dtype=np.float32)


def mean_ci(values, rng, n_bootstrap=400):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if len(values) else float('nan')
    if len(values) < 2:
        return mean, float('nan'), float('nan')
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(values), len(values))
        draws[index] = values[sample].mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return mean, float(low), float(high)


def effective_rank(values):
    centered = normalize(values).astype(np.float64)
    centered -= centered.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular ** 2
    probability = variance / np.maximum(variance.sum(), 1e-30)
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def crossmodal_neighbors(image, sfh, maximum_k, chunk_size=512):
    image, sfh = normalize(image), normalize(sfh)
    output = np.empty((len(image), maximum_k), dtype=np.int32)
    for start in range(0, len(image), chunk_size):
        stop = min(start + chunk_size, len(image))
        similarity = image[start:stop] @ sfh.T
        candidates = np.argpartition(
            -similarity, kth=maximum_k - 1, axis=1,
        )[:, :maximum_k]
        candidate_scores = np.take_along_axis(similarity, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        output[start:stop] = np.take_along_axis(candidates, order, axis=1)
    return output


def morphology_neighbor_ratio(values, neighbors, k, rng):
    values = np.asarray(values, dtype=np.float32)
    neighbor_values = values[neighbors[:, :k]]
    valid = np.isfinite(values)[:, None] & np.isfinite(neighbor_values)
    difference = np.abs(neighbor_values - values[:, None])
    numerator = np.divide(
        np.where(valid, difference, 0).sum(1), valid.sum(1),
        out=np.full(len(values), np.nan), where=valid.sum(1) > 0,
    )
    random_indices = rng.integers(0, len(values), size=(len(values), k))
    random_values = values[random_indices]
    random_valid = np.isfinite(values)[:, None] & np.isfinite(random_values)
    random_difference = np.abs(random_values - values[:, None])
    denominator = np.divide(
        np.where(random_valid, random_difference, 0).sum(1), random_valid.sum(1),
        out=np.full(len(values), np.nan), where=random_valid.sum(1) > 0,
    )
    ratio = numerator / denominator
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def sampled_geometry_correlations(image, sfh, reference, rng, n_pairs=300_000):
    image, sfh, reference = map(normalize, (image, sfh, reference))
    first = rng.integers(0, len(image), n_pairs)
    second = rng.integers(0, len(image), n_pairs)
    keep = first != second
    first, second = first[keep], second[keep]
    image_similarity = (image[first] * image[second]).sum(1)
    sfh_similarity = (sfh[first] * sfh[second]).sum(1)
    reference_similarity = (reference[first] * reference[second]).sum(1)
    cross_forward = (image[first] * sfh[second]).sum(1)
    cross_reverse = (image[second] * sfh[first]).sum(1)
    return {
        'image_vs_ae_spearman': float(spearmanr(image_similarity, reference_similarity).statistic),
        'sfh_vs_ae_spearman': float(spearmanr(sfh_similarity, reference_similarity).statistic),
        'image_vs_sfh_geometry_spearman': float(spearmanr(image_similarity, sfh_similarity).statistic),
        'crossmodal_symmetry_rmse': float(np.sqrt(np.mean((cross_forward - cross_reverse) ** 2))),
        'sampled_mean_image_cosine': float(image_similarity.mean()),
        'sampled_mean_sfh_cosine': float(sfh_similarity.mean()),
    }


def add_summary(records, run, metric, values, rng, k=None):
    mean, low, high = mean_ci(values, rng)
    records.append({
        'run': run,
        'metric': metric,
        'k': '' if k is None else int(k),
        'value': mean,
        'ci95_low': low,
        'ci95_high': high,
    })


def compare(archives, labels, output_dir, ks=(10, 50), seed=42):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    loaded = []
    for path in archives:
        with np.load(path) as archive:
            loaded.append({key: np.asarray(archive[key]) for key in archive.files})
    ids = loaded[0]['galaxy_id']
    for path, archive in zip(archives[1:], loaded[1:]):
        if not np.array_equal(ids, archive['galaxy_id']):
            raise ValueError(f'Galaxy IDs/order differ in {path}.')
    reference = loaded[0]['sfh_preprojection_embedding']
    for path, archive in zip(archives[1:], loaded[1:]):
        if not np.allclose(reference, archive['sfh_preprojection_embedding'], atol=1e-5):
            raise ValueError(f'Frozen SFH-AE embeddings differ in {path}.')

    maximum_k = max(ks)
    reference_neighbors = nearest_neighbors(reference, maximum_k)
    records, details = [], {'n_galaxies': int(len(ids)), 'runs': {}}
    details['reference_sfh_ae_effective_rank'] = effective_rank(reference)

    for label, archive in zip(labels, loaded):
        image = archive['image_embedding']
        sfh = archive['sfh_embedding']
        joint = archive['joint_embedding']
        image_neighbors = nearest_neighbors(image, maximum_k)
        sfh_neighbors = nearest_neighbors(sfh, maximum_k)
        joint_neighbors = nearest_neighbors(joint, maximum_k)
        cross_neighbors = crossmodal_neighbors(image, sfh, maximum_k)

        run_details = sampled_geometry_correlations(image, sfh, reference, rng)
        run_details.update({
            'unaligned_image_effective_rank': effective_rank(
                archive['image_preprojection_embedding'],
            ),
            'image_effective_rank': effective_rank(image),
            'sfh_effective_rank': effective_rank(sfh),
            'joint_effective_rank': effective_rank(joint),
            'mean_paired_cosine': float(np.sum(normalize(image) * normalize(sfh), axis=1).mean()),
        })
        details['runs'][label] = run_details
        for metric, value in run_details.items():
            records.append({
                'run': label, 'metric': metric, 'k': '', 'value': value,
                'ci95_low': '', 'ci95_high': '',
            })

        for k in ks:
            add_summary(
                records, label, 'image_sfh_neighbor_overlap',
                row_overlap(image_neighbors, sfh_neighbors, k), rng, k,
            )
            add_summary(
                records, label, 'image_ae_neighbor_overlap',
                row_overlap(image_neighbors, reference_neighbors, k), rng, k,
            )
            add_summary(
                records, label, 'sfh_ae_neighbor_overlap',
                row_overlap(sfh_neighbors, reference_neighbors, k), rng, k,
            )
            add_summary(
                records, label, 'joint_ae_neighbor_overlap',
                row_overlap(joint_neighbors, reference_neighbors, k), rng, k,
            )
            add_summary(
                records, label, 'crossmodal_semantic_recall',
                row_overlap(cross_neighbors, reference_neighbors, k), rng, k,
            )
            paired_hit = np.any(cross_neighbors[:, :k] == np.arange(len(ids))[:, None], axis=1)
            add_summary(records, label, 'image_to_sfh_paired_recall', paired_hit, rng, k)
            for field in MORPHOLOGY_FIELDS:
                if field not in archive:
                    continue
                for space, neighbors in (
                    ('image', image_neighbors), ('sfh', sfh_neighbors),
                    ('joint', joint_neighbors), ('ae', reference_neighbors),
                ):
                    ratio = morphology_neighbor_ratio(
                        archive[field], neighbors, k, rng,
                    )
                    add_summary(
                        records, label,
                        f'{space}_{field}_neighbor_mad_over_random', ratio, rng, k,
                    )

    with (output_dir / 'alignment_geometry_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (output_dir / 'alignment_geometry_details.json').write_text(
        json.dumps(details, indent=2) + '\n'
    )

    local_metrics = (
        'image_sfh_neighbor_overlap', 'image_ae_neighbor_overlap',
        'crossmodal_semantic_recall',
    )
    retained_metrics = ('sfh_ae_neighbor_overlap', 'joint_ae_neighbor_overlap')
    k = max(ks)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    width = 0.36
    x = np.arange(len(local_metrics))
    for offset, label in enumerate(labels):
        rows = {
            row['metric']: row for row in records
            if row['run'] == label and row['k'] == k
        }
        values = [rows[name]['value'] for name in local_metrics]
        low = [rows[name]['ci95_low'] for name in local_metrics]
        high = [rows[name]['ci95_high'] for name in local_metrics]
        error = np.array([np.array(values) - low, np.array(high) - values])
        axes[0].bar(
            x + (offset - 0.5) * width, values, width, yerr=error,
            capsize=3, label=label,
        )
    axes[0].set_xticks(x, [
        'Image-SFH\nneighbours', 'Image-AE\nneighbours',
        'Cross-modal\nsemantic recall',
    ])
    axes[0].set_ylabel(f'Local overlap / recall at k={k}')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.25)

    x_retained = np.arange(len(retained_metrics))
    for offset, label in enumerate(labels):
        rows = {
            row['metric']: row for row in records
            if row['run'] == label and row['k'] == k
        }
        values = [rows[name]['value'] for name in retained_metrics]
        low = [rows[name]['ci95_low'] for name in retained_metrics]
        high = [rows[name]['ci95_high'] for name in retained_metrics]
        error = np.array([np.array(values) - low, np.array(high) - values])
        axes[1].bar(
            x_retained + (offset - 0.5) * width, values, width, yerr=error,
            capsize=3, label=label,
        )
    axes[1].set_xticks(x_retained, ['SFH-AE\nneighbours', 'Joint-AE\nneighbours'])
    axes[1].set_ylabel(f'AE-neighbour retention at k={k}')
    axes[1].grid(axis='y', alpha=0.25)

    correlation_names = (
        'image_vs_ae_spearman', 'sfh_vs_ae_spearman',
        'image_vs_sfh_geometry_spearman',
    )
    for offset, label in enumerate(labels):
        values = [details['runs'][label][name] for name in correlation_names]
        axes[2].bar(x + (offset - 0.5) * width, values, width, label=label)
    axes[2].set_xticks(x, ['Image vs AE', 'SFH vs AE', 'Image vs SFH'])
    axes[2].set_ylabel('Sampled pairwise Spearman correlation')
    axes[2].grid(axis='y', alpha=0.25)
    fig.suptitle(f'Alignment geometry on {len(ids):,} common validation galaxies')
    fig.tight_layout()
    fig.savefig(output_dir / 'alignment_geometry_comparison.pdf')
    plt.close(fig)
    return records, details


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, action='append', required=True)
    parser.add_argument('--label', action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--k', type=int, action='append', default=[])
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.archive) != len(args.label):
        raise ValueError('Provide one --label per --archive.')
    ks = tuple(args.k) if args.k else (10, 50)
    if not ks or min(ks) < 1:
        raise ValueError('--k values must be positive.')
    records, details = compare(
        args.archive, args.label, args.output_dir, ks=ks, seed=args.seed,
    )
    print(json.dumps(details, indent=2), flush=True)
    print(f'Summary: {args.output_dir / "alignment_geometry_summary.csv"}')
    print(f'Figure: {args.output_dir / "alignment_geometry_comparison.pdf"}')


if __name__ == '__main__':
    main()
