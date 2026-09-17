"""UMAP diagnostics for saved Euclid image--SFH CLIP validation embeddings.

The script consumes ``validation_embeddings.npz`` from
``euclid.evaluate_zoobot_clip``.  It does not load a checkpoint or re-encode
images.  Four projections are produced: image, SFH, their normalized average,
and a shared projection fitted to both modalities at once.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .vis_selection import flux_ujy_to_ab_magnitude


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Property:
    key: str
    label: str
    values: np.ndarray
    cmap: str = 'viridis'


def _read_rows(dataset, rows):
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(dataset[rows[order]])[inverse]


def _normalize_rows(array):
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError('Embedding array contains a nonfinite or zero-norm row.')
    return array / norms


def load_embeddings(path):
    archive = np.load(path)
    required = {
        'galaxy_id', 'h5_row', 'redshift', 'image_embedding', 'sfh_embedding',
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        raise ValueError(f'Missing arrays in {path}: {missing}')
    galaxy_ids = np.asarray(archive['galaxy_id'], dtype=np.int64)
    rows = np.asarray(archive['h5_row'], dtype=np.int64)
    redshift = np.asarray(archive['redshift'], dtype=np.float32)
    image = _normalize_rows(archive['image_embedding'])
    sfh = _normalize_rows(archive['sfh_embedding'])
    sfh_preprojection = (
        _normalize_rows(archive['sfh_preprojection_embedding'])
        if 'sfh_preprojection_embedding' in archive else None
    )
    n_objects = len(galaxy_ids)
    if any(len(value) != n_objects for value in (rows, redshift, image, sfh)):
        raise ValueError('Embedding archive arrays have inconsistent lengths.')
    if (sfh_preprojection is not None
            and sfh_preprojection.shape != sfh.shape):
        raise ValueError(
            'Pre-projection and projected SFH embeddings must have equal shapes.'
        )
    if image.shape != sfh.shape or image.ndim != 2:
        raise ValueError('Image and SFH embeddings must have the same 2D shape.')
    if len(np.unique(galaxy_ids)) != n_objects:
        raise ValueError('Embedding archive contains duplicate galaxy IDs.')
    return galaxy_ids, rows, redshift, image, sfh, sfh_preprojection


def _catalog_property(source, rows, candidates, valid=None):
    for name in candidates:
        if name in source:
            values = _read_rows(source[name], rows).astype(np.float32)
            values[~np.isfinite(values)] = np.nan
            if valid is not None:
                values[~valid(values)] = np.nan
            return values, name
    return None, None


def _dirichlet_fraction(source, rows, numerator, answers):
    """Convert MER ZooBot Dirichlet concentrations to a question fraction."""
    if any(name not in source for name in answers):
        return None
    values = {
        name: _read_rows(source[name], rows).astype(np.float64)
        for name in answers
    }
    denominator = sum(values.values())
    selected = sum(values[name] for name in numerator)
    output = np.full(len(rows), np.nan, dtype=np.float32)
    valid = (
        np.isfinite(denominator) & (denominator > 0)
        & np.isfinite(selected)
    )
    output[valid] = (selected[valid] / denominator[valid]).astype(np.float32)
    return output


def _sfh_properties(source, rows):
    log_sfh = _read_rows(source['sfh'], rows).astype(np.float64)
    time = np.asarray(source['sfh_time_grid'][:], dtype=np.float64)
    epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))
    weights = np.maximum(10.0 ** log_sfh - epsilon, 0.0)
    totals = weights.sum(axis=1, keepdims=True)
    weights /= np.maximum(totals, epsilon)

    recent_10 = weights[:, time <= 0.1].sum(axis=1)
    recent_20 = weights[:, time <= 0.2].sum(axis=1)
    old_20 = weights[:, time >= 0.8].sum(axis=1)
    mean_lookback = np.sum(weights * time[None, :], axis=1)
    peak_lookback = time[np.argmax(weights, axis=1)]
    cumulative = np.cumsum(weights, axis=1)
    t50_lookback = time[np.argmax(cumulative >= 0.5, axis=1)]
    entropy = -np.sum(
        np.where(weights > 0, weights * np.log(np.maximum(weights, epsilon)), 0.0),
        axis=1,
    ) / np.log(weights.shape[1])
    log_old_recent = np.log10((old_20 + epsilon) / (recent_20 + epsilon))
    return {
        'sfh_recent_10': recent_10.astype(np.float32),
        'sfh_recent_20': recent_20.astype(np.float32),
        'sfh_old_20': old_20.astype(np.float32),
        'sfh_mean_lookback': mean_lookback.astype(np.float32),
        'sfh_peak_lookback': peak_lookback.astype(np.float32),
        'sfh_t50_lookback': t50_lookback.astype(np.float32),
        'sfh_entropy': entropy.astype(np.float32),
        'sfh_log_old_recent': log_old_recent.astype(np.float32),
    }


def load_rank_properties(path, galaxy_ids):
    if path is None:
        return {}
    by_id = {}
    with path.open(newline='') as stream:
        for record in csv.DictReader(stream):
            by_id[int(record['galaxy_id'])] = record
    missing = [int(value) for value in galaxy_ids if int(value) not in by_id]
    if missing:
        raise ValueError(f'{len(missing)} embedding IDs are absent from {path}.')

    def values(column):
        return np.array([
            float(by_id[int(galaxy_id)][column]) for galaxy_id in galaxy_ids
        ], dtype=np.float32)

    return {
        'image_to_sfh_rank_percentile': values('image_to_sfh_rank_percentile'),
        'sfh_to_image_rank_percentile': values('sfh_to_image_rank_percentile'),
        'paired_cosine': values('paired_cosine'),
    }


def load_properties(h5_path, rows, galaxy_ids, archive_redshift,
                    image_embedding, sfh_embedding, ranks_path=None):
    properties = {'redshift': archive_redshift.astype(np.float32)}
    sources = {'redshift': 'validation_embeddings.npz'}
    with h5py.File(h5_path, 'r') as source:
        h5_ids = _read_rows(source['galaxy_id'], rows).astype(np.int64)
        if not np.array_equal(h5_ids, galaxy_ids):
            raise ValueError('Embedding IDs do not match the requested HDF5 rows.')
        catalog_specs = {
            'log_stellar_mass': (
                ['phz_pp_median_stellarmass'], lambda x: (x > 0) & (x < 20),
            ),
            'sersic_index': (
                ['sersic_sersic_vis_index'], lambda x: (x > 0) & (x < 20),
            ),
            'sersic_radius': (
                ['sersic_sersic_vis_radius'], lambda x: x > 0,
            ),
            'axis_ratio': (
                ['sersic_sersic_vis_axis_ratio'], lambda x: (x > 0) & (x <= 1),
            ),
            'fwhm': (['fwhm'], lambda x: x > 0),
            'kron_radius': (['kron_radius'], lambda x: x > 0),
            'semimajor_axis': (['semimajor_axis'], lambda x: x > 0),
            'ellipticity': (
                ['ellipticity'], lambda x: (x >= 0) & (x < 1),
            ),
            'segmentation_area': (['segmentation_area'], lambda x: x > 0),
            'point_like_probability': (
                ['point_like_prob'], lambda x: (x >= 0) & (x <= 1),
            ),
            'concentration': (['concentration'], lambda x: np.isfinite(x)),
            'asymmetry': (['asymmetry'], lambda x: np.isfinite(x)),
            'smoothness': (['smoothness'], lambda x: np.isfinite(x)),
            'gini': (['gini'], lambda x: np.isfinite(x)),
            'moment_20': (['moment_20'], lambda x: np.isfinite(x)),
            't_type': (['t_type'], lambda x: np.isfinite(x)),
            'etg_or_ltg': (['etg_or_ltg'], lambda x: np.isfinite(x)),
            'major_merger_probability': (
                ['major_merger'], lambda x: (x >= 0) & (x <= 1),
            ),
        }
        for key, (candidates, valid) in catalog_specs.items():
            value, source_name = _catalog_property(source, rows, candidates, valid)
            if value is not None:
                properties[key] = value
                sources[key] = source_name
        flux, flux_name = _catalog_property(
            source, rows, ['flux_detection_total'], lambda x: x > 0,
        )
        if flux is not None:
            vis_magnitude = flux_ujy_to_ab_magnitude(flux).astype(np.float32)
            if 'vis_det' in source:
                vis_detected = _read_rows(source['vis_det'], rows) == 1
                vis_magnitude[~vis_detected] = np.nan
            properties['vis_magnitude'] = vis_magnitude
            sources['vis_magnitude'] = f'derived from {flux_name} (microJy)'

        zoo_questions = {
            'zoobot_smooth_probability': (
                ('smooth_or_featured_smooth',),
                ('smooth_or_featured_smooth',
                 'smooth_or_featured_featured_or_disk',
                 'smooth_or_featured_artifact_star_zoom'),
            ),
            'zoobot_featured_probability': (
                ('smooth_or_featured_featured_or_disk',),
                ('smooth_or_featured_smooth',
                 'smooth_or_featured_featured_or_disk',
                 'smooth_or_featured_artifact_star_zoom'),
            ),
            'zoobot_edge_on_probability': (
                ('disk_edge_on_yes',),
                ('disk_edge_on_yes', 'disk_edge_on_no'),
            ),
            'zoobot_spiral_probability': (
                ('has_spiral_arms_yes',),
                ('has_spiral_arms_yes', 'has_spiral_arms_no'),
            ),
            'zoobot_bar_probability': (
                ('bar_strong', 'bar_weak'),
                ('bar_strong', 'bar_weak', 'bar_no'),
            ),
            'zoobot_merger_probability': (
                ('merging_minor_disturbance', 'merging_major_disturbance',
                 'merging_merger'),
                ('merging_none', 'merging_minor_disturbance',
                 'merging_major_disturbance', 'merging_merger'),
            ),
        }
        for key, (numerator, answers) in zoo_questions.items():
            values = _dirichlet_fraction(source, rows, numerator, answers)
            if values is not None:
                properties[key] = values
                sources[key] = 'derived from MER ZooBot Dirichlet concentrations'
        properties.update(_sfh_properties(source, rows))
        sources.update({key: 'derived from sfh' for key in properties if key.startswith('sfh_')})

    properties['paired_cosine'] = np.sum(
        image_embedding * sfh_embedding, axis=1,
    ).astype(np.float32)
    sources['paired_cosine'] = 'derived from embeddings'
    if ranks_path is not None:
        properties.update(load_rank_properties(ranks_path, galaxy_ids))
        sources.update({
            key: str(ranks_path) for key in (
                'paired_cosine', 'image_to_sfh_rank_percentile',
                'sfh_to_image_rank_percentile',
            )
        })
    return properties, sources


def property_specs(properties):
    definitions = {
        'redshift': ('Redshift z', 'plasma'),
        'vis_magnitude': ('VIS total magnitude (AB)', 'viridis_r'),
        'log_stellar_mass': (r'log $M_\star/M_\odot$', 'inferno'),
        'sersic_index': ('Sérsic index n', 'viridis'),
        'sersic_radius': ('Sérsic radius', 'viridis'),
        'axis_ratio': ('Sérsic axis ratio b/a', 'viridis'),
        'fwhm': ('FWHM', 'magma'),
        'kron_radius': ('Kron radius', 'magma'),
        'semimajor_axis': ('Semimajor axis', 'magma'),
        'ellipticity': ('Ellipticity', 'viridis'),
        'segmentation_area': ('Segmentation area', 'magma'),
        'point_like_probability': ('Point-like probability', 'cividis'),
        'concentration': ('Concentration', 'viridis'),
        'asymmetry': ('Asymmetry', 'magma'),
        'smoothness': ('CAS smoothness', 'magma'),
        'gini': ('Gini coefficient', 'viridis'),
        'moment_20': (r'$M_{20}$', 'coolwarm'),
        't_type': ('MER T-type', 'coolwarm'),
        'etg_or_ltg': ('MER ETG/LTG score', 'coolwarm'),
        'major_merger_probability': ('Major-merger probability', 'magma'),
        'zoobot_smooth_probability': ('ZooBot P(smooth)', 'viridis'),
        'zoobot_featured_probability': ('ZooBot P(featured/disk)', 'viridis'),
        'zoobot_edge_on_probability': ('ZooBot P(edge-on)', 'magma'),
        'zoobot_spiral_probability': ('ZooBot P(spiral arms)', 'magma'),
        'zoobot_bar_probability': ('ZooBot P(bar)', 'magma'),
        'zoobot_merger_probability': ('ZooBot P(disturbed/merger)', 'magma'),
        'sfh_recent_10': ('SFH fraction: recent 10%', 'hot'),
        'sfh_recent_20': ('SFH fraction: recent 20%', 'hot'),
        'sfh_old_20': ('SFH fraction: oldest 20%', 'cividis'),
        'sfh_mean_lookback': ('Mean fractional lookback time', 'plasma'),
        'sfh_peak_lookback': ('Peak fractional lookback time', 'plasma'),
        'sfh_t50_lookback': ('SFH t50 fractional lookback', 'plasma'),
        'sfh_entropy': ('Normalized SFH entropy', 'viridis'),
        'sfh_log_old_recent': ('log(old 20% / recent 20%)', 'coolwarm'),
        'paired_cosine': ('Matched image-SFH cosine', 'coolwarm'),
        'image_to_sfh_rank_percentile': ('Image→SFH rank percentile', 'viridis_r'),
        'sfh_to_image_rank_percentile': ('SFH→image rank percentile', 'viridis_r'),
    }
    return {
        key: Property(key, definitions[key][0], properties[key], definitions[key][1])
        for key in definitions if key in properties
    }


def fit_umap(embedding, n_neighbors=15, min_dist=0.1, seed=42):
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            'umap-learn is required; use the same cosmos_visual environment as training.'
        ) from error
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=seed,
        low_memory=True,
        verbose=True,
    )
    return reducer.fit_transform(embedding).astype(np.float32)


def _color_limits(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    lower, upper = np.percentile(finite, [1, 99])
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if lower == upper:
        padding = max(abs(lower) * 0.01, 1e-6)
        lower -= padding
        upper += padding
    return float(lower), float(upper)


def scatter_property(ax, coordinates, prop, point_size=2.0):
    values = np.asarray(prop.values, dtype=float)
    mask = np.isfinite(values)
    limits = _color_limits(values)
    if limits is None or not np.any(mask):
        ax.text(0.5, 0.5, f'{prop.label}\nno finite values',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return
    scatter = ax.scatter(
        coordinates[mask, 0], coordinates[mask, 1], c=values[mask],
        s=point_size, alpha=0.68, linewidths=0, rasterized=True,
        cmap=prop.cmap, vmin=limits[0], vmax=limits[1],
    )
    colorbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
    colorbar.ax.tick_params(labelsize=6)
    colorbar.set_label(prop.label, fontsize=7)
    ax.set_title(prop.label, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def property_pages(pdf, coordinates, props, title):
    for page_start in range(0, len(props), 9):
        page = props[page_start:page_start + 9]
        fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), squeeze=False)
        fig.suptitle(title, fontsize=11)
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, prop in zip(axes.flat, page):
            ax.set_visible(True)
            scatter_property(ax, coordinates, prop)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=170)
        plt.close(fig)


def shared_manifold_page(pdf, image_xy, sfh_xy, rng, n_lines=500, run_label=None):
    n_objects = len(image_xy)
    selected = np.sort(rng.choice(n_objects, min(n_lines, n_objects), replace=False))
    paired_distance = np.linalg.norm(image_xy - sfh_xy, axis=1)
    shuffled = np.roll(sfh_xy, max(1, n_objects // 3), axis=0)
    shuffled_distance = np.linalg.norm(image_xy - shuffled, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].scatter(image_xy[:, 0], image_xy[:, 1], s=2, alpha=0.35,
                    color='tab:blue', rasterized=True, label='image')
    axes[0].scatter(sfh_xy[:, 0], sfh_xy[:, 1], s=2, alpha=0.35,
                    color='tab:orange', rasterized=True, label='SFH')
    axes[0].legend(markerscale=4, frameon=False)
    axes[0].set_title('Shared UMAP: modality occupancy')

    for index in selected:
        axes[1].plot(
            [image_xy[index, 0], sfh_xy[index, 0]],
            [image_xy[index, 1], sfh_xy[index, 1]],
            color='0.55', alpha=0.08, linewidth=0.35,
        )
    axes[1].scatter(image_xy[selected, 0], image_xy[selected, 1], s=3,
                    color='tab:blue', alpha=0.6, rasterized=True)
    axes[1].scatter(sfh_xy[selected, 0], sfh_xy[selected, 1], s=3,
                    color='tab:orange', alpha=0.6, rasterized=True)
    axes[1].set_title(f'{len(selected):,} matched pairs')

    upper = np.percentile(np.concatenate([paired_distance, shuffled_distance]), 99)
    bins = np.linspace(0, upper, 50)
    axes[2].hist(shuffled_distance, bins=bins, density=True, histtype='step',
                 linewidth=1.6, color='0.4', label='shuffled')
    axes[2].hist(paired_distance, bins=bins, density=True, histtype='step',
                 linewidth=1.6, color='tab:green', label='matched')
    axes[2].set_xlabel('2D shared-UMAP separation')
    axes[2].set_ylabel('Density')
    axes[2].legend(frameon=False)
    axes[2].set_title('Projection-space pair separation')
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    heading = (
        'Image and SFH embeddings fitted in one shared UMAP\n'
        '(2D distances are qualitative; retrieval metrics remain definitive)'
    )
    if run_label:
        heading += f'\n{run_label}'
    fig.suptitle(heading, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig, dpi=170)
    plt.close(fig)


def comparison_page(pdf, coordinate_sets, props, run_label=None):
    for page_start in range(0, len(props), 4):
        page = props[page_start:page_start + 4]
        fig, axes = plt.subplots(len(page), 3, figsize=(12, 3.0 * len(page)), squeeze=False)
        for row, prop in enumerate(page):
            for column, (name, coordinates) in enumerate(coordinate_sets):
                scatter_property(axes[row, column], coordinates, prop, point_size=1.8)
                axes[row, column].set_title(f'{name}: {prop.label}', fontsize=8)
        heading = 'The same property in independently fitted embedding spaces'
        if run_label:
            heading += f'\n{run_label}'
        fig.suptitle(heading, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=170)
        plt.close(fig)


def save_property_table(path, galaxy_ids, rows, properties):
    keys = list(properties)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['galaxy_id', 'h5_row', *keys])
        for index, galaxy_id in enumerate(galaxy_ids):
            writer.writerow([
                int(galaxy_id), int(rows[index]),
                *[float(properties[key][index]) for key in keys],
            ])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--per-object', type=Path,
                        help='Optional per_object.csv from checkpoint evaluation.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--npz-output', type=Path)
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-label', default='',
                        help='Checkpoint/run label printed on every PDF section.')
    parser.add_argument('--max-objects', type=int, default=0,
                        help='Random deterministic subset; zero uses all objects.')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    required_paths = {
        'validation embedding archive': args.embeddings,
        'preprocessed SFH dataset': args.dataset,
    }
    for label, path in required_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {label}: {path}')
    if args.per_object is not None and not args.per_object.is_file():
        raise FileNotFoundError(
            f'Missing per-object evaluation table: {args.per_object}'
        )
    if args.n_neighbors < 2 or not 0 <= args.min_dist <= 1:
        raise ValueError('Use n-neighbors >= 2 and min-dist in [0, 1].')

    (galaxy_ids, rows, redshift, image, sfh,
     sfh_preprojection) = load_embeddings(args.embeddings)
    rng = np.random.default_rng(args.seed)
    if args.max_objects > 0 and args.max_objects < len(galaxy_ids):
        selected = np.sort(rng.choice(len(galaxy_ids), args.max_objects, replace=False))
        galaxy_ids, rows, redshift, image, sfh = (
            value[selected] for value in (galaxy_ids, rows, redshift, image, sfh)
        )
        if sfh_preprojection is not None:
            sfh_preprojection = sfh_preprojection[selected]
    joint = _normalize_rows(image + sfh)
    properties, sources = load_properties(
        args.dataset, rows, galaxy_ids, redshift, image, sfh, args.per_object,
    )
    specs = property_specs(properties)
    log.info('Loaded %d objects and properties: %s', len(galaxy_ids), ', '.join(specs))
    for key, source in sources.items():
        log.info('  %-32s <- %s', key, source)

    log.info('Fitting image UMAP')
    xy_image = fit_umap(image, args.n_neighbors, args.min_dist, args.seed)
    log.info('Fitting SFH UMAP')
    xy_sfh = fit_umap(sfh, args.n_neighbors, args.min_dist, args.seed)
    xy_sfh_preprojection = None
    if sfh_preprojection is not None:
        log.info('Fitting frozen pre-projection SFH UMAP')
        xy_sfh_preprojection = fit_umap(
            sfh_preprojection, args.n_neighbors, args.min_dist, args.seed,
        )
    log.info('Fitting averaged joint UMAP')
    xy_joint = fit_umap(joint, args.n_neighbors, args.min_dist, args.seed)
    log.info('Fitting stacked shared-manifold UMAP')
    xy_shared = fit_umap(
        np.concatenate([image, sfh]), args.n_neighbors, args.min_dist, args.seed,
    )
    xy_shared_image = xy_shared[:len(image)]
    xy_shared_sfh = xy_shared[len(image):]

    morphology_keys = [
        'redshift', 'sersic_index', 'sersic_radius', 'axis_ratio', 'fwhm',
        'kron_radius', 'semimajor_axis', 'point_like_probability',
        'ellipticity', 'segmentation_area',
    ]
    physical_keys = [
        'redshift', 'log_stellar_mass', 'sfh_recent_10', 'sfh_recent_20',
        'sfh_old_20', 'sfh_mean_lookback', 'sfh_peak_lookback',
        'sfh_t50_lookback', 'sfh_entropy', 'sfh_log_old_recent',
    ]
    alignment_keys = [
        'paired_cosine', 'image_to_sfh_rank_percentile',
        'sfh_to_image_rank_percentile',
    ]
    morphology = [specs[key] for key in morphology_keys if key in specs]
    physical = [specs[key] for key in physical_keys if key in specs]
    alignment = [specs[key] for key in alignment_keys if key in specs]
    comparison = [
        specs[key] for key in (
            'redshift', 'log_stellar_mass', 'sersic_index', 'sersic_radius',
            'sfh_recent_20', 'sfh_mean_lookback', 'sfh_entropy', 'paired_cosine',
        ) if key in specs
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        shared_manifold_page(
            pdf, xy_shared_image, xy_shared_sfh, rng, run_label=args.run_label,
        )
        suffix = f' - {args.run_label}' if args.run_label else ''
        property_pages(pdf, xy_image, morphology,
                       'Euclid VIS image embedding: morphology' + suffix)
        property_pages(pdf, xy_image, physical,
                       'Euclid VIS image embedding: physical and SFH properties' + suffix)
        property_pages(pdf, xy_sfh, morphology,
                       'Euclid SFH embedding: morphology transferred from images' + suffix)
        property_pages(pdf, xy_sfh, physical,
                       'Euclid SFH embedding: physical and SFH properties' + suffix)
        if xy_sfh_preprojection is not None:
            property_pages(
                pdf, xy_sfh_preprojection, physical,
                'Frozen SFH autoencoder latent: physical and SFH properties' + suffix,
            )
        property_pages(pdf, xy_joint, morphology,
                       'Euclid averaged joint embedding: morphology' + suffix)
        property_pages(pdf, xy_joint, physical,
                       'Euclid averaged joint embedding: physical and SFH properties' + suffix)
        if alignment:
            property_pages(pdf, xy_joint, alignment,
                           'Euclid averaged joint embedding: alignment quality' + suffix)
        comparison_page(
            pdf,
            [('Image', xy_image), ('SFH', xy_sfh), ('Joint average', xy_joint)],
            comparison, run_label=args.run_label,
        )

    npz_output = args.npz_output or args.output.with_suffix('.npz')
    npz_data = {
        'galaxy_id': galaxy_ids,
        'h5_row': rows,
        'image_embedding': image,
        'sfh_embedding': sfh,
        'joint_embedding': joint,
        'xy_image': xy_image,
        'xy_sfh': xy_sfh,
        'xy_joint': xy_joint,
        'xy_shared_image': xy_shared_image,
        'xy_shared_sfh': xy_shared_sfh,
    }
    if sfh_preprojection is not None:
        npz_data.update({
            'sfh_preprojection_embedding': sfh_preprojection,
            'xy_sfh_preprojection': xy_sfh_preprojection,
        })
    npz_data.update({key: np.asarray(value, dtype=np.float32)
                     for key, value in properties.items()})
    np.savez_compressed(npz_output, **npz_data)
    save_property_table(args.output.with_suffix('.csv'), galaxy_ids, rows, properties)
    log.info('Saved PDF: %s', args.output)
    log.info('Saved coordinates/properties: %s', npz_output)


if __name__ == '__main__':
    main()
