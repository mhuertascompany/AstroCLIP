"""Interactive local explorer for Euclid image--SFH or SFH-only embeddings.

The app accepts one or more compatible UMAP NPZ archives.
Both panels share the same selected galaxy IDs, while their run, embedding
space, and color property can be changed independently. Lasso selections show
median SFHs with posterior intervals and the population SFH of the complete
selection. When ``--stamps`` is supplied, it also shows the corresponding VIS
stamps.

Run directly::

    python -m euclid.explore_embeddings \
        --h5 /path/to/euclid_explorer.h5 \
        --stamps /path/to/VIS \
        --umap /path/to/baseline_umap.npz \
        --umap /path/to/soft_w1_umap.npz

Or use ``panel serve euclid/explore_embeddings.py --show --args ...``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from PIL import Image
from euclid.cosmic_sfh import COSMOLOGY as _SFH_COSMOLOGY, cosmic_sfh_weights
from matplotlib.ticker import MaxNLocator


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import panel as pn
from bokeh.events import Tap
from bokeh.models import LabelSet
from euclid.umap_path import sample_segment
from euclid.sfh_migration import catalog_migration
from euclid.rejuvenation import catalog_diagnostics, LABELS as REJ_LABELS
from euclid.main_sequence_sfh import main_sequence_along_sfh
from euclid.ms_deviation import catalog_deviations
from euclid.recent_ms import WINDOWS, catalog_recent_offsets, matched_mask
from euclid.sfh_shape import sfh_duration_80, sfh_recent_activity
from bokeh.models import (
    BasicTicker,
    BooleanFilter,
    CDSView,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
)
from bokeh.palettes import Inferno256, Plasma256, Turbo256, Viridis256
from bokeh.plotting import figure as bk_figure


pn.extension(sizing_mode='stretch_width')

N_DISPLAY = 16
NCOLS = 4

_PALETTES = {
    'inferno': Inferno256,
    'plasma': Plasma256,
    'viridis': Viridis256,
    'coolwarm': Turbo256,
}

_PROPERTY_LABELS = {
    'redshift': 'Redshift z',
    'vis_magnitude': 'VIS total magnitude (AB)',
    'log_stellar_mass': 'log M★',
    'sersic_index': 'Sérsic index n',
    'sersic_radius': 'Sérsic radius',
    'axis_ratio': 'Sérsic axis ratio b/a',
    'fwhm': 'FWHM',
    'kron_radius': 'Kron radius',
    'semimajor_axis': 'Semimajor axis',
    'ellipticity': 'Ellipticity',
    'segmentation_area': 'Segmentation area',
    'point_like_probability': 'Point-like probability',
    'concentration': 'Concentration',
    'asymmetry': 'Asymmetry',
    'smoothness': 'CAS smoothness',
    'gini': 'Gini coefficient',
    'moment_20': 'M20',
    't_type': 'MER T-type',
    'etg_or_ltg': 'MER ETG/LTG score',
    'major_merger_probability': 'Major-merger probability',
    'zoobot_smooth_probability': 'ZooBot P(smooth)',
    'zoobot_smooth_conditional_fraction': 'Smooth fraction (smooth + featured)',
    'zoobot_featured_conditional_fraction': 'Featured fraction (smooth + featured)',
    'zoobot_featured_probability': 'ZooBot P(featured/disk)',
    'zoobot_edge_on_probability': 'ZooBot P(edge-on)',
    'zoobot_spiral_probability': 'ZooBot P(spiral arms)',
    'zoobot_bar_probability': 'ZooBot P(bar)',
    'zoobot_merger_probability': 'ZooBot P(disturbed/merger)',
    'sfh_recent_10': 'SFH fraction: recent 10%',
    'sfh_recent_20': 'SFH fraction: recent 20%',
    'sfh_old_20': 'SFH fraction: oldest 20%',
    'sfh_mean_lookback': 'Mean fractional lookback time',
    'sfh_peak_lookback': 'Peak fractional lookback time',
    'sfh_recent_birthrate': 'Recent SFR / lifetime mean (latest 10%)',
    'sfh_recent_trend': 'Recent SFH trend (+ rising, - declining)',
    'sfh_log_recent_sfr_per_formed_mass': 'log10 recent SFR / formed mass (yr⁻¹; floor −15)',
    'sfh_duration_80': 'SFH duration: central 80% (fractional time)',
    'sfh_t50_lookback': 'SFH t50 fractional lookback',
    'sfh_entropy': 'Normalized SFH entropy',
    'sfh_log_old_recent': 'log(old 20% / recent 20%)',
    'sfh_posterior_width': 'Mean SFH posterior width',
    'reconstruction_w1': 'Reconstruction W1',
    'reconstruction_mae': 'Reconstruction MAE',
    'paired_cosine': 'Matched image–SFH cosine',
    'image_to_sfh_rank_percentile': 'Image→SFH rank percentile',
    'sfh_to_image_rank_percentile': 'SFH→image rank percentile',
}

_PROPERTY_PALETTES = {
    'Redshift z': 'plasma',
    'VIS total magnitude (AB)': 'viridis',
    'log M★': 'inferno',
    'Matched image–SFH cosine': 'coolwarm',
    'Image→SFH rank percentile': 'viridis',
    'SFH→image rank percentile': 'viridis',
}

_COORDINATES = {
    'Joint average': 'xy_joint',
    'Image encoder': 'xy_image',
    'SFH encoder': 'xy_sfh',
    'SFH autoencoder latent': 'xy_sfh_preprojection',
    'Shared UMAP: image': 'xy_shared_image',
    'Shared UMAP: SFH': 'xy_shared_sfh',
}

_ARCHIVE_RESERVED = {
    'galaxy_id', 'h5_row', 'image_embedding', 'sfh_embedding',
    'sfh_preprojection_embedding', 'joint_embedding', *_COORDINATES.values(),
}

_BLANK_HTML = (
    '<div style="height:220px;display:flex;align-items:center;'
    'justify-content:center;color:#999;font-size:13px">'
    'Draw a selection on either UMAP.</div>'
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5', type=Path, required=True,
                        help='Compact euclid_explorer.h5 or full sfh_clip HDF5.')
    parser.add_argument('--stamps', type=Path,
                        help='Optional extracted VIS directory, or its parent.')
    parser.add_argument('--umap', type=Path, action='append', required=True,
                        help='Diagnostic NPZ; repeat to compare runs.')
    parser.add_argument('--label', action='append', default=[],
                        help='Optional display label for each --umap, in order.')
    parser.add_argument('--band', default='VIS')
    parser.add_argument('--port', type=int, default=5007)
    parser.add_argument('--selection-output', type=Path,
                        default=Path('euclid_selected_galaxies.csv'))
    args, _ = parser.parse_known_args()
    if args.label and len(args.label) != len(args.umap):
        parser.error('Provide either no --label arguments or one per --umap.')
    return args


def _read_rows(dataset, rows):
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(dataset[rows[order]])[inverse]


def _default_run_label(path):
    for parent in (path.parent.parent, path.parent):
        if parent.name.startswith('training_'):
            return parent.name.removeprefix('training_')
    return path.stem


def _load_archive(path):
    with np.load(path) as archive:
        required = {'galaxy_id'}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f'Missing arrays in {path}: {missing}')
        ids = np.asarray(archive['galaxy_id'], dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError(f'Invalid or duplicate galaxy_id values in {path}.')

        coordinates = {}
        for label, key in _COORDINATES.items():
            if key not in archive:
                continue
            values = np.asarray(archive[key], dtype=np.float32)
            if values.shape != (len(ids), 2):
                raise ValueError(f'{key} in {path} must have shape ({len(ids)}, 2).')
            coordinates[label] = values
        if not coordinates:
            raise ValueError(
                f'{path} contains none of the supported UMAP coordinate arrays.'
            )

        embeddings = {}
        for label, key in (
            ('Joint average', 'joint_embedding'),
            ('Image encoder', 'image_embedding'),
            ('SFH encoder', 'sfh_embedding'),
            ('SFH autoencoder latent', 'sfh_preprojection_embedding'),
        ):
            if key not in archive:
                continue
            values = np.asarray(archive[key], dtype=np.float32)
            if values.ndim != 2 or len(values) != len(ids):
                raise ValueError(f'{key} in {path} has an invalid shape.')
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
                raise ValueError(f'{key} in {path} contains invalid rows.')
            embeddings[label] = values / norms

        properties = {}
        for key in archive.files:
            if key == 'sfh_recent_sfr_per_formed_mass':
                continue  # Recomputed below as a logarithmic display property.
            if key in _ARCHIVE_RESERVED:
                continue
            values = np.asarray(archive[key])
            if (values.ndim != 1 or len(values) != len(ids)
                    or values.dtype.kind not in 'biuf'):
                continue
            label = _PROPERTY_LABELS.get(key, key.replace('_', ' ').capitalize())
            properties[label] = values.astype(np.float64)
    return ids, coordinates, embeddings, properties


def _align(values, positions):
    return np.asarray(values)[positions]


def load_data(h5_path, archive_paths, labels=None):
    loaded = [_load_archive(path) for path in archive_paths]
    first_ids = loaded[0][0]
    common = set(map(int, first_ids))
    for ids, _, _, _ in loaded[1:]:
        common.intersection_update(map(int, ids))
    with h5py.File(h5_path, 'r') as source:
        h5_ids = np.asarray(source['galaxy_id'][:], dtype=np.int64)
    common.intersection_update(map(int, h5_ids))
    galaxy_ids = np.array(
        [value for value in first_ids if int(value) in common], dtype=np.int64,
    )
    if not len(galaxy_ids):
        raise ValueError('The HDF5 file and UMAP archives have no IDs in common.')

    h5_by_id = {int(value): index for index, value in enumerate(h5_ids)}
    h5_rows = np.array([h5_by_id[int(value)] for value in galaxy_ids], dtype=np.int64)
    with h5py.File(h5_path, 'r') as source:
        source_rows = (
            _read_rows(source['source_h5_row'], h5_rows).astype(np.int64)
            if 'source_h5_row' in source else h5_rows.copy()
        )
        duration = sfh_duration_80(
            _read_rows(source['sfh'], h5_rows), source['sfh_time_grid'][:],
            float(source.attrs.get('sfh_log_epsilon', 1e-10)),
        )
        activity = sfh_recent_activity(
            _read_rows(source['sfh'], h5_rows), source['sfh_time_grid'][:],
            float(source.attrs.get('sfh_log_epsilon', 1e-10)),
            _read_rows(source['sfh_time_norm'], h5_rows) if 'sfh_time_norm' in source else None,
        )
    runs = {}
    used_labels = set()
    for index, (path, (ids, coordinates, embeddings, properties)) in enumerate(
        zip(archive_paths, loaded)
    ):
        label = labels[index] if labels else _default_run_label(path)
        original_label = label
        suffix = 2
        while label in used_labels:
            label = f'{original_label} ({suffix})'
            suffix += 1
        used_labels.add(label)
        by_id = {int(value): position for position, value in enumerate(ids)}
        positions = np.array([by_id[int(value)] for value in galaxy_ids], dtype=np.int64)
        runs[label] = {
            'coordinates': {
                key: _align(value, positions) for key, value in coordinates.items()
            },
            'embeddings': {
                key: _align(value, positions) for key, value in embeddings.items()
            },
            'properties': {
                key: _align(value, positions) for key, value in properties.items()
            },
            'path': path,
        }
        runs[label]['properties'][_PROPERTY_LABELS['sfh_duration_80']] = duration
        runs[label]['properties'].update({_PROPERTY_LABELS[k]: v for k, v in activity.items() if k in _PROPERTY_LABELS})

    first_properties = next(iter(runs.values()))['properties']
    redshift = first_properties.get('Redshift z')
    if redshift is None:
        with h5py.File(h5_path, 'r') as source:
            if 'redshift' not in source:
                redshift = None
            else:
                redshift = _read_rows(source['redshift'], h5_rows).astype(float)
    if redshift is None:
        redshift = np.full(len(galaxy_ids), np.nan)
    return {
        'galaxy_ids': galaxy_ids,
        'h5_rows': h5_rows,
        'source_h5_rows': source_rows,
        'redshift': np.asarray(redshift, dtype=float),
        'runs': runs,
    }


def _stamp_directory(path, band):
    if path is None:
        return None
    if (path / band).is_dir():
        return path / band
    return path


def _linear_sfh(log_sfh, epsilon):
    weights = np.maximum(10.0 ** np.asarray(log_sfh, dtype=float) - epsilon, epsilon)
    return weights / np.maximum(weights.sum(axis=-1, keepdims=True), epsilon)


def _render_sfh(ax, time, median_log, p16_log, p84_log, epsilon,
                reconstruction=None, redshift=None, time_norm_myr=None, cosmic_reference=False,
                ms_reference=False, log_mass=np.nan, return_fraction=0., mass_offset=0., rejuvenation=None, ms_sfr_offset=-.93):
    track = None
    median = _linear_sfh(median_log, epsilon)
    if p16_log is not None and p84_log is not None:
        lower = np.maximum(10.0 ** p16_log - epsilon, epsilon)
        upper = np.maximum(10.0 ** p84_log - epsilon, epsilon)
        ax.fill_between(time, lower, upper, color='steelblue', alpha=0.22,
                        linewidth=0, label='16–84% posterior')
    ax.plot(time, median, color='steelblue', linewidth=1.1, label='median')
    if reconstruction is not None:
        ax.plot(
            time, reconstruction, color='darkorange', linewidth=1.0,
            linestyle='--', label='decoder reconstruction',
        )
    if cosmic_reference and redshift is not None:
        reference = cosmic_sfh_weights(time, redshift, time_norm_myr)
        if np.isfinite(reference).all():
            ax.plot(time, reference, color='#7b3294', linestyle='--',
                    linewidth=1.0, label='Cosmic SFH (MD14), normalized')
    if ms_reference and redshift is not None:
        track = main_sequence_along_sfh(time, median_log, redshift, log_mass,
                                        time_norm_myr, return_fraction, mass_offset, epsilon,
                                        ms_sfr_offset=ms_sfr_offset)
        reference = track['weights']
        if np.isfinite(reference).any():
            ax.plot(time, reference, color='#238b45', linestyle=':', linewidth=1,
                    label=f'MS shifted {ms_sfr_offset:+.2f} dex: extrapolated')
            ax.plot(time, np.where(track['supported'], reference, np.nan),
                    color='#238b45', linestyle='--', linewidth=1.2,
                    label=f'MS shifted {ms_sfr_offset:+.2f} dex: conservative domain')
            ax.text(.02, .96, f'MS {ms_sfr_offset:+.2f} dex; R={return_fraction:.2f}; log M★={log_mass + mass_offset:.2f}',
                    transform=ax.transAxes, va='top', fontsize=4, color='#238b45')
        else:
            ax.text(.02, .96, 'MS reference unavailable (mass/z)',
                    transform=ax.transAxes, va='top', fontsize=4, color='#238b45')
    if rejuvenation is not None:
        start, width, recent_width = rejuvenation
        ax.axvspan(0, recent_width, color='#e6ab02', alpha=.15)
        ax.axvspan(start, start+width, color='gray', alpha=.2)
        ax.text(.98, .02, 'Rejuvenation candidate', transform=ax.transAxes,
                ha='right', fontsize=4, color='#996600')
    ax.set_yscale('linear')
    ax.set_ylim(bottom=0)
    ax.set_xlim(float(time.min()), float(time.max()))
    ax.set_xlabel('Fractional lookback time', fontsize=5)
    ax.set_ylabel('Normalized SFH weight', fontsize=5)
    ax.tick_params(labelsize=5)
    if redshift is not None and np.isfinite(redshift) and redshift >= 0:
        age = (float(time_norm_myr) / 1000 if time_norm_myr is not None
               else float(_SFH_COSMOLOGY.age(redshift).value))
        if np.isfinite(age) and age > 0:
            top = ax.secondary_xaxis(
                'top', functions=(lambda fraction: age * fraction,
                                  lambda lookback: lookback / age))
            top.set_xlabel('Time before observation [Gyr]', fontsize=5, labelpad=2)
            top.xaxis.set_major_locator(MaxNLocator(nbins=3))
            top.tick_params(labelsize=5, pad=1)

    return track


def _render_delta_ms(ax, time, track):
    """Display model log ratios; arrows denote clipped or zero-SFR bins."""
    ax.axhline(0, color='#238b45', linewidth=.8)
    ax.set(xlim=(float(time.min()), float(time.max())), ylim=(-3, 3))
    ax.set_xlabel('Fractional lookback time', fontsize=5)
    ax.set_ylabel('ΔMS [dex]', fontsize=5)
    ax.tick_params(labelsize=5)
    ax.set_yticks([-3, 0, 3])
    if track is None or 'delta_ms' not in track:
        ax.text(.5, .5, 'ΔMS unavailable (mass/z)', transform=ax.transAxes,
                ha='center', fontsize=5)
        return
    delta = track['delta_ms']
    finite = np.isfinite(delta)
    values = np.where(finite, np.clip(delta, -3, 3), np.nan)
    ax.plot(time, values, color='steelblue', linestyle=':', linewidth=.8)
    ax.plot(time, np.where(track['supported'], values, np.nan),
            color='steelblue', linewidth=1.)
    low = (delta < -3) | np.isneginf(delta)
    high = delta > 3
    ax.scatter(np.asarray(time)[low], np.full(low.sum(), -2.9), marker='v',
               s=5, color='steelblue')
    ax.scatter(np.asarray(time)[high], np.full(high.sum(), 2.9), marker='^',
               s=5, color='steelblue')
    if np.any(low | high):
        ax.text(.98, .03, 'Triangles: beyond ±3 or zero SFR',
                transform=ax.transAxes, ha='right', fontsize=3.5)


def _fig_to_html(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=105, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return (
        f'<img src="data:image/png;base64,{encoded}" '
        'style="width:100%;max-width:650px"/>'
    )


def _gallery(h5_path, stamp_dir, selected, data, rng, band, ordered=False, cosmic_reference=False,
             ms_reference=False, return_fraction=0., mass_offset=0., ms_sfr_offset=-.93, show_delta_ms=True):
    n = min(N_DISPLAY, len(selected))
    shown = (np.asarray(selected, dtype=int)[:n] if ordered else
             np.sort(rng.choice(np.asarray(selected, dtype=int), n, replace=False)))
    rows = data['h5_rows'][shown]
    ids = data['galaxy_ids'][shown]
    redshift = data['redshift'][shown]
    with h5py.File(h5_path, 'r') as source:
        masses = (_read_rows(source['phz_pp_median_stellarmass'], rows)
                  if 'phz_pp_median_stellarmass' in source else
                  next((run['properties']['log M★'][shown] for run in data['runs'].values()
                        if 'log M★' in run['properties']), np.full(n, np.nan)))
        median = _read_rows(source['sfh'], rows)
        p16 = _read_rows(source['sfh_p16'], rows) if 'sfh_p16' in source else None
        p84 = _read_rows(source['sfh_p84'], rows) if 'sfh_p84' in source else None
        reconstruction = (
            _read_rows(source['sfh_reconstruction'], rows)
            if 'sfh_reconstruction' in source else None
        )
        time_norm = (_read_rows(source['sfh_time_norm'], rows)
                     if 'sfh_time_norm' in source else None)
        time = np.asarray(source['sfh_time_grid'][:], dtype=float)
        epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))

    nrows = max(1, (n + NCOLS - 1) // NCOLS)
    if stamp_dir is not None:
        fig_images, image_axes = plt.subplots(
            nrows, NCOLS, figsize=(NCOLS * 1.7, nrows * 1.7), squeeze=False,
        )
    else:
        fig_images, image_axes = None, None
    with_delta = show_delta_ms and ms_reference
    if with_delta:
        fig_sfhs = plt.figure(figsize=(NCOLS * 1.9, nrows * 3.0))
        outer = fig_sfhs.add_gridspec(nrows, NCOLS)
        sfh_axes = np.empty((nrows, NCOLS), dtype=object)
        delta_axes = np.empty_like(sfh_axes)
        for row in range(nrows):
            for col in range(NCOLS):
                inner = outer[row, col].subgridspec(2, 1, height_ratios=[2, 1], hspace=.08)
                sfh_axes[row, col] = fig_sfhs.add_subplot(inner[0])
                delta_axes[row, col] = fig_sfhs.add_subplot(inner[1], sharex=sfh_axes[row, col])
    else:
        fig_sfhs, sfh_axes = plt.subplots(
            nrows, NCOLS, figsize=(NCOLS * 1.9, nrows * 2.15), squeeze=False,
        )
    axes_groups = [sfh_axes.flat]
    if with_delta:
        axes_groups.append(delta_axes.flat)
    if image_axes is not None:
        axes_groups.append(image_axes.flat)
    for axes in axes_groups:
        for axis in axes:
            axis.set_visible(False)

    for index, (galaxy_id, z) in enumerate(zip(ids, redshift)):
        if image_axes is not None:
            image_axis = image_axes.flat[index]
            image_axis.set_visible(True)
            stamp = stamp_dir / f'{band}_{int(galaxy_id)}.jpg'
            if stamp.is_file():
                with Image.open(stamp) as image:
                    image_axis.imshow(np.asarray(image.convert('L')), cmap='gray',
                                      interpolation='nearest')
            else:
                image_axis.text(0.5, 0.5, 'missing stamp', ha='center', va='center')
            image_axis.set_xticks([])
            image_axis.set_yticks([])
            image_axis.set_title(f'{index + 1}. {int(galaxy_id)}  z={z:.2f}', fontsize=5, pad=1)

        sfh_axis = sfh_axes.flat[index]
        sfh_axis.set_visible(True)
        rejuvenation = None
        diagnostic = data.get('rejuvenation')
        if diagnostic is not None and diagnostic['values']['candidate'][shown[index]] == 1:
            rejuvenation = (diagnostic['values']['lull_start'][shown[index]],
                            diagnostic['settings']['lull_width'],
                            diagnostic['settings']['recent_width'])
        track = _render_sfh(
            sfh_axis, time, median[index],
            p16[index] if p16 is not None else None,
            p84[index] if p84 is not None else None,
            epsilon,
            reconstruction[index] if reconstruction is not None else None,
            redshift=z, time_norm_myr=time_norm[index] if time_norm is not None else None,
            cosmic_reference=cosmic_reference, ms_reference=ms_reference,
            log_mass=masses[index], return_fraction=return_fraction, mass_offset=mass_offset,
            ms_sfr_offset=ms_sfr_offset,
            rejuvenation=rejuvenation,
        )
        if with_delta:
            delta_axis = delta_axes.flat[index]
            delta_axis.set_visible(True)
            _render_delta_ms(delta_axis, time, track)
            sfh_axis.set_xlabel('')
            sfh_axis.tick_params(labelbottom=False)
        sfh_axis.set_title(f'{index + 1}. {int(galaxy_id)}  z={z:.2f}', fontsize=5, pad=27)

    if fig_images is not None:
        fig_images.tight_layout(pad=0.4)
    fig_sfhs.tight_layout(pad=0.4)
    return fig_images, fig_sfhs


def _population_sfh(h5_path, rows):
    with h5py.File(h5_path, 'r') as source:
        log_sfhs = _read_rows(source['sfh'], rows)
        reconstruction = (
            _read_rows(source['sfh_reconstruction'], rows)
            if 'sfh_reconstruction' in source else None
        )
        time = np.asarray(source['sfh_time_grid'][:], dtype=float)
        epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))
    weights = _linear_sfh(log_sfhs, epsilon)
    lower, median, upper = np.percentile(weights, [16, 50, 84], axis=0)
    fig, axis = plt.subplots(figsize=(5, 2.6))
    axis.fill_between(time, lower, upper, color='steelblue', alpha=0.25,
                      label='16–84% of selected galaxies')
    axis.plot(time, median, color='steelblue', linewidth=1.5,
              label=f'population median (N={len(rows):,})')
    if reconstruction is not None:
        axis.plot(
            time, np.median(reconstruction, axis=0), color='darkorange',
            linewidth=1.2, linestyle='--', label='median reconstruction',
        )
    axis.set_yscale('linear')
    axis.set_ylim(bottom=0)
    axis.set_xlim(float(time.min()), float(time.max()))
    axis.set_xlabel('Fractional lookback time', fontsize=8)
    axis.set_ylabel('Normalized SFH weight', fontsize=8)
    axis.legend(fontsize=7)
    axis.tick_params(labelsize=7)
    fig.tight_layout(pad=0.6)
    return fig


def _finite_values(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.zeros(len(values), dtype=float), np.array([0.0])
    median = np.median(finite)
    return np.where(np.isfinite(values), values, median), finite


def _bounds(values):
    safe, finite = _finite_values(values)
    lower, upper = np.percentile(finite, [1, 99])
    full_lower, full_upper = finite.min(), finite.max()
    if full_lower == full_upper:
        padding = max(abs(full_lower) * 0.01, 1e-6)
        full_lower -= padding
        full_upper += padding
        lower, upper = full_lower, full_upper
    step = max((full_upper - full_lower) / 200, 1e-6)
    return safe, float(full_lower), float(full_upper), float(lower), float(upper), step


def build_app(h5_path, stamp_dir, data, band, selection_output):
    rng = np.random.default_rng(42)
    n_objects = len(data['galaxy_ids'])
    run_names = list(data['runs'])
    left_run_default = run_names[0]
    right_run_default = run_names[1] if len(run_names) > 1 else run_names[0]

    all_properties = sorted({
        key for run in data['runs'].values() for key in run['properties']
    })
    if not all_properties:
        raise ValueError('No scalar properties were found in the UMAP archives.')

    def property_values(run_name, property_name):
        return data['runs'][run_name]['properties'].get(
            property_name, np.full(n_objects, np.nan),
        )

    def coordinate_values(run_name, embedding_name):
        coordinates = data['runs'][run_name]['coordinates']
        if embedding_name not in coordinates:
            embedding_name = next(iter(coordinates))
        return coordinates[embedding_name]

    def latent_values(run_name, embedding_name):
        embeddings = data['runs'][run_name]['embeddings']
        if embedding_name in embeddings:
            return embeddings[embedding_name]
        if embedding_name == 'Shared UMAP: image' and 'Image encoder' in embeddings:
            return embeddings['Image encoder']
        if embedding_name == 'Shared UMAP: SFH' and 'SFH encoder' in embeddings:
            return embeddings['SFH encoder']
        return None

    left_embedding_default = next(iter(data['runs'][left_run_default]['coordinates']))
    right_embedding_default = next(iter(data['runs'][right_run_default]['coordinates']))
    if stamp_dir is None and 'Redshift z' in all_properties:
        left_color_default = 'Redshift z'
    else:
        left_color_default = (
            'Sérsic index n'
            if 'Sérsic index n' in all_properties else all_properties[0]
        )
    right_color_default = (
        'SFH fraction: recent 20%'
        if 'SFH fraction: recent 20%' in all_properties
        else all_properties[min(1, len(all_properties) - 1)]
    )

    left_xy = coordinate_values(left_run_default, left_embedding_default)
    right_xy = coordinate_values(right_run_default, right_embedding_default)
    source = ColumnDataSource(dict(
        x_left=left_xy[:, 0].tolist(),
        y_left=left_xy[:, 1].tolist(),
        x_right=right_xy[:, 0].tolist(),
        y_right=right_xy[:, 1].tolist(),
        color_left=property_values(left_run_default, left_color_default).tolist(),
        color_right=property_values(right_run_default, right_color_default).tolist(),
        galaxy_id=data['galaxy_ids'].tolist(),
        redshift=data['redshift'].tolist(),
        dynamic_color=['#bdbdbd'] * n_objects,
    ))

    z_safe, z_finite = _finite_values(data['redshift'])
    z_mask = np.ones(n_objects, dtype=bool)
    property_mask = np.ones(n_objects, dtype=bool)
    match_mask = np.ones(n_objects, dtype=bool)
    match_state = {'values': None, 'masses': None, 'settings': None}
    view = CDSView(filter=BooleanFilter(booleans=[True] * n_objects))

    def update_view():
        visible = z_mask & property_mask & match_mask
        view.filter = BooleanFilter(booleans=visible.tolist())
        source.selected.indices = [i for i in source.selected.indices if visible[i]]

    def make_plot(x_field, y_field, color_field, initial_values, title):
        _, _, _, lower, upper, _ = _bounds(initial_values)
        mapper = LinearColorMapper(
            palette=Plasma256, low=lower, high=upper, nan_color=(0, 0, 0, 0),
        )
        plot = bk_figure(
            width=510, height=480,
            tools='lasso_select,box_select,wheel_zoom,pan,reset',
            active_drag='lasso_select', title=title, output_backend='webgl',
        )
        continuous = plot.scatter(
            x_field, y_field, source=source, view=view,
            color={'field': color_field, 'transform': mapper},
            size=3, alpha=0.72, line_width=0,
            selection_fill_color={'field': color_field, 'transform': mapper},
            selection_alpha=1,
            selection_line_color='#222222', selection_line_width=0.6,
            nonselection_alpha=0.72,
        )
        dynamic = plot.scatter(
            x_field, y_field, source=source, view=view,
            fill_color='dynamic_color', size=3, alpha=0.8, line_width=0,
            selection_fill_color='dynamic_color', selection_alpha=1,
            selection_line_color='#222222',
            selection_line_width=0.6,
            nonselection_alpha=0.8, visible=False,
        )
        continuous.selection_glyph.size = 4
        dynamic.selection_glyph.size = 4
        plot.add_layout(ColorBar(
            color_mapper=mapper, ticker=BasicTicker(), label_standoff=8,
            width=12, location=(0, 0),
        ), 'right')
        plot.add_tools(HoverTool(tooltips=[
            ('galaxy ID', '@galaxy_id'), ('redshift', '@redshift{0.000}'),
        ]))
        plot.xaxis.axis_label = 'UMAP 1'
        plot.yaxis.axis_label = 'UMAP 2'
        return plot, mapper, continuous, dynamic

    left_plot, left_mapper, left_continuous, left_dynamic = make_plot(
        'x_left', 'y_left', 'color_left',
        property_values(left_run_default, left_color_default), 'Left UMAP',
    )
    right_plot, right_mapper, right_continuous, right_dynamic = make_plot(
        'x_right', 'y_right', 'color_right',
        property_values(right_run_default, right_color_default), 'Right UMAP',
    )

    left_run = pn.widgets.Select(name='Left run', options=run_names,
                                 value=left_run_default, width=260)
    right_run = pn.widgets.Select(name='Right run', options=run_names,
                                  value=right_run_default, width=260)
    left_embedding = pn.widgets.Select(
        name='Left embedding',
        options=list(data['runs'][left_run_default]['coordinates']),
        value=left_embedding_default, width=260,
    )
    right_embedding = pn.widgets.Select(
        name='Right embedding',
        options=list(data['runs'][right_run_default]['coordinates']),
        value=right_embedding_default, width=260,
    )
    left_color = pn.widgets.Select(name='Left color', options=all_properties,
                                   value=left_color_default, width=260)
    right_color = pn.widgets.Select(name='Right color', options=all_properties,
                                    value=right_color_default, width=260)

    def make_range(name, values):
        _, full_lower, full_upper, lower, upper, step = _bounds(values)
        return pn.widgets.RangeSlider(
            name=name, start=full_lower, end=full_upper, value=(lower, upper),
            step=step, width=260,
        )

    left_range = make_range(
        'Left color range', property_values(left_run_default, left_color_default),
    )
    right_range = make_range(
        'Right color range', property_values(right_run_default, right_color_default),
    )

    def update_color(run_widget, color_widget, range_widget, field, mapper,
                     continuous, dynamic):
        values = property_values(run_widget.value, color_widget.value)
        _, full_lower, full_upper, lower, upper, step = _bounds(values)
        source.data[field] = np.where(np.isfinite(values), values, np.nan).tolist()
        mapper.palette = _PALETTES[
            _PROPERTY_PALETTES.get(color_widget.value, 'plasma')
        ]
        mapper.low, mapper.high = lower, upper
        range_widget.start, range_widget.end = full_lower, full_upper
        range_widget.step, range_widget.value = step, (lower, upper)
        continuous.visible, dynamic.visible = True, False

    def update_left_coordinates(event=None):
        available = list(data['runs'][left_run.value]['coordinates'])
        left_embedding.options = available
        if left_embedding.value not in available:
            left_embedding.value = available[0]
        coordinates = coordinate_values(left_run.value, left_embedding.value)
        source.data['x_left'] = coordinates[:, 0].tolist()
        source.data['y_left'] = coordinates[:, 1].tolist()
        left_plot.title.text = f'{left_run.value}: {left_embedding.value}'
        update_color(left_run, left_color, left_range, 'color_left', left_mapper,
                     left_continuous, left_dynamic)
        if event is not None:
            update_filter_bounds()

    def update_right_coordinates(event=None):
        available = list(data['runs'][right_run.value]['coordinates'])
        right_embedding.options = available
        if right_embedding.value not in available:
            right_embedding.value = available[0]
        coordinates = coordinate_values(right_run.value, right_embedding.value)
        source.data['x_right'] = coordinates[:, 0].tolist()
        source.data['y_right'] = coordinates[:, 1].tolist()
        right_plot.title.text = f'{right_run.value}: {right_embedding.value}'
        update_color(right_run, right_color, right_range, 'color_right', right_mapper,
                     right_continuous, right_dynamic)

    redshift_finite = z_finite[np.isfinite(z_finite)]
    z_low = float(np.floor(redshift_finite.min() * 10) / 10) if len(redshift_finite) else 0
    z_high = float(np.ceil(redshift_finite.max() * 10) / 10) if len(redshift_finite) else 6
    redshift_slider = pn.widgets.RangeSlider(
        name='Redshift range', start=z_low, end=z_high,
        value=(z_low, z_high), step=0.01, width=220,
    )

    property_enabled = pn.widgets.Checkbox(name='Use additional property filter', value=False)
    filter_property = pn.widgets.Select(
        name='Filter property (left run)', options=all_properties,
        value=left_color_default, width=220,
    )
    filter_range = make_range(
        'Filter range', property_values(left_run_default, left_color_default),
    )
    filter_range.width = 220
    match_enabled = pn.widgets.Checkbox(name='Match mass + recent ΔMS', value=False)
    match_window = pn.widgets.Select(name='Recent activity window', options=list(WINDOWS), value='0.1 cosmic age')
    match_mass_low = pn.widgets.FloatInput(name='Minimum log M★', value=9.5, step=.1, width=105)
    match_mass_high = pn.widgets.FloatInput(name='Maximum log M★', value=10.5, step=.1, width=105)
    match_recent_low = pn.widgets.FloatInput(name='Minimum ΔMS', value=-.3, step=.1, width=105)
    match_recent_high = pn.widgets.FloatInput(name='Maximum ΔMS', value=.3, step=.1, width=105)
    match_compute = pn.widgets.Button(name='Compute recent ΔMS', button_type='primary')
    match_select = pn.widgets.Button(name='Select all visible', button_type='success')
    match_info = pn.pane.Markdown('Compute recent ΔMS, set mass/activity bounds and the redshift range, then enable matching.', width=220)
    cluster_count = pn.widgets.IntInput(
        name='k clusters', value=6, start=2, end=50, width=90,
    )
    cluster_button = pn.widgets.Button(
        name='Cluster visible subset', button_type='success', width=190,
    )
    reset_button = pn.widgets.Button(name='Show all', button_type='light', width=90)
    sample_button = pn.widgets.Button(name='New random sample', button_type='primary', width=190)
    save_button = pn.widgets.Button(name='Save selected IDs', button_type='primary', width=190)
    info = pn.pane.Markdown('_Draw a selection on either UMAP._', width=220)
    cluster_info = pn.pane.Markdown('', width=220, styles={'font-size': '11px'})
    image_pane = pn.pane.HTML(_BLANK_HTML, width=570)
    sfh_pane = pn.pane.HTML(_BLANK_HTML, width=650)
    population_pane = pn.pane.HTML(_BLANK_HTML, width=430)

    cosmic_toggle = pn.widgets.Checkbox(name='Cosmic SFH reference (MD14)', value=True)
    ms_toggle = pn.widgets.Checkbox(name='MS along inferred mass history', value=True)
    delta_ms_toggle = pn.widgets.Checkbox(name='Show ΔMS history below SFH', value=True)
    ms_return = pn.widgets.FloatSlider(name='MS returned mass fraction R', start=0., end=.6, step=.05, value=0.)
    ms_deviation_button = pn.widgets.Button(name='Compute MS statistics', button_type='primary')
    ms_delta_floor = pn.widgets.FloatInput(name='ΔMS lower floor (dex)', value=-3., step=.5)
    ms_deviation_info = pn.pane.Markdown('Uses the median SFH and current MS settings.')
    ms_sfr_shift = pn.widgets.FloatInput(name='Empirical MS SFR shift (dex)', value=-.93, step=.05)
    ms_offset = pn.widgets.FloatInput(name='Mass → Kroupa offset (dex)', value=0., step=.01)
    rej_recent = pn.widgets.FloatInput(name='Recent window Δf', value=.05, step=.01)
    rej_lull = pn.widgets.FloatInput(name='Lull window Δf', value=.05, step=.01)
    rej_contrast = pn.widgets.FloatInput(name='Minimum recovery/older contrast', value=5., step=1.)
    rej_old = pn.widgets.FloatInput(name='Minimum older mass fraction', value=.5, step=.05)
    rej_mass = pn.widgets.FloatInput(name='Minimum recent mass fraction', value=.01, step=.005)
    rej_button = pn.widgets.Button(name='Compute rejuvenation', button_type='primary')
    rej_info = pn.pane.Markdown('Uses fractional time and normalized mass. Compute to add color/filter properties.', width=220)
    migration_mode = pn.widgets.Select(name='Migration time units', options=['Fractional time', 'Gyr'])
    migration_lag = pn.widgets.FloatInput(name='Migration lag Δ', value=.1, step=.01)
    migration_window = pn.widgets.FloatInput(name='SFR averaging width', value=.02, step=.01)
    migration_button = pn.widgets.Button(name='Compute migration', button_type='primary')
    migration_info = pn.pane.Markdown('Median-SFH diagnostics inspired by Arango-Toro et al. (2025).', width=220)
    def set_migration_units(event):
        migration_lag.value = .7 if event.new == 'Gyr' else .1
        migration_window.value = .1 if event.new == 'Gyr' else .02
    migration_mode.param.watch(set_migration_units, 'value')
    gallery_state = {'rng_before': rng.bit_generator.state}
    path_mode = pn.widgets.Checkbox(name='Draw line (click start, then end)', value=False)
    path_count = pn.widgets.IntSlider(name='Samples along line', start=2, end=N_DISPLAY, value=min(12, N_DISPLAY))
    path_radius = pn.widgets.FloatSlider(name='Max distance (% of UMAP span)', start=0.5, end=20, step=0.5, value=3)
    path_state = {'start': None, 'side': None, 'indices': ()}
    path_sources = {}
    for side, plot in [('left', left_plot), ('right', right_plot)]:
        line = ColumnDataSource(dict(x=[], y=[]))
        markers = ColumnDataSource(dict(x=[], y=[], label=[]))
        plot.line('x', 'y', source=line, line_color='#00a6a6', line_width=3)
        plot.scatter('x', 'y', source=markers, size=9, fill_alpha=0,
                     line_color='#00a6a6', line_width=2)
        plot.add_layout(LabelSet(x='x', y='y', text='label', source=markers,
                                x_offset=5, y_offset=5, text_color='#008080', text_font_size='10pt'))
        path_sources[side] = (line, markers)

    def clear_path(event=None):
        path_state.update(start=None, side=None, indices=())
        for line, markers in path_sources.values():
            line.data = dict(x=[], y=[])
            markers.data = dict(x=[], y=[], label=[])

    def path_tap(side, event):
        if not path_mode.value:
            return
        point = np.array([event.x, event.y], dtype=float)
        if path_state['start'] is None or path_state['side'] != side:
            clear_path()
            path_state.update(start=point, side=side)
            path_sources[side][0].data = dict(x=[event.x], y=[event.y])
            info.object = 'Start set. Click the endpoint on the same UMAP.'
            return
        start = path_state['start']
        xy = np.column_stack([source.data[f'x_{side}'], source.data[f'y_{side}']])
        finite = np.isfinite(xy).all(axis=1)
        span = np.linalg.norm(np.ptp(xy[finite], axis=0)) if finite.any() else 0
        try:
            selected = sample_segment(xy, start, point, path_count.value,
                                      span * path_radius.value / 100, z_mask & property_mask & match_mask)
        except ValueError as error:
            info.object = str(error)
            return
        path_state.update(start=None, indices=tuple(selected))
        path_sources[side][0].data = dict(x=[start[0], point[0]], y=[start[1], point[1]])
        path_sources[side][1].data = dict(x=xy[selected, 0].tolist(), y=xy[selected, 1].tolist(),
                                         label=[str(i + 1) for i in range(len(selected))])
        source.selected.indices = selected
        previous_selection[0] = tuple(selected)
        refresh(selected, ordered=True)

    left_plot.on_event(Tap, lambda event: path_tap('left', event))
    right_plot.on_event(Tap, lambda event: path_tap('right', event))
    clear_path_button = pn.widgets.Button(name='Clear line', width=190)
    clear_path_button.on_click(clear_path)
    for widget in (left_run, right_run, left_embedding, right_embedding,
                   redshift_slider, filter_range, path_mode):
        widget.param.watch(clear_path, 'value')

    def refresh(selected, ordered=False):
        selected = list(selected)
        if not selected:
            info.object = '_No galaxies selected._'
            image_pane.object = sfh_pane.object = population_pane.object = _BLANK_HTML
            return
        info.object = (
            f'**{len(selected):,}** selected — showing '
            f'{min(N_DISPLAY, len(selected))} ' + ('examples in line order' if ordered else 'random examples')
        )
        gallery_state['rng_before'] = rng.bit_generator.state
        image_fig, sfh_fig = _gallery(
            h5_path, stamp_dir, selected, data, rng, band, ordered=ordered,
            cosmic_reference=cosmic_toggle.value, ms_reference=ms_toggle.value,
            return_fraction=ms_return.value, mass_offset=ms_offset.value,
            ms_sfr_offset=ms_sfr_shift.value, show_delta_ms=delta_ms_toggle.value,
        )
        if image_fig is not None:
            image_pane.object = _fig_to_html(image_fig)
        sfh_pane.object = _fig_to_html(sfh_fig)
        rows = data['h5_rows'][np.asarray(selected, dtype=int)]
        population_pane.object = _fig_to_html(_population_sfh(h5_path, rows))

    def update_cosmic_reference(event=None):
        rng.bit_generator.state = gallery_state['rng_before']
        selected = source.selected.indices
        refresh(selected, ordered=bool(selected) and tuple(selected) == path_state['indices'])

    for widget in (cosmic_toggle, ms_toggle, ms_return, ms_offset, ms_sfr_shift, delta_ms_toggle):
        widget.param.watch(update_cosmic_reference, 'value')

    def apply_match(event=None):
        nonlocal match_mask
        match_mask = np.ones(n_objects, dtype=bool)
        if match_enabled.value:
            if match_state['values'] is None:
                match_mask[:] = False
                match_info.object = 'Compute recent ΔMS first. No matched objects displayed.'
            else:
                j = list(WINDOWS).index(match_window.value)
                match_mask = matched_mask(match_state['masses'], data['redshift'],
                    match_state['values'][:, j], (match_mass_low.value, match_mass_high.value),
                    redshift_slider.value, (match_recent_low.value, match_recent_high.value))
                match_info.object = (f"**{np.count_nonzero(match_mask & property_mask):,} visible matches**. "
                    f"{match_window.value}; {match_state['settings']}. ΔMS ≤ −4 dex is censored to −4.")
        elif match_state['values'] is not None:
            match_info.object = 'Matching disabled. Cached recent ΔMS is available for colors; enable matching to apply the bounds.'
        clear_path()
        update_view()
        refresh(source.selected.indices)

    def compute_recent(event=None):
        match_compute.disabled = True
        try:
            settings = dict(return_fraction=ms_return.value, mass_offset=ms_offset.value,
                            ms_sfr_offset=ms_sfr_shift.value)
            with h5py.File(h5_path, 'r') as h:
                rows = data['h5_rows']
                masses = (_read_rows(h['phz_pp_median_stellarmass'], rows)
                          if 'phz_pp_median_stellarmass' in h else
                          next((run['properties']['log M★'] for run in data['runs'].values()
                                if 'log M★' in run['properties']), np.full(n_objects, np.nan)))
                values = catalog_recent_offsets(_read_rows(h['sfh'], rows), h['sfh_time_grid'][:],
                    data['redshift'], masses,
                    _read_rows(h['sfh_time_norm'], rows) if 'sfh_time_norm' in h else None,
                    epsilon=float(h.attrs.get('sfh_log_epsilon', 1e-10)), **settings)
            suffix = f"R={ms_return.value:g}, MS={ms_sfr_shift.value:g}, mass offset={ms_offset.value:g} dex"
            match_state.update(values=values, masses=masses, settings=suffix)
            labels = [f'Recent ΔMS {window} [floor=-4; {suffix}]' for window in WINDOWS]
            for run in data['runs'].values():
                run['properties'].update({label: values[:,j] for j,label in enumerate(labels)})
            properties = sorted({key for run in data['runs'].values() for key in run['properties']})
            for widget in (left_color, right_color, filter_property):
                widget.options = properties
            match_info.object = 'Recent ΔMS computed. Enable matching to apply the bounds.'
            apply_match()
        except ValueError as error:
            match_info.object = f'Cannot compute recent ΔMS: {error}'
        finally:
            match_compute.disabled = False

    def invalidate_recent(event=None):
        match_state.update(values=None, settings=None)
        apply_match()
        match_info.object = 'MS settings changed: recompute recent ΔMS before matching.'

    def select_visible(event=None):
        clear_path()
        selected = np.flatnonzero(z_mask & property_mask & match_mask).tolist()
        source.selected.indices = selected
        refresh(selected)

    match_compute.on_click(compute_recent)
    match_select.on_click(select_visible)
    for widget in (match_enabled, match_window, match_mass_low, match_mass_high,
                   match_recent_low, match_recent_high, redshift_slider):
        widget.param.watch(apply_match, 'value')
    for widget in (ms_return, ms_offset, ms_sfr_shift):
        widget.param.watch(invalidate_recent, 'value')

    def compute_ms_deviation(event=None):
        ms_deviation_button.disabled = True
        ms_deviation_info.object = 'Computing MS history statistics…'
        settings = dict(return_fraction=ms_return.value, mass_offset=ms_offset.value,
                        ms_sfr_offset=ms_sfr_shift.value)
        try:
            with h5py.File(h5_path, 'r') as h:
                rows = data['h5_rows']
                masses = (_read_rows(h['phz_pp_median_stellarmass'], rows)
                          if 'phz_pp_median_stellarmass' in h else
                          next((run['properties']['log M★'] for run in data['runs'].values()
                                if 'log M★' in run['properties']), np.full(n_objects, np.nan)))
                values = catalog_deviations(
                    _read_rows(h['sfh'], rows), h['sfh_time_grid'][:], data['redshift'], masses,
                    _read_rows(h['sfh_time_norm'], rows) if 'sfh_time_norm' in h else None,
                    epsilon=float(h.attrs.get('sfh_log_epsilon', 1e-10)),
                    include_summary=True, delta_floor=ms_delta_floor.value, **settings)
            suffix = (f" [R={settings['return_fraction']:g}, MS={settings['ms_sfr_offset']:g}, "
                      f"mass={settings['mass_offset']:g} dex]")
            labels = ['MS D+ excess'+suffix, 'MS D− deficit'+suffix,
                      'MS calibrated time fraction'+suffix]
            floor_suffix = f' [ΔMS floor={ms_delta_floor.value:g} dex]'+suffix
            labels += [name+floor_suffix for name in (
                'MS maximum positive ΔMS (dex)', 'MS minimum negative ΔMS (dex)',
                'MS positive peak ∫Hdt', 'MS negative peak ∫Hdt',
                'MS time-weighted mean ΔMS (dex)', 'MS below-floor time fraction')]
            for run in data['runs'].values():
                run['properties'].update({label: values[:, i] for i, label in enumerate(labels)})
            properties = sorted({key for run in data['runs'].values() for key in run['properties']})
            for widget in (left_color, right_color, filter_property):
                widget.options = properties
            left_color.value, right_color.value = labels[:2]
            update_color(left_run, left_color, left_range, 'color_left', left_mapper,
                         left_continuous, left_dynamic)
            update_color(right_run, right_color, right_range, 'color_right', right_mapper,
                         right_continuous, right_dynamic)
            update_filter_bounds()
            ms_deviation_info.object = (f"Computed for {np.isfinite(values[:, 0]).sum():,}/{n_objects:,} galaxies. "
                'Colors and filters retain settings in their names; recompute after changing MS controls.')
        except ValueError as error:
            ms_deviation_info.object = f'Cannot compute: {error}'
        finally:
            ms_deviation_button.disabled = False

    ms_deviation_button.on_click(compute_ms_deviation)

    def compute_rejuvenation(event=None):
        settings = dict(recent_width=rej_recent.value, lull_width=rej_lull.value,
                        contrast=rej_contrast.value, min_old_fraction=rej_old.value,
                        min_recent_fraction=rej_mass.value)
        rej_button.disabled = True
        try:
            with h5py.File(h5_path, 'r') as source_h5:
                values = catalog_diagnostics(source_h5, data['h5_rows'], settings)
            data['rejuvenation'] = dict(values=values, settings=settings)
            for run in data['runs'].values():
                run['properties'].update({REJ_LABELS[k]: v for k, v in values.items()})
            properties = sorted({key for run in data['runs'].values() for key in run['properties']})
            for widget in (left_color, right_color, filter_property):
                widget.options = properties
            left_color.value = REJ_LABELS['candidate']
            update_color(left_run, left_color, left_range, 'color_left', left_mapper,
                         left_continuous, left_dynamic)
            update_filter_bounds()
            count = int(np.nansum(values['candidate']))
            detail = ('Posterior fraction available (valid draws only).'
                      if 'posterior_fraction' in values else
                      'Median-SFH flags only: this bundle has no posterior draws.')
            rej_info.object = f'**{count:,}/{n_objects:,} median-SFH candidates.** {detail} Gray shading: lull; gold: recent window. Thresholds are exploratory, not a quenching classification.'
            update_cosmic_reference()
        except ValueError as error:
            rej_info.object = f'Cannot compute: {error}'
        finally:
            rej_button.disabled = False

    rej_button.on_click(compute_rejuvenation)

    def compute_migration(event=None):
        migration_button.disabled = True
        try:
            with h5py.File(h5_path, 'r') as source_h5:
                values = catalog_migration(source_h5, data['h5_rows'], migration_mode.value,
                                           migration_lag.value, migration_window.value)
            for run in data['runs'].values():
                run['properties'].update(values)
            properties = sorted({key for run in data['runs'].values() for key in run['properties']})
            for widget in (left_color, right_color, filter_property):
                widget.options = properties
            angle = next(key for key in values if 'angle Φ' in key)
            left_color.value = angle
            update_color(left_run, left_color, left_range, 'color_left', left_mapper,
                         left_continuous, left_dynamic)
            update_filter_bounds()
            valid_key = next(key for key in values if 'measurable endpoints' in key)
            count = int(np.nansum(values[valid_key]))
            migration_info.object = f'**{count:,}/{n_objects:,} measurable vectors.** Positive Φ: rising; negative: declining. Zero-SFR or unavailable endpoints are undefined (not floored). Median SFHs only; rising does not establish rejuvenation.'
        except ValueError as error:
            migration_info.object = f'Cannot compute: {error}'
        finally:
            migration_button.disabled = False
    migration_button.on_click(compute_migration)

    def update_filter_bounds(event=None):
        values = property_values(left_run.value, filter_property.value)
        _, full_lower, full_upper, lower, upper, step = _bounds(values)
        filter_range.start, filter_range.end = full_lower, full_upper
        filter_range.step, filter_range.value = step, (lower, upper)

    def apply_property_filter(event=None):
        nonlocal property_mask
        values, _ = _finite_values(property_values(
            left_run.value, filter_property.value,
        ))
        low, high = filter_range.value
        property_mask = ((values >= low) & (values <= high) if property_enabled.value
                         else np.ones(n_objects, dtype=bool))
        update_view()

    def apply_redshift_filter(event=None):
        nonlocal z_mask
        low, high = redshift_slider.value
        z_mask = (z_safe >= low) & (z_safe <= high)
        update_view()

    def cluster_visible(event=None):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            cluster_info.object = 'Install scikit-learn to use dynamic clustering.'
            return
        visible = np.flatnonzero(z_mask & property_mask & match_mask)
        k = cluster_count.value
        if len(visible) < k:
            cluster_info.object = f'Only {len(visible)} visible galaxies; need at least k={k}.'
            return
        latent = latent_values(left_run.value, left_embedding.value)
        if latent is None:
            features = np.column_stack([
                source.data['x_left'], source.data['y_left'],
            ])[visible]
            cluster_space = '2D UMAP fallback; raw embeddings unavailable'
        else:
            features = latent[visible]
            cluster_space = f'full {left_embedding.value} latent space'
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features)
        step = max(1, 256 // k)
        palette = [Turbo256[min(index * step, 255)] for index in range(k)]
        colors = ['#bdbdbd'] * n_objects
        for position, label in zip(visible, labels):
            colors[position] = palette[int(label)]
        source.data['dynamic_color'] = colors
        left_continuous.visible, left_dynamic.visible = False, True
        right_continuous.visible, right_dynamic.visible = False, True
        sizes = [int(np.count_nonzero(labels == label)) for label in range(k)]
        cluster_info.object = '  \n'.join(
            [f'**{len(visible):,} visible galaxies → {k} clusters**',
             f'using {cluster_space}']
            + [f'cluster {label}: {size:,}' for label, size in enumerate(sizes)]
        )

    def reset(event=None):
        nonlocal property_mask, z_mask, match_mask
        match_enabled.value = False
        property_enabled.value = False
        match_mask = np.ones(n_objects, dtype=bool)
        property_mask = np.ones(n_objects, dtype=bool)
        z_mask = np.ones(n_objects, dtype=bool)
        redshift_slider.value = (z_low, z_high)
        update_filter_bounds()
        update_view()
        clear_path()
        source.selected.indices = []
        refresh([])
        cluster_info.object = ''
        left_continuous.visible, left_dynamic.visible = True, False
        right_continuous.visible, right_dynamic.visible = True, False

    def save_selection(event=None):
        selected = np.asarray(source.selected.indices, dtype=int)
        if not len(selected):
            info.object = '_No galaxies selected to save._'
            return
        selection_output.parent.mkdir(parents=True, exist_ok=True)
        with selection_output.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow([
                'galaxy_id', 'redshift', 'source_h5_row', 'bundle_h5_row',
                'catalog_log_stellar_mass', 'recent_delta_ms', 'recent_window', 'recent_ms_settings',
            ])
            for index in selected:
                writer.writerow([
                    int(data['galaxy_ids'][index]), float(data['redshift'][index]),
                    int(data['source_h5_rows'][index]),
                    int(data['h5_rows'][index]),
                    (float(match_state['masses'][index]) if match_state['masses'] is not None else ''),
                    (float(match_state['values'][index, list(WINDOWS).index(match_window.value)])
                     if match_state['values'] is not None else ''),
                    match_window.value if match_state['values'] is not None else '',
                    match_state['settings'] or '',
                ])
        info.object = f'Saved **{len(selected):,}** IDs to `{selection_output}`.'

    left_run.param.watch(update_left_coordinates, 'value')
    right_run.param.watch(update_right_coordinates, 'value')
    left_embedding.param.watch(update_left_coordinates, 'value')
    right_embedding.param.watch(update_right_coordinates, 'value')
    left_color.param.watch(
        lambda event: update_color(
            left_run, left_color, left_range, 'color_left', left_mapper,
            left_continuous, left_dynamic,
        ), 'value',
    )
    right_color.param.watch(
        lambda event: update_color(
            right_run, right_color, right_range, 'color_right', right_mapper,
            right_continuous, right_dynamic,
        ), 'value',
    )
    left_range.param.watch(
        lambda event: setattr(left_mapper, 'low', float(event.new[0])), 'value',
    )
    left_range.param.watch(
        lambda event: setattr(left_mapper, 'high', float(event.new[1])), 'value',
    )
    right_range.param.watch(
        lambda event: setattr(right_mapper, 'low', float(event.new[0])), 'value',
    )
    right_range.param.watch(
        lambda event: setattr(right_mapper, 'high', float(event.new[1])), 'value',
    )
    property_enabled.param.watch(apply_property_filter, 'value')
    filter_property.param.watch(update_filter_bounds, 'value')
    filter_range.param.watch(apply_property_filter, 'value')
    redshift_slider.param.watch(apply_redshift_filter, 'value')
    cluster_button.on_click(cluster_visible)
    reset_button.on_click(reset)
    def random_sample(event=None):
        clear_path()
        refresh(source.selected.indices)

    sample_button.on_click(random_sample)
    save_button.on_click(save_selection)

    previous_selection = [()]

    def poll_selection():
        current = tuple(source.selected.indices)
        if current != previous_selection[0]:
            previous_selection[0] = current
            ordered = bool(current) and current == path_state['indices']
            if not ordered:
                clear_path()
            refresh(current, ordered=ordered)

    pn.state.add_periodic_callback(poll_selection, period=350)
    update_left_coordinates()
    update_right_coordinates()

    sidebar = pn.Column(
        pn.pane.Markdown('## Euclid embedding explorer'),
        pn.layout.Divider(),
        redshift_slider,
        pn.pane.Markdown('**Compare at matched mass, activity and redshift**'),
        match_compute, match_window,
        pn.Row(match_mass_low, match_mass_high),
        pn.Row(match_recent_low, match_recent_high),
        match_enabled, match_select, match_info,
        pn.pane.Markdown('ΔMS = log₁₀(∫SFR dt / ∫MS dt) in the chosen window. Uses the current shifted MS along inferred mass history. Matching uses catalog log mass; all filters intersect. Missing values are excluded. The generic property filter below also applies. Select all visible or lasso separated UMAP regions to compare shapes; Save selected IDs exports the selected subset.', styles={'font-size':'11px'}),
        pn.layout.Divider(),
        pn.pane.Markdown('**Filter and cluster in the left panel**'),
        property_enabled, filter_property,
        filter_range,
        pn.Row(cluster_count, cluster_button),
        reset_button,
        cluster_info,
        pn.layout.Divider(),
        pn.pane.Markdown('**Sample a UMAP line**'),
        path_mode, path_count, path_radius, clear_path_button,
        pn.pane.Markdown('Nearest visible galaxies at evenly spaced locations; gaps may yield fewer samples. Line order is not a physical time sequence.', styles={'font-size': '11px'}),
        pn.pane.Markdown('**SFH migration**'),
        migration_mode, migration_lag, migration_window, migration_button, migration_info,
        pn.pane.Markdown('SFR windows: [0, width] and [lag, lag + width]. Ratios use normalized formed mass; no MS calibration. T90 means 90% had already formed, so it is usually more recent than T50.', styles={'font-size': '11px'}),
        pn.pane.Markdown('**Rejuvenation candidates**'),
        rej_recent, rej_lull, rej_contrast, rej_old, rej_mass, rej_button, rej_info,
        ms_toggle, delta_ms_toggle, ms_return, ms_offset, ms_sfr_shift,
        ms_delta_floor, ms_deviation_button, ms_deviation_info,
        pn.pane.Markdown("D+ and D− integrate positive and negative SFR−MS differences separately, divided by the integral of SFR+MS. Both are in [0,1], with sum ≤1. Full inferred history including MS extrapolation; calibrated time fraction reports coverage. Undefined pre-formation bins are excluded; zero-SFR bins with nonzero MS count as deficit. Median histories only. ΔMS extrema and time-weighted mean use the adjustable lower floor, including zero SFR. Peak times are accumulated ∫Hdt = ln(a_obs/a_peak), evaluated at bin midpoints; proportional to elapsed fixed-overdensity halo dynamical times, with no halo prefactor applied. The mean remains weighted by physical time; ties use the most recent bin. If an excursion is absent its amplitude is zero and its time undefined. Below-floor time fraction indicates censoring. These statistics also include extrapolated MS epochs.", styles={'font-size': '11px'}),
        pn.pane.Markdown("ΔMS = log10(SFR / shifted MS SFR). Zero is the reference; positive is above, negative below. Dotted segments extrapolate the MS. Triangles mark values beyond ±3 dex or zero SFR (−∞); bins with no inferred mass are undefined and blank. Uses the median history, without posterior uncertainty propagation.", styles={'font-size': '11px'}),
        pn.pane.Markdown("Green: main-sequence SFR along this galaxy’s inferred mass history, on the SAME scale as blue (not independently normalized). The empirical SFR shift defaults to −0.93 dex, calibrated to the median 100 Myr SFH/MS offset at observation for 9,719 positive-rate bright validation galaxies with R=0. Set shift to 0 for the original MS. This constant shift across all epochs is illustrative, not a validated historical correction; blue/green compares against the shifted reference. Constant recycling and in-situ growth assumed; dotted portions extrapolate. IMF unverified: offset 0 assumes Kroupa; Chabrier +0.03 dex, Salpeter −0.21 dex.", styles={'font-size': '11px'}),
        cosmic_toggle,
        pn.pane.Markdown("Purple dashed: cosmic SFH integrated into the same bins and normalized to unit total mass, up to each galaxy’s redshift. A shape reference, not an individual-galaxy prediction. MD14 extrapolates at high redshift.", styles={'font-size': '11px'}),
        info,
        sample_button,
        save_button,
        pn.pane.Markdown(
            '_Selections are linked by galaxy ID across both runs and spaces._',
            styles={'font-size': '11px', 'color': '#777'},
        ),
        width=240,
    )

    detail_columns = []
    if stamp_dir is not None:
        detail_columns.append(
            pn.Column(pn.pane.Markdown(f'#### {band} stamps'), image_pane),
        )
    detail_columns.extend([
        pn.Column(pn.pane.Markdown('#### SFHs with posterior intervals'), sfh_pane),
        pn.Column(pn.pane.Markdown('#### Population SFH'), population_pane),
    ])

    return pn.Column(
        pn.Row(
            pn.Column(left_run, left_embedding, left_color, left_range,
                      pn.pane.Bokeh(left_plot)),
            pn.Column(right_run, right_embedding, right_color, right_range,
                      pn.pane.Bokeh(right_plot)),
            sidebar,
        ),
        pn.layout.Divider(),
        pn.Row(*detail_columns),
    )


_ARGS = _parse_args()
for _path in [_ARGS.h5, *_ARGS.umap]:
    if not _path.is_file():
        raise FileNotFoundError(_path)
_STAMP_DIR = _stamp_directory(_ARGS.stamps, _ARGS.band)
if _STAMP_DIR is not None and not _STAMP_DIR.is_dir():
    raise FileNotFoundError(_STAMP_DIR)
_DATA = load_data(_ARGS.h5, _ARGS.umap, _ARGS.label or None)
app = build_app(
    _ARGS.h5, _STAMP_DIR, _DATA, _ARGS.band, _ARGS.selection_output,
)
app.servable()


if __name__ == '__main__':
    app.show(port=_ARGS.port, open=True)
