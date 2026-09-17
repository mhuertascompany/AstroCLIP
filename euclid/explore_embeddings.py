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

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import panel as pn
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
    'sfh_recent_10': 'SFH fraction: recent 10%',
    'sfh_recent_20': 'SFH fraction: recent 20%',
    'sfh_old_20': 'SFH fraction: oldest 20%',
    'sfh_mean_lookback': 'Mean fractional lookback time',
    'sfh_peak_lookback': 'Peak fractional lookback time',
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
                reconstruction=None):
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
    ax.set_yscale('log')
    ax.set_xlim(float(time.min()), float(time.max()))
    ax.set_xlabel('Fractional lookback time', fontsize=5)
    ax.set_ylabel('Normalized SFH weight', fontsize=5)
    ax.tick_params(labelsize=5)


def _fig_to_html(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=105, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return (
        f'<img src="data:image/png;base64,{encoded}" '
        'style="width:100%;max-width:650px"/>'
    )


def _gallery(h5_path, stamp_dir, selected, data, rng, band):
    n = min(N_DISPLAY, len(selected))
    shown = np.sort(rng.choice(np.asarray(selected, dtype=int), n, replace=False))
    rows = data['h5_rows'][shown]
    ids = data['galaxy_ids'][shown]
    redshift = data['redshift'][shown]
    with h5py.File(h5_path, 'r') as source:
        median = _read_rows(source['sfh'], rows)
        p16 = _read_rows(source['sfh_p16'], rows) if 'sfh_p16' in source else None
        p84 = _read_rows(source['sfh_p84'], rows) if 'sfh_p84' in source else None
        reconstruction = (
            _read_rows(source['sfh_reconstruction'], rows)
            if 'sfh_reconstruction' in source else None
        )
        time = np.asarray(source['sfh_time_grid'][:], dtype=float)
        epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))

    nrows = max(1, (n + NCOLS - 1) // NCOLS)
    if stamp_dir is not None:
        fig_images, image_axes = plt.subplots(
            nrows, NCOLS, figsize=(NCOLS * 1.7, nrows * 1.7), squeeze=False,
        )
    else:
        fig_images, image_axes = None, None
    fig_sfhs, sfh_axes = plt.subplots(
        nrows, NCOLS, figsize=(NCOLS * 1.9, nrows * 1.7), squeeze=False,
    )
    axes_groups = [sfh_axes.flat]
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
            image_axis.set_title(f'{int(galaxy_id)}  z={z:.2f}', fontsize=5, pad=1)

        sfh_axis = sfh_axes.flat[index]
        sfh_axis.set_visible(True)
        _render_sfh(
            sfh_axis, time, median[index],
            p16[index] if p16 is not None else None,
            p84[index] if p84 is not None else None,
            epsilon,
            reconstruction[index] if reconstruction is not None else None,
        )
        sfh_axis.set_title(f'{int(galaxy_id)}  z={z:.2f}', fontsize=5, pad=1)

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
    axis.set_yscale('log')
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
    view = CDSView(filter=BooleanFilter(booleans=[True] * n_objects))

    def update_view():
        view.filter = BooleanFilter(booleans=(z_mask & property_mask).tolist())

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
            selection_color='white', selection_alpha=1,
            nonselection_alpha=0.1,
        )
        dynamic = plot.scatter(
            x_field, y_field, source=source, view=view,
            fill_color='dynamic_color', size=3, alpha=0.8, line_width=0,
            selection_fill_color='white', selection_alpha=1,
            nonselection_alpha=0.12, visible=False,
        )
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
        value=(z_low, z_high), step=0.1, width=220,
    )

    filter_property = pn.widgets.Select(
        name='Filter property (left run)', options=all_properties,
        value=left_color_default, width=220,
    )
    filter_range = make_range(
        'Filter range', property_values(left_run_default, left_color_default),
    )
    filter_range.width = 220
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

    def refresh(selected):
        selected = list(selected)
        if not selected:
            info.object = '_No galaxies selected._'
            image_pane.object = sfh_pane.object = population_pane.object = _BLANK_HTML
            return
        info.object = (
            f'**{len(selected):,}** selected — showing '
            f'{min(N_DISPLAY, len(selected))} random examples'
        )
        image_fig, sfh_fig = _gallery(
            h5_path, stamp_dir, selected, data, rng, band,
        )
        if image_fig is not None:
            image_pane.object = _fig_to_html(image_fig)
        sfh_pane.object = _fig_to_html(sfh_fig)
        rows = data['h5_rows'][np.asarray(selected, dtype=int)]
        population_pane.object = _fig_to_html(_population_sfh(h5_path, rows))

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
        property_mask = (values >= low) & (values <= high)
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
        visible = np.flatnonzero(z_mask & property_mask)
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
        nonlocal property_mask, z_mask
        property_mask = np.ones(n_objects, dtype=bool)
        z_mask = np.ones(n_objects, dtype=bool)
        redshift_slider.value = (z_low, z_high)
        update_filter_bounds()
        update_view()
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
            ])
            for index in selected:
                writer.writerow([
                    int(data['galaxy_ids'][index]), float(data['redshift'][index]),
                    int(data['source_h5_rows'][index]),
                    int(data['h5_rows'][index]),
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
    filter_property.param.watch(update_filter_bounds, 'value')
    filter_range.param.watch(apply_property_filter, 'value')
    redshift_slider.param.watch(apply_redshift_filter, 'value')
    cluster_button.on_click(cluster_visible)
    reset_button.on_click(reset)
    sample_button.on_click(lambda event: refresh(source.selected.indices))
    save_button.on_click(save_selection)

    previous_selection = [()]

    def poll_selection():
        current = tuple(sorted(source.selected.indices))
        if current != previous_selection[0]:
            previous_selection[0] = current
            refresh(current)

    pn.state.add_periodic_callback(poll_selection, period=350)
    update_left_coordinates()
    update_right_coordinates()

    sidebar = pn.Column(
        pn.pane.Markdown('## Euclid embedding explorer'),
        pn.layout.Divider(),
        redshift_slider,
        pn.layout.Divider(),
        pn.pane.Markdown('**Filter and cluster in the left panel**'),
        filter_property,
        filter_range,
        pn.Row(cluster_count, cluster_button),
        reset_button,
        cluster_info,
        pn.layout.Divider(),
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
