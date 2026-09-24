"""Refresh CLIP UMAP properties from an updated HDF5 without refitting UMAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .umap_zoobot_clip import (
    comparison_page,
    load_properties,
    property_pages,
    property_specs,
    save_property_table,
)


MORPHOLOGY_KEYS = [
    'vis_magnitude', 'concentration', 'asymmetry', 'smoothness', 'gini',
    'moment_20', 't_type', 'etg_or_ltg',
    'zoobot_smooth_conditional_fraction',
    'zoobot_featured_conditional_fraction', 'zoobot_smooth_probability',
    'zoobot_featured_probability', 'zoobot_edge_on_probability',
    'zoobot_spiral_probability', 'zoobot_bar_probability',
    'zoobot_merger_probability', 'major_merger_probability', 'redshift',
    'sersic_index', 'sersic_radius', 'axis_ratio', 'fwhm', 'kron_radius',
    'semimajor_axis', 'point_like_probability', 'ellipticity',
    'segmentation_area',
]
PHYSICAL_KEYS = [
    'sfh_recent_birthrate', 'sfh_recent_trend',
    'sfh_log_recent_sfr_per_formed_mass', 'redshift', 'log_stellar_mass',
    'sfh_recent_10', 'sfh_recent_20', 'sfh_old_20', 'sfh_mean_lookback',
    'sfh_peak_lookback', 'sfh_duration_80', 'sfh_t50_lookback',
    'sfh_entropy', 'sfh_log_old_recent',
]


def refresh(source_archive, dataset, output_archive, output_pdf,
            per_object=None, run_label=''):
    source_archive, dataset = Path(source_archive), Path(dataset)
    output_archive, output_pdf = Path(output_archive), Path(output_pdf)
    for path in (source_archive, dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    if per_object is not None and not Path(per_object).is_file():
        raise FileNotFoundError(per_object)
    for path in (output_archive, output_pdf):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')

    with np.load(source_archive) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    required = {
        'galaxy_id', 'h5_row', 'image_embedding', 'sfh_embedding',
        'xy_image', 'xy_sfh', 'xy_joint',
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f'Missing arrays in {source_archive}: {missing}')

    ids = np.asarray(arrays['galaxy_id'], dtype=np.int64)
    rows = np.asarray(arrays['h5_row'], dtype=np.int64)
    image = np.asarray(arrays['image_embedding'], dtype=np.float32)
    sfh = np.asarray(arrays['sfh_embedding'], dtype=np.float32)
    archive_redshift = np.asarray(
        arrays.get('redshift', np.full(len(ids), np.nan)), dtype=np.float32,
    )
    properties, sources = load_properties(
        dataset, rows, ids, archive_redshift, image, sfh, per_object,
    )
    arrays.update({
        key: np.asarray(value, dtype=np.float32)
        for key, value in properties.items()
    })

    output_archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_archive, **arrays)
    save_property_table(output_pdf.with_suffix('.csv'), ids, rows, properties)

    specs = property_specs(properties)
    morphology = [specs[key] for key in MORPHOLOGY_KEYS if key in specs]
    physical = [specs[key] for key in PHYSICAL_KEYS if key in specs]
    comparison = [
        specs[key] for key in (
            'redshift', 'log_stellar_mass', 'sersic_index', 'sersic_radius',
            'zoobot_smooth_conditional_fraction',
            'zoobot_spiral_probability', 'zoobot_merger_probability',
            'sfh_recent_20', 'sfh_mean_lookback', 'paired_cosine',
        ) if key in specs
    ]
    suffix = f' - {run_label}' if run_label else ''
    with PdfPages(output_pdf) as pdf:
        for key, title in (
            ('xy_image', 'Euclid VIS image embedding: morphology'),
            ('xy_sfh', 'Euclid SFH embedding: morphology transferred from images'),
            ('xy_joint', 'Euclid averaged joint embedding: morphology'),
        ):
            property_pages(pdf, arrays[key], morphology, title + suffix)
        property_pages(
            pdf, arrays['xy_sfh'], physical,
            'Euclid SFH embedding: physical and SFH properties' + suffix,
        )
        comparison_page(
            pdf,
            [('Image', arrays['xy_image']), ('SFH', arrays['xy_sfh']),
             ('Joint average', arrays['xy_joint'])],
            comparison, run_label=run_label,
        )

    manifest = {
        'source_archive': str(source_archive),
        'dataset': str(dataset),
        'per_object': str(per_object) if per_object is not None else None,
        'output_archive': str(output_archive),
        'output_pdf': str(output_pdf),
        'n_objects': int(len(ids)),
        'umap_recomputed': False,
        'property_sources': sources,
        'properties': sorted(properties),
    }
    output_pdf.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-archive', type=Path, required=True)
    parser.add_argument('--output-pdf', type=Path, required=True)
    parser.add_argument('--per-object', type=Path)
    parser.add_argument('--run-label', default='')
    args = parser.parse_args()
    manifest = refresh(
        args.archive, args.dataset, args.output_archive, args.output_pdf,
        args.per_object, args.run_label,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
