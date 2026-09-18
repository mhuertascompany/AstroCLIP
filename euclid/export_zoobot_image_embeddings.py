"""Export pretrained Euclid ZooBot image features and morphology diagnostics.

This is an image-only diagnostic.  It deliberately exports the frozen timm
backbone output before the randomly initialized AstroCLIP projection head, so
the resulting UMAP tests what the pretrained Euclid ZooBot encoder already
knows about morphology.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

from .dataset_zoobot import _inference_transform
from .umap_zoobot_clip import (
    fit_umap,
    load_catalog_properties,
    property_pages,
    property_specs,
    save_property_table,
)


log = logging.getLogger(__name__)


MORPHOLOGY_KEYS = [
    'concentration', 'asymmetry', 'smoothness', 'gini', 'moment_20',
    't_type', 'etg_or_ltg', 'major_merger_probability',
    'zoobot_smooth_probability', 'zoobot_featured_probability',
    'zoobot_smooth_conditional_fraction', 'zoobot_featured_conditional_fraction',
    'zoobot_edge_on_probability', 'zoobot_spiral_probability',
    'zoobot_bar_probability', 'zoobot_merger_probability',
    'sersic_index', 'sersic_radius', 'axis_ratio', 'ellipticity',
]
CONFOUNDER_KEYS = [
    'vis_magnitude', 'redshift', 'log_stellar_mass', 'fwhm',
    'kron_radius', 'semimajor_axis', 'segmentation_area',
    'point_like_probability',
]
SFH_KEYS = [
    'sfh_recent_10', 'sfh_recent_20', 'sfh_old_20',
    'sfh_mean_lookback', 'sfh_peak_lookback', 'sfh_t50_lookback',
    'sfh_entropy', 'sfh_log_old_recent',
]
ZOOBOT_KEYS = [key for key in MORPHOLOGY_KEYS if key.startswith('zoobot_')]


class StampDataset(Dataset):
    def __init__(self, stamp_dir, band, galaxy_ids, rows, image_size=224):
        self.stamp_dir = Path(stamp_dir)
        self.band = band
        self.galaxy_ids = np.asarray(galaxy_ids, dtype=np.int64)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.transform = _inference_transform(image_size)

    def __len__(self):
        return len(self.galaxy_ids)

    def __getitem__(self, index):
        galaxy_id = int(self.galaxy_ids[index])
        path = self.stamp_dir / f'{self.band}_{galaxy_id}.jpg'
        with Image.open(path) as image:
            tensor = self.transform(image.convert('L'))
        return tensor, galaxy_id, int(self.rows[index])


def select_stamp_rows(dataset_path, stamp_root, band='VIS', max_objects=30000,
                      seed=42):
    """Match HDF5 rows to existing JPEGs and choose a deterministic subset."""
    dataset_path = Path(dataset_path)
    stamp_root = Path(stamp_root)
    stamp_dir = stamp_root / band if (stamp_root / band).is_dir() else stamp_root
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not stamp_dir.is_dir():
        raise FileNotFoundError(f'Stamp directory not found: {stamp_dir}')

    prefix = f'{band}_'
    available_ids = set()
    for path in stamp_dir.glob(f'{prefix}*.jpg'):
        try:
            available_ids.add(int(path.stem[len(prefix):]))
        except ValueError:
            log.warning('Ignoring stamp with an unexpected name: %s', path)
    with h5py.File(dataset_path, 'r') as source:
        galaxy_ids = np.asarray(source['galaxy_id'][:], dtype=np.int64)
    mask = np.fromiter(
        (int(galaxy_id) in available_ids for galaxy_id in galaxy_ids),
        dtype=bool, count=len(galaxy_ids),
    )
    rows = np.flatnonzero(mask).astype(np.int64)
    paired_ids = galaxy_ids[rows]
    if not len(rows):
        raise ValueError('No HDF5 galaxy IDs have matching JPEG stamps.')
    if max_objects > 0 and max_objects < len(rows):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(rows), max_objects, replace=False))
        rows = rows[selected]
        paired_ids = paired_ids[selected]
    return stamp_dir, rows, paired_ids, int(mask.sum()), len(galaxy_ids)


def load_backbone(model_name, device):
    try:
        import timm
    except ImportError as error:
        raise ImportError('timm is required to load the ZooBot encoder.') from error
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


def encode_stamps(model, loader, device):
    embeddings = []
    galaxy_ids = []
    rows = []
    use_amp = device.type == 'cuda'
    with torch.inference_mode():
        for batch_index, (images, batch_ids, batch_rows) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp,
            ):
                features = model(images)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                features = features.flatten(1)
            features = F.normalize(features.float(), dim=1)
            embeddings.append(features.cpu().numpy())
            galaxy_ids.append(np.asarray(batch_ids, dtype=np.int64))
            rows.append(np.asarray(batch_rows, dtype=np.int64))
            if batch_index == 0 or (batch_index + 1) % 50 == 0:
                log.info('Encoded %,d images', sum(len(value) for value in galaxy_ids))
    return (
        np.concatenate(embeddings).astype(np.float32),
        np.concatenate(galaxy_ids),
        np.concatenate(rows),
    )


def write_diagnostic_products(dataset_path, output_dir, galaxy_ids, rows,
                              embedding, coordinates, require_zoobot=False):
    """Attach current HDF5 metadata and write image-only NPZ/PDF/CSV products."""
    output_dir = Path(output_dir)
    properties, sources = load_catalog_properties(
        dataset_path, rows, galaxy_ids,
    )
    specs = property_specs(properties)
    available_zoobot = [
        key for key in ZOOBOT_KEYS
        if key in specs and np.any(np.isfinite(properties[key]))
    ]
    if not available_zoobot:
        message = (
            'No MER ZooBot question columns are present in %s. The PDF will '
            'contain structural measurements only. Add the detailed MER '
            'catalog with euclid.restore_morphology_metadata, then run '
            'euclid.refresh_zoobot_image_diagnostics.'
        )
        if require_zoobot:
            raise ValueError(message % dataset_path)
        log.warning(message, dataset_path)
    else:
        log.info('Loaded MER ZooBot properties: %s', ', '.join(available_zoobot))

    pdf_path = output_dir / 'zoobot_image_umap.pdf'
    with PdfPages(pdf_path) as pdf:
        property_pages(
            pdf, coordinates,
            [specs[key] for key in MORPHOLOGY_KEYS if key in specs],
            'Pretrained Euclid ZooBot backbone: morphology',
        )
        property_pages(
            pdf, coordinates,
            [specs[key] for key in CONFOUNDER_KEYS if key in specs],
            'Pretrained Euclid ZooBot backbone: sample and image properties',
        )
        property_pages(
            pdf, coordinates,
            [specs[key] for key in SFH_KEYS if key in specs],
            'Pretrained Euclid ZooBot backbone: SFH properties',
        )

    npz_path = output_dir / 'zoobot_image_umap.npz'
    npz_data = {
        'galaxy_id': galaxy_ids,
        'h5_row': rows,
        'image_embedding': embedding,
        'xy_image': coordinates,
    }
    npz_data.update({
        key: np.asarray(values, dtype=np.float32)
        for key, values in properties.items()
    })
    np.savez_compressed(npz_path, **npz_data)
    save_property_table(
        output_dir / 'zoobot_image_properties.csv',
        galaxy_ids, rows, properties,
    )
    return {
        'property_sources': sources,
        'properties': sorted(properties),
        'zoobot_properties': available_zoobot,
        'archive': npz_path.name,
        'diagnostic_pdf': pdf_path.name,
    }


def export_embeddings(dataset_path, stamp_root, output_dir, model_name,
                      band='VIS', image_size=224, batch_size=256,
                      num_workers=8, max_objects=30000, seed=42,
                      n_neighbors=15, min_dist=0.1, device_name='auto'):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_objects < 0:
        raise ValueError('max_objects must be nonnegative; zero means all stamps.')

    stamp_dir, rows, galaxy_ids, n_paired, n_h5 = select_stamp_rows(
        dataset_path, stamp_root, band, max_objects, seed,
    )
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    log.info(
        'Selected %,d of %,d stamp-paired objects (%,d HDF5 rows); device=%s',
        len(rows), n_paired, n_h5, device,
    )

    stamps = StampDataset(stamp_dir, band, galaxy_ids, rows, image_size)
    loader = DataLoader(
        stamps, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == 'cuda', persistent_workers=num_workers > 0,
    )
    model = load_backbone(model_name, device)
    embedding, encoded_ids, encoded_rows = encode_stamps(model, loader, device)
    if not np.array_equal(encoded_ids, galaxy_ids) or not np.array_equal(
        encoded_rows, rows,
    ):
        raise RuntimeError('DataLoader changed the requested object order.')

    log.info('Fitting image-only UMAP for %,d raw ZooBot features', len(rows))
    coordinates = fit_umap(embedding, n_neighbors, min_dist, seed)
    diagnostic_manifest = write_diagnostic_products(
        dataset_path, output_dir, galaxy_ids, rows, embedding, coordinates,
    )
    manifest = {
        'model_name': model_name,
        'features': 'raw pretrained backbone output, flattened and L2-normalized',
        'dataset': str(Path(dataset_path)),
        'stamp_directory': str(stamp_dir),
        'band': band,
        'n_hdf5_objects': n_h5,
        'n_stamp_paired': n_paired,
        'n_exported': int(len(rows)),
        'embedding_dimension': int(embedding.shape[1]),
        'image_size': image_size,
        'seed': seed,
        'umap_n_neighbors': n_neighbors,
        'umap_min_dist': min_dist,
        **diagnostic_manifest,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    log.info('Saved image embedding archive: %s', output_dir / manifest['archive'])
    log.info('Saved diagnostic PDF: %s', output_dir / manifest['diagnostic_pdf'])
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--stamp-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--model-name', default='hf_hub:mwalmsley/zoobot-encoder-euclid',
    )
    parser.add_argument('--band', default='VIS')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-objects', type=int, default=30000,
                        help='Deterministic subset size; zero exports all pairs.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--device', default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )
    manifest = export_embeddings(
        args.dataset, args.stamp_root, args.output_dir, args.model_name,
        args.band, args.image_size, args.batch_size, args.num_workers,
        args.max_objects, args.seed, args.n_neighbors, args.min_dist,
        args.device,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
