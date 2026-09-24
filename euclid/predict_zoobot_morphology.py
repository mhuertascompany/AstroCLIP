"""Run the full pretrained Euclid ZooBot decision tree on VIS stamps.

The encoder-only Hugging Face model cannot produce Galaxy Zoo answers.  This
module downloads and loads the matching ``FinetuneableZoobotTree`` checkpoint,
runs the complete frozen encoder plus decision-tree head, and writes its
Dirichlet concentrations to an OBJECT_ID-aligned FITS catalog.  Column names
are normalized to the convention used by the Euclid MER morphology catalog so
the result can be consumed by ``euclid.restore_morphology_metadata``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import DataLoader

from .export_zoobot_image_embeddings import StampDataset, select_stamp_rows


log = logging.getLogger(__name__)

DEFAULT_REPO = 'mwalmsley/zoobot-finetuned-euclid'
DEFAULT_FILENAME = 'FinetuneableZoobotTree.ckpt'
REQUIRED_EXPLORER_COLUMNS = {
    'smooth_or_featured_smooth',
    'smooth_or_featured_featured_or_disk',
    'smooth_or_featured_artifact_star_zoom',
    'has_spiral_arms_yes',
    'has_spiral_arms_no',
    'merging_none',
    'merging_minor_disturbance',
    'merging_major_disturbance',
    'merging_merger',
}


def select_stamps_without_h5(stamp_root, band='VIS', max_objects=0, seed=42):
    """Select directly from JPEG filenames when no full SFH HDF5 is present."""
    stamp_root = Path(stamp_root)
    stamp_dir = stamp_root / band if (stamp_root / band).is_dir() else stamp_root
    if not stamp_dir.is_dir():
        raise FileNotFoundError(f'Stamp directory not found: {stamp_dir}')
    prefix = f'{band}_'
    ids = []
    for path in stamp_dir.glob(f'{prefix}*.jpg'):
        try:
            ids.append(int(path.stem[len(prefix):]))
        except ValueError:
            log.warning('Ignoring stamp with an unexpected name: %s', path)
    galaxy_ids = np.asarray(sorted(ids), dtype=np.int64)
    if not len(galaxy_ids):
        raise ValueError(f'No {band}_<object_id>.jpg stamps found in {stamp_dir}.')
    if len(np.unique(galaxy_ids)) != len(galaxy_ids):
        raise ValueError('Stamp directory contains duplicate object IDs.')
    if max_objects > 0 and max_objects < len(galaxy_ids):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(galaxy_ids), max_objects, replace=False))
        galaxy_ids = galaxy_ids[selected]
    rows = np.full(len(galaxy_ids), -1, dtype=np.int64)
    return stamp_dir, rows, galaxy_ids, len(ids), None


def normalize_answer_name(name):
    """Convert a ZooBot schema label to the MER/HDF5 naming convention."""
    normalized = re.sub(r'[^0-9a-zA-Z]+', '_', str(name)).strip('_').lower()
    # The released checkpoint namespaces each decision-tree question with
    # ``-euclid`` (e.g. smooth-or-featured-euclid_smooth), whereas the MER
    # catalog omits that survey tag.
    normalized = normalized.replace('_euclid_', '_')
    # Older internal Euclid schemas called the combined rejection answer
    # ``problem``; the released MER catalog calls it ``artifact_star_zoom``.
    if normalized == 'smooth_or_featured_problem':
        return 'smooth_or_featured_artifact_star_zoom'
    return normalized


def normalized_schema_columns(schema):
    raw = list(schema.label_cols)
    columns = [normalize_answer_name(value) for value in raw]
    if len(columns) != len(set(columns)):
        duplicates = sorted({value for value in columns if columns.count(value) > 1})
        raise ValueError(f'ZooBot schema labels collide after normalization: {duplicates}')
    return raw, columns


def compatible_checkpoint_hparams(hparams, tree_class, abstract_class):
    """Drop Lightning hyperparameters removed by the installed ZooBot version."""
    valid = set(inspect.signature(tree_class.__init__).parameters)
    valid.update(inspect.signature(abstract_class.__init__).parameters)
    valid.difference_update({'self', 'args', 'kwargs', 'super_kwargs'})
    clean = {key: value for key, value in dict(hparams).items() if key in valid}
    ignored = sorted(set(hparams).difference(clean))
    if 'schema' not in clean:
        raise ValueError('ZooBot checkpoint hyperparameters contain no schema.')
    return clean, ignored


def load_compatible_checkpoint(checkpoint, tree_class, abstract_class):
    """Reconstruct an older ZooBot checkpoint with the current public API."""
    try:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument
        saved = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' not in saved or 'hyper_parameters' not in saved:
        raise ValueError('Not a complete ZooBot Lightning checkpoint.')
    hparams, ignored = compatible_checkpoint_hparams(
        saved['hyper_parameters'], tree_class, abstract_class,
    )
    model = tree_class(**hparams)
    model.load_state_dict(saved['state_dict'], strict=True)
    return model, ignored


def load_full_model(repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME,
                    checkpoint=None):
    """Load a frozen FinetuneableZoobotTree from disk or Hugging Face."""
    try:
        from zoobot.pytorch.training.finetune import (
            FinetuneableZoobotAbstract,
            FinetuneableZoobotTree,
        )
    except ImportError as error:
        raise ImportError(
            'The full classifier requires Zoobot. Install it in the active '
            'environment with: python -m pip install "zoobot[pytorch]"'
        ) from error

    if checkpoint is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise ImportError('huggingface_hub is required to download ZooBot.') from error
        checkpoint = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    try:
        model = FinetuneableZoobotTree.load_from_checkpoint(
            str(checkpoint), map_location='cpu',
        )
    except TypeError as error:
        if 'unexpected keyword argument' not in str(error):
            raise
        log.warning(
            'Checkpoint uses obsolete ZooBot hyperparameters (%s); '
            'reconstructing with the installed API.', error,
        )
        model, ignored = load_compatible_checkpoint(
            checkpoint, FinetuneableZoobotTree, FinetuneableZoobotAbstract,
        )
        log.warning('Ignored obsolete checkpoint hyperparameters: %s', ignored)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if not hasattr(model, 'schema') or not hasattr(model.schema, 'label_cols'):
        raise TypeError('Loaded checkpoint has no ZooBot decision-tree schema.')
    return model, checkpoint


def predict_batches(model, loader, device):
    predictions, galaxy_ids, rows = [], [], []
    model = model.to(device)
    with torch.inference_mode():
        for batch_index, (images, batch_ids, batch_rows) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            # Keep the Dirichlet head in float32. Its positive activation can
            # underflow in float16 for low-concentration answers.
            output = model(images)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim != 2:
                raise ValueError(f'Expected 2D ZooBot output; received {output.shape}.')
            predictions.append(output.float().cpu().numpy())
            galaxy_ids.append(np.asarray(batch_ids, dtype=np.int64))
            rows.append(np.asarray(batch_rows, dtype=np.int64))
            if batch_index == 0 or (batch_index + 1) % 50 == 0:
                log.info('Classified %s stamps', f'{sum(len(x) for x in galaxy_ids):,}')
    return (
        np.concatenate(predictions).astype(np.float32),
        np.concatenate(galaxy_ids),
        np.concatenate(rows),
    )


def model_input_channels(model):
    """Read the expected channel count from the first convolution."""
    encoder = getattr(model, 'encoder', model)
    for module in encoder.modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.in_channels not in (1, 3):
                raise ValueError(
                    f'Unexpected ZooBot input channel count: {module.in_channels}'
                )
            return int(module.in_channels)
    raise TypeError('Could not find an input Conv2d in the ZooBot encoder.')


def prediction_table(galaxy_ids, rows, predictions, schema):
    raw_columns, columns = normalized_schema_columns(schema)
    missing_columns = sorted(REQUIRED_EXPLORER_COLUMNS.difference(columns))
    if missing_columns:
        raise ValueError(
            'ZooBot schema cannot supply the explorer morphology fields; '
            f'missing normalized columns: {missing_columns}'
        )
    if predictions.shape != (len(galaxy_ids), len(columns)):
        raise ValueError(
            f'Predictions have shape {predictions.shape}, but schema has '
            f'{len(columns)} answers.'
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError('ZooBot predictions contain nonfinite values.')
    if np.any(predictions <= 0):
        raise ValueError('ZooBot Dirichlet concentrations must be positive.')

    table = Table()
    table['object_id'] = np.asarray(galaxy_ids, dtype=np.int64)
    table['h5_row'] = np.asarray(rows, dtype=np.int64)
    for index, column in enumerate(columns):
        table[column] = predictions[:, index]
        table[column].description = (
            f'ZooBot Dirichlet concentration; schema label={raw_columns[index]}'
        )
    return table, raw_columns, columns


def run(dataset, stamp_root, output, band='VIS', image_size=224,
        batch_size=256, num_workers=8, max_objects=0, seed=42,
        repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME, checkpoint=None,
        device_name='auto', overwrite=False):
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite {output}')
    if max_objects < 0:
        raise ValueError('max_objects must be nonnegative; zero means all stamps.')
    if dataset is None:
        stamp_dir, rows, galaxy_ids, n_paired, n_h5 = select_stamps_without_h5(
            stamp_root, band, max_objects, seed,
        )
    else:
        stamp_dir, rows, galaxy_ids, n_paired, n_h5 = select_stamp_rows(
            dataset, stamp_root, band, max_objects, seed,
        )
    device = torch.device(
        'cuda' if device_name == 'auto' and torch.cuda.is_available()
        else 'cpu' if device_name == 'auto' else device_name
    )
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')

    model, checkpoint_path = load_full_model(repo_id, filename, checkpoint)
    num_channels = model_input_channels(model)
    log.info('ZooBot model expects %d-channel input', num_channels)
    dataset_object = StampDataset(
        stamp_dir, band, galaxy_ids, rows, image_size=image_size,
        num_channels=num_channels,
    )
    loader = DataLoader(
        dataset_object, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == 'cuda',
        persistent_workers=num_workers > 0,
    )
    predictions, output_ids, output_rows = predict_batches(model, loader, device)
    table, raw_columns, columns = prediction_table(
        output_ids, output_rows, predictions, model.schema,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    table.write(temporary, format='fits', overwrite=True)
    temporary.replace(output)

    manifest = {
        'dataset': str(Path(dataset).resolve()) if dataset is not None else None,
        'stamp_directory': str(stamp_dir.resolve()),
        'model_repository': repo_id,
        'checkpoint_filename': filename,
        'checkpoint_path': str(checkpoint_path),
        'device': str(device),
        'input_channels': num_channels,
        'n_h5_objects': n_h5,
        'n_stamps_available': n_paired,
        'n_classified': len(table),
        'schema_labels': raw_columns,
        'output_columns': columns,
        'output_catalog': str(output.resolve()),
    }
    manifest_path = output.with_suffix('.json')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path,
                        help='Optional HDF5 used only to record matching row numbers.')
    parser.add_argument('--stamp-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--band', default='VIS')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-objects', type=int, default=0,
                        help='Zero classifies every available stamp.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repo-id', default=DEFAULT_REPO)
    parser.add_argument('--filename', default=DEFAULT_FILENAME)
    parser.add_argument('--checkpoint', type=Path,
                        help='Use a local checkpoint instead of downloading it.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )
    run(
        args.dataset, args.stamp_root, args.output, args.band,
        args.image_size, args.batch_size, args.num_workers, args.max_objects,
        args.seed, args.repo_id, args.filename, args.checkpoint,
        args.device, args.overwrite,
    )


if __name__ == '__main__':
    main()
