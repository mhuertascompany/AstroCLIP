"""
Apply the trained Zoobot models (regular, family, binary) to the master catalog
stamps and save all probabilities into a FITS table.

Example usage:

```
python -m moprhology.zoobot.apply_models_to_master \
  --catalog /n07data/ilbert/COSMOS-Web/photoz_MASTER_v3.1.0/MORPHO/COSMOSWeb_mastercatalog_v1.fits \
  --stamp-root /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
  --ckpt-regular /n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/regular_6.ckpt \
  --ckpt-family /n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/family_2.ckpt \
  --ckpt-binary /n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/binary_3-v1.ckpt \
  --output ilbert_visual_zoobot_morphology.fits \
  --accelerator gpu --devices 1
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moprhology.zoobot.train_on_cosmos_visual import select_rest_frame_filter
from zoobot.pytorch.training import finetune


LABEL_SETS: Dict[str, List[str]] = {
    "regular": [
        'ELL_REGULAR', 'ELL_INTER', 'ELL_DISTURB',
        'S0_REGULAR', 'S0_INTER', 'S0_DISTURB',
        'EDISK_REGULAR', 'EDISK_INTER', 'EDISK_DISTURB',
        'LDISK_REGULAR', 'LDISK_INTER', 'LDISK_DISTURB'
    ],
    "family": ['ELLIPTICAL', 'S0', 'EARLY_DISK', 'LATE_DISK'],
    "binary": ['NOT_DISTURBED', 'DISTURBED'],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Zoobot checkpoints to master catalog stamps.")
    parser.add_argument("--catalog", type=Path, required=True, help="Master FITS catalog.")
    parser.add_argument("--id-column", default="id", help="Column with unique IDs in the catalog.")
    parser.add_argument("--redshift-column", default="zfinal", help="Column with redshift values.")
    parser.add_argument("--stamp-root", type=Path, required=True,
                        help="Root directory containing per-filter stamp folders.")
    parser.add_argument("--filters", nargs="+", default=["F150W", "F277W", "F444W"],
                        help="Filters available in stamp-root.")
    parser.add_argument("--filename-template", default="{filter}_{id}.jpg",
                        help="How stamps are named inside each filter folder.")
    parser.add_argument("--mag-column", default="mag_model_f277w",
                        help="Optional column for magnitude filtering (case-sensitive).")
    parser.add_argument("--mag-limit", type=float, default=None,
                        help="Keep objects with mag_column <= mag_limit (set None to skip).")
    parser.add_argument("--ckpt-regular", type=Path, required=True, help="Checkpoint for the regular model.")
    parser.add_argument("--ckpt-family", type=Path, required=True, help="Checkpoint for the family model.")
    parser.add_argument("--ckpt-binary", type=Path, required=True, help="Checkpoint for the binary model.")
    parser.add_argument("--output", type=Path, required=True, help="Output FITS file path.")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--image-size", type=int, default=224, help="Resize dimension (square).")
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator (e.g. gpu, cpu).")
    parser.add_argument("--devices", default="auto", help="Devices argument for Lightning.")
    parser.add_argument("--ignore-missing", action="store_true",
                        help="Skip IDs whose stamp file is missing instead of failing.")
    return parser.parse_args()


def load_master_subset(catalog_path: Path, id_col: str, z_col: str, extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
    logging.info("Reading master catalog from %s", catalog_path)

    def to_native(array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.dtype.kind in {"S", "U"}:
            return arr.astype(str)
        if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and np.little_endian is False):
            dtype = arr.dtype.newbyteorder("<")
            arr = arr.byteswap().view(dtype)
        return arr

    def fetch_column(hdus, column: str) -> np.ndarray:
        for hdu in hdus[1:]:
            data = getattr(hdu, "data", None)
            if data is None or column not in getattr(data, "names", []):
                continue
            table = Table(data)
            if column in table.colnames:
                return to_native(table[column])
        raise ValueError(f"Column {column} not found in FITS file {catalog_path}")

    with fits.open(catalog_path, memmap=True) as hdus:
        id_values = fetch_column(hdus, id_col)
        data = {id_col: id_values}
        length = len(id_values)
        data[z_col] = fetch_column(hdus, z_col)
        if extra_cols:
            for col in extra_cols:
                try:
                    data[col] = fetch_column(hdus, col)
                except ValueError:
                    logging.warning("Column %s not found; filling NaNs.", col)
                    data[col] = np.full(length, np.nan, dtype=float)

    return pd.DataFrame(data)


def sanitize_id(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return int(value)
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            numeric = float(stripped)
            if numeric.is_integer():
                return int(numeric)
            return int(numeric)
        except ValueError:
            return stripped
    return value


def build_inference_catalog(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stamp_root = args.stamp_root.expanduser().resolve()
    rows = []
    missing = 0
    for row in df.itertuples(index=False):
        obj_id = getattr(row, args.id_column)
        z = getattr(row, args.redshift_column)
        if pd.isna(z):
            continue
        filter_name = select_rest_frame_filter(float(z))
        if filter_name is None:
            continue
        clean_id = sanitize_id(obj_id)
        file_path = stamp_root / filter_name / args.filename_template.format(filter=filter_name, id=clean_id)
        if not file_path.exists():
            missing += 1
            continue
        rows.append({
            'id': str(clean_id),
            'file_loc': str(file_path),
            'filter_used': filter_name,
            args.redshift_column: z
        })
    logging.info("Prepared %d entries for inference (skipped %d missing files).", len(rows), missing)
    return pd.DataFrame(rows)


class StampDataset(Dataset):
    def __init__(self, catalog: pd.DataFrame, transform):
        self.catalog = catalog.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.catalog)

    def __getitem__(self, idx: int):
        row = self.catalog.iloc[idx]
        image = Image.open(row['file_loc']).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, row['id']


def get_inference_transform(image_size: int):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])


def choose_device(accelerator: str, devices: str) -> torch.device:
    accel = accelerator.lower()
    if accel in {'gpu', 'cuda'} and torch.cuda.is_available():
        return torch.device('cuda')
    if accel == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def determine_label_names(model: finetune.FinetuneableZoobotClassifier, fallback: List[str]) -> List[str]:
    for attr in ('label_names', 'class_names'):
        names = getattr(model, attr, None)
        if names:
            return list(names)
    num_classes = getattr(model, 'num_classes', None)
    if num_classes is not None and num_classes != len(fallback):
        logging.warning(
            "Checkpoint reports %d classes but fallback expects %d. "
            "Creating generic class names.", num_classes, len(fallback)
        )
        return [f"class_{i}" for i in range(num_classes)]
    return fallback


def canonical_regular_label(label_name: str, index: int) -> str:
    upper = label_name.upper()
    mappings = [
        ('EARLY', 'EARLY_DISK'),
        ('EDISK', 'EARLY_DISK'),
        ('LATE', 'LATE_DISK'),
        ('LDISK', 'LATE_DISK'),
        ('S0', 'S0'),
        ('ELL', 'ELL'),
    ]
    for key, value in mappings:
        if key in upper:
            return value
    fallback_order = ['ELL', 'S0', 'EARLY_DISK', 'LATE_DISK']
    if label_name.startswith('class_') and index < len(fallback_order):
        return fallback_order[index]
    return label_name


def canonical_label(label_set: str, label_name: str, index: int) -> str:
    if label_set == 'regular':
        return canonical_regular_label(label_name, index)
    return label_name


def run_model(model_path: Path, fallback_labels: List[str], catalog: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, List[str]]:
    logging.info("Loading model from %s", model_path)
    model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(str(model_path), strict=False)
    model.eval()
    device = choose_device(args.accelerator, str(args.devices))
    model.to(device)

    label_names = determine_label_names(model, fallback_labels)
    transform = get_inference_transform(args.image_size)
    dataset = StampDataset(catalog, transform)
    if len(dataset) == 0:
        raise RuntimeError("Inference catalog is empty. Nothing to run.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda'
    )

    all_probs = []
    ids = []
    with torch.no_grad():
        for batch_images, batch_ids in loader:
            batch_images = batch_images.to(device)
            logits = model(batch_images)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            ids.extend(batch_ids)

    predictions = torch.cat(all_probs, dim=0).numpy()
    num_outputs = predictions.shape[1]
    if len(label_names) != num_outputs:
        logging.warning(
            "Checkpoint produced %d outputs but we have %d label names. "
            "Adjusting names to match tensor shape.",
            num_outputs,
            len(label_names)
        )
        if len(label_names) >= num_outputs:
            label_names = label_names[:num_outputs]
        else:
            extras = [f"class_{i}" for i in range(len(label_names), num_outputs)]
            label_names = label_names + extras
    df = pd.DataFrame(predictions, columns=[f"{name}" for name in label_names])
    df.insert(0, 'id', ids)
    return df, label_names


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    extra_cols = [args.mag_column] if (args.mag_column and args.mag_limit is not None) else []
    df_master = load_master_subset(args.catalog, args.id_column, args.redshift_column, extra_cols)
    if args.mag_column and args.mag_column in df_master.columns and args.mag_limit is not None:
         before = len(df_master)
         df_master = df_master[df_master[args.mag_column].notna() & (df_master[args.mag_column] <= args.mag_limit)]
         logging.info("Magnitude cut %s <= %.2f reduced catalog from %d to %d objects.",
                      args.mag_column, args.mag_limit, before, len(df_master))
    inference_catalog = build_inference_catalog(df_master, args)
    if inference_catalog.empty:
        raise RuntimeError("No valid stamp entries prepared for inference.")

    outputs = {'id': inference_catalog['id'].astype(str)}

    model_configs = [
        ('regular', args.ckpt_regular),
        ('family', args.ckpt_family),
        ('binary', args.ckpt_binary),
    ]

    for label_set, ckpt in model_configs:
        fallback = LABEL_SETS[label_set]
        logging.info("Running %s model (fallback %d classes)", label_set, len(fallback))
        preds, label_names = run_model(ckpt, fallback, inference_catalog[['id', 'file_loc']], args)
        rename_map = {}
        for idx, name in enumerate(label_names):
            if name == 'id':
                continue
            canonical = canonical_label(label_set, name, idx)
            rename_map[name] = f"{label_set}_{canonical}"
        preds = preds.rename(columns=rename_map)
        values_only = preds.drop(columns=['id'])
        if not values_only.empty:
            values_only = values_only.groupby(level=0, axis=1).sum()
            if label_set == 'regular':
                row_sums = values_only.sum(axis=1)
                nonzero = row_sums > 0
                if nonzero.any():
                    values_only.loc[nonzero] = values_only.loc[nonzero].div(row_sums[nonzero], axis=0)
        preds = pd.concat([preds[['id']], values_only], axis=1)
        for col in preds.columns:
            if col == 'id':
                continue
            outputs[col] = preds[col].values

    final_df = pd.DataFrame(outputs)
    final_df['id'] = final_df['id'].astype(str)

    table = Table.from_pandas(final_df)
    table.write(args.output, overwrite=True)
    logging.info("Saved predictions to %s", args.output)


if __name__ == "__main__":
    main()
