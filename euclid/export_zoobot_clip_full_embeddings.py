"""Export CLIP embeddings for every SFH object with an available VIS stamp.

Unlike :mod:`euclid.evaluate_zoobot_clip`, this command does not compute the
quadratic full-gallery retrieval metrics and does not restrict extraction to
the held-out validation split.  It is intended for full-sample UMAPs and the
interactive explorer.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from .training_index import inspect_sfh_file


log = logging.getLogger(__name__)


def _read_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(dataset[rows[order]])[inverse]


def select_stamp_rows(
    dataset_path: Path, stamp_root: Path, band: str = "VIS",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return HDF5 rows and IDs for every object with a JPEG stamp."""
    galaxy_ids, n_bins, n_realizations = inspect_sfh_file(dataset_path)
    stamp_dir = stamp_root / band if (stamp_root / band).is_dir() else stamp_root
    if not stamp_dir.is_dir():
        raise FileNotFoundError(f"Stamp directory not found: {stamp_dir}")
    paired = np.fromiter(
        (
            (stamp_dir / f"{band}_{int(galaxy_id)}.jpg").is_file()
            for galaxy_id in galaxy_ids
        ),
        dtype=bool,
        count=len(galaxy_ids),
    )
    rows = np.flatnonzero(paired).astype(np.int64)
    if not len(rows):
        raise ValueError("No SFH objects have matching image stamps")
    return rows, galaxy_ids[rows].astype(np.int64), n_bins, n_realizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stamp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band", default="VIS")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    # Keep the heavy ZooBot/timm dependency out of metadata-only imports and tests.
    from cosmosweb.model_zoobot import CosmosWebZooBotCLIP
    from .evaluate_zoobot_clip import extract_embeddings, extraction_loader

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    for path in (args.checkpoint, args.dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Use a positive batch size and non-negative worker count")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    rows, galaxy_ids, n_bins, n_realizations = select_stamp_rows(
        args.dataset, args.stamp_root, args.band,
    )
    log.info(
        "Encoding full paired sample: %d objects, %d SFH bins, %d realizations",
        len(rows), n_bins, n_realizations,
    )
    model = CosmosWebZooBotCLIP.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu",
    ).to(device)
    loader = extraction_loader(
        args.dataset, args.stamp_root, rows, galaxy_ids, args.band,
        args.image_size, args.batch_size, args.num_workers,
    )
    image, sfh, sfh_preprojection, encoded_ids, _ = extract_embeddings(
        model, loader, device,
    )
    if not np.array_equal(encoded_ids, galaxy_ids):
        raise ValueError("Embedding extraction changed the full-sample ID order")
    with h5py.File(args.dataset, "r") as source:
        redshift = _read_rows(source["redshift"], rows).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        galaxy_id=galaxy_ids,
        h5_row=rows,
        redshift=redshift,
        image_embedding=image.astype(np.float32),
        sfh_embedding=sfh.astype(np.float32),
        sfh_preprojection_embedding=sfh_preprojection.astype(np.float32),
    )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "stamp_root": str(args.stamp_root.resolve()),
        "band": args.band,
        "device": str(device),
        "n_dataset_objects": int(len(inspect_sfh_file(args.dataset)[0])),
        "n_paired_objects": int(len(rows)),
        "n_sfh_bins": int(n_bins),
        "n_posterior_realizations": int(n_realizations),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
