"""Export multimodal-JEPA embeddings and UMAPs for the Euclid explorer."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .population_residual_dataset import PackedPopulationResidualDataset
from .train_multimodal_jepa import JEPAExperiment
from .umap_zoobot_clip import fit_umap


_RESERVED = {
    "galaxy_id", "h5_row", "image_embedding", "sfh_embedding",
    "joint_embedding", "xy_image", "xy_sfh", "xy_joint",
    "xy_shared_image", "xy_shared_sfh", "paired_cosine",
    "image_to_sfh_rank_percentile", "sfh_to_image_rank_percentile",
}


def _requested_ids(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            if "galaxy_id" not in archive:
                raise ValueError(f"{path} has no galaxy_id array")
            values = archive["galaxy_id"]
    else:
        values = np.loadtxt(path, dtype=np.int64, delimiter=",")
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if len(np.unique(values)) != len(values):
        raise ValueError("Requested galaxy IDs contain duplicates")
    return values


def _select_rows(dataset: Path, transform: Path, ids_path: Path | None):
    with h5py.File(dataset, "r") as source:
        source_ids = np.asarray(source["galaxy_id"], dtype=np.int64)
    with np.load(transform) as arrays:
        train_rows = np.asarray(arrays["train_rows"], dtype=np.int64)
        val_rows = np.asarray(arrays["val_rows"], dtype=np.int64)
        sfh_mean = np.asarray(arrays["sfh_mean"], dtype=np.float32)
        sfh_scale = np.asarray(arrays["sfh_scale"], dtype=np.float32)
    allowed_rows = np.concatenate((train_rows, val_rows))
    allowed = np.zeros(len(source_ids), dtype=bool)
    allowed[allowed_rows] = True
    if ids_path is None:
        rows = allowed_rows
    else:
        requested = _requested_ids(ids_path)
        row_by_id = {int(value): index for index, value in enumerate(source_ids)}
        rows = np.array(
            [row_by_id[int(value)] for value in requested if int(value) in row_by_id],
            dtype=np.int64,
        )
        rows = rows[allowed[rows]]
        if not len(rows):
            raise ValueError("None of the requested IDs belong to the selected sample")
        missing = len(requested) - len(rows)
        print(
            f"Requested {len(requested):,} IDs; exporting {len(rows):,}; "
            f"absent or excluded={missing:,}",
            flush=True,
        )
    return source_ids, rows, sfh_mean, sfh_scale


def _copy_properties(path: Path | None, galaxy_ids: np.ndarray) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    with np.load(path) as archive:
        if "galaxy_id" not in archive:
            raise ValueError(f"{path} has no galaxy_id array")
        source_ids = np.asarray(archive["galaxy_id"], dtype=np.int64)
        by_id = {int(value): index for index, value in enumerate(source_ids)}
        try:
            positions = np.array(
                [by_id[int(value)] for value in galaxy_ids], dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                f"Exported ID {int(error.args[0])} is absent from {path}"
            ) from error
        output = {}
        for key in archive.files:
            if key in _RESERVED:
                continue
            values = np.asarray(archive[key])
            if (
                values.ndim == 1 and len(values) == len(source_ids)
                and values.dtype.kind in "biuf"
            ):
                output[key] = values[positions].astype(np.float32)
    return output


def export(args):
    source_ids, rows, sfh_mean, sfh_scale = _select_rows(
        args.dataset, args.data_transform, args.ids,
    )
    covariates = np.zeros((len(source_ids), 5), dtype=np.float32)
    data = PackedPopulationResidualDataset(
        path=args.dataset,
        packed_vis=args.packed_vis,
        rows=rows,
        sfh_mean=sfh_mean,
        sfh_scale=sfh_scale,
        covariates=covariates,
        training=False,
    )
    loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, persistent_workers=args.workers > 0,
        pin_memory=args.device.startswith("cuda"),
    )
    experiment = JEPAExperiment.load_from_checkpoint(
        args.checkpoint, map_location="cpu",
    )
    model = experiment.model.to(args.device).eval()
    collected = {mode: [] for mode in ("image", "sfh", "joint")}
    collected_ids = []
    amp = args.device.startswith("cuda")
    with torch.inference_mode():
        for number, batch in enumerate(loader, start=1):
            image = batch["image"].to(args.device, non_blocking=True)
            sfh = batch["sfh"].to(args.device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp,
            ):
                for mode in collected:
                    values = model.embed(image, sfh, mode)
                    collected[mode].append(values.float().cpu().numpy())
            collected_ids.append(batch["galaxy_id"].numpy())
            if number % 25 == 0 or number == len(loader):
                done = min(number * args.batch_size, len(data))
                print(f"Encoded {done:,}/{len(data):,}", flush=True)
    embeddings = {
        mode: np.concatenate(parts).astype(np.float32)
        for mode, parts in collected.items()
    }
    galaxy_ids = np.concatenate(collected_ids).astype(np.int64)
    if not np.array_equal(galaxy_ids, source_ids[rows]):
        raise RuntimeError("Embedding export changed the requested row order")

    print("Fitting image-only UMAP", flush=True)
    xy_image = fit_umap(
        embeddings["image"], args.n_neighbors, args.min_dist, args.seed,
    )
    print("Fitting SFH-only UMAP", flush=True)
    xy_sfh = fit_umap(
        embeddings["sfh"], args.n_neighbors, args.min_dist, args.seed,
    )
    print("Fitting joint UMAP", flush=True)
    xy_joint = fit_umap(
        embeddings["joint"], args.n_neighbors, args.min_dist, args.seed,
    )
    print("Fitting common image/SFH UMAP", flush=True)
    shared = fit_umap(
        np.concatenate((embeddings["image"], embeddings["sfh"])),
        args.n_neighbors, args.min_dist, args.seed,
    )
    image_unit = embeddings["image"] / np.maximum(
        np.linalg.norm(embeddings["image"], axis=1, keepdims=True), 1e-12,
    )
    sfh_unit = embeddings["sfh"] / np.maximum(
        np.linalg.norm(embeddings["sfh"], axis=1, keepdims=True), 1e-12,
    )
    output = {
        "galaxy_id": galaxy_ids,
        "h5_row": rows.astype(np.int64),
        "image_embedding": embeddings["image"],
        "sfh_embedding": embeddings["sfh"],
        "joint_embedding": embeddings["joint"],
        "xy_image": xy_image,
        "xy_sfh": xy_sfh,
        "xy_joint": xy_joint,
        "xy_shared_image": shared[:len(galaxy_ids)],
        "xy_shared_sfh": shared[len(galaxy_ids):],
        "paired_cosine": np.sum(image_unit * sfh_unit, axis=1).astype(np.float32),
    }
    output.update(_copy_properties(args.properties, galaxy_ids))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    np.savez_compressed(args.output, **output)
    print(f"Saved {args.output} ({len(galaxy_ids):,} galaxies)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--packed-vis", type=Path, required=True)
    parser.add_argument("--data-transform", type=Path, required=True)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--properties", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
