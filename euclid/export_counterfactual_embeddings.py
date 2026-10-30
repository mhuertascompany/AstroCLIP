"""Export common and prior representations from a counterfactual-flow model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .sfh_conditional_diffusion import log_sfh_to_weights, weights_to_clr
from .train_counterfactual_flow import CounterfactualFlowExperiment


def _read_rows(dataset, rows):
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(dataset[rows[order]])[inverse]


class TargetEmbeddingDataset(Dataset):
    def __init__(self, stamp_root, galaxy_ids, sfhs, image_size=64, band="VIS"):
        self.stamp_root = Path(stamp_root) / band
        self.galaxy_ids = np.asarray(galaxy_ids, dtype=np.int64)
        self.sfhs = np.asarray(sfhs, dtype=np.float32)
        self.image_size = int(image_size)
        self.band = str(band)

    def __len__(self):
        return len(self.galaxy_ids)

    def __getitem__(self, index):
        galaxy_id = int(self.galaxy_ids[index])
        path = self.stamp_root / f"{self.band}_{galaxy_id}.jpg"
        with Image.open(path) as source:
            image = source.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC,
            )
            values = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return (
            torch.from_numpy(values.copy())[None],
            torch.from_numpy(self.sfhs[index].copy()),
            galaxy_id,
        )


def _standardize(values, epsilon=1e-6):
    values = np.asarray(values, dtype=np.float32)
    return (
        (values - values.mean(axis=0, keepdims=True))
        / np.maximum(values.std(axis=0, keepdims=True), epsilon)
    ).astype(np.float32)


def _effective_rank(values):
    singular = np.linalg.svd(_standardize(values), compute_uv=False)
    variance = np.square(singular.astype(np.float64))
    probability = variance / np.maximum(variance.sum(), 1e-12)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)))
    return float(np.exp(entropy))


def _linear_cka(left, right):
    left = _standardize(left).astype(np.float64)
    right = _standardize(right).astype(np.float64)
    cross = np.linalg.norm(left.T @ right, ord="fro") ** 2
    denominator = (
        np.linalg.norm(left.T @ left, ord="fro")
        * np.linalg.norm(right.T @ right, ord="fro")
    )
    return float(cross / max(denominator, 1e-12))


def _paired_retrieval(left, right, block_size=256):
    left = F.normalize(torch.from_numpy(_standardize(left)), dim=1)
    right = F.normalize(torch.from_numpy(_standardize(right)), dim=1)
    paired = torch.sum(left * right, dim=1)
    ranks = []
    for start in range(0, len(left), block_size):
        stop = min(start + block_size, len(left))
        similarities = left[start:stop] @ right.T
        ranks.append(
            1 + torch.sum(
                similarities > paired[start:stop, None] + 1e-6, dim=1,
            )
        )
    ranks = torch.cat(ranks).numpy()
    shuffled = torch.sum(left * right.roll(1, 0), dim=1)
    return {
        "paired_cosine_median": float(torch.median(paired)),
        "shuffled_cosine_median": float(torch.median(shuffled)),
        "median_rank": float(np.median(ranks)),
        "median_rank_percentile": float(np.median(ranks / len(left))),
        "rank1_fraction": float(np.mean(ranks == 1)),
        "rank10_fraction": float(np.mean(ranks <= 10)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stamps", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-objects", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.checkpoint, args.dataset, args.split):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (args.stamps / "VIS").is_dir():
        raise FileNotFoundError(args.stamps / "VIS")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    with np.load(args.split) as split:
        rows = np.asarray(split["val_rows"], dtype=np.int64)
        sfh_mean = np.asarray(split["sfh_mean"], dtype=np.float32)
        sfh_scale = np.asarray(split["sfh_scale"], dtype=np.float32)
        activity_all = np.asarray(split["activity_class"], dtype=np.int8)
    rng = np.random.default_rng(args.seed)
    if args.max_objects > 0 and len(rows) > args.max_objects:
        rows = rng.choice(rows, args.max_objects, replace=False)
    rows = np.sort(rows)

    with h5py.File(args.dataset, "r") as source:
        galaxy_ids = _read_rows(source["galaxy_id"], rows).astype(np.int64)
        log_sfhs = _read_rows(source["sfh"], rows).astype(np.float32)
        redshift = _read_rows(source["redshift"], rows).astype(np.float32)
        mass = _read_rows(source["phz_pp_median_stellarmass"], rows).astype(np.float32)
    clr = weights_to_clr(log_sfh_to_weights(log_sfhs))
    sfhs = ((clr - sfh_mean) / sfh_scale).astype(np.float32)
    activity = activity_all[rows]

    dataset = TargetEmbeddingDataset(
        args.stamps, galaxy_ids, sfhs, image_size=args.image_size,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.device != "cpu",
        persistent_workers=args.workers > 0,
    )
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    experiment = CounterfactualFlowExperiment.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu", weights_only=False,
    )
    model = experiment.model.eval().to(device)

    output = {name: [] for name in (
        "image_common", "sfh_common", "image_prior", "sfh_prior",
    )}
    encoded_ids = []
    with torch.inference_mode():
        for images, batch_sfhs, batch_ids in loader:
            images = images.to(device, non_blocking=True)
            batch_sfhs = batch_sfhs.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output["image_common"].append(
                    model.image_common_encoder(images).float().cpu().numpy()
                )
                output["sfh_common"].append(
                    model.sfh_common_encoder(batch_sfhs).float().cpu().numpy()
                )
                output["image_prior"].append(
                    model.image_prior_encoder(images).float().cpu().numpy()
                )
                output["sfh_prior"].append(
                    model.sfh_prior_encoder(batch_sfhs).float().cpu().numpy()
                )
            encoded_ids.append(np.asarray(batch_ids, dtype=np.int64))
    encoded_ids = np.concatenate(encoded_ids)
    if not np.array_equal(encoded_ids, galaxy_ids):
        raise RuntimeError("DataLoader changed galaxy order")
    output = {name: np.concatenate(parts).astype(np.float32)
              for name, parts in output.items()}

    common_image = _standardize(output["image_common"])
    common_sfh = _standardize(output["sfh_common"])
    prior_image = _standardize(output["image_prior"])
    prior_sfh = _standardize(output["sfh_prior"])
    shared = dict(
        galaxy_id=galaxy_ids, h5_row=rows, redshift=redshift,
        log_stellar_mass=mass, sfh_activity_class=activity,
    )
    np.savez_compressed(
        args.output_dir / "common_embeddings.npz", **shared,
        image_embedding=common_image, sfh_embedding=common_sfh,
        image_raw_embedding=output["image_common"],
        sfh_raw_embedding=output["sfh_common"],
    )
    np.savez_compressed(
        args.output_dir / "prior_embeddings.npz", **shared,
        image_embedding=prior_image, sfh_embedding=prior_sfh,
        image_raw_embedding=output["image_prior"],
        sfh_raw_embedding=output["sfh_prior"],
    )

    diagnostics = {
        "checkpoint": str(args.checkpoint),
        "n_objects": int(len(rows)),
        "common": {
            "image_effective_rank": _effective_rank(common_image),
            "sfh_effective_rank": _effective_rank(common_sfh),
            "linear_cka": _linear_cka(common_image, common_sfh),
            **_paired_retrieval(common_image, common_sfh),
        },
        "prior": {
            "image_effective_rank": _effective_rank(prior_image),
            "sfh_effective_rank": _effective_rank(prior_sfh),
            "linear_cka": _linear_cka(prior_image, prior_sfh),
            **_paired_retrieval(prior_image, prior_sfh),
        },
    }
    (args.output_dir / "embedding_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
