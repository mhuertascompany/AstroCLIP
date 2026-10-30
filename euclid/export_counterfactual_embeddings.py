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
    def __init__(self, stamp_root, galaxy_ids, sfhs, donor_ids, donor_sfhs,
                 image_size=64, band="VIS"):
        self.stamp_root = Path(stamp_root) / band
        self.galaxy_ids = np.asarray(galaxy_ids, dtype=np.int64)
        self.sfhs = np.asarray(sfhs, dtype=np.float32)
        self.donor_ids = np.asarray(donor_ids, dtype=np.int64)
        self.donor_sfhs = np.asarray(donor_sfhs, dtype=np.float32)
        self.image_size = int(image_size)
        self.band = str(band)
        if self.donor_ids.shape[:1] != (len(self.galaxy_ids),):
            raise ValueError("One donor set is required per target")
        if self.donor_sfhs.shape[:2] != self.donor_ids.shape:
            raise ValueError("Donor ID and SFH arrays are inconsistent")

    def __len__(self):
        return len(self.galaxy_ids)

    def _image(self, galaxy_id):
        path = self.stamp_root / f"{self.band}_{galaxy_id}.jpg"
        with Image.open(path) as source:
            image = source.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC,
            )
            values = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(values.copy())[None]

    def __getitem__(self, index):
        galaxy_id = int(self.galaxy_ids[index])
        donor_images = torch.stack([
            self._image(int(value)) for value in self.donor_ids[index]
        ])
        return (
            self._image(galaxy_id),
            torch.from_numpy(self.sfhs[index].copy()),
            donor_images,
            torch.from_numpy(self.donor_sfhs[index].copy()),
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
    parser.add_argument("--donors", type=int, default=8)
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
        all_val_rows = np.asarray(split["val_rows"], dtype=np.int64)
        all_donor_pools = np.asarray(split["val_donor_pools"], dtype=np.int64)
        sfh_mean = np.asarray(split["sfh_mean"], dtype=np.float32)
        sfh_scale = np.asarray(split["sfh_scale"], dtype=np.float32)
        activity_all = np.asarray(split["activity_class"], dtype=np.int8)
    if not 1 <= args.donors <= all_donor_pools.shape[1]:
        raise ValueError("donors is incompatible with the saved donor pools")
    selected = np.arange(len(all_val_rows))
    rng = np.random.default_rng(args.seed)
    if args.max_objects > 0 and len(selected) > args.max_objects:
        selected = rng.choice(selected, args.max_objects, replace=False)
    order = np.argsort(all_val_rows[selected])
    selected = selected[order]
    rows = all_val_rows[selected]
    donor_rows = all_donor_pools[selected, :args.donors]
    unique_rows = np.unique(np.concatenate((rows, donor_rows.ravel())))

    with h5py.File(args.dataset, "r") as source:
        unique_ids = _read_rows(source["galaxy_id"], unique_rows).astype(np.int64)
        unique_log_sfhs = _read_rows(source["sfh"], unique_rows).astype(np.float32)
        redshift = _read_rows(source["redshift"], rows).astype(np.float32)
        mass = _read_rows(source["phz_pp_median_stellarmass"], rows).astype(np.float32)
    unique_clr = weights_to_clr(log_sfh_to_weights(unique_log_sfhs))
    unique_sfhs = ((unique_clr - sfh_mean) / sfh_scale).astype(np.float32)
    target_positions = np.searchsorted(unique_rows, rows)
    donor_positions = np.searchsorted(unique_rows, donor_rows)
    galaxy_ids = unique_ids[target_positions]
    sfhs = unique_sfhs[target_positions]
    donor_ids = unique_ids[donor_positions]
    donor_sfhs = unique_sfhs[donor_positions]
    activity = activity_all[rows]

    dataset = TargetEmbeddingDataset(
        args.stamps, galaxy_ids, sfhs, donor_ids, donor_sfhs,
        image_size=args.image_size,
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
        for images, batch_sfhs, donor_images, batch_donor_sfhs, batch_ids in loader:
            images = images.to(device, non_blocking=True)
            batch_sfhs = batch_sfhs.to(device, non_blocking=True)
            donor_images = donor_images.to(device, non_blocking=True)
            batch_donor_sfhs = batch_donor_sfhs.to(device, non_blocking=True)
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
                batch, donors = donor_images.shape[:2]
                image_prior = model.image_prior_encoder(
                    donor_images.flatten(0, 1)
                ).unflatten(0, (batch, donors))
                sfh_prior = model.sfh_prior_encoder(
                    batch_donor_sfhs.flatten(0, 1)
                ).unflatten(0, (batch, donors))
                output["image_prior"].append(
                    model.image_prior_pool(image_prior).float().cpu().numpy()
                )
                output["sfh_prior"].append(
                    model.sfh_prior_pool(sfh_prior).float().cpu().numpy()
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
        "donors_per_object": int(args.donors),
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
