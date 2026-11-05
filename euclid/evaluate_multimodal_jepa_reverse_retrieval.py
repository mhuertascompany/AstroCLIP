"""Evaluate VIS-to-SFH retrieval through the trained multimodal JEPA predictor.

The image and SFH encoder outputs are not assumed to be directly aligned.  A
VIS image is therefore passed through the online encoder and the JEPA
``image_to_sfh`` predictor.  Its predicted SFH target tokens are compared with
SFH-only target tokens from the EMA teacher, which is the latent space used by
the reverse JEPA training objective.  Both cosine and CSLS retrievals are
saved for downstream, image-selected SFH diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_jepa_csls import (
    _comparison_figure,
    _csls_densities,
    _csls_retrieval_many,
)
from .evaluate_multimodal_jepa_retrieval import _retrieval, _summary
from .population_residual_dataset import PackedPopulationResidualDataset
from .train_multimodal_jepa import JEPAExperiment


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.data_transform) as arrays:
        rows = np.asarray(
            arrays["val_rows"] if args.split == "val"
            else np.concatenate((arrays["train_rows"], arrays["val_rows"])),
            dtype=np.int64,
        )
        sfh_mean = np.asarray(arrays["sfh_mean"], dtype=np.float32)
        sfh_scale = np.asarray(arrays["sfh_scale"], dtype=np.float32)
    with h5py.File(args.dataset, "r") as source:
        source_ids = np.asarray(source["galaxy_id"], dtype=np.int64)
    galaxy_ids = source_ids[rows]

    data = PackedPopulationResidualDataset(
        args.dataset, args.packed_vis, rows, sfh_mean, sfh_scale,
        np.zeros((len(source_ids), 5), dtype=np.float32), training=False,
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
    predicted_tokens, teacher_tokens = [], []
    amp = args.device.startswith("cuda")
    with torch.inference_mode():
        for number, batch in enumerate(loader, start=1):
            image = batch["image"].to(args.device, non_blocking=True)
            sfh = batch["sfh"].to(args.device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp,
            ):
                prediction, target, spec = model.predict(
                    image, sfh, "image_to_sfh",
                )
            predicted_tokens.append(
                prediction[:, spec.target].half().cpu().numpy(),
            )
            teacher_tokens.append(
                target[:, spec.target].half().cpu().numpy(),
            )
            if number % 25 == 0 or number == len(loader):
                print(
                    f"Predicted and encoded {min(number * args.batch_size, len(data)):,}"
                    f"/{len(data):,}", flush=True,
                )

    predicted_tokens = np.concatenate(predicted_tokens)
    teacher_tokens = np.concatenate(teacher_tokens)
    predicted_pooled = predicted_tokens.astype(np.float32).mean(1)
    teacher_pooled = teacher_tokens.astype(np.float32).mean(1)

    paired = np.arange(len(data), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(data))
    if np.any(shuffled == paired):
        shuffled = np.roll(paired, 1)
    cosine = _retrieval(
        predicted_pooled, teacher_pooled, paired, shuffled,
        args.top_k, args.device, args.query_chunk,
    )

    ks = tuple(sorted(set(args.k)))
    query_density, gallery_density = _csls_densities(
        predicted_pooled, teacher_pooled, ks,
        args.device, args.gallery_chunk,
    )
    csls = _csls_retrieval_many(
        predicted_pooled, teacher_pooled,
        query_density, gallery_density, paired, shuffled,
        args.top_k, args.device, args.query_chunk,
    )
    summary = {
        "direction": "vis_to_sfh",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "token_count": int(predicted_tokens.shape[1]),
        "token_dimension": int(predicted_tokens.shape[2]),
        "cosine": _summary(cosine, len(data)),
        **{
            f"csls_k{k}": _summary(result, len(data))
            for k, result in csls.items()
        },
    }
    for name, values in summary.items():
        if not isinstance(values, dict):
            continue
        result = cosine if name == "cosine" else csls[int(name.removeprefix("csls_k"))]
        counts = np.bincount(
            result["top_index"].reshape(-1), minlength=len(data),
        )
        values["maximum_hub_query_fraction"] = float(counts.max() / len(data))
        values["top_k_gallery_coverage"] = float(np.mean(counts > 0))

    saved = {
        "galaxy_id": galaxy_ids,
        "h5_row": rows,
        **{f"cosine_{name}": value for name, value in cosine.items()},
        **{f"query_density_k{k}": query_density[k] for k in ks},
        **{f"gallery_density_k{k}": gallery_density[k] for k in ks},
    }
    for k, result in csls.items():
        saved.update({f"csls_k{k}_{name}": value for name, value in result.items()})
    np.savez_compressed(args.output / "csls_results.npz", **saved)
    np.savez_compressed(
        args.output / "reverse_retrieval_embeddings.npz",
        galaxy_id=galaxy_ids,
        h5_row=rows,
        predicted_sfh_embedding=predicted_pooled,
        teacher_sfh_embedding=teacher_pooled,
    )
    (args.output / "csls_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
    )
    _comparison_figure(
        cosine, csls, len(data),
        args.output / "csls_retrieval_diagnostics.png",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved reverse retrieval to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--packed-vis", type=Path, required=True)
    parser.add_argument("--data-transform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "all"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gallery-chunk", type=int, default=1024)
    parser.add_argument("--query-chunk", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output: {args.output}")
    evaluate(args)


if __name__ == "__main__":
    main()
