"""Evaluate SFH-to-VIS retrieval through a trained multimodal JEPA predictor.

The raw image-only and SFH-only encoder outputs are not assumed to be aligned.
Instead, an SFH is passed through the online encoder and JEPA predictor to
produce image-target tokens.  Those predictions are compared with image-only
tokens from the EMA teacher, exactly the latent space used by the training
objective.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .population_residual_dataset import PackedPopulationResidualDataset
from .sfh_conditional_diffusion import clr_to_weights
from .train_multimodal_jepa import JEPAExperiment


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12,
    )


def _retrieval(
    query: np.ndarray,
    gallery: np.ndarray,
    paired_index: np.ndarray,
    control_index: np.ndarray,
    top_k: int,
    device: str,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Exact cosine retrieval, including paired and shuffled-control ranks."""
    query = _unit_rows(query)
    gallery = _unit_rows(gallery)
    paired_index = np.asarray(paired_index, dtype=np.int64)
    control_index = np.asarray(control_index, dtype=np.int64)
    if len(query) != len(paired_index) or len(query) != len(control_index):
        raise ValueError("Every query needs paired and control gallery indices")
    if top_k < 1 or top_k > len(gallery):
        raise ValueError("top_k must lie between one and the gallery size")

    gallery_tensor = torch.from_numpy(gallery).to(device)
    ranks, control_ranks, paired_similarity = [], [], []
    nearest_similarity, top_indices, top_scores = [], [], []
    with torch.inference_mode():
        for start in range(0, len(query), chunk_size):
            stop = min(start + chunk_size, len(query))
            q = torch.from_numpy(query[start:stop]).to(device)
            similarity = q @ gallery_tensor.T
            local = torch.arange(stop - start, device=device)
            paired = torch.as_tensor(
                paired_index[start:stop], device=device,
            )
            control = torch.as_tensor(
                control_index[start:stop], device=device,
            )
            paired_score = similarity[local, paired]
            control_score = similarity[local, control]
            ranks.append(
                (similarity > paired_score[:, None]).sum(1).add(1).cpu().numpy()
            )
            control_ranks.append(
                (similarity > control_score[:, None]).sum(1).add(1).cpu().numpy()
            )
            scores, indices = similarity.topk(top_k, dim=1)
            paired_similarity.append(paired_score.cpu().numpy())
            nearest_similarity.append(scores[:, 0].cpu().numpy())
            top_indices.append(indices.cpu().numpy())
            top_scores.append(scores.cpu().numpy())
    return {
        "rank": np.concatenate(ranks).astype(np.int32),
        "control_rank": np.concatenate(control_ranks).astype(np.int32),
        "paired_similarity": np.concatenate(paired_similarity).astype(np.float32),
        "nearest_similarity": np.concatenate(nearest_similarity).astype(np.float32),
        "top_index": np.concatenate(top_indices).astype(np.int32),
        "top_similarity": np.concatenate(top_scores).astype(np.float32),
    }


def _activity(properties: dict[str, np.ndarray]) -> np.ndarray | None:
    for name in ("sf_class", "sfr_class", "activity_class"):
        if name in properties:
            return np.asarray(properties[name])
    return None


def _load_properties(path: Path | None, galaxy_ids: np.ndarray):
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as archive:
        source_ids = np.asarray(archive["galaxy_id"], dtype=np.int64)
        position = {int(value): index for index, value in enumerate(source_ids)}
        indices = np.array(
            [position.get(int(value), -1) for value in galaxy_ids],
            dtype=np.int64,
        )
        matched = indices >= 0
        print(
            f"Morphology properties cover {matched.sum():,}/{len(galaxy_ids):,} "
            "retrieval galaxies", flush=True,
        )
        output = {}
        for name in archive.files:
            values = np.asarray(archive[name])
            if (
                name != "galaxy_id" and values.ndim == 1
                and len(values) == len(source_ids)
                and values.dtype.kind in "biuf"
            ):
                selected = np.full(len(galaxy_ids), np.nan, dtype=np.float32)
                selected[matched] = values[indices[matched]].astype(np.float32)
                output[name] = selected
    return output


def _matched_controls(
    properties: dict[str, np.ndarray], query_indices: np.ndarray,
    count: int, seed: int,
) -> np.ndarray | None:
    if not {"log_stellar_mass", "redshift"}.issubset(properties):
        return None
    mass = np.asarray(properties["log_stellar_mass"], dtype=float)
    redshift = np.asarray(properties["redshift"], dtype=float)
    valid = np.isfinite(mass) & np.isfinite(redshift)
    activity = _activity(properties)
    rng = np.random.default_rng(seed)
    controls = np.empty((len(query_indices), count), dtype=np.int32)
    for output_row, query in enumerate(query_indices):
        candidate = valid.copy()
        candidate[query] = False
        candidate &= np.abs(mass - mass[query]) <= 0.1
        candidate &= np.abs(redshift - redshift[query]) <= 0.05
        if activity is not None:
            candidate &= activity == activity[query]
        available = np.flatnonzero(candidate)
        if len(available) < count:
            distance = (
                ((mass - mass[query]) / 0.1) ** 2
                + ((redshift - redshift[query]) / 0.05) ** 2
            )
            distance[~valid] = np.inf
            distance[query] = np.inf
            if activity is not None:
                distance[activity != activity[query]] = np.inf
            available = np.argsort(distance)[:max(count, 100)]
            available = available[np.isfinite(distance[available])]
        if not len(available):
            return None
        controls[output_row] = rng.choice(
            available, size=count, replace=len(available) < count,
        )
    return controls


def _summary(result: dict[str, np.ndarray], gallery_size: int) -> dict[str, float]:
    rank = result["rank"]
    top = result["top_index"]
    counts = np.bincount(top.reshape(-1), minlength=gallery_size)
    output = {
        "queries": int(len(rank)),
        "gallery": int(gallery_size),
        "median_rank": float(np.median(rank)),
        "median_rank_percentile": float(np.median((rank - 1) / max(1, gallery_size - 1))),
        "mean_paired_similarity": float(np.mean(result["paired_similarity"])),
        "median_paired_similarity": float(np.median(result["paired_similarity"])),
        "median_nearest_similarity": float(np.median(result["nearest_similarity"])),
        "unique_top_k_gallery_fraction": float(np.count_nonzero(counts) / gallery_size),
        "maximum_top_k_occurrences": int(counts.max()),
    }
    for k in (1, 5, 10, 20, 50, 100):
        if k <= gallery_size:
            output[f"recall_at_{k}"] = float(np.mean(rank <= k))
    return output


def _diagnostic_figure(
    pooled: dict[str, np.ndarray], patch: dict[str, np.ndarray] | None,
    gallery_size: int, output: Path,
):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    percentile = (pooled["rank"] - 1) / max(1, gallery_size - 1)
    control = (pooled["control_rank"] - 1) / max(1, gallery_size - 1)
    bins = np.linspace(0, 1, 41)
    axes[0, 0].hist(control, bins=bins, density=True, histtype="step", lw=2,
                    color="0.45", label="shuffled association")
    axes[0, 0].hist(percentile, bins=bins, density=True, histtype="step", lw=2,
                    color="#0072B2", label="true pair")
    axes[0, 0].set(xlabel="True-pair rank percentile", ylabel="Density",
                   title="Mean-pooled predictor retrieval")
    axes[0, 0].legend()

    ks = np.array([1, 5, 10, 20, 50, 100])
    ks = ks[ks <= gallery_size]
    axes[0, 1].plot(ks, [np.mean(pooled["rank"] <= k) for k in ks],
                        marker="o", label="mean pooled")
    if patch is not None:
        axes[0, 1].plot(ks, [np.mean(patch["rank"] <= k) for k in ks],
                            marker="o", label="patch averaged")
    axes[0, 1].plot(ks, ks / gallery_size, ls="--", color="0.45", label="chance")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(xlabel="k", ylabel="Recall@k", title="True-pair retrieval")
    axes[0, 1].legend()

    axes[1, 0].hist(pooled["paired_similarity"], bins=40, density=True,
                    histtype="step", lw=2, label="paired image")
    axes[1, 0].hist(pooled["nearest_similarity"], bins=40, density=True,
                    histtype="step", lw=2, label="nearest image")
    axes[1, 0].set(xlabel="Cosine similarity", ylabel="Density",
                   title="Predicted-image latent similarity")
    axes[1, 0].legend()

    counts = np.bincount(
        pooled["top_index"].reshape(-1), minlength=gallery_size,
    )
    axes[1, 1].hist(counts, bins=np.arange(counts.max() + 2) - 0.5,
                    color="#D55E00", alpha=0.75)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(xlabel="Occurrences among all top-k lists",
                   ylabel="Number of VIS images", title="Retrieval hubness")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _morphology_figure(
    properties: dict[str, np.ndarray], query: np.ndarray,
    retrieved: np.ndarray, controls: np.ndarray | None, output: Path,
):
    requested = [
        ("zoobot_featured_probability", "P(featured)"),
        ("zoobot_spiral_probability", "P(spiral arms)"),
        ("zoobot_merger_probability", "P(merger)"),
        ("zoobot_edge_on_probability", "P(edge-on)"),
        ("sersic_index", "Sérsic n"),
        ("axis_ratio", "Axis ratio"),
        ("concentration", "Concentration"),
        ("asymmetry", "Asymmetry"),
    ]
    available = [(key, label) for key, label in requested if key in properties]
    if not available:
        return
    columns = 4
    rows = math.ceil(len(available) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.8 * rows), squeeze=False)
    for axis, (key, label) in zip(axes.flat, available):
        values = np.asarray(properties[key], dtype=float)
        true = values[query]
        retrieved_median = np.nanmedian(values[retrieved], axis=1)
        good = np.isfinite(true) & np.isfinite(retrieved_median)
        rho = spearmanr(true[good], retrieved_median[good]).statistic if good.sum() > 2 else np.nan
        axis.scatter(true[good], retrieved_median[good], s=8, alpha=0.35,
                     color="#0072B2", label=f"retrieval, rho={rho:.2f}")
        if controls is not None:
            control_median = np.nanmedian(values[controls], axis=1)
            control_good = np.isfinite(true) & np.isfinite(control_median)
            control_rho = (
                spearmanr(true[control_good], control_median[control_good]).statistic
                if control_good.sum() > 2 else np.nan
            )
            axis.scatter(true[control_good], control_median[control_good], s=8,
                         alpha=0.25, color="0.55",
                         label=f"mass-z control, rho={control_rho:.2f}")
        limits = np.nanpercentile(
            np.r_[true[good], retrieved_median[good]], [1, 99],
        )
        if np.all(np.isfinite(limits)) and limits[1] > limits[0]:
            axis.plot(limits, limits, ls="--", lw=1, color="0.2")
            axis.set_xlim(limits)
            axis.set_ylim(limits)
        axis.set(xlabel=f"True paired {label}", ylabel=f"Median top-k {label}",
                 title=label)
        axis.legend(fontsize=7)
    for axis in axes.flat[len(available):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _show_image(axis, image, title):
    lo, hi = np.nanpercentile(image, [1, 99.5])
    axis.imshow(image, cmap="gray", origin="lower", vmin=lo, vmax=hi)
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def _example_pdf(
    path: Path, query_indices: np.ndarray, galaxy_ids: np.ndarray,
    rows: np.ndarray, packed_vis: Path, pooled: dict[str, np.ndarray],
    patch: dict[str, np.ndarray] | None, sfh_mean: np.ndarray,
    sfh_scale: np.ndarray,
):
    images = np.load(packed_vis / "vis_images.npy", mmap_mode="r")
    clr = np.load(packed_vis / "sfh_clr.npy", mmap_mode="r")
    with PdfPages(path) as pdf:
        for local, query in enumerate(query_indices):
            pooled_top = pooled["top_index"][query, :5]
            patch_top = patch["top_index"][local, :5] if patch is not None else np.array([], int)
            figure = plt.figure(figsize=(15, 7))
            grid = figure.add_gridspec(3, 6, width_ratios=(1.4, 1, 1, 1, 1, 1))
            sfh_axis = figure.add_subplot(grid[:, 0])
            weights = clr_to_weights(np.asarray(clr[rows[query]], dtype=np.float64))
            sfh_axis.plot(np.linspace(0, 1, len(weights)), weights, color="#009E73")
            sfh_axis.set(xlabel="Fractional lookback time", ylabel="Normalized SFH weight",
                         title=f"Query SFH\nID {galaxy_ids[query]}")
            true_axis = figure.add_subplot(grid[0, 1])
            _show_image(true_axis, images[rows[query]], "True paired VIS")
            for column, gallery_index in enumerate(pooled_top, start=1):
                axis = figure.add_subplot(grid[1, column])
                _show_image(
                    axis, images[rows[gallery_index]],
                    f"Pooled #{column}\nID {galaxy_ids[gallery_index]}",
                )
            for column, gallery_index in enumerate(patch_top, start=1):
                axis = figure.add_subplot(grid[2, column])
                _show_image(
                    axis, images[rows[gallery_index]],
                    f"Patch #{column}\nID {galaxy_ids[gallery_index]}",
                )
            figure.text(0.27, 0.62, "Mean-pooled retrieval", fontsize=10, weight="bold")
            if patch is not None:
                figure.text(0.27, 0.31, "Patch-grid retrieval", fontsize=10, weight="bold")
            figure.suptitle(
                f"SFH→VIS JEPA retrieval · paired pooled rank {pooled['rank'][query]:,}",
                fontsize=13,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(figure)
            plt.close(figure)


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
                    image, sfh, "sfh_to_image",
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

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(data))
    if np.any(shuffled == np.arange(len(data))):
        shuffled = np.roll(np.arange(len(data)), 1)
    paired = np.arange(len(data), dtype=np.int64)
    pooled = _retrieval(
        predicted_pooled, teacher_pooled, paired, shuffled,
        args.top_k, args.device, args.retrieval_chunk,
    )

    patch = None
    patch_query = np.array([], dtype=np.int64)
    if args.patch_queries:
        patch_query = np.sort(rng.choice(
            len(data), min(args.patch_queries, len(data)), replace=False,
        ))
        predicted_patch = F.normalize(
            torch.from_numpy(predicted_tokens[patch_query].astype(np.float32)),
            dim=-1,
        ).numpy().reshape(len(patch_query), -1)
        teacher_patch = F.normalize(
            torch.from_numpy(teacher_tokens.astype(np.float32)), dim=-1,
        ).numpy().reshape(len(data), -1)
        scale = math.sqrt(predicted_tokens.shape[1])
        patch = _retrieval(
            predicted_patch / scale, teacher_patch / scale,
            patch_query, shuffled[patch_query], args.top_k,
            args.device, args.patch_chunk,
        )

    summary = {"pooled": _summary(pooled, len(data))}
    if patch is not None:
        summary["patch"] = _summary(patch, len(data))
    summary.update(
        split=args.split, checkpoint=str(args.checkpoint), seed=args.seed,
        token_count=int(predicted_tokens.shape[1]),
        token_dimension=int(predicted_tokens.shape[2]),
    )
    (args.output / "retrieval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
    )

    saved = {
        "galaxy_id": galaxy_ids,
        "h5_row": rows,
        "predicted_image_embedding": predicted_pooled,
        "teacher_image_embedding": teacher_pooled,
        **{f"pooled_{key}": value for key, value in pooled.items()},
        "patch_query_index": patch_query,
    }
    if patch is not None:
        saved.update({f"patch_{key}": value for key, value in patch.items()})
    np.savez_compressed(args.output / "retrieval_results.npz", **saved)
    _diagnostic_figure(
        pooled, patch, len(data), args.output / "retrieval_diagnostics.png",
    )

    properties = _load_properties(args.properties, galaxy_ids)
    morphology_query = rng.choice(
        len(data), min(args.morphology_queries, len(data)), replace=False,
    )
    morphology_top = pooled["top_index"][morphology_query, :args.morphology_k]
    controls = _matched_controls(
        properties, morphology_query, args.morphology_k, args.seed + 10,
    )
    _morphology_figure(
        properties, morphology_query, morphology_top, controls,
        args.output / "morphology_retrieval.png",
    )

    if patch is not None:
        example_count = min(args.examples, len(patch_query))
        example_local = np.linspace(
            0, len(patch_query) - 1, example_count, dtype=int,
        )
        examples = patch_query[example_local]
        example_patch = {
            key: value[example_local] if len(value) == len(patch_query) else value
            for key, value in patch.items()
        }
    else:
        examples = np.linspace(0, len(data) - 1, args.examples, dtype=int)
        example_patch = None
    _example_pdf(
        args.output / "retrieval_examples.pdf", examples, galaxy_ids, rows,
        args.packed_vis, pooled, example_patch, sfh_mean, sfh_scale,
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved diagnostics to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--packed-vis", type=Path, required=True)
    parser.add_argument("--data-transform", type=Path, required=True)
    parser.add_argument("--properties", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "all"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retrieval-chunk", type=int, default=512)
    parser.add_argument("--patch-chunk", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--patch-queries", type=int, default=256)
    parser.add_argument("--morphology-queries", type=int, default=1000)
    parser.add_argument("--morphology-k", type=int, default=20)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output: {args.output}")
    evaluate(args)


if __name__ == "__main__":
    main()
