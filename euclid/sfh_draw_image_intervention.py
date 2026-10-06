"""Propagate conditional-SFH draws through the VIS diffusion and image encoders.

For each selected physical condition (Mstar, redshift, SFR100), this experiment
chooses posterior-predictive SFHs with closely matched realized SFR100 and
maximally different older cumulative histories. It then generates VIS images
with identical initial diffusion noise across SFH variants and measures their
motion in frozen ZooBot and aligned image-embedding spaces.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from .prepare_diffusion_conditions import sha256


def recent_log_sfr(weights, mass, age_myr, time):
    """Compute R=0 log SFR averaged over the latest 100 Myr."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None]
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
    time = np.asarray(time, dtype=np.float64)
    edges = np.r_[0.0, (time[:-1] + time[1:]) / 2.0, 1.0]
    upper = min(1.0e8 / (float(age_myr) * 1.0e6), 1.0)
    overlap = np.maximum(0.0, np.minimum(edges[1:], upper) - edges[:-1])
    recent_mass = weights @ (overlap / np.diff(edges))
    return float(mass) + np.log10(np.maximum(recent_mass, 1e-15)) - 8.0


def select_diverse_draws(draws, realized_sfr, target_sfr, time, age_myr,
                         count, tolerance):
    """Choose SFR-matched draws with diverse cumulative histories before 100 Myr."""
    difference = np.abs(np.asarray(realized_sfr) - float(target_sfr))
    eligible = np.flatnonzero(difference <= tolerance)
    relaxed = False
    if len(eligible) < count:
        eligible = np.argsort(difference)[:max(count, min(len(draws), 4 * count))]
        relaxed = True
    if not len(eligible):
        raise ValueError("No finite conditional-SFH draws are available.")
    recent_boundary = min(1.0e8 / (float(age_myr) * 1.0e6), 1.0)
    old = np.asarray(draws[eligible], dtype=np.float64).copy()
    old[:, np.asarray(time) <= recent_boundary] = 0.0
    old /= np.maximum(old.sum(axis=1, keepdims=True), 1e-30)
    cumulative = np.cumsum(old, axis=1)

    # Start with the closest SFR match, then use farthest-point sampling in
    # cumulative-SFH space. This favors distinct assembly histories rather
    # than isolated noisy bins.
    chosen_local = [int(np.argmin(difference[eligible]))]
    while len(chosen_local) < min(count, len(eligible)):
        remaining = np.setdiff1d(np.arange(len(eligible)), chosen_local)
        distances = np.mean(
            np.abs(cumulative[remaining, None] - cumulative[chosen_local][None]),
            axis=2,
        )
        chosen_local.append(int(remaining[np.argmax(distances.min(axis=1))]))
    return eligible[np.asarray(chosen_local)], relaxed


def cosine_distance_rows(values):
    values = np.asarray(values, dtype=np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-30)
    return 1.0 - values @ values.T


def _read_rows(dataset, rows):
    order = np.argsort(rows)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(dataset[np.asarray(rows)[order]])[inverse]


def choose_galaxies(condition, n_galaxies, seed):
    """Sample across the SFR distribution rather than only its dense center."""
    if n_galaxies > len(condition):
        raise ValueError(f"Requested {n_galaxies} galaxies from {len(condition)} rows.")
    rng = np.random.default_rng(seed)
    order = np.argsort(condition[:, 2])
    groups = np.array_split(order, n_galaxies)
    return np.asarray([rng.choice(group) for group in groups], dtype=np.int64)


def verify_provenance(cache_path, clip_checkpoint, pixel_model):
    cache_digest = sha256(cache_path)
    if str(pixel_model.hparams.cache_sha256) != cache_digest:
        raise ValueError("Pixel-diffusion checkpoint was trained with another condition cache.")
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
    if metadata.get("checkpoint_sha256") != sha256(clip_checkpoint):
        raise ValueError("CLIP checkpoint differs from the one used by the image diffusion.")
    return metadata, cache_digest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "predictive", "dataset", "condition-cache", "clip-checkpoint",
        "pixel-checkpoint", "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--n-galaxies", type=int, default=6)
    parser.add_argument("--n-draws", type=int, default=5,
                        help="Generated SFH draws per galaxy; observed SFH is added separately.")
    parser.add_argument("--sfr-tolerance", type=float, default=0.15)
    parser.add_argument("--image-seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(args):
    # Keep model imports local so the numerical selection utilities remain
    # usable in lightweight environments without ZooBot/timm installed.
    from .model_zoobot import EuclidZooBotCLIP
    from .train_pixel_diffusion import PixelDiffusion

    paths = (
        args.predictive, args.dataset, args.condition_cache,
        args.clip_checkpoint, args.pixel_checkpoint,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.n_galaxies, args.n_draws, args.steps) < 1:
        raise ValueError("Galaxy, draw, and sampling counts must be positive.")
    if not args.image_seeds or args.sfr_tolerance <= 0:
        raise ValueError("Provide image seeds and a positive SFR tolerance.")

    device = torch.device(args.device)
    pixel = PixelDiffusion.load_from_checkpoint(
        str(args.pixel_checkpoint), map_location="cpu",
    ).to(device).eval().requires_grad_(False)
    clip = EuclidZooBotCLIP.load_from_checkpoint(
        str(args.clip_checkpoint), map_location="cpu",
    ).to(device).eval().requires_grad_(False)
    if int(getattr(clip.hparams, "unfreeze_blocks", 0)) != 0:
        raise ValueError(
            "The CLIP checkpoint changed the ZooBot backbone; its latent is not an "
            "unaligned frozen-ZooBot control."
        )
    metadata, cache_digest = verify_provenance(
        args.condition_cache, args.clip_checkpoint, pixel,
    )
    if not 2 <= args.steps <= len(pixel.schedule.alpha):
        raise ValueError("Sampling steps lie outside the pixel-diffusion schedule.")

    with h5py.File(args.predictive, "r") as source:
        required = (
            "galaxy_id", "source_h5_row", "sfh_time_grid", "condition",
            "observed_sfh", "sfh_draws",
        )
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"Predictive file is missing datasets: {missing}")
        ids_all = np.asarray(source["galaxy_id"], dtype=np.int64)
        rows_all = np.asarray(source["source_h5_row"], dtype=np.int64)
        condition_all = np.asarray(source["condition"], dtype=np.float64)
        time = np.asarray(source["sfh_time_grid"], dtype=np.float64)
        sfr_source = str(source.attrs.get("sfr_source", ""))
        if sfr_source != "sfh":
            raise ValueError("This intervention requires the SFH-SFR100 conditional model.")
        selected = choose_galaxies(condition_all, args.n_galaxies, args.selection_seed)
        observed = _read_rows(source["observed_sfh"], selected).astype(np.float64)
        candidate_draws = _read_rows(source["sfh_draws"], selected).astype(np.float64)
    ids, rows = ids_all[selected], rows_all[selected]
    condition = condition_all[selected]

    with h5py.File(args.dataset, "r") as source:
        if not np.array_equal(_read_rows(source["galaxy_id"], rows), ids):
            raise ValueError("Predictive rows do not match the source dataset IDs.")
        age_myr = _read_rows(source["sfh_time_norm"], rows).astype(np.float64)
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))

    variants, variant_indices, actual_sfr, relaxed = [], [], [], []
    for index in range(len(ids)):
        draw_sfr = recent_log_sfr(
            candidate_draws[index], condition[index, 0], age_myr[index], time,
        )
        finite = np.all(np.isfinite(candidate_draws[index]), axis=1) & np.isfinite(draw_sfr)
        valid_indices = np.flatnonzero(finite)
        chosen_local, was_relaxed = select_diverse_draws(
            candidate_draws[index, valid_indices], draw_sfr[valid_indices],
            condition[index, 2], time, age_myr[index], args.n_draws,
            args.sfr_tolerance,
        )
        chosen = valid_indices[chosen_local]
        sfhs = np.concatenate([observed[index:index + 1], candidate_draws[index, chosen]])
        variants.append(sfhs)
        variant_indices.append(np.r_[-1, chosen])
        actual_sfr.append(recent_log_sfr(
            sfhs, condition[index, 0], age_myr[index], time,
        ))
        relaxed.append(bool(was_relaxed))
    if len({len(value) for value in variants}) != 1:
        raise ValueError("Not enough valid draws to construct equal-size comparisons.")
    variants = np.asarray(variants, dtype=np.float32)
    variant_indices = np.asarray(variant_indices, dtype=np.int64)
    actual_sfr = np.asarray(actual_sfr, dtype=np.float64)
    n_variants = variants.shape[1]

    log_sfh = np.log10(np.maximum(variants, 0.0) + epsilon).reshape(-1, variants.shape[-1])
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda",
    ):
        sfh_condition = F.normalize(
            clip.encode_sfh(torch.as_tensor(log_sfh, device=device)), dim=1,
        ).float()
    sfh_condition = sfh_condition.reshape(len(ids), n_variants, -1)

    args.output.mkdir(parents=True)
    image_dir = args.output / "images"
    image_dir.mkdir()
    all_pixels = np.empty(
        (len(ids), len(args.image_seeds), n_variants, 224, 224), dtype=np.uint8,
    )
    raw_embeddings, aligned_embeddings = [], []
    for galaxy_index, galaxy_id in enumerate(ids):
        raw_by_seed, aligned_by_seed = [], []
        for seed_index, seed in enumerate(args.image_seeds):
            generated = []
            for variant_index in range(n_variants):
                # Batch size one plus the repeated seed guarantees the exact
                # same initial pixel noise for every SFH variant.
                image = pixel.schedule.sample(
                    pixel.ema, sfh_condition[galaxy_index, variant_index:variant_index + 1],
                    steps=args.steps, guidance=args.guidance, seed=int(seed),
                )
                generated.append(image)
            generated = torch.cat(generated, dim=0)
            pixels = (
                (generated[:, 0].clamp(-1, 1) + 1) * 127.5
            ).round().byte().cpu().numpy()
            all_pixels[galaxy_index, seed_index] = pixels
            encoder_input = ((generated + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                raw = F.normalize(clip.encode_image_latent(encoder_input).float(), dim=1)
                aligned = F.normalize(clip.encode_image(encoder_input).float(), dim=1)
            raw_by_seed.append(raw.cpu().numpy())
            aligned_by_seed.append(aligned.cpu().numpy())
            for variant_index, image in enumerate(pixels):
                label = "observed" if variant_index == 0 else f"draw{variant_indices[galaxy_index, variant_index]}"
                Image.fromarray(image).save(
                    image_dir / f"{galaxy_index + 1:02d}_{int(galaxy_id)}_{label}_seed{seed}.png"
                )
        raw_embeddings.append(raw_by_seed)
        aligned_embeddings.append(aligned_by_seed)
        print(f"Generated and encoded {galaxy_index + 1}/{len(ids)}: {galaxy_id}", flush=True)
    raw_embeddings = np.asarray(raw_embeddings)
    aligned_embeddings = np.asarray(aligned_embeddings)

    # Pairwise table: SFH distances are shared by image seeds; image distances
    # are evaluated separately for each fixed-noise realization.
    pair_records = []
    for galaxy_index, galaxy_id in enumerate(ids):
        sfh_distance = np.mean(
            np.abs(
                np.cumsum(variants[galaxy_index], axis=1)[:, None]
                - np.cumsum(variants[galaxy_index], axis=1)[None]
            ), axis=2,
        )
        for seed_index, seed in enumerate(args.image_seeds):
            raw_distance = cosine_distance_rows(raw_embeddings[galaxy_index, seed_index])
            aligned_distance = cosine_distance_rows(aligned_embeddings[galaxy_index, seed_index])
            for left in range(n_variants):
                for right in range(left + 1, n_variants):
                    pair_records.append({
                        "galaxy_id": int(galaxy_id), "image_seed": int(seed),
                        "left_variant": int(left), "right_variant": int(right),
                        "sfh_w1": float(sfh_distance[left, right]),
                        "zoobot_cosine_distance": float(raw_distance[left, right]),
                        "aligned_cosine_distance": float(aligned_distance[left, right]),
                    })
    with (args.output / "pairwise_distances.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_records[0].keys())
        writer.writeheader()
        writer.writerows(pair_records)

    # Visual comparison pages for the first image-noise seed.
    with PdfPages(args.output / "sfh_draw_image_comparisons.pdf") as pdf:
        for galaxy_index, galaxy_id in enumerate(ids):
            fig, axes = plt.subplots(2, n_variants, figsize=(2.55 * n_variants, 5.0))
            for variant_index in range(n_variants):
                axes[0, variant_index].plot(time, variants[galaxy_index, variant_index], lw=1.4)
                axes[0, variant_index].axvline(
                    min(1e8 / (age_myr[galaxy_index] * 1e6), 1), color="0.5", ls="--", lw=0.8,
                )
                title = "Observed SFH" if variant_index == 0 else f"Draw {variant_indices[galaxy_index, variant_index]}"
                axes[0, variant_index].set_title(
                    f"{title}\nlog SFR100={actual_sfr[galaxy_index, variant_index]:.2f}", fontsize=8,
                )
                axes[0, variant_index].set(xlim=(0, 1), ylim=(0, None))
                axes[1, variant_index].imshow(
                    all_pixels[galaxy_index, 0, variant_index], cmap="gray", vmin=0, vmax=255,
                )
                axes[1, variant_index].axis("off")
            axes[0, 0].set_ylabel("Normalized SFH weight")
            axes[0, 0].set_xlabel("Fractional lookback time")
            fig.suptitle(
                f"ID {galaxy_id} | log M*={condition[galaxy_index, 0]:.2f}, "
                f"z={condition[galaxy_index, 1]:.2f}, target log SFR100={condition[galaxy_index, 2]:.2f}\n"
                f"Fixed image noise seed {args.image_seeds[0]}", fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            pdf.savefig(fig)
            plt.close(fig)

    # PCA maps are descriptive projections of this generated sample only.
    fig, axes = plt.subplots(len(args.image_seeds), 2, figsize=(11, 4.5 * len(args.image_seeds)), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(ids)))
    for seed_index, seed in enumerate(args.image_seeds):
        for column, (name, values) in enumerate((
            ("Frozen ZooBot", raw_embeddings[:, seed_index]),
            ("Aligned image embedding", aligned_embeddings[:, seed_index]),
        )):
            flat = values.reshape(-1, values.shape[-1])
            xy = PCA(n_components=2).fit_transform(flat).reshape(len(ids), n_variants, 2)
            ax = axes[seed_index, column]
            for galaxy_index, galaxy_id in enumerate(ids):
                ax.plot(xy[galaxy_index, :, 0], xy[galaxy_index, :, 1], "-", color=colors[galaxy_index], alpha=0.55)
                ax.scatter(xy[galaxy_index, 1:, 0], xy[galaxy_index, 1:, 1], s=28, color=colors[galaxy_index])
                ax.scatter(xy[galaxy_index, 0, 0], xy[galaxy_index, 0, 1], s=90, marker="*", color=colors[galaxy_index], label=str(galaxy_id))
            ax.set_title(f"{name} | fixed image seed {seed}")
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output / "embedding_motion.pdf", bbox_inches="tight")
    fig.savefig(args.output / "embedding_motion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sfh_distance = np.asarray([row["sfh_w1"] for row in pair_records])
    raw_distance = np.asarray([row["zoobot_cosine_distance"] for row in pair_records])
    aligned_distance = np.asarray([row["aligned_cosine_distance"] for row in pair_records])
    raw_rho = spearmanr(sfh_distance, raw_distance).statistic
    aligned_rho = spearmanr(sfh_distance, aligned_distance).statistic
    figure, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.scatter(sfh_distance, raw_distance, s=13, alpha=0.45, label=f"Frozen ZooBot (rho={raw_rho:.2f})")
    ax.scatter(sfh_distance, aligned_distance, s=13, alpha=0.45, label=f"Aligned (rho={aligned_rho:.2f})")
    ax.set(xlabel="SFH cumulative W1 distance", ylabel="Image-embedding cosine distance")
    ax.grid(alpha=0.2)
    ax.legend()
    figure.tight_layout()
    figure.savefig(args.output / "sfh_vs_image_displacement.pdf", bbox_inches="tight")
    plt.close(figure)

    np.savez_compressed(
        args.output / "intervention_embeddings.npz",
        galaxy_id=ids, source_h5_row=rows, condition=condition,
        sfh=variants, sfh_draw_index=variant_indices, realized_log_sfr100=actual_sfr,
        image_seed=np.asarray(args.image_seeds), zoobot_embedding=raw_embeddings,
        aligned_image_embedding=aligned_embeddings,
    )
    report = {
        "n_galaxies": len(ids), "n_generated_draws_per_galaxy": args.n_draws,
        "n_variants_including_observed": n_variants,
        "image_seeds": args.image_seeds, "guidance": args.guidance,
        "sampling_steps": args.steps, "sfr_tolerance_dex": args.sfr_tolerance,
        "galaxy_ids": ids.tolist(), "target_conditions": condition.tolist(),
        "realized_log_sfr100": actual_sfr.tolist(),
        "sfr_constraint_relaxed": relaxed,
        "mean_pairwise_zoobot_cosine_distance": float(raw_distance.mean()),
        "mean_pairwise_aligned_cosine_distance": float(aligned_distance.mean()),
        "sfh_distance_vs_zoobot_distance_spearman": float(raw_rho),
        "sfh_distance_vs_aligned_distance_spearman": float(aligned_rho),
        "condition_cache_sha256": cache_digest,
        "condition_cache_metadata": metadata,
        "note": (
            "Within each galaxy and image seed, all VIS samples start from identical noise. "
            "The star marks the image conditioned on the observed SFH; circles use conditional "
            "SFH draws. PCA is fitted only for visualization and is not a distance metric."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
