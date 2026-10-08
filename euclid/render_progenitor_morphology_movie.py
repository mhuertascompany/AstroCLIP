"""Generate morphology movies along a progenitor analogue track."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .prepare_diffusion_conditions import sha256


def slerp(left, right, fraction):
    """Spherical interpolation between batches of unit vectors."""
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    dot = (left * right).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    sine = torch.sin(angle)
    fraction = torch.as_tensor(fraction, device=left.device, dtype=left.dtype)
    while fraction.ndim < left.ndim:
        fraction = fraction.unsqueeze(-1)
    linear = F.normalize((1 - fraction) * left + fraction * right, dim=-1)
    spherical = (
        torch.sin((1 - fraction) * angle) / sine.clamp_min(1e-7) * left
        + torch.sin(fraction * angle) / sine.clamp_min(1e-7) * right
    )
    return torch.where(sine.abs() < 1e-6, linear, spherical)


def load_condition_cache(path):
    with np.load(path, allow_pickle=False) as source:
        ids = np.concatenate([source["train_ids"], source["val_ids"]]).astype(np.int64)
        conditions = np.concatenate([
            source["train_condition"], source["val_condition"],
        ]).astype(np.float32)
        metadata = json.loads(str(source["metadata"]))
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Condition cache contains duplicate galaxy IDs.")
    if not np.allclose(np.linalg.norm(conditions, axis=1), 1, atol=1e-4):
        raise ValueError("Condition cache contains non-unit embeddings.")
    return ids, conditions, metadata


def build_track(candidates_path, descendant_id, ids, conditions, n_analogues=5):
    table = pd.read_csv(candidates_path)
    if "descendant_id" in table:
        available = table.descendant_id.astype(np.int64).unique()
        if descendant_id is None:
            if len(available) != 1:
                raise ValueError(
                    "The candidate table contains multiple descendants; pass --descendant-id."
                )
            descendant_id = int(available[0])
        table = table[table.descendant_id.astype(np.int64) == int(descendant_id)]
    elif descendant_id is None:
        raise ValueError("Pass --descendant-id for an individual analogue table.")
    if table.empty:
        raise ValueError(f"No analogue rows found for descendant {descendant_id}.")
    required = {"stage", "rank", "galaxy_id", "state_lookback_gyr", "formed_mass_fraction"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Candidate table is missing columns: {missing}")

    lookup = {int(gid): index for index, gid in enumerate(ids)}
    if int(descendant_id) not in lookup:
        raise ValueError("Descendant is absent from the pixel-diffusion condition cache.")
    anchor_rows = [{
        "stage": 0, "lookback_gyr": 0.0, "formed_mass_fraction": 1.0,
        "n_analogues": 1, "galaxy_ids": [int(descendant_id)],
        "condition": conditions[lookup[int(descendant_id)]],
    }]
    for stage, group in table.groupby("stage", sort=True):
        # A track may have been searched in a larger parent sample than the
        # diffusion cache (for example, before an edge-on ablation). Select
        # the best-ranked candidates that the trained generator can actually
        # condition on.
        group = group[
            group.galaxy_id.astype(np.int64).map(lambda value: int(value) in lookup)
        ].sort_values("rank").head(n_analogues)
        if group.empty:
            raise ValueError(
                f"Stage {stage} has no analogue present in the condition cache."
            )
        analogue_ids = group.galaxy_id.astype(np.int64).to_numpy()
        vectors = conditions[[lookup[int(gid)] for gid in analogue_ids]]
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError(f"Degenerate embedding centroid at stage {stage}.")
        anchor_rows.append({
            "stage": int(stage),
            "lookback_gyr": float(group.state_lookback_gyr.median()),
            "formed_mass_fraction": float(group.formed_mass_fraction.median()),
            "n_analogues": len(group),
            "galaxy_ids": analogue_ids.astype(int).tolist(),
            "condition": (centroid / norm).astype(np.float32),
        })
    anchor_rows.sort(key=lambda row: row["lookback_gyr"])
    times = np.asarray([row["lookback_gyr"] for row in anchor_rows])
    if np.any(np.diff(times) <= 0):
        raise ValueError("Track lookback times must be unique and increasing.")
    return int(descendant_id), anchor_rows


def interpolate_track(anchor_rows, frames, device):
    anchor_time = np.asarray([row["lookback_gyr"] for row in anchor_rows])
    anchor_fraction = np.asarray([row["formed_mass_fraction"] for row in anchor_rows])
    anchor = torch.as_tensor(
        np.stack([row["condition"] for row in anchor_rows]), device=device,
    )
    # A movie runs forward in cosmic time: earliest available progenitor to descendant.
    frame_time = np.linspace(anchor_time[-1], 0.0, frames)
    ascending_time = frame_time[::-1].copy()
    indices = np.searchsorted(anchor_time, ascending_time, side="right") - 1
    indices = np.clip(indices, 0, len(anchor_time) - 2)
    denominator = anchor_time[indices + 1] - anchor_time[indices]
    local = (ascending_time - anchor_time[indices]) / denominator
    embedding = slerp(anchor[indices], anchor[indices + 1], torch.as_tensor(local, device=device))
    embedding = embedding.flip(0)
    frame_fraction = np.interp(ascending_time, anchor_time, anchor_fraction)[::-1]
    return frame_time, frame_fraction, embedding


def nearest_reference(frame_embeddings, reference, chunk=32768):
    query = np.asarray(frame_embeddings, dtype=np.float32)
    best = np.full(len(query), -np.inf, dtype=np.float32)
    best_index = np.full(len(query), -1, dtype=np.int64)
    for start in range(0, len(reference), chunk):
        similarity = query @ reference[start:start + chunk].T
        local_index = similarity.argmax(axis=1)
        local_best = similarity[np.arange(len(query)), local_index]
        update = local_best > best
        best[update] = local_best[update]
        best_index[update] = start + local_index[update]
    return best, best_index


def nearest_similarity(frame_embeddings, reference, chunk=32768):
    """Return only the cosine value for backward compatibility."""
    return nearest_reference(frame_embeddings, reference, chunk=chunk)[0]


def load_descendant_sfh(path, descendant_id):
    """Load one normalized median SFH on a physical lookback-time grid."""
    with h5py.File(path, "r") as source:
        ids = np.asarray(source["galaxy_id"], dtype=np.int64)
        match = np.flatnonzero(ids == int(descendant_id))
        if len(match) != 1:
            raise ValueError(
                f"Expected one SFH row for descendant {descendant_id}; found {len(match)}."
            )
        row = int(match[0])
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))
        weights = np.maximum(10.0 ** np.asarray(source["sfh"][row], dtype=float) - epsilon, 0)
        weights /= weights.sum()
        fractional_time = np.asarray(source["sfh_time_grid"], dtype=float)
        time_norm_gyr = float(source["sfh_time_norm"][row]) / 1000.0
    edges = np.empty(len(fractional_time) + 1, dtype=float)
    edges[1:-1] = 0.5 * (fractional_time[:-1] + fractional_time[1:])
    edges[0], edges[-1] = 0.0, 1.0
    widths_gyr = np.diff(edges) * time_norm_gyr
    return fractional_time * time_norm_gyr, weights / widths_gyr, time_norm_gyr


def load_condition_umap(path, condition_ids, cache_digest):
    """Load and align a UMAP fitted to the exact diffusion-condition cache."""
    with np.load(path, allow_pickle=False) as source:
        ids = np.asarray(source["galaxy_id"], dtype=np.int64)
        xy = np.asarray(source["xy"], dtype=np.float32)
        metadata = json.loads(str(source["metadata"]))
    if metadata.get("condition_cache_sha256") != cache_digest:
        raise ValueError("Condition UMAP was fitted to a different condition cache.")
    if xy.shape != (len(ids), 2) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid condition UMAP arrays.")
    lookup = {int(gid): index for index, gid in enumerate(ids)}
    positions = np.array([lookup.get(int(gid), -1) for gid in condition_ids])
    if np.any(positions < 0):
        raise ValueError("Condition UMAP does not contain every condition-cache ID.")
    return xy[positions], metadata


def umap_density(xy, bins=320):
    """Build a reusable density background for fast per-frame UMAP panels."""
    xy = np.asarray(xy, dtype=float)
    low = np.nanpercentile(xy, 0.2, axis=0)
    high = np.nanpercentile(xy, 99.8, axis=0)
    padding = np.maximum((high - low) * 0.04, 1e-3)
    low, high = low - padding, high + padding
    density, x_edges, y_edges = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=bins,
        range=((low[0], high[0]), (low[1], high[1])),
    )
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    return np.log1p(density.T), extent


def evolution_frame(
    pixels, index, total, lookback, mass_fraction, nearest, descendant_id,
    analogue_id, noise_seed, sfh_time, sfh_rate, time_norm_gyr,
    track_xy=None, density=None, extent=None, image_label="Generated morphology",
    morphology_id=None, morphology_similarity=None,
):
    """Render a generated stamp beside the progressively revealed descendant SFH."""
    has_umap = track_xy is not None
    figure = plt.figure(
        figsize=(15.5 if has_umap else 11, 5.2), dpi=100, facecolor="#111318",
    )
    widths = (1.0, 1.25, 1.12) if has_umap else (1.0, 1.25)
    grid = figure.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.20)
    image_axis = figure.add_subplot(grid[0, 0])
    sfh_axis = figure.add_subplot(grid[0, 1])
    image_axis.imshow(pixels, cmap="gray", vmin=0, vmax=255)
    image_axis.axis("off")
    subtitle = f"SFH state ID {analogue_id}"
    if morphology_id is not None:
        subtitle += f" · morphology donor {morphology_id}"
    if noise_seed is not None:
        subtitle += f" · noise seed {noise_seed}"
    image_axis.set_title(f"{image_label}\n{subtitle}", color="white", fontsize=10)

    revealed = sfh_time >= lookback - 1e-10
    sfh_axis.plot(sfh_time, sfh_rate, color="#657080", lw=1.2, alpha=0.35,
                  label="descendant SFH not yet reached")
    sfh_axis.plot(
        sfh_time[revealed], sfh_rate[revealed], color="#4fd09b", lw=2.4,
        label="history formed by this epoch",
    )
    sfh_axis.axvline(lookback, color="#ffcf66", lw=1.5, ls="--",
                     label="current track epoch")
    sfh_axis.axvspan(0, lookback, color="#111318", alpha=0.34, lw=0)
    sfh_axis.set_xlim(0, time_norm_gyr)
    sfh_axis.set_ylim(bottom=0)
    sfh_axis.set_xlabel("Time before descendant observation [Gyr]", color="white")
    sfh_axis.set_ylabel("Normalized SFR [fraction Gyr$^{-1}$]", color="white")
    sfh_axis.tick_params(colors="white")
    sfh_axis.grid(alpha=0.15, color="white")
    for spine in sfh_axis.spines.values():
        spine.set_color("#737b88")
    sfh_axis.set_facecolor("#111318")
    legend = sfh_axis.legend(loc="upper right", fontsize=8, framealpha=0.75)
    legend.get_frame().set_facecolor("#222630")
    for text in legend.get_texts():
        text.set_color("white")
    similarity_text = f"SFH snap cosine={nearest:.3f}"
    if morphology_similarity is not None:
        similarity_text += f" · SFH→morph cosine={morphology_similarity:.3f}"
    sfh_axis.set_title(
        f"Descendant {descendant_id} · frame {index + 1}/{total}\n"
        f"lookback={lookback:.2f} Gyr · formed mass={mass_fraction:.3f} · "
        f"{similarity_text}",
        color="white", fontsize=10,
    )
    if has_umap:
        umap_axis = figure.add_subplot(grid[0, 2])
        umap_axis.imshow(
            density, origin="lower", extent=extent, cmap="Greys", aspect="auto",
            interpolation="nearest", alpha=0.68,
        )
        umap_axis.plot(
            track_xy[:, 0], track_xy[:, 1], color="#788291", lw=1.0, alpha=0.5,
        )
        umap_axis.plot(
            track_xy[:index + 1, 0], track_xy[:index + 1, 1],
            color="#4fd09b", lw=2.0,
        )
        umap_axis.scatter(
            track_xy[index, 0], track_xy[index, 1], marker="*", s=130,
            color="#ffcf66", edgecolor="black", linewidth=0.6, zorder=5,
        )
        umap_axis.set_xlim(extent[0], extent[1])
        umap_axis.set_ylim(extent[2], extent[3])
        umap_axis.set_xlabel("UMAP 1", color="white")
        umap_axis.set_ylabel("UMAP 2", color="white")
        umap_axis.tick_params(colors="white")
        for spine in umap_axis.spines.values():
            spine.set_color("#737b88")
        umap_axis.set_facecolor("#111318")
        umap_axis.set_title(
            "AE-adjacency aligned-SFH UMAP\nexact track-condition space",
            color="white", fontsize=10,
        )
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return Image.fromarray(rgba[:, :, :3])


def annotate_frame(pixels, index, total, lookback, mass_fraction, nearest, descendant_id):
    image = Image.fromarray(pixels).resize((384, 384), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (640, 450), "#111318")
    canvas.paste(image.convert("RGB"), (16, 34))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((416, 44), f"Descendant {descendant_id}", fill="white", font=font)
    draw.text((416, 82), f"Frame {index + 1}/{total}", fill="#d7dde8", font=font)
    draw.text((416, 112), f"Lookback: {lookback:.2f} Gyr", fill="#d7dde8", font=font)
    draw.text((416, 142), f"Formed mass: {mass_fraction:.3f}", fill="#d7dde8", font=font)
    draw.text((416, 172), f"Nearest real cosine: {nearest:.3f}", fill="#d7dde8", font=font)
    left, top, right, bottom = 416, 224, 616, 246
    draw.rectangle((left, top, right, bottom), outline="#707887", width=1)
    progress = (index + 1) / total
    draw.rectangle((left + 2, top + 2, left + 2 + int((right - left - 4) * progress), bottom - 2), fill="#4cae8a")
    draw.text((416, 258), "earlier                         descendant", fill="#aeb6c4", font=font)
    draw.text((16, 12), "Fixed-noise VIS diffusion along interpolated aligned-SFH track", fill="white", font=font)
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="analogue_candidates.csv from progenitor tracking.")
    parser.add_argument("--descendant-id", type=int)
    parser.add_argument("--condition-cache", type=Path, required=True)
    parser.add_argument(
        "--generation-condition-cache", type=Path,
        help=("Optional second cache used by the pixel generator. Dense track "
              "states remain defined in the SFH cache and retrieve their "
              "nearest aligned condition from this cache."),
    )
    parser.add_argument("--pixel-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-analogues", type=int, default=5)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument(
        "--snap-to-reference", action="store_true",
        help="Use the nearest real full-cache condition at every dense track point.",
    )
    parser.add_argument(
        "--independent-noise", action="store_true",
        help="Use noise-seed + frame index instead of shared noise at every epoch.",
    )
    parser.add_argument(
        "--sfh-dataset", type=Path,
        help="Source HDF5 used to add a progressively revealed descendant-SFH panel.",
    )
    parser.add_argument(
        "--condition-umap", type=Path,
        help="Full UMAP fitted to the exact diffusion-condition cache.",
    )
    parser.add_argument(
        "--real-stamps", type=Path,
        help="If given, also render a movie from the selected real VIS stamps.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(args):
    from .train_pixel_diffusion import PixelDiffusion

    paths = [args.candidates, args.condition_cache, args.pixel_checkpoint]
    if args.generation_condition_cache is not None:
        paths.append(args.generation_condition_cache)
    if args.sfh_dataset is not None:
        paths.append(args.sfh_dataset)
    if args.condition_umap is not None:
        paths.append(args.condition_umap)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.n_analogues, args.frames, args.batch_size, args.steps, args.fps) < 1:
        raise ValueError("Counts, steps, and fps must be positive.")
    device = torch.device(args.device)
    ids, reference, metadata = load_condition_cache(args.condition_cache)
    if metadata.get("modality") not in {None, "sfh"}:
        raise ValueError("The track condition cache must contain aligned SFH embeddings.")
    descendant_id, anchor_rows = build_track(
        args.candidates, args.descendant_id, ids, reference, args.n_analogues,
    )
    pixel = PixelDiffusion.load_from_checkpoint(
        str(args.pixel_checkpoint), map_location="cpu",
    ).to(device).eval().requires_grad_(False)
    cache_digest = sha256(args.condition_cache)
    if args.generation_condition_cache is None:
        generation_ids = ids
        generation_reference = reference
        generation_metadata = metadata
        generation_cache_digest = cache_digest
    else:
        generation_ids, generation_reference, generation_metadata = load_condition_cache(
            args.generation_condition_cache,
        )
        if generation_metadata.get("modality") != "image":
            raise ValueError(
                "The generation condition cache must contain aligned image embeddings."
            )
        if generation_reference.shape[1] != reference.shape[1]:
            raise ValueError("SFH and image alignment spaces have different dimensions.")
        for key in ("checkpoint_sha256", "split_sha256"):
            if generation_metadata.get(key) != metadata.get(key):
                raise ValueError(
                    f"SFH and image condition caches use different {key}."
                )
        generation_cache_digest = sha256(args.generation_condition_cache)
    if str(pixel.hparams.cache_sha256) != generation_cache_digest:
        raise ValueError("Pixel checkpoint was trained with a different condition cache.")
    if not 2 <= args.steps <= len(pixel.schedule.alpha):
        raise ValueError("Sampling steps lie outside the diffusion schedule.")
    if args.real_stamps is not None and not args.real_stamps.is_dir():
        raise FileNotFoundError(args.real_stamps)

    frame_time, frame_fraction, track_embedding = interpolate_track(
        anchor_rows, args.frames, device,
    )
    nearest, nearest_index = nearest_reference(
        track_embedding.float().cpu().numpy(), reference,
    )
    analogue_ids = ids[nearest_index]
    if args.snap_to_reference:
        track_embedding = torch.as_tensor(reference[nearest_index], device=device)
        # Exact references have unit self-cosine; retain the target-to-reference
        # similarity above as a diagnostic of the snapping approximation.
    if args.generation_condition_cache is None:
        morphology_similarity = nearest.copy()
        morphology_ids = analogue_ids.copy()
        generation_embedding = track_embedding
    else:
        morphology_similarity, morphology_index = nearest_reference(
            track_embedding.float().cpu().numpy(), generation_reference,
        )
        morphology_ids = generation_ids[morphology_index]
        generation_embedding = torch.as_tensor(
            generation_reference[morphology_index], device=device,
        )
    if args.sfh_dataset is not None:
        sfh_time, sfh_rate, sfh_time_norm = load_descendant_sfh(
            args.sfh_dataset, descendant_id,
        )
    else:
        sfh_time = sfh_rate = sfh_time_norm = None
    if args.condition_umap is not None:
        reference_xy, umap_metadata = load_condition_umap(
            args.condition_umap, ids, cache_digest,
        )
        track_xy = reference_xy[nearest_index]
        density, extent = umap_density(reference_xy)
    else:
        reference_xy = track_xy = density = extent = umap_metadata = None
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)
    base_noise = torch.randn((1, 1, 224, 224), device=device, generator=generator)
    frame_seeds = (
        np.arange(args.frames, dtype=np.int64) + args.noise_seed
        if args.independent_noise
        else np.full(args.frames, args.noise_seed, dtype=np.int64)
    )
    pixel_frames = []
    for start in range(0, args.frames, args.batch_size):
        condition = generation_embedding[start:start + args.batch_size]
        if args.independent_noise:
            noise = []
            for seed in frame_seeds[start:start + len(condition)]:
                current_generator = torch.Generator(device=device).manual_seed(int(seed))
                noise.append(torch.randn(
                    (1, 1, 224, 224), device=device, generator=current_generator,
                ))
            initial_noise = torch.cat(noise)
        else:
            initial_noise = base_noise.repeat(len(condition), 1, 1, 1)
        generated = pixel.schedule.sample(
            pixel.ema, condition, steps=args.steps, guidance=args.guidance,
            initial_noise=initial_noise,
        )
        pixels = (
            (generated[:, 0].clamp(-1, 1) + 1) * 127.5
        ).round().byte().cpu().numpy()
        pixel_frames.extend(pixels)
        print(f"Generated frames {start + 1}-{start + len(condition)}/{args.frames}", flush=True)

    args.output.mkdir(parents=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir()
    annotated = []
    for index, pixels in enumerate(pixel_frames):
        if args.sfh_dataset is None:
            frame = annotate_frame(
                pixels, index, args.frames, frame_time[index], frame_fraction[index],
                nearest[index], descendant_id,
            )
        else:
            frame = evolution_frame(
                pixels, index, args.frames, frame_time[index], frame_fraction[index],
                nearest[index], descendant_id, int(analogue_ids[index]),
                int(frame_seeds[index]), sfh_time, sfh_rate, sfh_time_norm,
                track_xy=track_xy, density=density, extent=extent,
                morphology_id=(
                    int(morphology_ids[index])
                    if args.generation_condition_cache is not None else None
                ),
                morphology_similarity=(
                    float(morphology_similarity[index])
                    if args.generation_condition_cache is not None else None
                ),
            )
        frame.save(frame_dir / f"frame_{index:04d}.png")
        annotated.append(frame)
    annotated[0].save(
        args.output / "morphology_track.gif", save_all=True,
        append_images=annotated[1:], duration=round(1000 / args.fps), loop=0,
        optimize=False,
    )
    mp4 = None
    if shutil.which("ffmpeg"):
        mp4 = args.output / "morphology_track.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
            "-i", str(frame_dir / "frame_%04d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(mp4),
        ], check=True)

    real_gif = real_mp4 = None
    if args.real_stamps is not None:
        real_frame_dir = args.output / "real_frames"
        real_frame_dir.mkdir()
        real_frames = []
        for index, galaxy_id in enumerate(morphology_ids):
            stamp = args.real_stamps / f"VIS_{int(galaxy_id)}.jpg"
            if not stamp.is_file():
                raise FileNotFoundError(stamp)
            with Image.open(stamp) as source:
                real_pixels = np.asarray(source.convert("L"))
            frame = evolution_frame(
                real_pixels, index, args.frames, frame_time[index],
                frame_fraction[index], nearest[index], descendant_id,
                int(analogue_ids[index]), None, sfh_time, sfh_rate,
                sfh_time_norm, track_xy=track_xy, density=density, extent=extent,
                image_label=(
                    "Real retrieved morphology donor"
                    if args.generation_condition_cache is not None
                    else "Real nearest analogue"
                ),
                morphology_id=(
                    int(galaxy_id)
                    if args.generation_condition_cache is not None else None
                ),
                morphology_similarity=(
                    float(morphology_similarity[index])
                    if args.generation_condition_cache is not None else None
                ),
            )
            frame.save(real_frame_dir / f"frame_{index:04d}.png")
            real_frames.append(frame)
        real_gif = args.output / "real_morphology_track.gif"
        real_frames[0].save(
            real_gif, save_all=True, append_images=real_frames[1:],
            duration=round(1000 / args.fps), loop=0, optimize=False,
        )
        if shutil.which("ffmpeg"):
            real_mp4 = args.output / "real_morphology_track.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                "-i", str(real_frame_dir / "frame_%04d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(real_mp4),
            ], check=True)

    sfh_real_gif = sfh_real_mp4 = None
    if args.real_stamps is not None and args.generation_condition_cache is not None:
        sfh_real_frame_dir = args.output / "real_sfh_frames"
        sfh_real_frame_dir.mkdir()
        sfh_real_frames = []
        for index, galaxy_id in enumerate(analogue_ids):
            stamp = args.real_stamps / f"VIS_{int(galaxy_id)}.jpg"
            if not stamp.is_file():
                raise FileNotFoundError(stamp)
            with Image.open(stamp) as source:
                real_pixels = np.asarray(source.convert("L"))
            frame = evolution_frame(
                real_pixels, index, args.frames, frame_time[index],
                frame_fraction[index], nearest[index], descendant_id,
                int(galaxy_id), None, sfh_time, sfh_rate, sfh_time_norm,
                track_xy=track_xy, density=density, extent=extent,
                image_label="Real SFH-track analogue",
                morphology_id=int(morphology_ids[index]),
                morphology_similarity=float(morphology_similarity[index]),
            )
            frame.save(sfh_real_frame_dir / f"frame_{index:04d}.png")
            sfh_real_frames.append(frame)
        sfh_real_gif = args.output / "real_sfh_track.gif"
        sfh_real_frames[0].save(
            sfh_real_gif, save_all=True, append_images=sfh_real_frames[1:],
            duration=round(1000 / args.fps), loop=0, optimize=False,
        )
        if shutil.which("ffmpeg"):
            sfh_real_mp4 = args.output / "real_sfh_track.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                "-i", str(sfh_real_frame_dir / "frame_%04d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(sfh_real_mp4),
            ], check=True)

    sample = np.unique(np.linspace(0, args.frames - 1, min(12, args.frames)).round().astype(int))
    figure, axes = plt.subplots(2, int(np.ceil(len(sample) / 2)), figsize=(18, 6), squeeze=False)
    for ax, index in zip(axes.ravel(), sample):
        ax.imshow(pixel_frames[index], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{frame_time[index]:.2f} Gyr | f={frame_fraction[index]:.3f}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(sample):]:
        ax.axis("off")
    figure.suptitle(
        f"{'Nearest-real' if args.snap_to_reference else 'Interpolated'} SFH track → "
        f"{'nearest aligned morphology' if args.generation_condition_cache else 'SFH condition'} "
        f"for descendant {descendant_id}"
    )
    figure.tight_layout()
    figure.savefig(args.output / "contact_sheet.pdf", bbox_inches="tight")
    figure.savefig(args.output / "contact_sheet.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    anchor_time = np.asarray([row["lookback_gyr"] for row in anchor_rows])
    anchor_similarity = np.asarray([
        float(np.max(np.asarray(row["condition"]) @ reference.T)) for row in anchor_rows
    ])
    diagnostic_rows = 3 if args.generation_condition_cache is not None else 2
    figure, axes = plt.subplots(
        diagnostic_rows, 1, figsize=(7, 8 if diagnostic_rows == 3 else 6),
        sharex=True,
    )
    axes[0].plot(frame_time, nearest, color="#276f9f")
    axes[0].scatter(anchor_time, anchor_similarity, color="black", s=25, zorder=4)
    axes[0].set_ylabel("Nearest real cosine")
    axes[0].axhline(0.9, color="0.5", ls="--", lw=0.8)
    fraction_axis = axes[-1]
    if diagnostic_rows == 3:
        axes[1].plot(frame_time, morphology_similarity, color="#9b55b5")
        axes[1].set_ylabel("SFH→morph cosine")
    fraction_axis.plot(frame_time, frame_fraction, color="#278c66")
    fraction_axis.scatter(
        anchor_time, [row["formed_mass_fraction"] for row in anchor_rows],
        color="black", s=25,
    )
    fraction_axis.set(
        xlabel="Descendant lookback time [Gyr]", ylabel="Formed mass fraction",
    )
    for ax in axes:
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output / "track_diagnostics.pdf", bbox_inches="tight")
    plt.close(figure)

    track_name = "snapped_track.npz" if args.snap_to_reference else "interpolated_track.npz"
    np.savez_compressed(
        args.output / track_name,
        frame_lookback_gyr=frame_time, formed_mass_fraction=frame_fraction,
        aligned_sfh_embedding=track_embedding.float().cpu().numpy(),
        aligned_generation_embedding=generation_embedding.float().cpu().numpy(),
        nearest_real_cosine=nearest,
        nearest_real_id=analogue_ids,
        nearest_morphology_cosine=morphology_similarity,
        nearest_morphology_id=morphology_ids,
        frame_noise_seed=frame_seeds,
        condition_umap_xy=(track_xy if track_xy is not None else np.empty((0, 2))),
        anchor_lookback_gyr=anchor_time,
        anchor_embedding=np.stack([row["condition"] for row in anchor_rows]),
    )
    serializable_anchors = [
        {key: value for key, value in row.items() if key != "condition"}
        for row in anchor_rows
    ]
    report = {
        "descendant_id": descendant_id, "candidate_table": str(args.candidates),
        "n_anchors": len(anchor_rows), "n_frames": args.frames,
        "n_analogues_per_anchor": args.n_analogues,
        "noise_seed": args.noise_seed, "guidance": args.guidance,
        "independent_noise": args.independent_noise,
        "snap_to_reference": args.snap_to_reference,
        "sfh_dataset": str(args.sfh_dataset) if args.sfh_dataset else None,
        "condition_umap": str(args.condition_umap) if args.condition_umap else None,
        "condition_umap_metadata": umap_metadata,
        "real_stamps": str(args.real_stamps) if args.real_stamps else None,
        "sampling_steps": args.steps, "fps": args.fps,
        "condition_cache_sha256": cache_digest,
        "condition_cache_metadata": metadata,
        "generation_condition_cache": (
            str(args.generation_condition_cache)
            if args.generation_condition_cache is not None else None
        ),
        "generation_condition_cache_sha256": generation_cache_digest,
        "generation_condition_cache_metadata": generation_metadata,
        "minimum_nearest_real_cosine": float(nearest.min()),
        "median_nearest_real_cosine": float(np.median(nearest)),
        "unique_nearest_real_objects": int(len(np.unique(analogue_ids))),
        "minimum_sfh_to_morphology_cosine": float(morphology_similarity.min()),
        "median_sfh_to_morphology_cosine": float(np.median(morphology_similarity)),
        "unique_morphology_donors": int(len(np.unique(morphology_ids))),
        "anchors": serializable_anchors,
        "gif": str(args.output / "morphology_track.gif"),
        "mp4": str(mp4) if mp4 is not None else None,
        "real_gif": str(real_gif) if real_gif is not None else None,
        "real_mp4": str(real_mp4) if real_mp4 is not None else None,
        "real_sfh_gif": str(sfh_real_gif) if sfh_real_gif is not None else None,
        "real_sfh_mp4": str(sfh_real_mp4) if sfh_real_mp4 is not None else None,
        "interpretation": (
            "A model-generated counterfactual sequence. Dense target points follow the "
            "analogue track; with snap_to_reference they are rendered through the nearest "
            "real full-cache SFH condition rather than an interpolated condition. When a "
            "generation cache is supplied, each SFH state retrieves its nearest aligned "
            "image embedding, which conditions the pixel generator. It is not an observed "
            "evolutionary movie."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
