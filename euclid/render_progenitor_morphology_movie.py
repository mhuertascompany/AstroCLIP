"""Generate a fixed-noise morphology movie along a progenitor analogue track."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        group = group.sort_values("rank").head(n_analogues)
        analogue_ids = group.galaxy_id.astype(np.int64).to_numpy()
        absent = [int(gid) for gid in analogue_ids if int(gid) not in lookup]
        if absent:
            raise ValueError(
                f"Stage {stage} has {len(absent)} analogues absent from the condition cache."
            )
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


def nearest_similarity(frame_embeddings, reference, chunk=32768):
    query = np.asarray(frame_embeddings, dtype=np.float32)
    best = np.full(len(query), -np.inf, dtype=np.float32)
    for start in range(0, len(reference), chunk):
        best = np.maximum(best, (query @ reference[start:start + chunk].T).max(axis=1))
    return best


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
    parser.add_argument("--pixel-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-analogues", type=int, default=5)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(args):
    from .train_pixel_diffusion import PixelDiffusion

    for path in (args.candidates, args.condition_cache, args.pixel_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.n_analogues, args.frames, args.batch_size, args.steps, args.fps) < 1:
        raise ValueError("Counts, steps, and fps must be positive.")
    device = torch.device(args.device)
    ids, reference, metadata = load_condition_cache(args.condition_cache)
    descendant_id, anchor_rows = build_track(
        args.candidates, args.descendant_id, ids, reference, args.n_analogues,
    )
    pixel = PixelDiffusion.load_from_checkpoint(
        str(args.pixel_checkpoint), map_location="cpu",
    ).to(device).eval().requires_grad_(False)
    cache_digest = sha256(args.condition_cache)
    if str(pixel.hparams.cache_sha256) != cache_digest:
        raise ValueError("Pixel checkpoint was trained with a different condition cache.")
    if not 2 <= args.steps <= len(pixel.schedule.alpha):
        raise ValueError("Sampling steps lie outside the diffusion schedule.")

    frame_time, frame_fraction, embedding = interpolate_track(
        anchor_rows, args.frames, device,
    )
    nearest = nearest_similarity(embedding.float().cpu().numpy(), reference)
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)
    base_noise = torch.randn((1, 1, 224, 224), device=device, generator=generator)
    pixel_frames = []
    for start in range(0, args.frames, args.batch_size):
        condition = embedding[start:start + args.batch_size]
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
        frame = annotate_frame(
            pixels, index, args.frames, frame_time[index], frame_fraction[index],
            nearest[index], descendant_id,
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

    sample = np.unique(np.linspace(0, args.frames - 1, min(12, args.frames)).round().astype(int))
    figure, axes = plt.subplots(2, int(np.ceil(len(sample) / 2)), figsize=(18, 6), squeeze=False)
    for ax, index in zip(axes.ravel(), sample):
        ax.imshow(pixel_frames[index], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{frame_time[index]:.2f} Gyr | f={frame_fraction[index]:.3f}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(sample):]:
        ax.axis("off")
    figure.suptitle(f"Fixed-noise morphology sequence for descendant {descendant_id}")
    figure.tight_layout()
    figure.savefig(args.output / "contact_sheet.pdf", bbox_inches="tight")
    figure.savefig(args.output / "contact_sheet.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    anchor_time = np.asarray([row["lookback_gyr"] for row in anchor_rows])
    anchor_similarity = np.asarray([
        float(np.max(np.asarray(row["condition"]) @ reference.T)) for row in anchor_rows
    ])
    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(frame_time, nearest, color="#276f9f")
    axes[0].scatter(anchor_time, anchor_similarity, color="black", s=25, zorder=4)
    axes[0].set_ylabel("Nearest real cosine")
    axes[0].axhline(0.9, color="0.5", ls="--", lw=0.8)
    axes[1].plot(frame_time, frame_fraction, color="#278c66")
    axes[1].scatter(anchor_time, [row["formed_mass_fraction"] for row in anchor_rows], color="black", s=25)
    axes[1].set(xlabel="Descendant lookback time [Gyr]", ylabel="Formed mass fraction")
    for ax in axes:
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output / "track_diagnostics.pdf", bbox_inches="tight")
    plt.close(figure)

    np.savez_compressed(
        args.output / "interpolated_track.npz",
        frame_lookback_gyr=frame_time, formed_mass_fraction=frame_fraction,
        aligned_sfh_embedding=embedding.float().cpu().numpy(),
        nearest_real_cosine=nearest,
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
        "sampling_steps": args.steps, "fps": args.fps,
        "condition_cache_sha256": cache_digest,
        "condition_cache_metadata": metadata,
        "minimum_nearest_real_cosine": float(nearest.min()),
        "median_nearest_real_cosine": float(np.median(nearest)),
        "anchors": serializable_anchors,
        "gif": str(args.output / "morphology_track.gif"),
        "mp4": str(mp4) if mp4 is not None else None,
        "interpretation": (
            "A model-generated counterfactual sequence along spherical interpolation of "
            "analogue-centroid aligned SFH embeddings; it is not an observed evolutionary movie."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
