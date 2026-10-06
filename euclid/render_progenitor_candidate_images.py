"""Render exact progenitor-candidate conditions with one shared noise realization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from .prepare_diffusion_conditions import sha256
from .render_progenitor_morphology_movie import load_condition_cache


def load_candidates(path, descendant_id, ids, conditions, n_analogues=5):
    table = pd.read_csv(path)
    required = {
        "descendant_id", "stage", "rank", "galaxy_id",
        "state_lookback_gyr", "formed_mass_fraction",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Candidate table is missing columns: {missing}")
    table = table[table.descendant_id.astype(np.int64) == int(descendant_id)].copy()
    if table.empty:
        raise ValueError(f"No analogue rows found for descendant {descendant_id}.")
    table = (
        table.sort_values(["stage", "rank"])
        .groupby("stage", sort=True, group_keys=False)
        .head(n_analogues)
    )
    lookup = {int(gid): index for index, gid in enumerate(ids)}
    requested = [int(descendant_id), *table.galaxy_id.astype(np.int64).tolist()]
    absent = sorted(set(requested).difference(lookup))
    if absent:
        raise ValueError(f"{len(absent)} requested IDs are absent from the condition cache.")

    records = [{
        "kind": "descendant", "stage": 0, "rank": 0,
        "galaxy_id": int(descendant_id), "lookback_gyr": 0.0,
        "formed_mass_fraction": 1.0,
    }]
    for row in table.itertuples(index=False):
        records.append({
            "kind": "analogue", "stage": int(row.stage), "rank": int(row.rank),
            "galaxy_id": int(row.galaxy_id),
            "lookback_gyr": float(row.state_lookback_gyr),
            "formed_mass_fraction": float(row.formed_mass_fraction),
        })
    vectors = np.stack([conditions[lookup[row["galaxy_id"]]] for row in records])
    return records, vectors.astype(np.float32)


def annotate(pixels, record, rms_from_descendant):
    image = Image.fromarray(pixels).resize((384, 384), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (520, 440), "#111318")
    canvas.paste(image.convert("RGB"), (12, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if record["kind"] == "descendant":
        title = f"Descendant {record['galaxy_id']}"
    else:
        title = f"Stage {record['stage']} · rank {record['rank']} · ID {record['galaxy_id']}"
    draw.text((12, 12), title, fill="white", font=font)
    draw.text(
        (405, 58),
        f"t={record['lookback_gyr']:.2f} Gyr\nf={record['formed_mass_fraction']:.3f}\n"
        f"RMS vs desc.\n{rms_from_descendant:.4f}",
        fill="#d7dde8", font=font, spacing=5,
    )
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--descendant-id", type=int, required=True)
    parser.add_argument("--condition-cache", type=Path, required=True)
    parser.add_argument("--pixel-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-analogues", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=314159)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(args):
    from .train_pixel_diffusion import PixelDiffusion

    for path in (args.candidates, args.condition_cache, args.pixel_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.n_analogues, args.batch_size, args.steps) < 1:
        raise ValueError("Counts and sampling steps must be positive.")

    device = torch.device(args.device)
    ids, reference, metadata = load_condition_cache(args.condition_cache)
    records, condition = load_candidates(
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

    condition_tensor = torch.as_tensor(condition, device=device)
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)
    base_noise = torch.randn((1, 1, 224, 224), device=device, generator=generator)
    generated = []
    for start in range(0, len(records), args.batch_size):
        current = condition_tensor[start:start + args.batch_size]
        images = pixel.schedule.sample(
            pixel.ema, current, steps=args.steps, guidance=args.guidance,
            initial_noise=base_noise.repeat(len(current), 1, 1, 1),
        )
        pixels = ((images[:, 0].clamp(-1, 1) + 1) * 127.5).cpu().numpy()
        generated.extend(pixels)
        print(f"Generated {start + 1}-{start + len(current)}/{len(records)}", flush=True)
    generated = np.asarray(generated, dtype=np.float32)
    descendant = generated[0]
    rms = np.sqrt(np.mean((generated - descendant[None]) ** 2, axis=(1, 2))) / 255.0

    args.output.mkdir(parents=True)
    image_dir = args.output / "images"
    image_dir.mkdir()
    for pixels, record, difference in zip(generated, records, rms):
        name = (
            f"descendant_{record['galaxy_id']}.png" if record["kind"] == "descendant"
            else f"stage_{record['stage']:02d}_rank_{record['rank']}_id_{record['galaxy_id']}.png"
        )
        annotate(pixels.round().astype(np.uint8), record, float(difference)).save(image_dir / name)

    # One page per epoch shows every exact candidate embedding at that epoch.
    with PdfPages(args.output / "candidate_images_by_stage.pdf") as pdf:
        descendant_page = plt.figure(figsize=(4, 4), layout="constrained")
        ax = descendant_page.add_subplot()
        ax.imshow(descendant, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Descendant {args.descendant_id}\nshared noise seed={args.noise_seed}")
        ax.axis("off")
        pdf.savefig(descendant_page, dpi=180)
        plt.close(descendant_page)
        stages = sorted({row["stage"] for row in records if row["kind"] == "analogue"}, reverse=True)
        for stage in stages:
            selected = [i for i, row in enumerate(records) if row["stage"] == stage and row["kind"] == "analogue"]
            fig, axes = plt.subplots(1, len(selected), figsize=(3 * len(selected), 3.35), squeeze=False, layout="constrained")
            for ax, index in zip(axes[0], selected):
                row = records[index]
                ax.imshow(generated[index], cmap="gray", vmin=0, vmax=255)
                ax.set_title(
                    f"rank {row['rank']} · {row['galaxy_id']}\n"
                    f"RMS(desc)={rms[index]:.4f}", fontsize=8,
                )
                ax.axis("off")
            row = records[selected[0]]
            fig.suptitle(
                f"Exact candidate conditions · t={row['lookback_gyr']:.2f} Gyr · "
                f"formed fraction={row['formed_mass_fraction']:.3f}", fontsize=11,
            )
            pdf.savefig(fig, dpi=180)
            plt.close(fig)

    # A compact forward-time sequence containing only the rank-1 candidates.
    top = [i for i, row in enumerate(records) if row["kind"] == "analogue" and row["rank"] == 1]
    top.sort(key=lambda i: records[i]["lookback_gyr"], reverse=True)
    top.append(0)
    fig, axes = plt.subplots(2, int(np.ceil(len(top) / 2)), figsize=(18, 6), squeeze=False, layout="constrained")
    for ax, index in zip(axes.ravel(), top):
        row = records[index]
        ax.imshow(generated[index], cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{row['lookback_gyr']:.2f} Gyr · ID {row['galaxy_id']}\nRMS(desc)={rms[index]:.4f}",
            fontsize=7.5,
        )
        ax.axis("off")
    for ax in axes.ravel()[len(top):]:
        ax.axis("off")
    fig.suptitle(f"No interpolation: rank-1 progenitor conditions → descendant · noise seed {args.noise_seed}")
    fig.savefig(args.output / "rank1_discrete_sequence.pdf", dpi=190, bbox_inches="tight")
    fig.savefig(args.output / "rank1_discrete_sequence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    output = pd.DataFrame(records)
    output["condition_cosine_to_descendant"] = condition @ condition[0]
    output["pixel_rms_from_descendant"] = rms
    output.to_csv(args.output / "candidate_image_metrics.csv", index=False)
    report = {
        "descendant_id": args.descendant_id,
        "n_images": len(records),
        "n_analogue_images": len(records) - 1,
        "noise_seed": args.noise_seed,
        "guidance": args.guidance,
        "sampling_steps": args.steps,
        "condition_cache_sha256": cache_digest,
        "condition_cache_metadata": metadata,
        "median_pixel_rms_from_descendant": float(np.median(rms[1:])),
        "maximum_pixel_rms_from_descendant": float(np.max(rms[1:])),
        "note": "Each image uses an exact candidate condition and identical initial noise; no centroiding or interpolation is applied.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
