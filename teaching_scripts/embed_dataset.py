from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from teaching_scripts.data_utils import build_multimodal_dataloader, load_astroclip_dataset
from teaching_scripts.models import ImageAutoencoder, SmallCLIPModel, SpectrumAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed the AstroCLIP dataset using a trained CLIP model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path or HuggingFace identifier for the dataset.")
    parser.add_argument("--clip-ckpt", type=Path, required=True, help="Checkpoint from train_clip_alignment.py.")
    parser.add_argument("--image-ckpt", type=Path, required=True, help="Image encoder checkpoint.")
    parser.add_argument("--spectrum-ckpt", type=Path, required=True, help="Spectrum encoder checkpoint.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to embed.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/embeddings.h5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_astroclip_dataset(args.dataset)
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not present in dataset.")

    dataloader = build_multimodal_dataloader(
        ds[args.split],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    image_encoder = ImageAutoencoder.load_from_checkpoint(args.image_ckpt)
    spectrum_encoder = SpectrumAutoencoder.load_from_checkpoint(args.spectrum_ckpt)
    clip_model = SmallCLIPModel.load_from_checkpoint(
        args.clip_ckpt,
        image_encoder=image_encoder,
        spectrum_encoder=spectrum_encoder,
    )
    clip_model.eval()
    clip_model.to("cuda" if torch.cuda.is_available() else "cpu")

    all_image_embeddings = []
    all_spectrum_embeddings = []
    all_redshift = []
    all_targetid = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Embedding split '{args.split}'"):
            images = batch["image"].to(clip_model.device)
            spectra = batch["spectrum"].to(clip_model.device)
            img_embeddings, sp_embeddings = clip_model(images, spectra)

            all_image_embeddings.append(img_embeddings.cpu().numpy())
            all_spectrum_embeddings.append(sp_embeddings.cpu().numpy())
            if "redshift" in batch:
                all_redshift.append(batch["redshift"].numpy())
            if "targetid" in batch:
                all_targetid.append(batch["targetid"].numpy())

    image_matrix = np.concatenate(all_image_embeddings, axis=0)
    spectrum_matrix = np.concatenate(all_spectrum_embeddings, axis=0)
    redshift_array = np.concatenate(all_redshift, axis=0) if all_redshift else None
    targetid_array = np.concatenate(all_targetid, axis=0) if all_targetid else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("image_embeddings", data=image_matrix)
        f.create_dataset("spectrum_embeddings", data=spectrum_matrix)
        if redshift_array is not None:
            f.create_dataset("redshift", data=redshift_array)
        if targetid_array is not None:
            f.create_dataset("targetid", data=targetid_array)


if __name__ == "__main__":
    main()
