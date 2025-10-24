from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from teaching_scripts.data_utils import build_multimodal_dataloader, load_astroclip_dataset
from teaching_scripts.models import ImageAutoencoder, SmallCLIPModel, SpectrumAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small CLIP-style alignment model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path or HuggingFace identifier for the dataset.")
    parser.add_argument("--image-ckpt", type=Path, required=True, help="Checkpoint from train_image_encoder.py.")
    parser.add_argument("--spectrum-ckpt", type=Path, required=True, help="Checkpoint from train_spectrum_encoder.py.")
    parser.add_argument("--output", type=Path, default=Path("outputs/clip_alignment.ckpt"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--finetune-encoders", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_astroclip_dataset(args.dataset)
    if "train" not in ds:
        raise ValueError("Dataset must contain a 'train' split.")

    train_loader = build_multimodal_dataloader(
        ds["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if "test" in ds:
        val_loader = build_multimodal_dataloader(
            ds["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    image_encoder = ImageAutoencoder.load_from_checkpoint(args.image_ckpt)
    spectrum_encoder = SpectrumAutoencoder.load_from_checkpoint(args.spectrum_ckpt)

    model = SmallCLIPModel(
        image_encoder=image_encoder,
        spectrum_encoder=spectrum_encoder,
        projection_dim=args.projection_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        finetune_encoders=args.finetune_encoders,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output.parent,
        filename=args.output.stem,
        save_last=True,
        save_top_k=1,
        monitor="val_loss" if val_loader is not None else None,
        mode="min",
    )

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        deterministic=args.deterministic,
        callbacks=[checkpoint_callback],
        log_every_n_steps=25,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(args.output)


if __name__ == "__main__":
    main()
