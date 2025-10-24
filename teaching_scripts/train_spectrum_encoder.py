from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

import numpy as np

from teaching_scripts.data_utils import load_astroclip_dataset, build_spectrum_dataloader
from teaching_scripts.models import SpectrumAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight spectrum autoencoder for AstroCLIP.")
    parser.add_argument("--dataset", type=str, required=True, help="Path or HuggingFace identifier for the dataset.")
    parser.add_argument("--output", type=Path, default=Path("outputs/spectrum_autoencoder.ckpt"), help="Checkpoint destination.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
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

    sample = ds["train"][0]["spectrum"]
    input_dim = int(np.prod(sample.shape))

    train_loader = build_spectrum_dataloader(
        ds["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = None
    if "test" in ds:
        val_loader = build_spectrum_dataloader(
            ds["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

    model = SpectrumAutoencoder(
        input_dim=input_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
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
