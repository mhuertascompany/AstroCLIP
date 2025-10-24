from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from teaching_scripts.data_utils import build_image_dataloader, load_astroclip_dataset
from teaching_scripts.models import ImageAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight image autoencoder for AstroCLIP.")
    parser.add_argument("--dataset", type=str, required=True, help="Path or HuggingFace identifier for the dataset.")
    parser.add_argument("--output", type=Path, default=Path("outputs/image_autoencoder.ckpt"), help="Checkpoint destination.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/image_autoencoder"))
    parser.add_argument("--run-name", type=str, default="image_autoencoder")
    parser.add_argument("--plot-path", type=Path, default=None, help="Optional path to save the loss curve PNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_astroclip_dataset(args.dataset)
    if "train" not in ds:
        raise ValueError("Dataset must contain a 'train' split.")

    train_loader = build_image_dataloader(
        ds["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = None
    if "test" in ds:
        val_loader = build_image_dataloader(
            ds["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

    model = ImageAutoencoder(
        embed_dim=args.embed_dim,
        hidden_channels=args.hidden_channels,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=str(args.log_dir), name=args.run_name)

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
        logger=logger,
        log_every_n_steps=25,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(args.output)

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    plot_path = args.plot_path or Path(logger.log_dir) / "loss_curve.png"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        fig, ax = plt.subplots()
        if "train_loss" in df.columns:
            train_df = df.dropna(subset=["train_loss"])
            if not train_df.empty:
                ax.plot(train_df["epoch"], train_df["train_loss"], label="train_loss")
        if "val_loss" in df.columns:
            val_df = df.dropna(subset=["val_loss"])
            if not val_df.empty:
                ax.plot(val_df["epoch"], val_df["val_loss"], label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Image autoencoder training")
        ax.legend()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved loss curve to {plot_path}")
    else:
        print(f"No metrics.csv found at {metrics_path}; skipping plot.")


if __name__ == "__main__":
    main()
