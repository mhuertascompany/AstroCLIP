"""
Train CosmosWebCLIP: contrastive alignment of COSMOS-Web images and CIGALE SFHs.

Usage:
    python train.py \\
        --dataset    /path/to/cosmosweb_dataset.h5 \\
        --output     outputs/cosmosweb_clip.ckpt \\
        --embed_dim  256 \\
        --batch_size 256 \\
        --max_epochs 100 \\
        --lr         1e-4

Tips:
    - Larger batches improve CLIP training (more negatives).  Try 512 if VRAM allows.
    - For a quick smoke-test:  --max_epochs 5 --batch_size 64
    - Resume from a checkpoint with --resume_from /path/to/last.ckpt
"""

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .dataset import CosmosWebDataModule
from .model import CosmosWebCLIP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train CosmosWebCLIP')

    # Data
    p.add_argument('--dataset',      type=Path, required=True,
                   help='HDF5 file from prepare_dataset.py')
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--sfh_noise_std', type=float, default=0.0,
                   help='Gaussian noise std added to SFH (log10 space) for augmentation')

    # Model
    p.add_argument('--embed_dim',    type=int,   default=256)
    p.add_argument('--sfh_input_dim', type=int,  default=50,
                   help='Length of SFH vector (= SFH_N_BINS in prepare_dataset.py)')
    p.add_argument('--temperature',  type=float, default=0.07)
    p.add_argument('--no_pretrained', action='store_true',
                   help='Disable ImageNet pre-training for the image encoder')

    # Optimiser
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int,  default=5)
    p.add_argument('--batch_size',   type=int,   default=256)
    p.add_argument('--max_epochs',   type=int,   default=100)

    # Hardware
    p.add_argument('--accelerator',  type=str,   default='auto')
    p.add_argument('--devices',      type=str,   default='auto')
    p.add_argument('--precision',    type=str,   default='16-mixed',
                   help='16-mixed recommended for A100 / V100 GPUs')

    # Logging & checkpointing
    p.add_argument('--output',       type=Path,  default=Path('outputs/cosmosweb_clip.ckpt'))
    p.add_argument('--log_dir',      type=Path,  default=Path('logs/cosmosweb_clip'))
    p.add_argument('--run_name',     type=str,   default='cosmosweb_clip')
    p.add_argument('--resume_from',  type=Path,  default=None,
                   help='Path to a Lightning checkpoint to resume from')

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── data ──────────────────────────────────────────────────────────────────
    datamodule = CosmosWebDataModule(
        h5_path=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        sfh_noise_std=args.sfh_noise_std,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = CosmosWebCLIP(
        embed_dim=args.embed_dim,
        sfh_input_dim=args.sfh_input_dim,
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        pretrained_image_encoder=not args.no_pretrained,
    )

    # ── callbacks ─────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output.parent,
        filename=args.output.stem + '-{epoch:03d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True,
    )
    lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

    logger = CSVLogger(save_dir=str(args.log_dir), name=args.run_name)

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor_cb],
        logger=logger,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=str(args.resume_from) if args.resume_from else None,
    )

    print(f'\nBest checkpoint : {checkpoint_cb.best_model_path}')
    print(f'Logs            : {logger.log_dir}')


if __name__ == '__main__':
    main()
