"""
Train CosmosWebZooBotCLIP v5: same model as v2/v4 but each galaxy is fed the
JWST stamp from its rest-frame optical filter instead of always F277W.

The ZooBOT backbone (family_2.ckpt) was already trained with per-galaxy filter
routing (z<1→F150W, 1≤z<3→F277W, z≥3→F444W), so a single checkpoint covers
all three bands.  The only change vs v4 is in the data loader.

Usage (from repo root):
    python -m cosmosweb.train_zoobot_multifilter \\
        --dataset        /path/to/cosmosweb_dataset_v4.h5 \\
        --stamp_root     /path/to/stamps_ilbert \\
        --zoobot_ckpt    /path/to/family_2.ckpt \\
        --output         /path/to/checkpoints/cosmosweb_zoobot_v5.ckpt
"""

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .dataset_zoobot import MultiFilterCosmosWebZooBotDataModule
from .model_zoobot import CosmosWebZooBotCLIP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train CosmosWebZooBotCLIP v5 (multi-filter stamps)')

    # Data
    p.add_argument('--dataset',     type=Path, required=True,
                   help='HDF5 file from prepare_dataset.py')
    p.add_argument('--stamp_root',  type=Path, required=True,
                   help='Root dir with per-filter JPEG stamp folders (F150W/, F277W/, F444W/)')
    p.add_argument('--z_low',       type=float, default=1.0,
                   help='Redshift boundary between F150W and F277W (default 1.0)')
    p.add_argument('--z_high',      type=float, default=3.0,
                   help='Redshift boundary between F277W and F444W (default 3.0)')
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--sfh_noise_std', type=float, default=0.0)
    p.add_argument('--image_size',  type=int,   default=224)

    # Model
    p.add_argument('--zoobot_ckpt', type=Path, required=True,
                   help='Single FinetuneableZoobotClassifier checkpoint (e.g. family_2.ckpt). '
                        'This model was already trained with per-galaxy filter routing.')
    p.add_argument('--embed_dim',     type=int,   default=256)
    p.add_argument('--sfh_input_dim', type=int,   default=50)
    p.add_argument('--temperature',   type=float, default=0.07)
    p.add_argument('--queue_size',    type=int,   default=1024)
    p.add_argument('--momentum',      type=float, default=0.995)

    # Optimiser
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int,   default=3)
    p.add_argument('--batch_size',    type=int,   default=128)
    p.add_argument('--max_epochs',    type=int,   default=100)

    # Hardware
    p.add_argument('--accelerator', type=str, default='auto')
    p.add_argument('--devices',     type=str, default='auto')
    p.add_argument('--precision',   type=str, default='16-mixed')

    # Logging & checkpointing
    p.add_argument('--output',      type=Path,
                   default=Path('outputs/cosmosweb_zoobot_v5.ckpt'))
    p.add_argument('--log_dir',     type=Path,
                   default=Path('logs/cosmosweb_zoobot_v5'))
    p.add_argument('--run_name',    type=str,  default='cosmosweb_zoobot_v5')
    p.add_argument('--resume_from', type=Path, default=None)
    p.add_argument('--patience',    type=int,  default=20)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── data ──────────────────────────────────────────────────────────────────
    datamodule = MultiFilterCosmosWebZooBotDataModule(
        h5_path=args.dataset,
        stamp_root=args.stamp_root,
        z_low=args.z_low,
        z_high=args.z_high,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        sfh_noise_std=args.sfh_noise_std,
        image_size=args.image_size,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = CosmosWebZooBotCLIP(
        zoobot_ckpt=str(args.zoobot_ckpt),
        embed_dim=args.embed_dim,
        sfh_input_dim=args.sfh_input_dim,
        temperature=args.temperature,
        queue_size=args.queue_size,
        momentum=args.momentum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
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
        patience=args.patience,
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
