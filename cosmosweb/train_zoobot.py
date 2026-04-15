"""
Train CosmosWebZooBotCLIP: contrastive alignment of COSMOS-Web JPEG stamps
and CIGALE SFHs using a frozen ZooBOT EfficientNet backbone.

Usage (from repo root):
    python -m cosmosweb.train_zoobot \\
        --dataset    /n03data/huertas/COSMOS-Web/cosmosweb_clip/cosmosweb_dataset_v2.h5 \\
        --stamp_root /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \\
        --filter     F277W \\
        --zoobot_ckpt /n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/family_2.ckpt \\
        --output     /n03data/huertas/COSMOS-Web/cosmosweb_clip/checkpoints/cosmosweb_zoobot.ckpt

Tips:
    - Default batch size is 128 (224×224 images use more VRAM than 64×64).
    - Only the MLP projection head and the SFH encoder are trained; the
      ZooBOT backbone is frozen.  Training is therefore fast.
    - For a smoke-test: --max_epochs 3 --batch_size 32
"""

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .dataset_zoobot import CosmosWebZooBotDataModule
from .model_zoobot import CosmosWebZooBotCLIP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train CosmosWebZooBotCLIP')

    # Data
    p.add_argument('--dataset',     type=Path, required=True,
                   help='HDF5 file from prepare_dataset.py')
    p.add_argument('--stamp_root',  type=Path, required=True,
                   help='Root dir with per-filter JPEG stamp folders')
    p.add_argument('--filter',      type=str,  default='F277W',
                   help='JWST filter to use (must exist under stamp_root)')
    p.add_argument('--num_workers', type=int,  default=4)
    p.add_argument('--sfh_noise_std', type=float, default=0.0)
    p.add_argument('--image_size',  type=int,  default=224)

    # Model
    p.add_argument('--zoobot_ckpt', type=Path, required=True,
                   help='FinetuneableZoobotClassifier checkpoint (e.g. family_2.ckpt)')
    p.add_argument('--embed_dim',     type=int,   default=256)
    p.add_argument('--sfh_input_dim', type=int,   default=50,
                   help='Length of SFH vector (= SFH_N_BINS in prepare_dataset.py)')
    p.add_argument('--temperature',   type=float, default=0.07)

    # Optimiser
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int,  default=5)
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--max_epochs',   type=int,   default=50)

    # Hardware
    p.add_argument('--accelerator', type=str, default='auto')
    p.add_argument('--devices',     type=str, default='auto')
    p.add_argument('--precision',   type=str, default='16-mixed')

    # Logging & checkpointing
    p.add_argument('--output',      type=Path, default=Path('outputs/cosmosweb_zoobot.ckpt'))
    p.add_argument('--log_dir',     type=Path, default=Path('logs/cosmosweb_zoobot'))
    p.add_argument('--run_name',    type=str,  default='cosmosweb_zoobot')
    p.add_argument('--resume_from', type=Path, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── data ──────────────────────────────────────────────────────────────────
    datamodule = CosmosWebZooBotDataModule(
        h5_path=args.dataset,
        stamp_root=args.stamp_root,
        filter_name=args.filter,
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
