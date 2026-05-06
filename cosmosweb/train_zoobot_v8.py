"""
Train CosmosWebZooBotCLIP v8: transformer SFH encoder with random token sampling.

The SFH is represented as 9 (t_frac, log_sfr) tokens — one per CIGALE bin.
At each training step t_frac is drawn uniformly within the bin's temporal
extent, preventing the encoder from reading plateau widths or exact bin
positions as a redshift proxy.  A small Set-Transformer (3-layer pre-LN,
d_model=64, CLS pooling) encodes the 9 tokens → 256-d embedding.

Usage (from repo root):
    python -m cosmosweb.train_zoobot_v8 \\
        --dataset   /path/to/cosmosweb_dataset_v6.h5 \\
        --stamp_root /path/to/stamps_ilbert \\
        --zoobot_ckpt /path/to/family_2.ckpt \\
        --output    /path/to/checkpoints/cosmosweb_zoobot_v8.ckpt
"""

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .dataset_zoobot import SFHTokenCosmosWebZooBotDataModule
from .model_zoobot import CosmosWebZooBotCLIPv8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train CosmosWebZooBotCLIP v8 (transformer SFH encoder)')

    # Data
    p.add_argument('--dataset',      type=Path, required=True)
    p.add_argument('--stamp_root',   type=Path, required=True)
    p.add_argument('--filter',       type=str,  default='F277W')
    p.add_argument('--num_workers',  type=int,  default=4)
    p.add_argument('--image_size',   type=int,  default=224)

    # ZooBOT backbone
    p.add_argument('--zoobot_ckpt',  type=Path, required=True)
    p.add_argument('--unfreeze_blocks',   type=int,   default=0)
    p.add_argument('--backbone_lr_scale', type=float, default=0.1)

    # SFH transformer
    p.add_argument('--sfh_n_bins',   type=int,   default=9)
    p.add_argument('--sfh_d_model',  type=int,   default=64)
    p.add_argument('--sfh_n_heads',  type=int,   default=4)
    p.add_argument('--sfh_n_layers', type=int,   default=3)

    # CLIP
    p.add_argument('--embed_dim',    type=int,   default=256)
    p.add_argument('--temperature',  type=float, default=0.07)
    p.add_argument('--queue_size',   type=int,   default=1024)
    p.add_argument('--momentum',     type=float, default=0.995)

    # Optimiser
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs',type=int,   default=3)
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--max_epochs',   type=int,   default=100)
    p.add_argument('--patience',     type=int,   default=20)

    # Hardware
    p.add_argument('--accelerator',  type=str, default='auto')
    p.add_argument('--devices',      type=str, default='auto')
    p.add_argument('--precision',    type=str, default='16-mixed')

    # Logging
    p.add_argument('--output',       type=Path, default=Path('outputs/cosmosweb_zoobot_v8.ckpt'))
    p.add_argument('--log_dir',      type=Path, default=Path('logs/cosmosweb_zoobot_v8'))
    p.add_argument('--run_name',     type=str,  default='cosmosweb_zoobot_v8')
    p.add_argument('--resume_from',  type=Path, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    datamodule = SFHTokenCosmosWebZooBotDataModule(
        h5_path=args.dataset,
        stamp_root=args.stamp_root,
        filter_name=args.filter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        image_size=args.image_size,
    )

    model = CosmosWebZooBotCLIPv8(
        zoobot_ckpt=str(args.zoobot_ckpt),
        embed_dim=args.embed_dim,
        sfh_n_bins=args.sfh_n_bins,
        sfh_d_model=args.sfh_d_model,
        sfh_n_heads=args.sfh_n_heads,
        sfh_n_layers=args.sfh_n_layers,
        temperature=args.temperature,
        queue_size=args.queue_size,
        momentum=args.momentum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        unfreeze_blocks=args.unfreeze_blocks,
        backbone_lr_scale=args.backbone_lr_scale,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output.parent,
        filename=args.output.stem + '-{epoch:03d}-{val_loss:.4f}',
        monitor='val_loss', mode='min', save_top_k=3, save_last=True,
    )
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=args.patience,
                                  mode='min', verbose=True)
    lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')
    logger = CSVLogger(save_dir=str(args.log_dir), name=args.run_name)

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
