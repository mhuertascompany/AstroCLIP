"""Train image--SFH CLIP on Euclid ZooBot JPEG stamps.

The SFH dimension is read from the preprocessed HDF5 file. During training the
data loader can draw a valid posterior SFH realization for each galaxy; the
validation set always uses the posterior-median SFH.
"""

import argparse
from pathlib import Path

import lightning as L
import numpy as np
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from cosmosweb.model_zoobot import CosmosWebZooBotCLIP

from .dataset_zoobot import EuclidZooBotDataModule
from .training_index import inspect_sfh_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a ZooBot--SFH contrastive model on Euclid data.',
    )
    data = parser.add_argument_group('data')
    data.add_argument('--dataset', type=Path, required=True,
                      help='Preprocessed Euclid SFH HDF5 file.')
    data.add_argument('--stamp-root', type=Path, required=True,
                      help='Directory containing BAND/BAND_object_id.jpg.')
    data.add_argument('--band', default='VIS')
    data.add_argument('--image-size', type=int, default=224)
    data.add_argument('--val-fraction', type=float, default=0.1)
    data.add_argument('--seed', type=int, default=42)
    data.add_argument('--max-pairs', type=int,
                      help='Limit matched pairs for a smoke test.')
    data.add_argument(
        '--sample-posterior',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Draw a valid SFH posterior realization in training (default: on).',
    )
    data.add_argument('--batch-size', type=int, default=128)
    data.add_argument('--num-workers', type=int, default=8)

    model = parser.add_argument_group('model')
    model.add_argument('--zoobot-ckpt', type=Path, required=True)
    model.add_argument('--embed-dim', type=int, default=256)
    model.add_argument('--temperature', type=float, default=0.07)
    model.add_argument('--queue-size', type=int, default=4096)
    model.add_argument('--momentum', type=float, default=0.995)
    model.add_argument('--unfreeze-blocks', type=int, default=0)
    model.add_argument('--backbone-lr-scale', type=float, default=0.1)

    optimization = parser.add_argument_group('optimization')
    optimization.add_argument('--lr', type=float, default=1e-4)
    optimization.add_argument('--weight-decay', type=float, default=0.05)
    optimization.add_argument('--warmup-epochs', type=int, default=5)
    optimization.add_argument('--max-epochs', type=int, default=100)
    optimization.add_argument('--patience', type=int, default=20)

    runtime = parser.add_argument_group('runtime')
    runtime.add_argument('--accelerator', default='auto')
    runtime.add_argument('--devices', default='auto')
    runtime.add_argument('--precision', default='16-mixed')
    runtime.add_argument('--output-dir', type=Path, required=True)
    runtime.add_argument('--run-name', default='euclid_zoobot_clip')
    runtime.add_argument('--resume-from', type=Path)
    return parser.parse_args()


def validate_args(args):
    if not args.dataset.is_file():
        raise FileNotFoundError(f'SFH dataset not found: {args.dataset}')
    stamp_dir = args.stamp_root / args.band
    if not stamp_dir.is_dir():
        raise FileNotFoundError(f'JPEG directory not found: {stamp_dir}')
    if not args.zoobot_ckpt.is_file():
        raise FileNotFoundError(f'ZooBot checkpoint not found: {args.zoobot_ckpt}')
    if args.resume_from is not None and not args.resume_from.is_file():
        raise FileNotFoundError(f'Resume checkpoint not found: {args.resume_from}')
    if args.batch_size < 2:
        raise ValueError('--batch-size must be at least 2.')
    if args.queue_size < args.batch_size:
        raise ValueError('--queue-size must be at least --batch-size.')
    if args.queue_size % args.batch_size:
        raise ValueError('--queue-size must be divisible by --batch-size.')


def main():
    args = parse_args()
    validate_args(args)
    _, n_bins, n_realizations = inspect_sfh_file(args.dataset)
    L.seed_everything(args.seed, workers=True)

    datamodule = EuclidZooBotDataModule(
        sfh_path=args.dataset,
        stamp_root=args.stamp_root,
        band=args.band,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        sample_posterior=args.sample_posterior,
        max_pairs=args.max_pairs,
    )
    datamodule.setup('fit')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_index = datamodule.pair_index
    np.savez_compressed(
        args.output_dir / 'pair_split.npz',
        train_rows=pair_index.train_rows,
        train_ids=pair_index.train_ids,
        val_rows=pair_index.val_rows,
        val_ids=pair_index.val_ids,
    )
    model = CosmosWebZooBotCLIP(
        zoobot_ckpt=str(args.zoobot_ckpt),
        embed_dim=args.embed_dim,
        sfh_input_dim=n_bins,
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

    checkpoint_dir = args.output_dir / 'checkpoints'
    log_dir = args.output_dir / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=args.run_name + '-{epoch:03d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            mode='min',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    logger = CSVLogger(save_dir=str(log_dir), name=args.run_name)

    print(
        f'Training with {n_bins} SFH bins and {n_realizations} posterior '
        f'realizations; posterior sampling={args.sample_posterior}',
        flush=True,
    )
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=str(args.resume_from) if args.resume_from else None,
    )
    print(f'Best checkpoint: {checkpoint_callback.best_model_path}', flush=True)
    print(f'Last checkpoint: {checkpoint_callback.last_model_path}', flush=True)
    print(f'Logs: {logger.log_dir}', flush=True)


if __name__ == '__main__':
    main()
