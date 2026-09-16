"""Pretrain the Euclid SFH transformer as a denoising autoencoder.

Training inputs are random valid posterior realizations.  The target is the
posterior-median SFH, so the global latent learns the stable history rather
than realization-specific fluctuations.  A contiguous interval is masked
before encoding, and a time-query decoder reconstructs the full 250-bin SFH.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset

from cosmosweb.sfh_autoencoder import (
    FixedGridSFHDecoder,
    mask_contiguous_bins,
    sfh_reconstruction_loss,
)
from cosmosweb.sfh_transformer import FixedGridSFHTransformerEncoder

from .training_index import inspect_sfh_file


class EuclidSFHAutoencoderDataset(Dataset):
    """Lazy HDF5 dataset returning noisy inputs and median targets."""

    def __init__(self, path, rows, training=True):
        self.path = str(Path(path))
        self.rows = np.asarray(rows, dtype=np.int64)
        self.training = training
        self._h5 = None
        self._h5_pid = None

    def __len__(self):
        return len(self.rows)

    def _source(self):
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(self.path, 'r')
            self._h5_pid = pid
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        state['_h5_pid'] = None
        return state

    def __del__(self):
        source = getattr(self, '_h5', None)
        if source is not None:
            source.close()

    def __getitem__(self, index):
        source = self._source()
        row = int(self.rows[index])
        target = np.asarray(source['sfh'][row], dtype=np.float32)
        valid = np.flatnonzero(source['sfh_realization_valid'][row])
        if not len(valid):
            raise ValueError(f'No valid SFH realization at HDF5 row {row}.')
        if self.training:
            realization_index = int(valid[int(torch.randint(len(valid), size=()).item())])
        else:
            # Deterministic across epochs, but distributed across realization
            # indices instead of always testing the first posterior draw.
            realization_index = int(valid[row % len(valid)])
        input_sfh = np.asarray(
            source['sfh_realizations'][row, realization_index], dtype=np.float32,
        )
        arrays = (input_sfh, target, source['sfh_p16'][row], source['sfh_p84'][row])
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError(f'Nonfinite SFH data at HDF5 row {row}.')
        return {
            'sfh': torch.from_numpy(input_sfh.copy()),
            'target': torch.from_numpy(target.copy()),
            'p16': torch.from_numpy(np.asarray(arrays[2], dtype=np.float32).copy()),
            'p84': torch.from_numpy(np.asarray(arrays[3], dtype=np.float32).copy()),
            'row': torch.tensor(row, dtype=torch.int64),
            'realization': torch.tensor(realization_index, dtype=torch.int64),
        }


def split_rows(dataset_path, split_path=None, val_fraction=0.1, seed=42,
               max_objects=None):
    ids, _, _ = inspect_sfh_file(dataset_path)
    if split_path is not None:
        with np.load(split_path) as split:
            required = {'train_rows', 'train_ids', 'val_rows', 'val_ids'}
            missing = sorted(required.difference(split.files))
            if missing:
                raise ValueError(f'Missing split arrays: {missing}')
            train_rows = np.asarray(split['train_rows'], dtype=np.int64)
            val_rows = np.asarray(split['val_rows'], dtype=np.int64)
            train_ids = np.asarray(split['train_ids'], dtype=np.int64)
            val_ids = np.asarray(split['val_ids'], dtype=np.int64)
        if not np.array_equal(ids[train_rows], train_ids):
            raise ValueError('Saved training rows do not match their galaxy IDs.')
        if not np.array_equal(ids[val_rows], val_ids):
            raise ValueError('Saved validation rows do not match their galaxy IDs.')
    else:
        if not 0 < val_fraction < 1:
            raise ValueError('val_fraction must lie strictly between zero and one.')
        order = np.random.default_rng(seed).permutation(len(ids))
        n_val = max(1, int(round(val_fraction * len(ids))))
        val_rows, train_rows = order[:n_val], order[n_val:]

    if max_objects is not None and len(train_rows) + len(val_rows) > max_objects:
        if max_objects < 2:
            raise ValueError('max_objects must be at least two.')
        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(max_objects * val_fraction)))
        n_train = max_objects - n_val
        train_rows = rng.permutation(train_rows)[:n_train]
        val_rows = rng.permutation(val_rows)[:n_val]
    if not len(train_rows) or not len(val_rows):
        raise ValueError('SFH pretraining requires non-empty train and validation sets.')
    return np.asarray(train_rows), np.asarray(val_rows)


class EuclidSFHAutoencoderDataModule(L.LightningDataModule):
    def __init__(self, dataset, split=None, batch_size=256, num_workers=8,
                 val_fraction=0.1, seed=42, max_objects=None):
        super().__init__()
        self.dataset = Path(dataset)
        self.split = Path(split) if split is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.max_objects = max_objects

    def setup(self, stage=None):
        train_rows, val_rows = split_rows(
            self.dataset, self.split, self.val_fraction, self.seed, self.max_objects,
        )
        self.train_rows, self.val_rows = train_rows, val_rows
        self.train_ds = EuclidSFHAutoencoderDataset(self.dataset, train_rows, True)
        self.val_ds = EuclidSFHAutoencoderDataset(self.dataset, val_rows, False)
        print(
            f'[EuclidSFHAutoencoder] train={len(train_rows):,}, '
            f'val={len(val_rows):,}', flush=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class EuclidSFHAutoencoder(L.LightningModule):
    def __init__(
        self,
        n_bins=250,
        embed_dim=256,
        d_model=128,
        n_heads=4,
        encoder_layers=4,
        decoder_layers=2,
        mask_fraction=0.35,
        w1_weight=0.5,
        sfh_log_epsilon=1e-10,
        lr=1e-4,
        weight_decay=0.01,
        epochs=50,
        warmup_epochs=3,
    ):
        super().__init__()
        if not 0 <= mask_fraction < 1:
            raise ValueError('mask_fraction must lie in [0, 1).')
        self.save_hyperparameters()
        self.encoder = FixedGridSFHTransformerEncoder(
            n_bins=n_bins, d_model=d_model, n_heads=n_heads,
            n_layers=encoder_layers, embed_dim=embed_dim,
        )
        self.decoder = FixedGridSFHDecoder(
            n_bins=n_bins, embed_dim=embed_dim, d_model=d_model,
            n_heads=n_heads, n_layers=decoder_layers,
        )
        self.mask_value = nn.Parameter(
            torch.tensor(float(np.log10(sfh_log_epsilon)), dtype=torch.float32),
        )

    def forward(self, sfh, apply_mask=False):
        if apply_mask:
            sfh, mask = mask_contiguous_bins(
                sfh, self.hparams.mask_fraction, self.mask_value,
            )
        else:
            mask = torch.zeros_like(sfh, dtype=torch.bool)
        latent = self.encoder(sfh)
        return self.decoder(latent), latent, mask

    def _shared_step(self, batch, stage, apply_mask):
        reconstruction, _, mask = self(batch['sfh'], apply_mask=apply_mask)
        loss, parts = sfh_reconstruction_loss(
            reconstruction, batch['target'], batch['p16'], batch['p84'],
            epsilon=self.hparams.sfh_log_epsilon,
            w1_weight=self.hparams.w1_weight,
        )
        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_w1', parts['w1'], prog_bar=stage == 'val', sync_dist=True)
        self.log(f'{stage}_huber', parts['huber'], sync_dist=True)
        self.log(f'{stage}_masked_fraction', mask.float().mean(), sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train', apply_mask=True)

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val', apply_mask=False)
        clean_reconstruction, _, _ = self(batch['target'], apply_mask=False)
        clean_loss, clean_parts = sfh_reconstruction_loss(
            clean_reconstruction, batch['target'], batch['p16'], batch['p84'],
            epsilon=self.hparams.sfh_log_epsilon,
            w1_weight=self.hparams.w1_weight,
        )
        self.log('val_clean_loss', clean_loss, sync_dist=True)
        self.log('val_clean_w1', clean_parts['w1'], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(epoch):
            warmup = self.hparams.warmup_epochs
            total = self.hparams.epochs
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', type=Path,
                        help='Optional CLIP pair_split.npz for leakage-free comparison.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-name', default='euclid_sfh_autoencoder')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-objects', type=int)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--encoder-layers', type=int, default=4)
    parser.add_argument('--decoder-layers', type=int, default=2)
    parser.add_argument('--mask-fraction', type=float, default=0.35)
    parser.add_argument('--w1-weight', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--accelerator', default='auto')
    parser.add_argument('--devices', default='auto')
    parser.add_argument('--precision', default='16-mixed')
    parser.add_argument('--resume-from', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.dataset, args.split, args.resume_from):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('Invalid batch size or worker count.')
    if not 0 <= args.w1_weight <= 1:
        raise ValueError('--w1-weight must lie in [0, 1].')
    _, n_bins, n_realizations = inspect_sfh_file(args.dataset)
    with h5py.File(args.dataset, 'r') as source:
        missing = [name for name in ('sfh_p16', 'sfh_p84') if name not in source]
        if missing:
            raise ValueError(f'Missing uncertainty datasets: {missing}')
        epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))
    L.seed_everything(args.seed, workers=True)
    datamodule = EuclidSFHAutoencoderDataModule(
        args.dataset, args.split, args.batch_size, args.num_workers,
        args.val_fraction, args.seed, args.max_objects,
    )
    model = EuclidSFHAutoencoder(
        n_bins=n_bins, embed_dim=args.embed_dim, d_model=args.d_model,
        n_heads=args.n_heads, encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers, mask_fraction=args.mask_fraction,
        w1_weight=args.w1_weight, sfh_log_epsilon=epsilon, lr=args.lr,
        weight_decay=args.weight_decay, epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
    )
    checkpoint_dir = args.output_dir / 'checkpoints'
    log_dir = args.output_dir / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=args.run_name + '-{epoch:03d}-{val_loss:.5f}',
            monitor='val_loss', mode='min', save_top_k=3, save_last=True,
        ),
        EarlyStopping(monitor='val_loss', patience=args.patience, mode='min', verbose=True),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    logger = CSVLogger(save_dir=str(log_dir), name=args.run_name)
    print(
        f'Pretraining SFH encoder-decoder: {n_bins} bins, '
        f'{n_realizations} posterior realizations, mask={args.mask_fraction:g}, '
        f'W1 weight={args.w1_weight:g}', flush=True,
    )
    trainer = L.Trainer(
        accelerator=args.accelerator, devices=args.devices,
        max_epochs=args.max_epochs, precision=args.precision,
        callbacks=callbacks, logger=logger, log_every_n_steps=20,
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model, datamodule=datamodule,
        ckpt_path=str(args.resume_from) if args.resume_from else None,
    )
    checkpoint = callbacks[0]
    print(f'Best checkpoint: {checkpoint.best_model_path}', flush=True)
    print(f'Last checkpoint: {checkpoint.last_model_path}', flush=True)
    print(f'Logs: {logger.log_dir}', flush=True)


if __name__ == '__main__':
    main()
