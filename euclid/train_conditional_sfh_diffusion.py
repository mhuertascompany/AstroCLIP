"""Train a physical-property-conditioned diffusion model for normalized SFHs.

The three default conditions are catalog log stellar mass, photometric
redshift, and catalog log SFR averaged over 100 Myr. Training can draw a random
valid SFH posterior realization per object, so generated samples represent the
conditional population distribution convolved with the stored SFH uncertainty.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .sfh_conditional_diffusion import (
    ConditionalSFHDiffusionModule,
    log_sfh_to_weights,
    weights_to_clr,
)
from .sfh_shape import sfh_recent_activity


BASE_CONDITION_COLUMNS = ('phz_pp_median_stellarmass', 'phz_pp_median_redshift')
PHZ_SFR_COLUMN = 'phz_pp_median_sfr'
CONDITION_LABELS = ('log_stellar_mass', 'redshift', 'log_sfr_100myr')


def _read_rows(dataset, rows):
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(dataset[rows[order]])[inverse]


def load_conditions(dataset_path, sfr_source='sfh'):
    if sfr_source not in {'sfh', 'phz'}:
        raise ValueError("sfr_source must be 'sfh' or 'phz'.")
    with h5py.File(dataset_path, 'r') as source:
        required = list(BASE_CONDITION_COLUMNS)
        if sfr_source == 'phz':
            required.append(PHZ_SFR_COLUMN)
        else:
            required.extend(('sfh', 'sfh_time_grid', 'sfh_time_norm'))
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f'Dataset is missing physical conditions: {missing}')
        mass = np.asarray(source[BASE_CONDITION_COLUMNS[0]], dtype=np.float32)
        redshift = np.asarray(source[BASE_CONDITION_COLUMNS[1]], dtype=np.float32)
        if sfr_source == 'phz':
            log_sfr = np.asarray(source[PHZ_SFR_COLUMN], dtype=np.float32)
        else:
            recent = sfh_recent_activity(
                np.asarray(source['sfh'], dtype=np.float32),
                np.asarray(source['sfh_time_grid'], dtype=np.float32),
                age_myr=np.asarray(source['sfh_time_norm'], dtype=np.float32),
            )
            log_sfr = recent['sfh_log_sfr_per_stellar_mass_100myr_r0'] + mass
        conditions = np.column_stack([mass, redshift, log_sfr]).astype(np.float32)
    valid = np.all(np.isfinite(conditions), axis=1)
    valid &= (conditions[:, 0] > 0) & (conditions[:, 0] < 20)
    valid &= (conditions[:, 1] >= 0) & (conditions[:, 1] < 20)
    valid &= (conditions[:, 2] > -20) & (conditions[:, 2] < 20)
    return conditions, valid


def load_saved_split(dataset_path, split_path):
    with h5py.File(dataset_path, 'r') as source:
        ids = np.asarray(source['galaxy_id'], dtype=np.int64)
    with np.load(split_path) as split:
        arrays = {
            key: np.asarray(split[key], dtype=np.int64)
            for key in ('train_rows', 'train_ids', 'val_rows', 'val_ids')
        }
    for partition in ('train', 'val'):
        rows, expected = arrays[f'{partition}_rows'], arrays[f'{partition}_ids']
        if not np.array_equal(ids[rows], expected):
            raise ValueError(f'{partition} rows do not match saved galaxy IDs.')
    if np.intersect1d(arrays['train_ids'], arrays['val_ids']).size:
        raise ValueError('Training and validation IDs overlap.')
    return arrays


def fit_transforms(dataset_path, train_rows, conditions, epsilon=1e-10,
                   clr_floor=1e-8):
    raw_condition = conditions[train_rows].astype(np.float64)
    condition_mean = raw_condition.mean(axis=0)
    condition_scale = raw_condition.std(axis=0)
    with h5py.File(dataset_path, 'r') as source:
        log_sfh = _read_rows(source['sfh'], train_rows)
    clr = weights_to_clr(log_sfh_to_weights(log_sfh, epsilon), clr_floor)
    clr_mean = clr.mean(axis=0, dtype=np.float64)
    clr_scale = clr.std(axis=0, dtype=np.float64)
    condition_scale = np.maximum(condition_scale, 1e-6)
    clr_scale = np.maximum(clr_scale, 1e-3)
    return tuple(value.astype(np.float32) for value in (
        condition_mean, condition_scale, clr_mean, clr_scale,
    ))


class ConditionalSFHDataset(Dataset):
    def __init__(self, path, rows, conditions, condition_mean, condition_scale,
                 clr_mean, clr_scale, input_mode='posterior', training=True,
                 epsilon=1e-10, clr_floor=1e-8, sfr_source='sfh'):
        self.path = str(Path(path))
        self.rows = np.asarray(rows, dtype=np.int64)
        self.conditions = np.asarray(conditions[self.rows], dtype=np.float32)
        self.condition_mean = np.asarray(condition_mean, dtype=np.float32)
        self.condition_scale = np.asarray(condition_scale, dtype=np.float32)
        self.clr_mean = np.asarray(clr_mean, dtype=np.float32)
        self.clr_scale = np.asarray(clr_scale, dtype=np.float32)
        self.input_mode = input_mode
        self.training = training
        self.epsilon = epsilon
        self.clr_floor = clr_floor
        self.sfr_source = sfr_source
        self._h5 = None
        self._pid = None
        if input_mode not in {'median', 'posterior'}:
            raise ValueError("input_mode must be 'median' or 'posterior'.")
        if sfr_source not in {'sfh', 'phz'}:
            raise ValueError("sfr_source must be 'sfh' or 'phz'.")
        with h5py.File(self.path, 'r') as source:
            self.time_grid = np.asarray(source['sfh_time_grid'], dtype=np.float32)
            self.age_myr = _read_rows(source['sfh_time_norm'], self.rows).astype(np.float32)
        self.time_edges = np.r_[
            0.0, (self.time_grid[:-1] + self.time_grid[1:]) / 2, 1.0,
        ]

    def __len__(self):
        return len(self.rows)

    def _source(self):
        pid = os.getpid()
        if self._h5 is None or self._pid != pid:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(self.path, 'r')
            self._pid = pid
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        state['_pid'] = None
        return state

    def __getitem__(self, index):
        source = self._source()
        row = int(self.rows[index])
        realization = -1
        if self.input_mode == 'posterior' and self.training:
            valid = np.flatnonzero(source['sfh_realization_valid'][row])
            if not len(valid):
                raise ValueError(f'No valid posterior SFH at row {row}.')
            realization = int(valid[int(torch.randint(len(valid), ()).item())])
            log_sfh = np.asarray(
                source['sfh_realizations'][row, realization], dtype=np.float32,
            )
        else:
            log_sfh = np.asarray(source['sfh'][row], dtype=np.float32)
        weights = log_sfh_to_weights(log_sfh, self.epsilon)
        clr = weights_to_clr(weights, self.clr_floor)
        target = (clr - self.clr_mean) / self.clr_scale
        raw_condition = self.conditions[index].copy()
        if self.sfr_source == 'sfh':
            age_myr = float(self.age_myr[index])
            upper = min(1.0e8 / (age_myr * 1.0e6), 1.0)
            widths = np.diff(self.time_edges)
            overlap = np.maximum(
                0.0,
                np.minimum(self.time_edges[1:], upper) - self.time_edges[:-1],
            )
            recent_fraction = float(weights @ (overlap / widths))
            raw_condition[2] = (
                raw_condition[0] + np.log10(max(recent_fraction, 1e-15)) - 8.0
            )
        condition = (raw_condition - self.condition_mean) / self.condition_scale
        return {
            'target': torch.from_numpy(target.copy()),
            'condition': torch.from_numpy(condition.copy()),
            'galaxy_id': torch.tensor(int(source['galaxy_id'][row]), dtype=torch.int64),
            'row': torch.tensor(row, dtype=torch.int64),
            'realization': torch.tensor(realization, dtype=torch.int64),
        }


class ConditionalSFHDiffusion(L.LightningModule):
    def __init__(self, n_bins, condition_mean, condition_scale, clr_mean,
                 clr_scale, d_model=128, n_heads=4, n_layers=4, dropout=0.1,
                 diffusion_steps=1000, condition_dropout=0.15, lr=1e-4,
                 weight_decay=0.01, ema_decay=0.999, sfr_source='sfh'):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConditionalSFHDiffusionModule(
            n_bins=n_bins, condition_mean=condition_mean,
            condition_scale=condition_scale, clr_mean=clr_mean,
            clr_scale=clr_scale, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
            diffusion_steps=diffusion_steps,
        )

    def train(self, mode=True):
        super().train(mode)
        self.model.ema.eval()
        return self

    def _loss(self, target, condition, timestep, noise, drop=None, ema=False):
        noisy, velocity = self.model.schedule.noisy_target(target, noise, timestep)
        network = self.model.ema if ema else self.model.net
        prediction = network(noisy, timestep, condition, drop)
        return F.mse_loss(prediction.float(), velocity.float())

    def training_step(self, batch, batch_idx):
        target, condition = batch['target'], batch['condition']
        timestep = torch.randint(
            len(self.model.schedule.alpha), (len(target),), device=self.device,
        )
        drop = torch.rand(len(target), device=self.device) < self.hparams.condition_dropout
        loss = self._loss(target, condition, timestep, torch.randn_like(target), drop)
        self.log('train_v_mse', loss, prog_bar=True, batch_size=len(target))
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        with torch.no_grad():
            for target, value in zip(
                self.model.ema.parameters(), self.model.net.parameters(), strict=True,
            ):
                target.lerp_(value, 1 - self.hparams.ema_decay)

    def validation_step(self, batch, batch_idx):
        target, condition = batch['target'], batch['condition']
        generator = torch.Generator(device=self.device).manual_seed(10000 + batch_idx)
        timestep = torch.randint(
            len(self.model.schedule.alpha), (len(target),),
            generator=generator, device=self.device,
        )
        noise = torch.randn(target.shape, generator=generator, device=self.device)
        null = torch.ones(len(target), device=self.device, dtype=torch.bool)
        losses = {
            'val_v_mse': self._loss(target, condition, timestep, noise, ema=True),
            'val_shuffled_v_mse': self._loss(
                target, condition.roll(1, 0), timestep, noise, ema=True,
            ),
            'val_unconditional_v_mse': self._loss(
                target, condition, timestep, noise, drop=null, ema=True,
            ),
        }
        for name, loss in losses.items():
            self.log(name, loss, prog_bar=name == 'val_v_mse', batch_size=len(target),
                     sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.net.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--input-mode', choices=('median', 'posterior'), default='posterior')
    parser.add_argument('--sfr-source', choices=('sfh', 'phz'), default='sfh')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--diffusion-steps', type=int, default=1000)
    parser.add_argument('--condition-dropout', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accelerator', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.dataset, args.split):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        raise ValueError(f'Output is nonempty; pass --resume or use a new path: {args.output}')
    L.seed_everything(42, workers=True)
    split = load_saved_split(args.dataset, args.split)
    conditions, valid = load_conditions(args.dataset, args.sfr_source)
    train_rows = split['train_rows'][valid[split['train_rows']]]
    val_rows = split['val_rows'][valid[split['val_rows']]]
    if args.smoke:
        train_rows, val_rows = train_rows[:512], val_rows[:128]
    transforms = fit_transforms(args.dataset, train_rows, conditions)
    condition_mean, condition_scale, clr_mean, clr_scale = transforms
    with h5py.File(args.dataset, 'r') as source:
        n_bins = source['sfh'].shape[1]
        ids = np.asarray(source['galaxy_id'], dtype=np.int64)
    dataset_args = (
        args.dataset, conditions, condition_mean, condition_scale, clr_mean, clr_scale,
    )
    train = ConditionalSFHDataset(
        dataset_args[0], train_rows, *dataset_args[1:],
        input_mode=args.input_mode, training=True, sfr_source=args.sfr_source,
    )
    val = ConditionalSFHDataset(
        dataset_args[0], val_rows, *dataset_args[1:],
        input_mode=args.input_mode, training=False, sfr_source=args.sfr_source,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / 'conditional_sfh_split.npz',
        train_rows=train_rows, train_ids=ids[train_rows],
        val_rows=val_rows, val_ids=ids[val_rows],
    )
    model_args = dict(
        n_bins=n_bins,
        condition_mean=condition_mean.tolist(),
        condition_scale=condition_scale.tolist(),
        clr_mean=clr_mean.tolist(),
        clr_scale=clr_scale.tolist(),
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        diffusion_steps=args.diffusion_steps,
        condition_dropout=args.condition_dropout, lr=args.lr,
        sfr_source=args.sfr_source,
    )
    model = ConditionalSFHDiffusion(**model_args)
    if args.resume is not None:
        model = ConditionalSFHDiffusion.load_from_checkpoint(
            str(args.resume), map_location='cpu',
        )
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    settings.update(
        condition_columns=(
            (*BASE_CONDITION_COLUMNS, PHZ_SFR_COLUMN)
            if args.sfr_source == 'phz'
            else (*BASE_CONDITION_COLUMNS, 'SFH-derived SFR100; R=0')
        ),
        condition_labels=CONDITION_LABELS, sfr_source=args.sfr_source,
        train_count=len(train), val_count=len(val), n_bins=n_bins,
        removed_for_invalid_conditions=int((~valid).sum()),
        sfh_representation='centered log-ratio; inverse softmax sums to one',
        generated_distribution=(
            'population conditional plus stored SFH posterior uncertainty'
            if args.input_mode == 'posterior' else 'population conditional medians'
        ),
    )
    (args.output / 'run.json').write_text(json.dumps(settings, indent=2) + '\n')
    loader = dict(
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=args.accelerator == 'gpu',
        persistent_workers=args.workers > 0,
    )
    checkpoint = ModelCheckpoint(
        dirpath=args.output / 'checkpoints', monitor='val_v_mse', mode='min',
        save_last=True, save_top_k=3,
        filename='conditional-sfh-{epoch:03d}-{val_v_mse:.5f}',
    )
    callbacks = [
        checkpoint,
        EarlyStopping('val_v_mse', mode='min', patience=args.patience),
        LearningRateMonitor('epoch'),
    ]
    trainer = L.Trainer(
        accelerator=args.accelerator, devices=1,
        precision='16-mixed' if args.accelerator == 'gpu' else '32-true',
        max_epochs=2 if args.smoke else args.epochs,
        gradient_clip_val=1.0, callbacks=callbacks,
        logger=CSVLogger(str(args.output), name='logs'),
        log_every_n_steps=5, num_sanity_val_steps=0,
    )
    start = time.monotonic()
    trainer.fit(
        model,
        DataLoader(train, shuffle=True, drop_last=True, **loader),
        DataLoader(val, shuffle=False, **loader),
        ckpt_path=str(args.resume) if args.resume is not None else None,
    )
    report = {
        'elapsed_seconds': time.monotonic() - start,
        'optimizer_steps': trainer.global_step,
        'best_checkpoint': checkpoint.best_model_path,
    }
    (args.output / 'runtime.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
