"""Train raw VIS/SFH counterfactual flows with matched-class donor priors."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from .counterfactual_dataset import (
    CounterfactualEuclidDataset,
    activity_classes,
    fit_sfh_transform,
    matched_donor_pools,
)
from .counterfactual_flow import EuclidCounterfactualFlow
from .training_index import build_pair_index


class CounterfactualFlowExperiment(L.LightningModule):
    def __init__(self, model_kwargs, lr=1e-4, weight_decay=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.model = EuclidCounterfactualFlow(**model_kwargs)

    def training_step(self, batch, batch_idx):
        loss, metrics = self.model.loss(batch, training=True)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch["image"]))
        for name, value in metrics.items():
            self.log(f"train_{name}", value, batch_size=len(batch["image"]))
        return loss

    def validation_step(self, batch, batch_idx):
        count = len(batch["image"])
        generator = torch.Generator(device=self.device).manual_seed(10000 + batch_idx)
        time_values = torch.rand(count, generator=generator, device=self.device)
        image_noise = torch.randn(
            batch["image"].shape, generator=generator, device=self.device,
        )
        sfh_noise = torch.randn(
            batch["sfh"].shape, generator=generator, device=self.device,
        )
        results = {}
        for mode in ("paired", "shuffled", "prior_only"):
            loss, metrics = self.model.loss(
                batch, common_mode=mode, training=False, time=time_values,
                image_noise=image_noise, sfh_noise=sfh_noise,
            )
            prefix = "val" if mode == "paired" else f"val_{mode}"
            results[mode] = loss
            self.log(f"{prefix}_loss", loss, prog_bar=mode == "paired",
                     batch_size=count, sync_dist=True)
            for name, value in metrics.items():
                self.log(f"{prefix}_{name}", value, batch_size=count,
                         sync_dist=True)
        self.log(
            "val_common_gain", results["prior_only"] - results["paired"],
            prog_bar=True, batch_size=count, sync_dist=True,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def _read_metadata(dataset):
    with h5py.File(dataset, "r") as source:
        required = (
            "galaxy_id", "sfh", "sfh_time_grid", "sfh_time_norm",
            "phz_pp_median_stellarmass", "redshift",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"Missing counterfactual-flow datasets: {missing}")
        log_sfh = np.asarray(source["sfh"], dtype=np.float32)
        time_grid = np.asarray(source["sfh_time_grid"], dtype=np.float32)
        age_myr = np.asarray(source["sfh_time_norm"], dtype=np.float32)
        mass = np.asarray(source["phz_pp_median_stellarmass"], dtype=np.float32)
        redshift = np.asarray(source["redshift"], dtype=np.float32)
    activity = activity_classes(log_sfh, time_grid, age_myr)
    return log_sfh, mass, redshift, activity


def _prepare(args):
    index = build_pair_index(
        args.dataset, args.stamps, val_fraction=args.val_fraction,
        seed=args.seed, max_pairs=args.max_pairs,
        max_vis_mag=args.max_vis_mag,
        exclude_edge_on_axis_ratio_below=args.exclude_edge_on_axis_ratio_below,
        exclude_edge_on_probability_above=args.exclude_edge_on_probability_above,
    )
    log_sfh, mass, redshift, activity = _read_metadata(args.dataset)
    valid = (
        np.isfinite(mass) & np.isfinite(redshift) & (mass > 0)
        & (redshift >= 0) & (activity >= 0)
    )
    train_rows = index.train_rows[valid[index.train_rows]]
    val_rows = index.val_rows[valid[index.val_rows]]
    if min(len(train_rows), len(val_rows)) < 2:
        raise ValueError("Metadata filtering left an empty split")
    sfh_mean, sfh_scale = fit_sfh_transform(log_sfh, train_rows)
    covariate_mean = np.array([
        np.mean(mass[train_rows]), np.mean(redshift[train_rows]),
    ], dtype=np.float32)
    covariate_scale = np.maximum(np.array([
        np.std(mass[train_rows]), np.std(redshift[train_rows]),
    ], dtype=np.float32), (1e-3, 1e-3))
    train_pools = matched_donor_pools(
        train_rows, train_rows, mass, redshift, activity, args.pool_size,
    )
    val_pools = matched_donor_pools(
        train_rows, val_rows, mass, redshift, activity, args.pool_size,
    )
    shared = dict(
        path=args.dataset, stamp_root=args.stamps, mass=mass,
        redshift=redshift, activity=activity, sfh_mean=sfh_mean,
        sfh_scale=sfh_scale, covariate_mean=covariate_mean,
        covariate_scale=covariate_scale,
        donors_per_item=args.donors, image_size=args.image_size,
    )
    train = CounterfactualEuclidDataset(
        rows=train_rows, donor_pools=train_pools, training=True, **shared,
    )
    val = CounterfactualEuclidDataset(
        rows=val_rows, donor_pools=val_pools, training=False, **shared,
    )
    arrays = dict(
        train_rows=train_rows, val_rows=val_rows,
        train_donor_pools=train_pools, val_donor_pools=val_pools,
        activity_class=activity, sfh_mean=sfh_mean, sfh_scale=sfh_scale,
        covariate_mean=covariate_mean, covariate_scale=covariate_scale,
    )
    return train, val, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-vis-mag", type=float, default=22.0)
    parser.add_argument("--exclude-edge-on-axis-ratio-below", type=float, default=0.5)
    parser.add_argument("--exclude-edge-on-probability-above", type=float, default=0.8)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--donors", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--common-dim", type=int, default=64)
    parser.add_argument("--prior-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--image-encoder-base", type=int, default=16)
    parser.add_argument("--image-flow-base", type=int, default=16)
    parser.add_argument("--sfh-d-model", type=int, default=96)
    parser.add_argument("--sfh-layers", type=int, default=3)
    parser.add_argument("--common-loss-weight", type=float, default=0.05)
    parser.add_argument("--common-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.image_size % 16:
        parser.error("image-size must be divisible by 16")
    if args.donors > args.pool_size:
        parser.error("donors cannot exceed pool-size")
    if args.smoke and args.max_pairs is None:
        args.max_pairs = 2048
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Output directory is nonempty; use --resume or a new output")

    L.seed_everything(args.seed, workers=True)
    train, val, arrays = _prepare(args)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "matched_donors.npz", **arrays)
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    settings.update(train_count=len(train), val_count=len(val),
                    n_sfh_bins=train.sfh_mean.size)
    (args.output / "run.json").write_text(json.dumps(settings, indent=2))

    model_kwargs = dict(
        n_sfh_bins=train.sfh_mean.size, common_dim=args.common_dim,
        prior_dim=args.prior_dim, condition_dim=args.condition_dim,
        image_encoder_base=args.image_encoder_base,
        image_flow_base=args.image_flow_base,
        sfh_d_model=args.sfh_d_model, sfh_layers=args.sfh_layers,
        common_loss_weight=args.common_loss_weight,
        common_dropout=args.common_dropout,
    )
    model = CounterfactualFlowExperiment(model_kwargs, lr=args.lr)
    checkpoint = ModelCheckpoint(
        dirpath=args.output / "checkpoints", monitor="val_loss", mode="min",
        save_last=True, save_top_k=2,
        filename="counterfactual-{epoch:03d}-{val_loss:.5f}",
    )
    callbacks = [checkpoint, EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience,
    )]
    loader = dict(
        batch_size=args.batch_size, num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=args.accelerator == "gpu",
    )
    trainer = L.Trainer(
        accelerator=args.accelerator, devices=1,
        precision="16-mixed" if args.accelerator == "gpu" else "32-true",
        max_epochs=2 if args.smoke else args.epochs,
        callbacks=callbacks,
        logger=CSVLogger(str(args.output), name="logs"),
        gradient_clip_val=1.0, log_every_n_steps=5,
        num_sanity_val_steps=0,
    )
    start = time.monotonic()
    trainer.fit(
        model,
        DataLoader(train, shuffle=True, drop_last=True, **loader),
        DataLoader(val, shuffle=False, **loader),
        ckpt_path=str(args.resume) if args.resume else None,
    )
    report = dict(
        elapsed_seconds=time.monotonic() - start,
        best_checkpoint=checkpoint.best_model_path,
        optimizer_steps=trainer.global_step,
    )
    (args.output / "runtime.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
