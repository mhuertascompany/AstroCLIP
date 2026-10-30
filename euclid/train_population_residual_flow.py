"""Train population VIS/SFH flows, then frozen cross-modal residual flows.

Stage 1 learns p(image | mass, z, activity class) and
p(SFH | mass, z, activity class).  Stage 2 freezes those vector fields and
learns only the paired cross-modal correction R(z)-R(0).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, default_collate

from .counterfactual_dataset import fit_sfh_transform
from .population_residual_dataset import PackedPopulationResidualDataset
from .population_residual_flow import PopulationFlow, ResidualVelocity
from .train_counterfactual_single_neighbor import _metadata, _split_rows


def _prepare(args):
    train_rows, val_rows = _split_rows(args)
    log_sfh, mass, redshift, activity = _metadata(args.dataset)
    good = (
        np.isfinite(mass) & np.isfinite(redshift) &
        (mass > 0) & (redshift >= 0) & (activity >= 0)
    )
    train_rows = train_rows[good[train_rows]]
    val_rows = val_rows[good[val_rows]]
    sfh_mean, sfh_scale = fit_sfh_transform(log_sfh, train_rows)
    mass_mean, mass_scale = float(mass[train_rows].mean()), float(mass[train_rows].std())
    z_mean, z_scale = float(redshift[train_rows].mean()), float(redshift[train_rows].std())
    if min(mass_scale, z_scale) <= 0:
        raise ValueError("Mass and redshift must vary in the training split")
    covariates = np.zeros((len(mass), 5), dtype=np.float32)
    covariates[:, 0] = (mass - mass_mean) / mass_scale
    covariates[:, 1] = (redshift - z_mean) / z_scale
    covariates[np.arange(len(mass)), 2 + np.clip(activity, 0, 2)] = 1
    common = dict(
        path=args.dataset, packed_vis=args.packed_vis,
        sfh_mean=sfh_mean, sfh_scale=sfh_scale, covariates=covariates,
    )
    train = PackedPopulationResidualDataset(
        rows=train_rows, training=True, **common,
    )
    val = PackedPopulationResidualDataset(
        rows=val_rows, training=False, **common,
    )
    arrays = dict(
        train_rows=train_rows, val_rows=val_rows,
        sfh_mean=sfh_mean, sfh_scale=sfh_scale,
        mass_mean=np.float32(mass_mean), mass_scale=np.float32(mass_scale),
        redshift_mean=np.float32(z_mean), redshift_scale=np.float32(z_scale),
        activity_class=activity,
    )
    return train, val, arrays


def _population_kwargs(args, n_sfh_bins):
    return dict(
        n_sfh_bins=n_sfh_bins, covariate_dim=5,
        image_base=args.image_flow_base, sfh_d_model=args.sfh_d_model,
        sfh_heads=args.sfh_heads, sfh_layers=args.sfh_layers,
    )


class PopulationExperiment(L.LightningModule):
    def __init__(self, model_kwargs, lr=1e-4, condition_dropout=0.15,
                 control_batches=8):
        super().__init__()
        self.save_hyperparameters()
        self.model = PopulationFlow(**model_kwargs)

    def training_step(self, batch, batch_idx):
        anchor = "image" if self.global_step % 2 == 0 else "sfh"
        loss = self.model.loss(
            batch, anchor, condition_dropout=self.hparams.condition_dropout,
        )
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch[anchor]))
        self.log(f"train_{anchor}_loss", loss, batch_size=len(batch[anchor]))
        return loss

    def validation_step(self, batch, batch_idx):
        losses = []
        for offset, anchor in enumerate(("image", "sfh")):
            generator = torch.Generator(device=self.device).manual_seed(
                10000 + 2 * batch_idx + offset,
            )
            count = len(batch[anchor])
            time_value = torch.rand(count, generator=generator, device=self.device)
            noise = torch.randn(
                batch[anchor].shape, generator=generator, device=self.device,
            )
            loss = self.model.loss(batch, anchor, time_value, noise)
            self.log(f"val_{anchor}_loss", loss, batch_size=count, sync_dist=True)
            losses.append(loss)
            if batch_idx < self.hparams.control_batches:
                for mode in ("shuffled", "zero"):
                    control = self.model.loss(
                        batch, anchor, time_value, noise, mode=mode,
                    )
                    self.log(
                        f"val_control_{anchor}_{mode}_loss", control,
                        batch_size=count, sync_dist=True,
                    )
        value = torch.stack(losses).mean()
        self.log("val_loss", value, prog_bar=True, batch_size=len(batch["image"]), sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class ResidualExperiment(L.LightningModule):
    def __init__(self, population_checkpoint, model_kwargs, lr=1e-4,
                 control_batches=8):
        super().__init__()
        self.save_hyperparameters()
        population_run = PopulationExperiment.load_from_checkpoint(
            population_checkpoint, map_location="cpu",
        )
        population = copy.deepcopy(population_run.model.velocity)
        self.model = ResidualVelocity(population=population, **model_kwargs)

    def training_step(self, batch, batch_idx):
        anchor = "image" if self.global_step % 2 == 0 else "sfh"
        loss, metrics = self.model.loss(batch, anchor)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch[anchor]))
        self.log(f"train_{anchor}_loss", loss, batch_size=len(batch[anchor]))
        for name, value in metrics.items():
            self.log(f"train_{name}", value, batch_size=len(batch[anchor]))
        return loss

    def validation_step(self, batch, batch_idx):
        paired_losses = []
        for offset, anchor in enumerate(("image", "sfh")):
            generator = torch.Generator(device=self.device).manual_seed(
                20000 + 2 * batch_idx + offset,
            )
            count = len(batch[anchor])
            time_value = torch.rand(count, generator=generator, device=self.device)
            noise = torch.randn(
                batch[anchor].shape, generator=generator, device=self.device,
            )
            paired, metrics = self.model.loss(
                batch, anchor, time_value, noise, "paired",
            )
            self.log(f"val_{anchor}_loss", paired, batch_size=count, sync_dist=True)
            for name, value in metrics.items():
                self.log(f"val_{anchor}_{name}", value, batch_size=count, sync_dist=True)
            paired_losses.append(paired)
            if batch_idx < self.hparams.control_batches:
                for mode in ("shuffled", "zero"):
                    value, branch = self.model.loss(
                        batch, anchor, time_value, noise, mode,
                    )
                    self.log(
                        f"val_control_{anchor}_{mode}_loss", value,
                        batch_size=count, sync_dist=True,
                    )
                    self.log(
                        f"val_control_{anchor}_{mode}_generative_loss",
                        branch["generative_loss"], batch_size=count, sync_dist=True,
                    )
        value = torch.stack(paired_losses).mean()
        self.log("val_loss", value, prog_bar=True, batch_size=len(batch["image"]), sync_dist=True)

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(parameters, lr=self.hparams.lr, weight_decay=0.01)


class Preview(L.Callback):
    def __init__(self, dataset, output, phase, every=2, samples=6, steps=32,
                 seed=1042):
        self.dataset = dataset
        self.output = Path(output)
        self.phase = phase
        self.every = int(every)
        self.steps = int(steps)
        self.seed = int(seed)
        indices = np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int)
        self.batch = default_collate([dataset[int(index)] for index in indices])

    def on_validation_epoch_end(self, trainer, module):
        if not trainer.is_global_zero or ((trainer.current_epoch + 1) % self.every):
            return
        import matplotlib.pyplot as plt

        batch = {name: value.to(module.device) for name, value in self.batch.items()}
        generator = torch.Generator(device=module.device).manual_seed(self.seed)
        noises = {
            anchor: torch.randn(
                batch[anchor].shape, generator=generator, device=module.device,
            ) for anchor in ("image", "sfh")
        }
        if self.phase == "population":
            image_rows = {
                mode: module.model.sample(
                    batch, "image", self.steps, noises["image"], mode,
                ).cpu() for mode in ("paired", "shuffled", "zero")
            }
            sfh_rows = {
                mode: module.model.sample(
                    batch, "sfh", self.steps, noises["sfh"], mode,
                ).cpu() for mode in ("paired", "shuffled", "zero")
            }
        else:
            image_rows = {
                mode: module.model.sample(
                    batch, "image", mode, self.steps, noises["image"],
                ).cpu() for mode in ("paired", "shuffled", "zero")
            }
            sfh_rows = {
                mode: module.model.sample(
                    batch, "sfh", mode, self.steps, noises["sfh"],
                ).cpu() for mode in ("paired", "shuffled", "zero")
            }
        self.output.mkdir(parents=True, exist_ok=True)
        image_path = self.output / f"epoch_{trainer.current_epoch:03d}_images.png"
        rows = {"target": batch["image"].cpu(), **image_rows}
        figure, axes = plt.subplots(
            len(rows), len(batch["image"]),
            figsize=(2 * len(batch["image"]), 2 * len(rows)), squeeze=False,
        )
        for row, (label, images) in enumerate(rows.items()):
            for column, image in enumerate(images[:, 0]):
                axes[row, column].imshow(image, cmap="gray", vmin=-1, vmax=1)
                axes[row, column].set_axis_off()
                if column == 0:
                    axes[row, column].set_title(label, loc="left", fontsize=8)
        figure.tight_layout()
        figure.savefig(image_path, dpi=150)
        plt.close(figure)

        sfh_path = self.output / f"epoch_{trainer.current_epoch:03d}_sfhs.png"
        mean = torch.as_tensor(self.dataset.sfh_mean)
        scale = torch.as_tensor(self.dataset.sfh_scale)
        target = torch.softmax(batch["sfh"].cpu() * scale + mean, dim=-1)
        curves = {
            label: torch.softmax(values * scale + mean, dim=-1)
            for label, values in sfh_rows.items()
        }
        x = np.linspace(0, 1, target.shape[1])
        figure, axes = plt.subplots(
            len(target), 1, figsize=(8, 2 * len(target)), squeeze=False, sharex=True,
        )
        colors = {"paired": "#0072B2",
                  "shuffled": "#D55E00", "zero": "#999999"}
        for index, axis in enumerate(axes[:, 0]):
            axis.plot(x, target[index], color="black", lw=1.5, label="target")
            for label, values in curves.items():
                axis.plot(x, values[index], color=colors[label], lw=1, label=label)
        axes[0, 0].legend(ncol=4, fontsize=7)
        axes[-1, 0].set_xlabel("Fractional lookback time")
        figure.tight_layout()
        figure.savefig(sfh_path, dpi=150)
        plt.close(figure)
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                logger.experiment.log({
                    "generations/images": wandb.Image(str(image_path)),
                    "generations/sfhs": wandb.Image(str(sfh_path)),
                    "epoch": trainer.current_epoch,
                })


def _effective_rank(values):
    values = values - values.mean(0, keepdims=True)
    singular = np.linalg.svd(values, compute_uv=False)
    probability = singular**2
    probability /= max(probability.sum(), np.finfo(float).eps)
    return float(np.exp(-np.sum(probability * np.log(probability + 1e-12))))


class EmbeddingPreview(L.Callback):
    def __init__(self, dataset, output, every=2, samples=512, seed=2042):
        self.dataset = dataset
        self.output = Path(output)
        self.every = int(every)
        rng = np.random.default_rng(seed)
        self.indices = rng.choice(len(dataset), min(samples, len(dataset)), replace=False)

    def on_validation_epoch_end(self, trainer, module):
        if not trainer.is_global_zero or ((trainer.current_epoch + 1) % self.every):
            return
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        batch = default_collate([self.dataset[int(index)] for index in self.indices])
        with torch.inference_mode():
            image, sfh = module.model.embeddings(
                batch["image"].to(module.device), batch["sfh"].to(module.device),
            )
        image = image.float().cpu().numpy()
        sfh = sfh.float().cpu().numpy()
        combined = np.r_[image, sfh]
        coordinates = PCA(2, random_state=42).fit_transform(combined)
        count = len(image)
        paired = np.sum(image * sfh, axis=1) / (
            np.linalg.norm(image, axis=1) * np.linalg.norm(sfh, axis=1) + 1e-8
        )
        shuffled = np.sum(image * np.roll(sfh, 1, axis=0), axis=1) / (
            np.linalg.norm(image, axis=1) * np.linalg.norm(np.roll(sfh, 1, axis=0), axis=1) + 1e-8
        )
        metrics = {
            "image_effective_rank": _effective_rank(image),
            "sfh_effective_rank": _effective_rank(sfh),
            "paired_cosine_median": float(np.median(paired)),
            "shuffled_cosine_median": float(np.median(shuffled)),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.output / f"epoch_{trainer.current_epoch:03d}.npz",
            galaxy_id=batch["galaxy_id"].numpy(), image_embedding=image,
            sfh_embedding=sfh, **metrics,
        )
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(coordinates[:count, 0], coordinates[:count, 1], s=8,
                     alpha=0.6, label="VIS")
        axis.scatter(coordinates[count:, 0], coordinates[count:, 1], s=8,
                     alpha=0.6, label="SFH")
        for index in range(min(count, 100)):
            axis.plot(
                [coordinates[index, 0], coordinates[count + index, 0]],
                [coordinates[index, 1], coordinates[count + index, 1]],
                color="0.8", lw=0.35, zorder=0,
            )
        axis.legend()
        axis.set_title(
            f"Residual embeddings · paired cos={np.median(paired):.3f} · "
            f"shuffled={np.median(shuffled):.3f}"
        )
        figure.tight_layout()
        path = self.output / f"epoch_{trainer.current_epoch:03d}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        for name, value in metrics.items():
            module.log(f"embedding_{name}", value)
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                logger.experiment.log({
                    "embeddings/paired_space": wandb.Image(str(path)),
                    **{f"embeddings/{name}": value for name, value in metrics.items()},
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("population", "residual"), required=True)
    parser.add_argument("--population-checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--packed-vis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-vis-mag", type=float, default=22.0)
    parser.add_argument("--exclude-edge-on-axis-ratio-below", type=float, default=0.5)
    parser.add_argument("--exclude-edge-on-probability-above", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--image-encoder-base", type=int, default=16)
    parser.add_argument("--image-flow-base", type=int, default=16)
    parser.add_argument("--sfh-d-model", type=int, default=96)
    parser.add_argument("--sfh-heads", type=int, default=4)
    parser.add_argument("--sfh-layers", type=int, default=3)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--condition-dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--control-batches", type=int, default=8)
    parser.add_argument("--preview-every", type=int, default=2)
    parser.add_argument("--preview-samples", type=int, default=6)
    parser.add_argument("--preview-steps", type=int, default=32)
    parser.add_argument("--embedding-every", type=int, default=2)
    parser.add_argument("--embedding-samples", type=int, default=512)
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    args = parser.parse_args()
    if args.phase == "residual" and args.population_checkpoint is None:
        parser.error("--population-checkpoint is required for the residual phase")
    if args.smoke and args.max_pairs is None:
        args.max_pairs = 2048
    if args.wandb_mode != "disabled" and not args.wandb_project:
        parser.error("--wandb-project is required for W&B logging")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Output is nonempty; pass --resume or use a new output")
    torch.set_float32_matmul_precision("high")
    L.seed_everything(args.seed, workers=True)
    train, val, arrays = _prepare(args)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "data_transform.npz", **arrays)
    run = {key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()}
    run.update(train_count=len(train), val_count=len(val),
               residual_definition="R(z)-R(0)" if args.phase == "residual" else None)
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    if args.phase == "population":
        model = PopulationExperiment(
            _population_kwargs(args, train.sfh_mean.size), args.lr,
            args.condition_dropout, args.control_batches,
        )
    else:
        kwargs = dict(
            n_sfh_bins=train.sfh_mean.size, embedding_dim=args.embedding_dim,
            image_encoder_base=args.image_encoder_base,
            image_flow_base=args.image_flow_base,
            sfh_d_model=args.sfh_d_model, sfh_heads=args.sfh_heads,
            sfh_layers=args.sfh_layers, alignment_weight=args.alignment_weight,
            variance_weight=args.variance_weight,
        )
        model = ResidualExperiment(
            str(args.population_checkpoint), kwargs, args.lr,
            args.control_batches,
        )
    checkpoint = ModelCheckpoint(
        dirpath=args.output / "checkpoints", monitor="val_loss", mode="min",
        save_last=True, save_top_k=2,
        filename=f"{args.phase}-{{epoch:03d}}-{{val_loss:.5f}}",
    )
    callbacks = [
        checkpoint,
        EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
        LearningRateMonitor(logging_interval="epoch"),
        Preview(
            val, args.output / "previews", args.phase,
            1 if args.smoke else args.preview_every,
            args.preview_samples, args.preview_steps, args.seed + 1000,
        ),
    ]
    if args.phase == "residual":
        callbacks.append(EmbeddingPreview(
            val, args.output / "embeddings",
            1 if args.smoke else args.embedding_every,
            args.embedding_samples, args.seed + 2000,
        ))
    loggers = [CSVLogger(str(args.output), name="logs")]
    if args.wandb_mode != "disabled":
        loggers.append(WandbLogger(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.output.name, save_dir=str(args.output),
            offline=args.wandb_mode == "offline", log_model=False,
        ))
    loader = dict(
        batch_size=args.batch_size, num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=args.accelerator == "gpu",
    )
    if args.workers > 0:
        loader["prefetch_factor"] = 4
    trainer = L.Trainer(
        accelerator=args.accelerator, devices=1,
        precision="16-mixed" if args.accelerator == "gpu" else "32-true",
        max_epochs=2 if args.smoke else args.epochs,
        callbacks=callbacks, logger=loggers, gradient_clip_val=1.0,
        log_every_n_steps=5, num_sanity_val_steps=0,
    )
    start = time.monotonic()
    trainer.fit(
        model,
        DataLoader(train, shuffle=True, drop_last=True, **loader),
        DataLoader(val, shuffle=False, **loader),
        ckpt_path=str(args.resume) if args.resume else None,
    )
    (args.output / "runtime.json").write_text(json.dumps({
        "elapsed_seconds": time.monotonic() - start,
        "best_checkpoint": checkpoint.best_model_path,
        "optimizer_steps": trainer.global_step,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30
            if args.accelerator == "gpu" else None
        ),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
