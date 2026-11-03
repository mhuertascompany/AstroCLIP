"""Train the masked multimodal JEPA on paired Euclid VIS images and SFHs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, default_collate

from .multimodal_jepa import MultimodalJEPA, TASKS
from .train_population_residual_flow import _prepare


class JEPAExperiment(L.LightningModule):
    def __init__(self, model_kwargs, lr=2e-4, weight_decay=0.05,
                 ema_start=0.996, ema_end=0.9999,
                 control_batches=8):
        super().__init__()
        self.save_hyperparameters()
        self.model = MultimodalJEPA(**model_kwargs)

    def training_step(self, batch, batch_idx):
        task = TASKS[int(self.global_step) % len(TASKS)]
        loss, metrics = self.model.loss(batch["image"], batch["sfh"], task)
        count = len(batch["image"])
        self.log("train_loss", loss, prog_bar=True, batch_size=count)
        self.log(f"train_{task}_loss", loss, batch_size=count)
        for name, value in metrics.items():
            self.log(f"train_{task}_{name}", value, batch_size=count)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        total = max(1, int(self.trainer.estimated_stepping_batches))
        progress = min(1.0, float(self.global_step) / total)
        momentum = (
            self.hparams.ema_start
            + (self.hparams.ema_end - self.hparams.ema_start) * progress
        )
        self.model.update_target(momentum)
        self.log("ema_momentum", momentum, prog_bar=False)

    def _generator(self, seed):
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def validation_step(self, batch, batch_idx):
        image, sfh = batch["image"], batch["sfh"]
        count = len(image)
        losses = []
        for task_index, task in enumerate(TASKS):
            seed = 100000 + 100 * batch_idx + task_index
            loss, metrics = self.model.loss(
                image, sfh, task, self._generator(seed),
            )
            losses.append(loss)
            self.log(f"val_{task}_loss", loss, batch_size=count, sync_dist=True)
            for name, value in metrics.items():
                self.log(
                    f"val_{task}_{name}", value,
                    batch_size=count, sync_dist=True,
                )
            if (
                task in ("sfh_to_image", "image_to_sfh")
                and batch_idx < self.hparams.control_batches
            ):
                shuffled, shuffled_metrics = self.model.loss(
                    image, sfh, task, self._generator(seed),
                    shuffle_source=True,
                )
                self.log(
                    f"val_{task}_shuffled_loss", shuffled,
                    batch_size=count, sync_dist=True,
                )
                self.log(
                    f"val_{task}_shuffled_penalty", shuffled - loss,
                    batch_size=count, sync_dist=True,
                )
                self.log(
                    f"val_{task}_shuffled_token_cosine",
                    shuffled_metrics["token_cosine"],
                    batch_size=count, sync_dist=True,
                )
        value = torch.stack(losses).mean()
        self.log("val_loss", value, prog_bar=True, batch_size=count, sync_dist=True)

    def configure_optimizers(self):
        parameters = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad
        ]
        return torch.optim.AdamW(
            parameters, lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )


def _effective_rank(values):
    values = values - values.mean(0, keepdims=True)
    singular = np.linalg.svd(values, compute_uv=False)
    probability = singular**2
    probability /= max(probability.sum(), np.finfo(float).eps)
    return float(np.exp(-np.sum(probability * np.log(probability + 1e-12))))


class EmbeddingPreview(L.Callback):
    def __init__(self, dataset, output, every=2, samples=512,
                 batch_size=64, seed=2042):
        self.dataset = dataset
        self.output = Path(output)
        self.every = int(every)
        self.batch_size = int(batch_size)
        rng = np.random.default_rng(seed)
        self.indices = rng.choice(
            len(dataset), min(samples, len(dataset)), replace=False,
        )

    def on_validation_epoch_end(self, trainer, module):
        if not trainer.is_global_zero or ((trainer.current_epoch + 1) % self.every):
            return
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        collected = {mode: [] for mode in ("joint", "image", "sfh")}
        ids = []
        was_training = module.training
        module.eval()
        with torch.inference_mode():
            for start in range(0, len(self.indices), self.batch_size):
                batch = default_collate([
                    self.dataset[int(index)]
                    for index in self.indices[start:start + self.batch_size]
                ])
                image = batch["image"].to(module.device)
                sfh = batch["sfh"].to(module.device)
                for mode in collected:
                    collected[mode].append(
                        module.model.embed(image, sfh, mode).float().cpu().numpy()
                    )
                ids.append(batch["galaxy_id"].numpy())
        module.train(was_training)
        values = {
            mode: np.concatenate(parts) for mode, parts in collected.items()
        }
        galaxy_id = np.concatenate(ids)
        count = len(galaxy_id)
        combined = np.r_[values["joint"], values["image"], values["sfh"]]
        coordinates = PCA(2, random_state=42).fit_transform(combined)
        joint_xy = coordinates[:count]
        image_xy = coordinates[count:2 * count]
        sfh_xy = coordinates[2 * count:]

        def cosine(first, second):
            return np.sum(first * second, axis=1) / (
                np.linalg.norm(first, axis=1)
                * np.linalg.norm(second, axis=1) + 1e-8
            )

        metrics = {
            f"{mode}_effective_rank": _effective_rank(value)
            for mode, value in values.items()
        }
        metrics.update(
            image_sfh_cosine_median=float(np.median(cosine(
                values["image"], values["sfh"],
            ))),
            image_joint_cosine_median=float(np.median(cosine(
                values["image"], values["joint"],
            ))),
            sfh_joint_cosine_median=float(np.median(cosine(
                values["sfh"], values["joint"],
            ))),
        )
        self.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.output / f"epoch_{trainer.current_epoch:03d}.npz",
            galaxy_id=galaxy_id, joint_embedding=values["joint"],
            image_embedding=values["image"], sfh_embedding=values["sfh"],
            **metrics,
        )
        figure, axis = plt.subplots(figsize=(7, 6))
        for index in range(min(count, 100)):
            axis.plot(
                [image_xy[index, 0], joint_xy[index, 0], sfh_xy[index, 0]],
                [image_xy[index, 1], joint_xy[index, 1], sfh_xy[index, 1]],
                color="0.85", lw=0.35, zorder=0,
            )
        axis.scatter(joint_xy[:, 0], joint_xy[:, 1], s=8, alpha=0.55,
                     label="paired joint")
        axis.scatter(image_xy[:, 0], image_xy[:, 1], s=8, alpha=0.55,
                     label="VIS only")
        axis.scatter(sfh_xy[:, 0], sfh_xy[:, 1], s=8, alpha=0.55,
                     label="SFH only")
        axis.set_xlabel("Joint PCA 1")
        axis.set_ylabel("Joint PCA 2")
        axis.set_title(
            f"Epoch {trainer.current_epoch} · shared online encoder · "
            f"rank(joint)={metrics['joint_effective_rank']:.1f}"
        )
        axis.legend()
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
                    "embeddings/joint_space": wandb.Image(str(path)),
                    **{f"embeddings/{name}": value
                       for name, value in metrics.items()},
                    "epoch": trainer.current_epoch,
                })


class CrossModalErrorPreview(L.Callback):
    def __init__(self, dataset, output, every=2, samples=32, seed=3042):
        self.output = Path(output)
        self.every = int(every)
        indices = np.linspace(
            0, len(dataset) - 1, min(samples, len(dataset)), dtype=int,
        )
        self.batch = default_collate([
            dataset[int(index)] for index in indices
        ])

    def on_validation_epoch_end(self, trainer, module):
        if not trainer.is_global_zero or ((trainer.current_epoch + 1) % self.every):
            return
        import matplotlib.pyplot as plt

        batch = {
            name: value.to(module.device)
            for name, value in self.batch.items()
        }
        image, sfh = batch["image"], batch["sfh"]
        was_training = module.training
        module.eval()
        with torch.inference_mode():
            image_true = module.model.cross_modal_errors(
                image, sfh, "sfh_to_image",
            ).mean(0).cpu().numpy()
            image_shuffled = module.model.cross_modal_errors(
                image, sfh, "sfh_to_image", True,
            ).mean(0).cpu().numpy()
            sfh_true = module.model.cross_modal_errors(
                image, sfh, "image_to_sfh",
            ).mean(0).cpu().numpy()
            sfh_shuffled = module.model.cross_modal_errors(
                image, sfh, "image_to_sfh", True,
            ).mean(0).cpu().numpy()
        module.train(was_training)
        side = module.model.online.image_side
        self.output.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(2, 3, figsize=(13, 7))
        maps = (
            (image_true.reshape(side, side), "SFH→VIS: true pair"),
            (image_shuffled.reshape(side, side), "SFH→VIS: shuffled"),
            ((image_shuffled - image_true).reshape(side, side), "shuffled − true"),
        )
        for axis, (values, title) in zip(axes[0], maps):
            artist = axis.imshow(values, cmap="magma" if "−" not in title else "coolwarm")
            axis.set_title(title)
            figure.colorbar(artist, ax=axis, fraction=0.046)
        x = np.linspace(0, 1, len(sfh_true))
        axes[1, 0].plot(x, sfh_true, label="true")
        axes[1, 0].plot(x, sfh_shuffled, label="shuffled")
        axes[1, 0].set_title("VIS→SFH latent error")
        axes[1, 0].legend()
        axes[1, 1].plot(x, sfh_shuffled - sfh_true, color="#D55E00")
        axes[1, 1].axhline(0, color="0.5", lw=0.7)
        axes[1, 1].set_title("Shuffled − true error")
        axes[1, 2].axis("off")
        for axis in axes[1, :2]:
            axis.set_xlabel("Fractional lookback time")
            axis.set_ylabel("1 − latent cosine")
        figure.tight_layout()
        path = self.output / f"epoch_{trainer.current_epoch:03d}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                logger.experiment.log({
                    "prediction/cross_modal_error": wandb.Image(str(path)),
                    "epoch": trainer.current_epoch,
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--no-early-stopping", action="store_true",
        help="Run to --epochs even when the moving-target validation loss rises.",
    )
    parser.add_argument("--image-patch", type=int, default=8)
    parser.add_argument("--sfh-patch", type=int, default=5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--predictor-width", type=int, default=192)
    parser.add_argument("--predictor-heads", type=int, default=6)
    parser.add_argument("--predictor-layers", type=int, default=3)
    parser.add_argument("--global-weight", type=float, default=0.0)
    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--ema-end", type=float, default=0.9999)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--control-batches", type=int, default=8)
    parser.add_argument("--preview-every", type=int, default=2)
    parser.add_argument("--embedding-samples", type=int, default=512)
    parser.add_argument("--error-samples", type=int, default=32)
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    args = parser.parse_args()
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
    run = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run.update(train_count=len(train), val_count=len(val), tasks=list(TASKS))
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    model_kwargs = dict(
        image_size=train.image_size, image_patch=args.image_patch,
        n_sfh_bins=train.sfh_mean.size, sfh_patch=args.sfh_patch,
        width=args.width, heads=args.heads, layers=args.layers,
        predictor_width=args.predictor_width,
        predictor_heads=args.predictor_heads,
        predictor_layers=args.predictor_layers,
        global_weight=args.global_weight,
    )
    model = JEPAExperiment(
        model_kwargs, args.lr, args.weight_decay,
        args.ema_start, args.ema_end, args.control_batches,
    )
    checkpoint = ModelCheckpoint(
        dirpath=args.output / "checkpoints", monitor="val_loss", mode="min",
        save_last=True, save_top_k=2,
        filename="multimodal-jepa-{epoch:03d}-{val_loss:.5f}",
    )
    callbacks = [
        checkpoint,
        LearningRateMonitor(logging_interval="epoch"),
        EmbeddingPreview(
            val, args.output / "embeddings",
            every=1 if args.smoke else args.preview_every,
            samples=args.embedding_samples, seed=args.seed + 2000,
        ),
        CrossModalErrorPreview(
            val, args.output / "prediction_errors",
            every=1 if args.smoke else args.preview_every,
            samples=args.error_samples, seed=args.seed + 3000,
        ),
    ]
    if not args.no_early_stopping:
        callbacks.insert(
            1,
            EarlyStopping(
                monitor="val_loss", mode="min", patience=args.patience,
            ),
        )
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
