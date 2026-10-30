"""Train the paper-faithful galaxy-counter VIS branch with SFH conditioning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, default_collate

from .galaxy_counter_sfh_vis import (
    GalaxyCounterSFHToVIS, UPSTREAM_COMMIT, UPSTREAM_URL,
)
from .train_counterfactual_single_neighbor import _prepare


class Experiment(L.LightningModule):
    def __init__(self, model_kwargs, lr=1e-4, control_batches=8):
        super().__init__()
        self.save_hyperparameters()
        self.model = GalaxyCounterSFHToVIS(**model_kwargs)

    def training_step(self, batch, batch_idx):
        # Match galaxy-counter's idx-parity survey swap: exactly one modality
        # is the generative anchor on each optimizer step.
        anchor = "image" if int(self.global_step) % 2 == 0 else "sfh"
        loss, metrics = self.model.loss(batch, anchor=anchor)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch["image"]))
        self.log("train_anchor_is_image", float(anchor == "image"),
                 batch_size=len(batch["image"]))
        for name, value in metrics.items():
            self.log(f"train_{name}", value, batch_size=len(batch["image"]))
        return loss

    def validation_step(self, batch, batch_idx):
        count = len(batch["image"])
        generator = torch.Generator(device=self.device).manual_seed(10000 + batch_idx)
        time = torch.rand(count, generator=generator, device=self.device)
        image_noise = torch.randn(
            batch["image"].shape, generator=generator, device=self.device,
        )
        sfh_noise = torch.randn(
            batch["sfh"].shape, generator=generator, device=self.device,
        )
        image_loss, image_metrics = self.model.loss(
            batch, anchor="image", time=time,
            image_noise=image_noise, sfh_noise=sfh_noise,
        )
        sfh_loss, sfh_metrics = self.model.loss(
            batch, anchor="sfh", time=time,
            image_noise=image_noise, sfh_noise=sfh_noise,
        )
        loss = 0.5 * (image_loss + sfh_loss)
        self.log("val_loss", loss, prog_bar=True, batch_size=count, sync_dist=True)
        self.log("val_image_anchor_loss", image_loss, batch_size=count, sync_dist=True)
        self.log("val_sfh_anchor_loss", sfh_loss, batch_size=count, sync_dist=True)
        self.log("val_image_flow_loss", image_metrics["image_flow_loss"],
                 batch_size=count, sync_dist=True)
        self.log("val_sfh_flow_loss", sfh_metrics["sfh_flow_loss"],
                 batch_size=count, sync_dist=True)
        self.log("val_geometric_loss", image_metrics["geometric_loss"],
                 batch_size=count, sync_dist=True)
        if batch_idx < self.hparams.control_batches:
            for name, common, private in (
                ("wrong_common", "shuffled", "paired"),
                ("zero_common", "zero", "paired"),
                ("wrong_private", "paired", "shuffled"),
                ("zero_private", "paired", "zero"),
            ):
                controlled_image_loss, controlled_image = self.model.loss(
                    batch, "image", common, private, time=time,
                    image_noise=image_noise, sfh_noise=sfh_noise,
                )
                controlled_sfh_loss, controlled_sfh = self.model.loss(
                    batch, "sfh", common, private, time=time,
                    image_noise=image_noise, sfh_noise=sfh_noise,
                )
                self.log(f"val_control_{name}_image_anchor_loss",
                         controlled_image_loss,
                         batch_size=count, sync_dist=True)
                self.log(f"val_control_{name}_sfh_anchor_loss",
                         controlled_sfh_loss,
                         batch_size=count, sync_dist=True)
                self.log(f"val_control_{name}_image_flow_loss",
                         controlled_image["image_flow_loss"],
                         batch_size=count, sync_dist=True)
                self.log(f"val_control_{name}_sfh_flow_loss",
                         controlled_sfh["sfh_flow_loss"],
                         batch_size=count, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class Preview(L.Callback):
    def __init__(self, dataset, output, every=1, samples=6, steps=250, seed=1042):
        self.dataset = dataset
        self.batch = default_collate([
            dataset[int(index)] for index in np.linspace(
                0, len(dataset) - 1, min(samples, len(dataset)), dtype=int,
            )
        ])
        self.output = Path(output)
        self.every, self.steps, self.seed = int(every), int(steps), int(seed)

    def on_validation_epoch_end(self, trainer, module):
        if not trainer.is_global_zero or (trainer.current_epoch + 1) % self.every:
            return
        import matplotlib.pyplot as plt
        batch = {key: value.to(module.device) for key, value in self.batch.items()}
        generator = torch.Generator(device=module.device).manual_seed(self.seed)
        image_noise = torch.randn(
            batch["image"].shape, generator=generator, device=module.device,
        )
        sfh_noise = torch.randn(
            batch["sfh"].shape, generator=generator, device=module.device,
        )
        modes = {
            "paired": ("paired", "paired"),
            "wrong common": ("shuffled", "paired"),
            "zero common": ("zero", "paired"),
            "wrong private": ("paired", "shuffled"),
            "zero private": ("paired", "zero"),
        }
        generated = {
            name: tuple(value.cpu() for value in module.model.sample(
                batch, common, private, self.steps, image_noise, sfh_noise,
            ))
            for name, (common, private) in modes.items()
        }
        rows = {
            "real": batch["image"].cpu(),
            "private donor": batch["donor_image"].cpu(),
            **{name: values[0] for name, values in generated.items()},
        }
        figure, axes = plt.subplots(
            len(rows), len(batch["image"]),
            figsize=(2 * len(batch["image"]), 2 * len(rows)), squeeze=False,
        )
        for row, (label, images) in enumerate(rows.items()):
            for column, image in enumerate(images[:, 0]):
                axes[row, column].imshow(image, cmap="gray", vmin=-1, vmax=1)
                axes[row, column].axis("off")
                if column == 0:
                    axes[row, column].set_title(label, loc="left", fontsize=8)
        figure.tight_layout()
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"epoch_{trainer.current_epoch:03d}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)

        sfh_path = self.output / f"epoch_{trainer.current_epoch:03d}_sfhs.png"
        mean = torch.as_tensor(self.dataset.sfh_mean)
        scale = torch.as_tensor(self.dataset.sfh_scale)
        target = torch.softmax(batch["sfh"].cpu() * scale + mean, dim=-1)
        donor = torch.softmax(batch["donor_sfh"].cpu() * scale + mean, dim=-1)
        generated_sfh = {
            name: torch.softmax(values[1] * scale + mean, dim=-1)
            for name, values in generated.items()
        }
        x = np.linspace(0, 1, target.shape[1])
        figure, axes = plt.subplots(
            len(target), 1, figsize=(8, 2 * len(target)),
            squeeze=False, sharex=True,
        )
        colors = {
            "paired": "#0072B2", "wrong common": "#D55E00",
            "zero common": "#999999", "wrong private": "#009E73",
            "zero private": "#CC79A7",
        }
        for index, axis in enumerate(axes[:, 0]):
            axis.plot(x, target[index], color="black", lw=1.6, label="target")
            axis.plot(x, donor[index], color="0.65", lw=1, ls="--",
                      label="private donor")
            for name, values in generated_sfh.items():
                axis.plot(x, values[index], color=colors[name], lw=1, label=name)
        axes[0, 0].legend(ncol=4, fontsize=7)
        axes[-1, 0].set_xlabel("Fractional lookback time")
        figure.tight_layout()
        figure.savefig(sfh_path, dpi=150)
        plt.close(figure)

        paired_image, paired_sfh = generated["paired"]
        sensitivity = {}
        for name, (images, sfhs) in generated.items():
            if name == "paired":
                continue
            key = name.replace(" ", "_")
            sensitivity[f"conditioning/{key}_image_mean_abs_delta"] = (
                images - paired_image
            ).abs().mean().item()
            sensitivity[f"conditioning/{key}_sfh_mean_abs_delta"] = (
                sfhs - paired_sfh
            ).abs().mean().item()
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                logger.experiment.log({
                    "generations/fixed_noise_images": wandb.Image(str(path)),
                    "generations/fixed_noise_sfhs": wandb.Image(str(sfh_path)),
                    **sensitivity,
                    "epoch": trainer.current_epoch,
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--packed-vis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-vis-mag", type=float, default=22.0)
    parser.add_argument("--exclude-edge-on-axis-ratio-below", type=float, default=0.5)
    parser.add_argument("--exclude-edge-on-probability-above", type=float, default=0.8)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--donor-reference-all", action="store_true")
    parser.add_argument("--donor-matching", choices=("mass_redshift_activity", "mass_window"), default="mass_window")
    parser.add_argument("--mass-window", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model-channels", type=int, default=128)
    parser.add_argument("--common-stem-dim", type=int, default=128)
    parser.add_argument("--common-trunk-layers", type=int, default=2)
    parser.add_argument("--common-trunk-heads", type=int, default=8)
    parser.add_argument("--geometric-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--control-batches", type=int, default=8)
    parser.add_argument("--fixed-training-donors", action="store_true")
    parser.add_argument("--preview-training", action="store_true")
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--preview-every", type=int, default=1)
    parser.add_argument("--preview-samples", type=int, default=6)
    parser.add_argument("--preview-steps", type=int, default=250)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output is nonempty: {args.output}")
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    train, val, arrays, diagnostics = _prepare(args)
    if args.fixed_training_donors:
        # Diagnostic memorization mode: donor identity and image orientation
        # must remain fixed so conditioning is the only changing signal.
        train.training = False
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "matched_neighbors.npz", **arrays)
    (args.output / "run.json").write_text(json.dumps({
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "upstream_url": UPSTREAM_URL, "upstream_commit": UPSTREAM_COMMIT,
        "anchor_strategy": "alternate_image_sfh_per_optimizer_step",
        "generative_losses_per_training_step": 1,
        "train_count": len(train), "val_count": len(val),
        "donor_diagnostics": diagnostics,
    }, indent=2) + "\n")
    model = Experiment(dict(
        n_sfh_bins=train.sfh_mean.size, image_size=train.image_size,
        model_channels=args.model_channels, geometric_weight=args.geometric_weight,
        common_stem_dim=args.common_stem_dim,
        common_trunk_layers=args.common_trunk_layers,
        common_trunk_heads=args.common_trunk_heads,
    ), lr=args.lr, control_batches=args.control_batches)
    checkpoint = ModelCheckpoint(
        dirpath=args.output / "checkpoints", monitor="val_loss", mode="min",
        save_last=True, save_top_k=2,
        filename="galaxy-counter-sfh-vis-{epoch:03d}-{val_loss:.5f}",
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
        persistent_workers=args.workers > 0, pin_memory=True,
    )
    if args.workers:
        loader["prefetch_factor"] = 4
    preview_dataset = train if args.preview_training else val
    trainer = L.Trainer(
        accelerator="gpu", devices=1, precision="bf16-mixed",
        max_epochs=args.epochs, callbacks=[checkpoint, Preview(
            preview_dataset, args.output / "previews", args.preview_every,
            args.preview_samples, args.preview_steps, args.seed + 1000,
        )], logger=loggers, log_every_n_steps=5, num_sanity_val_steps=0,
        check_val_every_n_epoch=args.validate_every,
    )
    start = time.monotonic()
    trainer.fit(
        model,
        DataLoader(train, shuffle=True, drop_last=True, **loader),
        DataLoader(val, shuffle=False, **loader),
    )
    (args.output / "runtime.json").write_text(json.dumps({
        "elapsed_seconds": time.monotonic() - start,
        "best_checkpoint": checkpoint.best_model_path,
        "optimizer_steps": trainer.global_step,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
