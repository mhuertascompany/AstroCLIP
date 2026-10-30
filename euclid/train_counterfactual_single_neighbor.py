"""Train the one-neighbour VIS/SFH counterfactual flow on packed VIS arrays."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, default_collate

from .counterfactual_dataset import (
    activity_classes,
    fit_sfh_transform,
    mass_window_donor_pools,
    matched_donor_pools,
)
from .counterfactual_packed_dataset import PackedCounterfactualDataset
from .counterfactual_single_neighbor_flow import EuclidSingleNeighborFlow
from .training_index import edge_on_exclusion_mask, inspect_sfh_file
from .vis_selection import bright_row_mask


def _split_rows(args):
    ids, _, _ = inspect_sfh_file(args.dataset)
    packed_ids = np.load(args.packed_vis / "galaxy_id.npy", mmap_mode="r")
    valid = np.asarray(np.load(args.packed_vis / "valid.npy", mmap_mode="r"), dtype=bool)
    if not np.array_equal(ids, packed_ids):
        raise ValueError("Packed VIS IDs do not match the training HDF5")
    bright, _ = bright_row_mask(
        args.dataset, args.max_vis_mag, "flux_detection_total", "vis_det", True,
    )
    eligible = valid & bright
    if args.exclude_edge_on_axis_ratio_below is not None:
        eligible &= ~edge_on_exclusion_mask(
            args.dataset, args.exclude_edge_on_axis_ratio_below,
            args.exclude_edge_on_probability_above,
        )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ids))
    is_validation = np.zeros(len(ids), dtype=bool)
    is_validation[order[:max(1, round(args.val_fraction * len(ids)))]] = True
    train = np.flatnonzero(eligible & ~is_validation)
    val = np.flatnonzero(eligible & is_validation)
    if args.max_pairs is not None and len(train) + len(val) > args.max_pairs:
        n_val = max(1, round(args.max_pairs * args.val_fraction))
        val = rng.permutation(val)[:n_val]
        train = rng.permutation(train)[:args.max_pairs - n_val]
    if min(len(train), len(val)) < 2:
        raise ValueError("Packed VIS selection left an empty split")
    return train.astype(np.int64), val.astype(np.int64)


def _metadata(path):
    with h5py.File(path, "r") as source:
        log_sfh = np.asarray(source["sfh"], dtype=np.float32)
        time_grid = np.asarray(source["sfh_time_grid"], dtype=np.float32)
        age_myr = np.asarray(source["sfh_time_norm"], dtype=np.float32)
        mass = np.asarray(source["phz_pp_median_stellarmass"], dtype=np.float32)
        redshift = np.asarray(source["redshift"], dtype=np.float32)
    return log_sfh, mass, redshift, activity_classes(log_sfh, time_grid, age_myr)


def _prepare(args):
    train_rows, val_rows = _split_rows(args)
    log_sfh, mass, redshift, activity = _metadata(args.dataset)
    good = np.isfinite(mass) & np.isfinite(redshift) & (mass > 0) & (redshift >= 0) & (activity >= 0)
    train_rows = train_rows[good[train_rows]]
    val_rows = val_rows[good[val_rows]]
    sfh_mean, sfh_scale = fit_sfh_transform(log_sfh, train_rows)
    if args.donor_matching == "mass_window":
        train_pools = mass_window_donor_pools(
            train_rows, train_rows, mass, args.mass_window,
            args.pool_size, args.seed + 10,
        )
        val_pools = mass_window_donor_pools(
            train_rows, val_rows, mass, args.mass_window,
            args.pool_size, args.seed + 20,
        )
    else:
        train_pools = matched_donor_pools(
            train_rows, train_rows, mass, redshift, activity, args.pool_size,
        )
        val_pools = matched_donor_pools(
            train_rows, val_rows, mass, redshift, activity, args.pool_size,
        )
    val_target = np.repeat(val_rows[:, None], val_pools.shape[1], axis=1)
    donor_diagnostics = {
        "median_abs_delta_log_mass": float(np.median(np.abs(
            mass[val_pools] - mass[val_target]
        ))),
        "median_abs_delta_redshift": float(np.median(np.abs(
            redshift[val_pools] - redshift[val_target]
        ))),
        "same_activity_fraction": float(np.mean(
            activity[val_pools] == activity[val_target]
        )),
    }
    common = dict(
        path=args.dataset, packed_vis=args.packed_vis,
        sfh_mean=sfh_mean, sfh_scale=sfh_scale,
    )
    train = PackedCounterfactualDataset(
        rows=train_rows, donor_pools=train_pools, training=True, **common,
    )
    val = PackedCounterfactualDataset(
        rows=val_rows, donor_pools=val_pools, training=False, **common,
    )
    arrays = dict(
        train_rows=train_rows, val_rows=val_rows,
        train_donor_pools=train_pools, val_donor_pools=val_pools,
        activity_class=activity, sfh_mean=sfh_mean, sfh_scale=sfh_scale,
        donor_median_abs_delta_log_mass=np.float32(
            donor_diagnostics["median_abs_delta_log_mass"]
        ),
        donor_median_abs_delta_redshift=np.float32(
            donor_diagnostics["median_abs_delta_redshift"]
        ),
        donor_same_activity_fraction=np.float32(
            donor_diagnostics["same_activity_fraction"]
        ),
    )
    return train, val, arrays, donor_diagnostics


class Experiment(L.LightningModule):
    def __init__(self, model_kwargs, lr=1e-4, control_batches=8):
        super().__init__()
        self.save_hyperparameters()
        self.model = EuclidSingleNeighborFlow(**model_kwargs)

    def training_step(self, batch, batch_idx):
        loss, metrics = self.model.loss(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch["image"]))
        for name, value in metrics.items():
            self.log(f"train_{name}", value, batch_size=len(batch["image"]))
        return loss

    def validation_step(self, batch, batch_idx):
        count = len(batch["image"])
        generator = torch.Generator(device=self.device).manual_seed(10000 + batch_idx)
        time_values = torch.rand(count, generator=generator, device=self.device)
        image_noise = torch.randn(batch["image"].shape, generator=generator, device=self.device)
        sfh_noise = torch.randn(batch["sfh"].shape, generator=generator, device=self.device)
        paired, metrics = self.model.loss(
            batch, time=time_values, image_noise=image_noise, sfh_noise=sfh_noise,
        )
        self.log("val_loss", paired, prog_bar=True, batch_size=count, sync_dist=True)
        for name, value in metrics.items():
            self.log(f"val_{name}", value, batch_size=count, sync_dist=True)
        if batch_idx < self.hparams.control_batches:
            controls = {
                "wrong_common": ("shuffled", "paired"),
                "zero_common": ("zero", "paired"),
                "wrong_private": ("paired", "shuffled"),
                "zero_private": ("paired", "zero"),
            }
            for label, (common_mode, private_mode) in controls.items():
                value, branch = self.model.loss(
                    batch, common_mode, private_mode, time_values,
                    image_noise, sfh_noise,
                )
                self.log(f"val_control_{label}_loss", value, batch_size=count, sync_dist=True)
                self.log(f"val_control_{label}_image_flow_loss", branch["image_flow_loss"], batch_size=count, sync_dist=True)
                self.log(f"val_control_{label}_sfh_flow_loss", branch["sfh_flow_loss"], batch_size=count, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class Preview(L.Callback):
    def __init__(self, dataset, output, every=5, samples=6, steps=32, seed=1042):
        self.dataset = dataset
        self.batch = default_collate([
            dataset[int(index)] for index in np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int)
        ])
        self.output, self.every, self.steps, self.seed = Path(output), every, steps, seed

    def on_validation_epoch_end(self, trainer, module):
        if not trainer.is_global_zero or ((trainer.current_epoch + 1) % self.every):
            return
        import matplotlib.pyplot as plt
        batch = {key: value.to(module.device) for key, value in self.batch.items()}
        generator = torch.Generator(device=module.device).manual_seed(self.seed)
        image_noise = torch.randn(batch["image"].shape, generator=generator, device=module.device)
        sfh_noise = torch.randn(batch["sfh"].shape, generator=generator, device=module.device)
        modes = {
            "paired": ("paired", "paired"),
            "wrong common": ("shuffled", "paired"),
            "zero common": ("zero", "paired"),
            "wrong private": ("paired", "shuffled"),
        }
        generated = {
            name: tuple(value.cpu() for value in module.model.sample(
                batch, common, private, self.steps, image_noise, sfh_noise,
            ))
            for name, (common, private) in modes.items()
        }
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"epoch_{trainer.current_epoch:03d}_images.png"
        rows = {"real": batch["image"].cpu(), **{
            name: values[0] for name, values in generated.items()
        }}
        figure, axes = plt.subplots(len(rows), len(batch["image"]), figsize=(2 * len(batch["image"]), 2 * len(rows)), squeeze=False)
        for row, (label, images) in enumerate(rows.items()):
            for column, image in enumerate(images[:, 0]):
                axes[row, column].imshow(image, cmap="gray", vmin=-1, vmax=1)
                axes[row, column].set_axis_off()
                if column == 0:
                    axes[row, column].set_title(label, loc="left", fontsize=8)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        sfh_path = self.output / f"epoch_{trainer.current_epoch:03d}_sfhs.png"
        mean = torch.as_tensor(self.dataset.sfh_mean)
        scale = torch.as_tensor(self.dataset.sfh_scale)
        target = torch.softmax(batch["sfh"].cpu() * scale + mean, dim=-1)
        generated_sfh = {
            name: torch.softmax(values[1] * scale + mean, dim=-1)
            for name, values in generated.items()
        }
        x = np.linspace(0, 1, target.shape[1])
        figure, axes = plt.subplots(len(target), 1, figsize=(8, 2 * len(target)), squeeze=False, sharex=True)
        colors = {"paired": "#0072B2", "wrong common": "#D55E00", "zero common": "#999999", "wrong private": "#009E73"}
        for index, axis in enumerate(axes[:, 0]):
            axis.plot(x, target[index], color="black", lw=1.5, label="target")
            for name, values in generated_sfh.items():
                axis.plot(x, values[index], color=colors[name], lw=1, label=name)
        axes[0, 0].legend(ncol=3, fontsize=7)
        axes[-1, 0].set_xlabel("Fractional lookback time")
        figure.tight_layout()
        figure.savefig(sfh_path, dpi=150)
        plt.close(figure)
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                logger.experiment.log({
                    "generations/images": wandb.Image(str(path)),
                    "generations/sfhs": wandb.Image(str(sfh_path)),
                    "epoch": trainer.current_epoch,
                })


def _effective_rank(values):
    centered = values - values.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    probability = variance / max(variance.sum(), np.finfo(float).eps)
    return float(np.exp(-np.sum(probability * np.log(probability + 1e-12))))


class EmbeddingGeometryPreview(L.Callback):
    """Plot true-target common/private spaces; donor priors are not included."""

    def __init__(self, dataset, output, every=5, samples=512, batch_size=64, seed=2042):
        self.dataset = dataset
        self.output = Path(output)
        self.every = int(every)
        self.batch_size = int(batch_size)
        rng = np.random.default_rng(seed)
        self.indices = rng.choice(len(dataset), size=min(samples, len(dataset)), replace=False)

    def on_validation_epoch_end(self, trainer, module):
        if not trainer.is_global_zero or ((trainer.current_epoch + 1) % self.every):
            return
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        names = ("image_common", "sfh_common", "image_private", "sfh_private")
        collected = {name: [] for name in names}
        was_training = module.training
        module.eval()
        with torch.inference_mode():
            for start in range(0, len(self.indices), self.batch_size):
                batch = default_collate([
                    self.dataset[int(index)] for index in self.indices[start:start + self.batch_size]
                ])
                embeddings = module.model.target_embeddings(
                    batch["image"].to(module.device), batch["sfh"].to(module.device),
                )
                for name in names:
                    collected[name].append(embeddings[name].float().cpu().numpy())
        module.train(was_training)
        values = {name: np.concatenate(parts) for name, parts in collected.items()}

        common = np.r_[values["image_common"], values["sfh_common"]]
        private = np.r_[values["image_private"], values["sfh_private"]]
        # PCA is used here deliberately: unlike separately fitted UMAPs it keeps
        # both modalities in one explicit coordinate system and is stable across
        # epochs. Explorer UMAPs can still be produced from saved embeddings.
        common_xy = PCA(2, random_state=42).fit_transform(common)
        private_xy = PCA(2, random_state=42).fit_transform(private)
        count = len(values["image_common"])
        labels = np.r_[np.zeros(count, dtype=int), np.ones(count, dtype=int)]

        def modality_accuracy(array):
            train_x, test_x, train_y, test_y = train_test_split(
                array, labels, test_size=0.3, random_state=42, stratify=labels,
            )
            return float(LogisticRegression(max_iter=1000).fit(train_x, train_y).score(test_x, test_y))

        paired = np.linalg.norm(values["image_common"] - values["sfh_common"], axis=1)
        shuffled = np.linalg.norm(values["image_common"] - np.roll(values["sfh_common"], 1, axis=0), axis=1)
        metrics = {
            "epoch": int(trainer.current_epoch),
            "common_image_effective_rank": _effective_rank(values["image_common"]),
            "common_sfh_effective_rank": _effective_rank(values["sfh_common"]),
            "private_image_effective_rank": _effective_rank(values["image_private"]),
            "private_sfh_effective_rank": _effective_rank(values["sfh_private"]),
            "common_modality_accuracy": modality_accuracy(common),
            "private_modality_accuracy": modality_accuracy(private),
            "common_paired_distance_median": float(np.median(paired)),
            "common_shuffled_distance_median": float(np.median(shuffled)),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        json_path = self.output / f"epoch_{trainer.current_epoch:03d}.json"
        json_path.write_text(json.dumps(metrics, indent=2) + "\n")
        for name, value in metrics.items():
            if name != "epoch":
                module.log(f"geometry_{name}", value, sync_dist=False)

        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ("#0072B2", "#D55E00")
        for index in range(min(count, 150)):
            axes[0].plot(
                [common_xy[index, 0], common_xy[count + index, 0]],
                [common_xy[index, 1], common_xy[count + index, 1]],
                color="0.8", lw=0.35, zorder=0,
            )
        for axis, coordinates, title in (
            (axes[0], common_xy, "Common target embeddings"),
            (axes[1], private_xy, "Private target embeddings"),
        ):
            axis.scatter(coordinates[:count, 0], coordinates[:count, 1], s=7, alpha=0.6, color=colors[0], label="VIS")
            axis.scatter(coordinates[count:, 0], coordinates[count:, 1], s=7, alpha=0.6, color=colors[1], label="SFH")
            axis.set_title(title)
            axis.set_xlabel("Joint PCA 1")
            axis.set_ylabel("Joint PCA 2")
            axis.legend()
        figure.suptitle(
            f"Epoch {trainer.current_epoch} · target encodings only · "
            f"common modality acc={metrics['common_modality_accuracy']:.3f} · "
            f"private={metrics['private_modality_accuracy']:.3f}"
        )
        figure.tight_layout()
        figure_path = self.output / f"epoch_{trainer.current_epoch:03d}.png"
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                logger.experiment.log({
                    "geometry/target_embeddings": wandb.Image(str(figure_path)),
                    **{f"geometry/{key}": value for key, value in metrics.items()},
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
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument(
        "--donor-matching",
        choices=("mass_redshift_activity", "mass_window"),
        default="mass_redshift_activity",
    )
    parser.add_argument("--mass-window", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--common-dim", type=int, default=64)
    parser.add_argument("--private-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--image-encoder-base", type=int, default=16)
    parser.add_argument("--image-flow-base", type=int, default=16)
    parser.add_argument("--sfh-d-model", type=int, default=96)
    parser.add_argument("--sfh-layers", type=int, default=3)
    parser.add_argument("--geometric-weight", type=float, default=0.1)
    parser.add_argument("--sinkhorn-blur", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--control-batches", type=int, default=8)
    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--preview-samples", type=int, default=6)
    parser.add_argument("--preview-steps", type=int, default=32)
    parser.add_argument("--geometry-every", type=int, default=5)
    parser.add_argument("--geometry-samples", type=int, default=512)
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
    train, val, arrays, donor_diagnostics = _prepare(args)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "matched_single_neighbors.npz", **arrays)
    (args.output / "run.json").write_text(json.dumps({
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_count": len(train), "val_count": len(val),
        "donor_diagnostics": donor_diagnostics,
    }, indent=2) + "\n")
    kwargs = dict(
        n_sfh_bins=train.sfh_mean.size, common_dim=args.common_dim,
        private_dim=args.private_dim, condition_dim=args.condition_dim,
        image_encoder_base=args.image_encoder_base, image_flow_base=args.image_flow_base,
        sfh_d_model=args.sfh_d_model, sfh_layers=args.sfh_layers,
        geometric_weight=args.geometric_weight, sinkhorn_blur=args.sinkhorn_blur,
    )
    model = Experiment(kwargs, args.lr, args.control_batches)
    checkpoint = ModelCheckpoint(
        dirpath=args.output / "checkpoints", monitor="val_loss", mode="min",
        save_last=True, save_top_k=2, filename="single-neighbor-{epoch:03d}-{val_loss:.5f}",
    )
    callbacks = [
        checkpoint, EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
        LearningRateMonitor(logging_interval="epoch"),
        Preview(val, args.output / "previews", 1 if args.smoke else args.preview_every,
                args.preview_samples, args.preview_steps, args.seed + 1000),
        EmbeddingGeometryPreview(
            val, args.output / "embedding_geometry",
            1 if args.smoke else args.geometry_every,
            args.geometry_samples, seed=args.seed + 2000,
        ),
    ]
    loggers = [CSVLogger(str(args.output), name="logs")]
    if args.wandb_mode != "disabled":
        loggers.append(WandbLogger(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.output.name, save_dir=str(args.output),
            offline=args.wandb_mode == "offline", log_model=False,
        ))
    loader = dict(
        batch_size=args.batch_size, num_workers=args.workers,
        persistent_workers=args.workers > 0, pin_memory=args.accelerator == "gpu",
    )
    if args.workers > 0:
        loader["prefetch_factor"] = 4
    trainer = L.Trainer(
        accelerator=args.accelerator, devices=1,
        precision="16-mixed" if args.accelerator == "gpu" else "32-true",
        max_epochs=2 if args.smoke else args.epochs, callbacks=callbacks,
        logger=loggers, gradient_clip_val=1.0, log_every_n_steps=5,
        num_sanity_val_steps=0,
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
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if args.accelerator == "gpu" else None,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
