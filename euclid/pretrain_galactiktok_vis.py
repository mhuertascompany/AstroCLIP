"""Self-supervised GalaxyTikTok pretraining on native-flux Euclid VIS FITS."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from .training_index import build_pair_index
from .vis_fits import (
    VISFitsTransform,
    estimate_image_stats,
    load_image_stats,
    resolve_fits_directory,
    vis_fits_path,
    write_image_stats,
)


class VISFitsDataset(Dataset):
    def __init__(self, fits_root: Path, galaxy_ids, band: str,
                 image_size: int, training: bool, image_stats: Path,
                 asinh_scale: float) -> None:
        self.stamp_dir = resolve_fits_directory(fits_root, band)
        self.galaxy_ids = galaxy_ids
        self.band = band
        p_lo, p_hi = load_image_stats(image_stats, 'euclid-vis')
        self.transform = VISFitsTransform(
            image_size, p_lo, p_hi, asinh_scale, training,
        )

    def __len__(self):
        return len(self.galaxy_ids)

    def __getitem__(self, index):
        galaxy_id = int(self.galaxy_ids[index])
        path = vis_fits_path(self.stamp_dir, galaxy_id, self.band)
        tensor = self.transform(path)
        return tensor, galaxy_id


class GalaxyTikTokVISPretrainer(L.LightningModule):
    def __init__(self, tokenizer, band: str, lr: float,
                 weight_decay: float, epochs: int, warmup_epochs: int,
                 target_gaussian_sigma: float = 0.0) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.band = band
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.target_gaussian_sigma = float(target_gaussian_sigma)

    @staticmethod
    def _gaussian_target(flux: torch.Tensor, sigma: float) -> torch.Tensor:
        """Return a fixed low-pass target without smoothing across channels."""
        if sigma <= 0:
            return flux
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinates = torch.arange(
            -radius, radius + 1, device=flux.device, dtype=flux.dtype,
        )
        kernel_1d = torch.exp(-0.5 * (coordinates / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.expand(flux.size(1), 1, -1, -1)
        padded = F.pad(flux, (radius, radius, radius, radius), mode='reflect')
        return F.conv2d(padded, kernel, groups=flux.size(1))

    def _loss(self, batch, prefix: str):
        try:
            from galactiktok.models.image_transformer import Image
        except ImportError as exc:
            raise ImportError('GalaxyTikTok is not importable.') from exc
        flux, _ = batch
        image = Image(flux=flux, bands=[self.band])
        if self.target_gaussian_sigma <= 0:
            loss, _ = self.tokenizer.loss_fn(image)
        else:
            # Masking prevents the encoder from directly observing the patch
            # whose loss is evaluated. The smoothed target additionally stops
            # exact high-frequency background noise from being rewarded.
            mask_fraction = (
                self.tokenizer.encoder_mask_fraction if self.training else 0.0
            )
            tokens = self.tokenizer.encode(image, mask_fraction=mask_fraction)
            reconstruction = self.tokenizer.decode(tokens).flux
            target = self._gaussian_target(
                flux, self.target_gaussian_sigma,
            ).detach()
            if tokens.mask_idx is None:
                loss = F.mse_loss(reconstruction, target)
            else:
                from galactiktok.models.image_transformer.modeling_image_transformer import (
                    patchify,
                )
                predicted_patches = patchify(
                    reconstruction,
                    (self.tokenizer.patch_size, self.tokenizer.patch_size),
                )
                target_patches = patchify(
                    target,
                    (self.tokenizer.patch_size, self.tokenizer.patch_size),
                )
                batch_index = torch.arange(
                    flux.size(0), device=flux.device,
                ).unsqueeze(-1)
                loss = F.mse_loss(
                    predicted_patches[batch_index, tokens.mask_idx],
                    target_patches[batch_index, tokens.mask_idx],
                )
        self.log(
            f'{prefix}_recon_loss', loss,
            prog_bar=True,
            on_step=prefix == 'train',
            on_epoch=True,
            batch_size=flux.size(0),
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._loss(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.tokenizer.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def schedule(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / max(1, self.warmup_epochs)
            progress = (epoch - self.warmup_epochs) / max(
                1, self.epochs - self.warmup_epochs,
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pretrain a GalaxyTikTok transformer tokenizer on Euclid VIS stamps.',
    )
    parser.add_argument('--dataset', type=Path, required=True,
                        help='SFH HDF5 used only to reproduce the paired train/val split.')
    parser.add_argument('--fits-root', type=Path, required=True,
                        help='Cutout run containing cutouts/VIS/<object_id>.fits.')
    parser.add_argument('--band', default='VIS')
    parser.add_argument('--tokenizer-band', default='euclid-vis')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-name', default='euclid_vis_galactiktok')
    parser.add_argument('--image-size', type=int, default=96)
    parser.add_argument('--patch-size', type=int, default=8)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-encoder-blocks', type=int, default=6)
    parser.add_argument('--num-decoder-blocks', type=int, default=6)
    parser.add_argument('--bottleneck-dim', type=int, default=8)
    parser.add_argument('--mask-fraction', type=float, default=0.5)
    parser.add_argument(
        '--target-gaussian-sigma', type=float, default=0.0,
        help=(
            'Gaussian sigma in normalized-image pixels for the reconstruction '
            'target. Zero retains exact noisy-pixel reconstruction.'
        ),
    )
    parser.add_argument('--image-stats', type=Path,
                        help='Existing GalaxyTikTok-style VIS percentile JSON.')
    parser.add_argument('--stats-images', type=int, default=4096)
    parser.add_argument('--stats-pixels-per-image', type=int, default=4096)
    parser.add_argument('--asinh-scale', type=float, default=20.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--max-pairs', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--accelerator', default='auto')
    parser.add_argument('--devices', default='auto')
    parser.add_argument('--precision', default='16-mixed')
    parser.add_argument('--resume-from', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    fits_directory = resolve_fits_directory(args.fits_root, args.band)
    if args.resume_from is not None and not args.resume_from.is_file():
        raise FileNotFoundError(args.resume_from)
    if args.image_size % args.patch_size:
        raise ValueError('--image-size must be divisible by --patch-size.')
    if not 0 <= args.mask_fraction < 1:
        raise ValueError('--mask-fraction must lie in [0, 1).')
    if min(args.batch_size, args.num_heads, args.embed_dim,
           args.bottleneck_dim, args.max_epochs) < 1:
        raise ValueError('Model and training dimensions must be positive.')
    if min(args.stats_images, args.stats_pixels_per_image) < 1:
        raise ValueError('Image-statistics sample sizes must be positive.')
    if not np.isfinite(args.asinh_scale) or args.asinh_scale <= 0:
        raise ValueError('--asinh-scale must be finite and positive.')
    if (not np.isfinite(args.target_gaussian_sigma)
            or args.target_gaussian_sigma < 0):
        raise ValueError('--target-gaussian-sigma must be finite and non-negative.')
    if args.embed_dim % args.num_heads:
        raise ValueError('--embed-dim must be divisible by --num-heads.')

    try:
        from galactiktok import ImageTransformerConfig, ImageTransformerTokenizer
    except ImportError as exc:
        raise ImportError(
            'GalaxyTikTok is required. Add GALACTIKTOK/src to PYTHONPATH.'
        ) from exc

    L.seed_everything(args.seed, workers=True)
    index = build_pair_index(
        args.dataset,
        args.fits_root,
        band=args.band,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_pairs=args.max_pairs,
        image_format='fits',
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_stats = args.image_stats or args.output_dir / 'image_stats.json'
    if args.image_stats is None and not image_stats.exists():
        p_lo, p_hi, n_images, n_pixels = estimate_image_stats(
            fits_directory,
            index.train_ids,
            band=args.band,
            max_images=args.stats_images,
            max_pixels_per_image=args.stats_pixels_per_image,
            seed=args.seed,
        )
        write_image_stats(
            image_stats, p_lo, p_hi, band=args.tokenizer_band,
            n_images=n_images,
        )
        print(
            f'VIS flux statistics from {n_images:,} images/{n_pixels:,} pixels: '
            f'p1={p_lo:.7g}, p99={p_hi:.7g}; wrote {image_stats}',
            flush=True,
        )
    else:
        load_image_stats(image_stats, args.tokenizer_band)
    train_dataset = VISFitsDataset(
        args.fits_root, index.train_ids, args.band, args.image_size, True,
        image_stats, args.asinh_scale,
    )
    val_dataset = VISFitsDataset(
        args.fits_root, index.val_ids, args.band, args.image_size, False,
        image_stats, args.asinh_scale,
    )
    loaders = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loaders,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loaders)

    config = ImageTransformerConfig(
        canonical_bands=[args.tokenizer_band],
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        encoder_mask_fraction=args.mask_fraction,
        band_dropout_p=0.0,
    )
    tokenizer = ImageTransformerTokenizer(config)
    model = GalaxyTikTokVISPretrainer(
        tokenizer,
        band=args.tokenizer_band,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        target_gaussian_sigma=args.target_gaussian_sigma,
    )

    checkpoint_dir = args.output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=args.run_name + '-{epoch:03d}-{val_recon_loss:.6f}',
        monitor='val_recon_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[
            callback,
            EarlyStopping(
                monitor='val_recon_loss', mode='min', patience=args.patience,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        logger=CSVLogger(
            save_dir=str(args.output_dir / 'logs'), name=args.run_name,
        ),
    )
    print(
        f'[GalaxyTikTok VIS] train={len(train_dataset):,}, '
        f'val={len(val_dataset):,}; image={args.image_size}, '
        f'patch={args.patch_size}, grid={args.image_size // args.patch_size}x'
        f'{args.image_size // args.patch_size}, bottleneck={args.bottleneck_dim}, '
        f'mask={args.mask_fraction:g}, target_sigma={args.target_gaussian_sigma:g}',
        flush=True,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)

    if callback.best_model_path:
        state = torch.load(callback.best_model_path, map_location='cpu')['state_dict']
        model.load_state_dict(state)
    tokenizer_dir = args.output_dir / 'tokenizer'
    model.tokenizer.cpu().eval()
    model.tokenizer.save_pretrained(str(tokenizer_dir))
    from galactiktok.models.image_transformer import Image
    preview_loader = DataLoader(
        val_dataset, batch_size=min(8, len(val_dataset)), shuffle=False,
        num_workers=0,
    )
    preview, _ = next(iter(preview_loader))
    preview_target = model._gaussian_target(
        preview, model.target_gaussian_sigma,
    )
    with torch.no_grad():
        reconstruction = model.tokenizer(
            Image(flux=preview, bands=[args.tokenizer_band]),
        ).flux
    save_image(
        torch.cat((preview, preview_target, reconstruction.clamp(0, 1)), dim=0),
        args.output_dir / 'reconstruction_examples.png',
        nrow=len(preview),
    )
    print(f'Best checkpoint: {callback.best_model_path}', flush=True)
    print(f'Alignment-ready tokenizer: {tokenizer_dir}', flush=True)
    print(
        'Reconstruction preview (input / target / reconstruction): '
        f'{args.output_dir / "reconstruction_examples.png"}',
        flush=True,
    )


if __name__ == '__main__':
    main()
