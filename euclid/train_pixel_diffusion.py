"""Train 224-pixel VIS diffusion conditioned on cached aligned SFH embeddings."""
import argparse
import copy
import json
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .pixel_diffusion import ConditionalUNet, DiffusionSchedule
from .prepare_diffusion_conditions import sha256


class ConditionedStamps(Dataset):
    def __init__(self, cache, stamps, split, augment=False, limit=0):
        with np.load(cache, allow_pickle=False) as source:
            ids = {name: source[name + '_ids'] for name in ('train', 'val')}
            if np.intersect1d(ids['train'], ids['val']).size:
                raise ValueError('Condition cache has train/validation overlap.')
            self.ids = ids[split].copy()
            self.conditions = source[split + '_condition'].astype(np.float32)
        if (self.ids.ndim != 1 or self.ids.dtype.kind not in 'iu'
                or len(np.unique(self.ids)) != len(self.ids)
                or self.conditions.ndim != 2 or len(self.conditions) != len(self.ids)
                or not len(self.ids) or not np.isfinite(self.conditions).all()
                or not np.allclose(np.linalg.norm(self.conditions, axis=1), 1, atol=1e-4)):
            raise ValueError('Invalid IDs or conditions in cache.')
        if limit:
            self.ids, self.conditions = self.ids[:limit], self.conditions[:limit]
        self.stamps, self.augment = Path(stamps) / 'VIS', augment
        for gid in self.ids:
            path = self.stamps / f'VIS_{int(gid)}.jpg'
            if not path.is_file():
                raise FileNotFoundError(path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        with Image.open(self.stamps / f'VIS_{int(self.ids[index])}.jpg') as image:
            if image.size != (224, 224):
                raise ValueError(f'Expected 224x224 stamp for {self.ids[index]}, got {image.size}')
            image = np.asarray(image.convert('L'), dtype=np.float32) / 127.5 - 1
        image = torch.from_numpy(image.copy())[None]
        if self.augment:
            image = torch.rot90(image, int(torch.randint(4, ()).item()), (-2, -1))
            if torch.rand(()) < .5:
                image = image.flip(-1)
        return image, torch.from_numpy(self.conditions[index].copy())


class PixelDiffusion(L.LightningModule):
    def __init__(self, condition_dim=256, base=32, steps=1000, lr=1e-4,
                 condition_dropout=.15, ema_decay=.999, cache_sha256=''):
        super().__init__()
        self.save_hyperparameters()
        self.net = ConditionalUNet(condition_dim, base)
        self.ema = copy.deepcopy(self.net).eval().requires_grad_(False)
        self.schedule = DiffusionSchedule(steps)

    def train(self, mode=True):
        super().train(mode)
        self.ema.eval()
        return self

    def training_step(self, batch, batch_idx):
        image, condition = batch
        t = torch.randint(len(self.schedule.alpha), (len(image),), device=self.device)
        noisy, target = self.schedule.noisy_target(image, torch.randn_like(image), t)
        drop = torch.rand(len(image), device=self.device) < self.hparams.condition_dropout
        prediction = self.net(noisy, t, condition, drop)
        loss = F.mse_loss(prediction.float(), target)
        self.log('train_v_mse', loss, prog_bar=True, batch_size=len(image))
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        with torch.no_grad():
            for target, value in zip(self.ema.parameters(), self.net.parameters()):
                target.lerp_(value, 1 - self.hparams.ema_decay)

    def validation_step(self, batch, batch_idx):
        image, condition = batch
        # Same validation noise/timesteps each epoch; no stochastic image augmentation.
        rng = torch.Generator(device=self.device).manual_seed(10000 + batch_idx)
        t = torch.randint(len(self.schedule.alpha), (len(image),), generator=rng, device=self.device)
        noise = torch.randn(image.shape, generator=rng, device=self.device)
        noisy, target = self.schedule.noisy_target(image, noise, t)
        for label, cond, drop in (
            ('val_v_mse', condition, None),
            ('val_shuffled_v_mse', condition.roll(1, dims=0), None),
            ('val_unconditional_v_mse', condition, torch.ones(len(image), dtype=torch.bool, device=self.device)),
        ):
            loss = F.mse_loss(self.ema(noisy, t, cond, drop).float(), target)
            self.log(label, loss, prog_bar=label == 'val_v_mse', batch_size=len(image))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr, weight_decay=.01)


def write_samples(model, dataset, output, steps=100, seed=42, n=8):
    """Same initial noise for correct, permuted, and null conditions; EMA weights."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n = min(n, len(dataset))
    indices = np.sort(np.random.default_rng(seed).choice(len(dataset), n, replace=False))
    batch = [dataset[int(index)] for index in indices]
    real = torch.stack([row[0] for row in batch])
    condition = torch.stack([row[1] for row in batch]).to(model.device)
    model.eval()
    panels = [real]
    for cond, guidance in [(condition, 1.), (condition, 2.),
                           (condition.roll(1, dims=0), 2.), (condition, 0.)]:
        panels.append(model.schedule.sample(model.ema, cond, steps=steps,
                                            guidance=guidance, seed=seed).cpu())
    pixels = [((panel.clamp(-1, 1) + 1) * 127.5).byte().numpy()[:, 0] for panel in panels]
    canvas = np.concatenate([np.concatenate(row, axis=1) for row in pixels], axis=0)
    Image.fromarray(canvas).save(output / 'conditioning_comparison.png')
    (output / 'samples.json').write_text(json.dumps({
        'galaxy_ids': dataset.ids[indices].tolist(),
        'shuffled_condition_ids': np.roll(dataset.ids[indices], 1).tolist(),
        'rows': ['real', 'conditional guidance=1', 'conditional guidance=2',
                 'shuffled guidance=2', 'unconditional'],
        'seed': seed, 'ddim_steps': steps, 'weights': 'EMA',
        'note': 'Same noise per column across generated rows; not paired reconstructions.',
    }, indent=2))


class Preview(L.Callback):
    def __init__(self, dataset, output):
        self.dataset, self.output = dataset, Path(output)

    def on_train_epoch_end(self, trainer, module):
        if (trainer.current_epoch + 1) % 5 == 0:
            write_samples(module, self.dataset, self.output / f'epoch_{trainer.current_epoch:03d}', steps=50, n=4)
            module.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--stamps', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--sample-checkpoint', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--base', type=int, default=32)
    parser.add_argument('--accelerator', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--sample-steps', type=int, default=100)
    args = parser.parse_args()
    if min(args.batch_size, args.accumulate, args.epochs) < 1 or args.workers < 0:
        parser.error('Batch, accumulation and epochs must be positive; workers nonnegative.')
    if args.resume and args.sample_checkpoint:
        parser.error('Use either resume or sample-checkpoint.')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error('Output is nonempty; use a new directory or --resume.')
    L.seed_everything(42, workers=True)
    digest = sha256(args.conditions)
    val = ConditionedStamps(args.conditions, args.stamps, 'val', limit=64 if args.smoke else 0)
    if args.sample_checkpoint:
        model = PixelDiffusion.load_from_checkpoint(str(args.sample_checkpoint), map_location='cpu')
        if model.hparams.cache_sha256 != digest:
            raise ValueError('Checkpoint and condition cache do not match.')
        model.to('cuda' if args.accelerator == 'gpu' else 'cpu')
        write_samples(model, val, args.output, steps=args.sample_steps)
        return
    train = ConditionedStamps(args.conditions, args.stamps, 'train', augment=True,
                              limit=256 if args.smoke else 0)
    if args.resume:
        model = PixelDiffusion.load_from_checkpoint(str(args.resume), map_location='cpu')
        if model.hparams.cache_sha256 != digest:
            raise ValueError('Resume checkpoint uses a different condition cache.')
    else:
        model = PixelDiffusion(condition_dim=train.conditions.shape[1], base=args.base,
                               cache_sha256=digest)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(condition_sha256=digest, train_count=len(train), val_count=len(val),
                    condition_dim=train.conditions.shape[1])
    with np.load(args.conditions, allow_pickle=False) as source:
        settings['condition_provenance'] = json.loads(str(source['metadata']))
    (args.output / ('resume_run.json' if args.resume else 'run.json')).write_text(json.dumps(settings, indent=2))
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=args.accelerator == 'gpu', persistent_workers=args.workers > 0)
    checkpoint = ModelCheckpoint(dirpath=args.output / 'checkpoints', monitor='val_v_mse',
                                 mode='min', save_last=True, save_top_k=1,
                                 filename='pixel-sfh-{epoch:03d}-{val_v_mse:.5f}')
    trainer = L.Trainer(accelerator=args.accelerator, devices=1,
                        precision='16-mixed' if args.accelerator == 'gpu' else '32-true',
                        max_epochs=2 if args.smoke else args.epochs,
                        accumulate_grad_batches=args.accumulate, gradient_clip_val=1.,
                        callbacks=[checkpoint, Preview(val, args.output / 'samples')],
                        logger=CSVLogger(str(args.output), name='logs'), log_every_n_steps=5,
                        num_sanity_val_steps=0)
    start = time.monotonic()
    if args.accelerator == 'gpu':
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, DataLoader(train, shuffle=True, **loader_args),
                DataLoader(val, shuffle=False, **loader_args), ckpt_path=str(args.resume) if args.resume else None)
    report = {'elapsed_seconds': time.monotonic() - start,
              'optimizer_steps': trainer.global_step,
              'best_checkpoint': checkpoint.best_model_path,
              'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated() / 2**30 if args.accelerator == 'gpu' else None}
    (args.output / 'runtime.json').write_text(json.dumps(report, indent=2))
    best = PixelDiffusion.load_from_checkpoint(checkpoint.best_model_path, map_location='cpu')
    best.to('cuda' if args.accelerator == 'gpu' else 'cpu')
    write_samples(best, val, args.output / 'samples_best', steps=20 if args.smoke else args.sample_steps)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
