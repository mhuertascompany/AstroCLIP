"""
CosmosWebZooBotCLIP: contrastive alignment of COSMOS-Web images and CIGALE SFHs
using a frozen ZooBOT ConvNeXt backbone as the image encoder.

v2 adds a MoCo-style momentum encoder queue for both modalities.  The queue
extends the effective number of negatives per step from batch_size-1 to
batch_size + queue_size - 1, dramatically sharpening the contrastive signal.

Architecture
------------
  Main image encoder  : frozen ZooBOT backbone + trainable MLP projection
  Main SFH encoder    : trainable transformer/MLP (SFHEncoder)
  Momentum encoders   : EMA copies of both main encoders (no gradient)
  Queue               : two circular buffers (img, sfh) of size Q

Loss (per step)
---------------
  Query    : main-encoder embeddings (get gradients)
  Keys     : momentum-encoder embeddings for current batch + queue
  Logits   : (B, B+Q) for each direction
  Labels   : arange(B)  (positive is always the diagonal of the batch block)
  Loss     : mean of cross_entropy(logits_i2s, labels) and cross_entropy(logits_s2i, labels)

Validation loss uses only the current batch (no queue) so it stays comparable.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sfh_encoder import SFHEncoder
from .zoobot_encoder import MultiFilterZooBotImageEncoder, ZooBotImageEncoder


class CosmosWebZooBotCLIP(L.LightningModule):

    def __init__(
        self,
        zoobot_ckpt:   str,
        embed_dim:     int   = 256,
        sfh_input_dim: int   = 50,
        temperature:   float = 0.07,
        queue_size:    int   = 4096,
        momentum:      float = 0.995,
        lr:            float = 1e-4,
        weight_decay:  float = 0.05,
        epochs:        int   = 50,
        warmup_epochs: int   = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ── main encoders (receive gradients) ────────────────────────────────
        self.image_encoder = ZooBotImageEncoder(
            ckpt_path=zoobot_ckpt,
            embed_dim=embed_dim,
        )
        self.sfh_encoder = SFHEncoder(
            input_dim=sfh_input_dim,
            embed_dim=embed_dim,
        )

        # ── momentum encoders (EMA, no gradient) ─────────────────────────────
        self.image_encoder_m = copy.deepcopy(self.image_encoder)
        self.sfh_encoder_m   = copy.deepcopy(self.sfh_encoder)
        for p in self.image_encoder_m.parameters():
            p.requires_grad_(False)
        for p in self.sfh_encoder_m.parameters():
            p.requires_grad_(False)

        # ── learnable temperature ─────────────────────────────────────────────
        self.log_temp = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))

        # ── circular queues (registered as buffers → saved in checkpoint) ────
        Q = queue_size
        D = embed_dim
        self.register_buffer('queue_img', F.normalize(torch.randn(D, Q), dim=0))
        self.register_buffer('queue_sfh', F.normalize(torch.randn(D, Q), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    # ── momentum update ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _momentum_update(self) -> None:
        m = self.hparams.momentum
        for p, pm in zip(self.image_encoder.parameters(),
                         self.image_encoder_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)
        for p, pm in zip(self.sfh_encoder.parameters(),
                         self.sfh_encoder_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)

    # ── queue management ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self,
        img_keys: torch.Tensor,   # (B, D)  from momentum image encoder
        sfh_keys: torch.Tensor,   # (B, D)  from momentum SFH encoder
    ) -> None:
        B   = img_keys.shape[0]
        Q   = self.hparams.queue_size
        ptr = int(self.queue_ptr)
        # queue_size must be divisible by batch_size; checked in training_step
        self.queue_img[:, ptr:ptr + B] = img_keys.T
        self.queue_sfh[:, ptr:ptr + B] = sfh_keys.T
        self.queue_ptr[0] = (ptr + B) % Q

    # ── encoders (public API, used by evaluate / umap scripts) ───────────────

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_sfh(self, sfh: torch.Tensor) -> torch.Tensor:
        return self.sfh_encoder(sfh)

    def forward(
        self, image: torch.Tensor, sfh: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(image), self.encode_sfh(sfh)

    # ── loss helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _infoNCE(
        queries:  torch.Tensor,   # (B, D)  L2-normalised
        all_keys: torch.Tensor,   # (B+Q, D) L2-normalised
        T:        torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy over logits (B, B+Q); positives at indices 0..B-1."""
        logits = queries @ all_keys.T * T          # (B, B+Q)  T = logit_scale = 1/temp
        labels = torch.arange(queries.size(0),
                              device=queries.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

    # ── training step ─────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, sfhs = batch['image'], batch['sfh']
        B = images.size(0)
        Q = self.hparams.queue_size

        if B > Q:
            raise ValueError(
                f'batch_size ({B}) > queue_size ({Q}). '
                'Reduce batch_size or increase queue_size.')

        T = self.log_temp.exp().clamp(min=1.0, max=100.0)

        # ── query embeddings (main encoders, receive gradients) ───────────────
        img_q = F.normalize(self.encode_image(images), dim=-1)   # (B, D)
        sfh_q = F.normalize(self.encode_sfh(sfhs),    dim=-1)    # (B, D)

        # ── key embeddings (momentum encoders, no gradient) ───────────────────
        with torch.no_grad():
            self._momentum_update()
            img_k = F.normalize(self.image_encoder_m(images), dim=-1)  # (B, D)
            sfh_k = F.normalize(self.sfh_encoder_m(sfhs),    dim=-1)   # (B, D)

        # ── all keys = current batch (momentum) + queue ───────────────────────
        # Shape: (B + Q, D)
        all_sfh = torch.cat([sfh_k, self.queue_sfh.T.clone().detach()], dim=0)
        all_img = torch.cat([img_k, self.queue_img.T.clone().detach()], dim=0)

        # ── contrastive loss (symmetric) ──────────────────────────────────────
        loss_i2s = self._infoNCE(img_q, all_sfh, T)
        loss_s2i = self._infoNCE(sfh_q, all_img, T)
        loss     = (loss_i2s + loss_s2i) / 2.0

        # ── within-batch rank-1 (cheap training-time diagnostic) ─────────────
        with torch.no_grad():
            labels   = torch.arange(B, device=images.device)
            logits_b = img_q @ sfh_k.T * T              # (B, B) batch-only
            r1       = ((logits_b.argmax(1) == labels).float().mean() +
                        (logits_b.T.argmax(1) == labels).float().mean()) / 2

        # ── update queue ──────────────────────────────────────────────────────
        self._dequeue_and_enqueue(img_k, sfh_k)

        self.log('train_loss',       loss, prog_bar=True,  sync_dist=True)
        self.log('train_rank1_batch', r1,  prog_bar=True,  sync_dist=True)
        self.log('train_logit_scale', T,   prog_bar=False, sync_dist=True)
        return loss

    # ── validation step (batch-only InfoNCE, no queue) ────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images, sfhs = batch['image'], batch['sfh']
        T = self.log_temp.exp().clamp(min=1.0, max=100.0)

        img_e = F.normalize(self.encode_image(images), dim=-1)
        sfh_e = F.normalize(self.encode_sfh(sfhs),    dim=-1)

        logits  = T * (img_e @ sfh_e.T)
        labels  = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        val_loss = (F.cross_entropy(logits,   labels) +
                    F.cross_entropy(logits.T, labels)) / 2.0

        with torch.no_grad():
            r1 = ((logits.argmax(1)   == labels).float().mean() +
                  (logits.T.argmax(1) == labels).float().mean()) / 2

        self.log('val_loss',   val_loss, prog_bar=True,  sync_dist=True)
        self.log('val_rank1',  r1,       prog_bar=True,  sync_dist=True)

    # ── optimiser & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        # Momentum encoder parameters have requires_grad=False → excluded automatically
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer':    optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


class MultiFilterCosmosWebZooBotCLIP(L.LightningModule):
    """
    MoCo-style CLIP model using three frozen ZooBOT backbones (F150W / F277W / F444W)
    routed by redshift, sharing a single MLP projection head.

    Identical training dynamics to CosmosWebZooBotCLIP; the only difference is
    that each batch also carries a 'redshift' tensor used to route images to the
    correct backbone inside MultiFilterZooBotImageEncoder.

    Parameters
    ----------
    zoobot_ckpt_f150w, zoobot_ckpt_f277w, zoobot_ckpt_f444w : str
        Paths to FinetuneableZoobotClassifier checkpoints for each filter.
    z_low, z_high : float
        Redshift boundaries for filter routing (default 1.0 and 3.0).
    All other parameters same as CosmosWebZooBotCLIP.
    """

    def __init__(
        self,
        zoobot_ckpt_f150w: str,
        zoobot_ckpt_f277w: str,
        zoobot_ckpt_f444w: str,
        z_low:         float = 1.0,
        z_high:        float = 3.0,
        embed_dim:     int   = 256,
        sfh_input_dim: int   = 50,
        temperature:   float = 0.07,
        queue_size:    int   = 4096,
        momentum:      float = 0.995,
        lr:            float = 1e-4,
        weight_decay:  float = 0.05,
        epochs:        int   = 50,
        warmup_epochs: int   = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = MultiFilterZooBotImageEncoder(
            ckpt_f150w=zoobot_ckpt_f150w,
            ckpt_f277w=zoobot_ckpt_f277w,
            ckpt_f444w=zoobot_ckpt_f444w,
            embed_dim=embed_dim,
            z_low=z_low,
            z_high=z_high,
        )
        self.sfh_encoder = SFHEncoder(
            input_dim=sfh_input_dim,
            embed_dim=embed_dim,
        )

        self.image_encoder_m = copy.deepcopy(self.image_encoder)
        self.sfh_encoder_m   = copy.deepcopy(self.sfh_encoder)
        for p in self.image_encoder_m.parameters():
            p.requires_grad_(False)
        for p in self.sfh_encoder_m.parameters():
            p.requires_grad_(False)

        self.log_temp = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))

        Q = queue_size
        D = embed_dim
        self.register_buffer('queue_img', F.normalize(torch.randn(D, Q), dim=0))
        self.register_buffer('queue_sfh', F.normalize(torch.randn(D, Q), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self) -> None:
        m = self.hparams.momentum
        for p, pm in zip(self.image_encoder.parameters(),
                         self.image_encoder_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)
        for p, pm in zip(self.sfh_encoder.parameters(),
                         self.sfh_encoder_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_keys, sfh_keys) -> None:
        B   = img_keys.shape[0]
        Q   = self.hparams.queue_size
        ptr = int(self.queue_ptr)
        self.queue_img[:, ptr:ptr + B] = img_keys.T
        self.queue_sfh[:, ptr:ptr + B] = sfh_keys.T
        self.queue_ptr[0] = (ptr + B) % Q

    def encode_image(self, image: torch.Tensor, redshifts: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image, redshifts)

    def encode_sfh(self, sfh: torch.Tensor) -> torch.Tensor:
        return self.sfh_encoder(sfh)

    @staticmethod
    def _infoNCE(queries, all_keys, T) -> torch.Tensor:
        logits = queries @ all_keys.T * T
        labels = torch.arange(queries.size(0), device=queries.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images    = batch['image']
        sfhs      = batch['sfh']
        redshifts = batch['redshift']
        B = images.size(0)
        Q = self.hparams.queue_size

        if B > Q:
            raise ValueError(f'batch_size ({B}) > queue_size ({Q}).')

        T = self.log_temp.exp().clamp(min=1.0, max=100.0)

        img_q = F.normalize(self.encode_image(images, redshifts), dim=-1)
        sfh_q = F.normalize(self.encode_sfh(sfhs),                dim=-1)

        with torch.no_grad():
            self._momentum_update()
            img_k = F.normalize(self.image_encoder_m(images, redshifts), dim=-1)
            sfh_k = F.normalize(self.sfh_encoder_m(sfhs),                dim=-1)

        all_sfh = torch.cat([sfh_k, self.queue_sfh.T.clone().detach()], dim=0)
        all_img = torch.cat([img_k, self.queue_img.T.clone().detach()], dim=0)

        loss_i2s = self._infoNCE(img_q, all_sfh, T)
        loss_s2i = self._infoNCE(sfh_q, all_img, T)
        loss     = (loss_i2s + loss_s2i) / 2.0

        with torch.no_grad():
            labels   = torch.arange(B, device=images.device)
            logits_b = img_q @ sfh_k.T * T
            r1       = ((logits_b.argmax(1) == labels).float().mean() +
                        (logits_b.T.argmax(1) == labels).float().mean()) / 2

        self._dequeue_and_enqueue(img_k, sfh_k)

        self.log('train_loss',        loss, prog_bar=True,  sync_dist=True)
        self.log('train_rank1_batch', r1,   prog_bar=True,  sync_dist=True)
        self.log('train_logit_scale', T,    prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images    = batch['image']
        sfhs      = batch['sfh']
        redshifts = batch['redshift']
        T = self.log_temp.exp().clamp(min=1.0, max=100.0)

        img_e = F.normalize(self.encode_image(images, redshifts), dim=-1)
        sfh_e = F.normalize(self.encode_sfh(sfhs),                dim=-1)

        logits   = T * (img_e @ sfh_e.T)
        labels   = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        val_loss = (F.cross_entropy(logits,   labels) +
                    F.cross_entropy(logits.T, labels)) / 2.0

        with torch.no_grad():
            r1 = ((logits.argmax(1)   == labels).float().mean() +
                  (logits.T.argmax(1) == labels).float().mean()) / 2

        self.log('val_loss',  val_loss, prog_bar=True,  sync_dist=True)
        self.log('val_rank1', r1,       prog_bar=True,  sync_dist=True)

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer':    optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
