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
An opt-in SFH-aware mode replaces one-hot targets with a mixture of the exact
pair and Wasserstein-nearest SFHs. Its default weight is zero, preserving the
original objective and existing checkpoints.
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
from .sfh_autoencoder import FixedGridSFHDecoder, sfh_reconstruction_loss
from .sfh_transformer import (
    FixedGridSFHTransformerEncoder,
    SFHTransformerEncoder,
)
from .sfh_similarity import soft_cross_entropy, wasserstein_soft_targets
from .zoobot_encoder import MultiFilterZooBotImageEncoder, ZooBotImageEncoder


def make_sfh_projection(kind: str, embed_dim: int) -> nn.Module:
    """Build the optional map from the SFH latent into the CLIP space.

    The linear option starts as the identity, so an autoencoder latent is
    unchanged before contrastive training.  Keeping this map outside the SFH
    encoder lets the latter remain a fixed, reconstructive representation.
    """
    if kind == 'identity':
        return nn.Identity()
    if kind == 'linear':
        projection = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(projection.weight)
        return projection
    raise ValueError(
        f'Unknown sfh_projection_type={kind!r}; choose identity or linear.'
    )


class CosmosWebZooBotCLIP(L.LightningModule):

    def __init__(
        self,
        zoobot_ckpt:      str | None = None,
        zoobot_model_name: str | None = None,
        embed_dim:        int   = 256,
        sfh_input_dim:    int   = 50,
        temperature:      float = 0.07,
        queue_size:       int   = 4096,
        momentum:         float = 0.995,
        lr:               float = 1e-4,
        weight_decay:     float = 0.05,
        epochs:           int   = 50,
        warmup_epochs:    int   = 5,
        unfreeze_blocks:  int   = 0,
        backbone_lr_scale: float = 0.1,
        sfh_encoder_type: str = 'mlp',
        sfh_d_model:      int = 128,
        sfh_n_heads:      int = 4,
        sfh_n_layers:     int = 4,
        sfh_lr_scale:     float = 1.0,
        soft_positive_weight: float = 0.0,
        soft_positive_k:  int = 8,
        sfh_log_epsilon:  float = 1e-10,
        sfh_reconstruction_weight: float = 0.0,
        sfh_reconstruction_w1_weight: float = 0.5,
        sfh_decoder_layers: int = 2,
        sfh_projection_type: str = 'identity',
        freeze_sfh_encoder: bool = False,
        freeze_sfh_decoder: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= soft_positive_weight <= 1.0:
            raise ValueError('soft_positive_weight must lie in [0, 1].')
        if soft_positive_k < 1:
            raise ValueError('soft_positive_k must be positive.')
        if sfh_log_epsilon <= 0:
            raise ValueError('sfh_log_epsilon must be positive.')
        if sfh_reconstruction_weight < 0:
            raise ValueError('sfh_reconstruction_weight cannot be negative.')
        if not 0 <= sfh_reconstruction_w1_weight <= 1:
            raise ValueError('sfh_reconstruction_w1_weight must lie in [0, 1].')
        if sfh_projection_type not in {'identity', 'linear'}:
            raise ValueError('sfh_projection_type must be identity or linear.')
        if freeze_sfh_decoder and sfh_reconstruction_weight <= 0:
            raise ValueError(
                'freeze_sfh_decoder requires a positive reconstruction weight '
                'so that a decoder is constructed.'
            )
        if soft_positive_weight > 0 and queue_size:
            raise ValueError(
                'SFH soft positives currently require queue_size=0 because '
                'the embedding queue does not store reference SFHs.'
            )
        self.save_hyperparameters()

        # ── main encoders (receive gradients) ────────────────────────────────
        self.image_encoder = ZooBotImageEncoder(
            ckpt_path=zoobot_ckpt,
            model_name=zoobot_model_name,
            embed_dim=embed_dim,
            unfreeze_blocks=unfreeze_blocks,
        )
        if sfh_encoder_type == 'mlp':
            self.sfh_encoder = SFHEncoder(
                input_dim=sfh_input_dim,
                embed_dim=embed_dim,
            )
        elif sfh_encoder_type == 'transformer':
            self.sfh_encoder = FixedGridSFHTransformerEncoder(
                n_bins=sfh_input_dim,
                d_model=sfh_d_model,
                n_heads=sfh_n_heads,
                n_layers=sfh_n_layers,
                embed_dim=embed_dim,
            )
        else:
            raise ValueError(
                f'Unknown sfh_encoder_type={sfh_encoder_type!r}; '
                "choose 'mlp' or 'transformer'."
            )
        self.sfh_projection = make_sfh_projection(
            sfh_projection_type, embed_dim,
        )
        if sfh_reconstruction_weight > 0:
            if sfh_encoder_type != 'transformer':
                raise ValueError('SFH reconstruction requires the transformer encoder.')
            self.sfh_decoder = FixedGridSFHDecoder(
                n_bins=sfh_input_dim,
                embed_dim=embed_dim,
                d_model=sfh_d_model,
                n_heads=sfh_n_heads,
                n_layers=sfh_decoder_layers,
            )
        else:
            self.sfh_decoder = None

        if freeze_sfh_encoder:
            for parameter in self.sfh_encoder.parameters():
                parameter.requires_grad_(False)
            self.sfh_encoder.eval()
        if freeze_sfh_decoder:
            for parameter in self.sfh_decoder.parameters():
                parameter.requires_grad_(False)
            self.sfh_decoder.eval()

        # ── momentum encoders (EMA, no gradient) ─────────────────────────────
        self.image_encoder_m = copy.deepcopy(self.image_encoder)
        self.sfh_encoder_m   = copy.deepcopy(self.sfh_encoder)
        self.sfh_projection_m = copy.deepcopy(self.sfh_projection)
        for p in self.image_encoder_m.parameters():
            p.requires_grad_(False)
        for p in self.sfh_encoder_m.parameters():
            p.requires_grad_(False)
        for p in self.sfh_projection_m.parameters():
            p.requires_grad_(False)
        self.image_encoder_m.eval()
        self.sfh_encoder_m.eval()
        self.sfh_projection_m.eval()

        # ── learnable temperature ─────────────────────────────────────────────
        self.log_temp = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))

        # ── circular queues (registered as buffers → saved in checkpoint) ────
        Q = queue_size
        D = embed_dim
        if Q < 0:
            raise ValueError('queue_size cannot be negative.')
        queue_img = F.normalize(torch.randn(D, Q), dim=0) if Q else torch.empty(D, 0)
        queue_sfh = F.normalize(torch.randn(D, Q), dim=0) if Q else torch.empty(D, 0)
        self.register_buffer('queue_img', queue_img)
        self.register_buffer('queue_sfh', queue_sfh)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    def train(self, mode: bool = True):
        """Keep momentum targets deterministic while training query encoders."""
        super().train(mode)
        self.image_encoder_m.eval()
        self.sfh_encoder_m.eval()
        self.sfh_projection_m.eval()
        if self.hparams.freeze_sfh_encoder:
            self.sfh_encoder.eval()
        if self.sfh_decoder is not None and self.hparams.freeze_sfh_decoder:
            self.sfh_decoder.eval()
        return self

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
        for p, pm in zip(self.sfh_projection.parameters(),
                         self.sfh_projection_m.parameters()):
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

    def encode_sfh_latent(self, sfh: torch.Tensor) -> torch.Tensor:
        return self.sfh_encoder(sfh)

    def project_sfh(self, latent: torch.Tensor) -> torch.Tensor:
        return self.sfh_projection(latent)

    def encode_sfh(self, sfh: torch.Tensor) -> torch.Tensor:
        return self.project_sfh(self.encode_sfh_latent(sfh))

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
        sfh_reference = batch.get('sfh_reference', sfhs)
        B = images.size(0)
        Q = self.hparams.queue_size

        if Q and B > Q:
            raise ValueError(
                f'batch_size ({B}) > queue_size ({Q}). '
                'Reduce batch_size or increase queue_size.')

        T = self.log_temp.exp().clamp(min=1.0, max=100.0)

        # ── query embeddings (main encoders, receive gradients) ───────────────
        img_q = F.normalize(self.encode_image(images), dim=-1)   # (B, D)
        sfh_latent = self.encode_sfh_latent(sfhs)
        sfh_q = F.normalize(self.project_sfh(sfh_latent), dim=-1)  # (B, D)

        # ── key embeddings (momentum encoders, no gradient) ───────────────────
        with torch.no_grad():
            self._momentum_update()
            img_k = F.normalize(self.image_encoder_m(images), dim=-1)  # (B, D)
            sfh_k = F.normalize(
                self.sfh_projection_m(self.sfh_encoder_m(sfhs)), dim=-1,
            )

        # ── all keys = current batch (momentum) + queue ───────────────────────
        # Shape: (B + Q, D)
        all_sfh = torch.cat([sfh_k, self.queue_sfh.T.clone().detach()], dim=0)
        all_img = torch.cat([img_k, self.queue_img.T.clone().detach()], dim=0)

        # ── contrastive loss (symmetric) ──────────────────────────────────────
        logits_i2s = img_q @ all_sfh.T * T
        logits_s2i = sfh_q @ all_img.T * T
        labels = torch.arange(B, device=images.device, dtype=torch.long)
        exact_loss = (
            F.cross_entropy(logits_i2s, labels)
            + F.cross_entropy(logits_s2i, labels)
        ) / 2.0
        if self.hparams.soft_positive_weight > 0:
            targets, sfh_w1 = wasserstein_soft_targets(
                sfh_reference,
                soft_weight=self.hparams.soft_positive_weight,
                n_neighbors=self.hparams.soft_positive_k,
                epsilon=self.hparams.sfh_log_epsilon,
            )
            contrastive_loss = (
                soft_cross_entropy(logits_i2s, targets)
                + soft_cross_entropy(logits_s2i, targets)
            ) / 2.0
        else:
            sfh_w1 = None
            contrastive_loss = exact_loss

        if self.sfh_decoder is not None:
            reconstruction = self.sfh_decoder(sfh_latent)
            reconstruction_loss, reconstruction_parts = sfh_reconstruction_loss(
                reconstruction,
                sfh_reference,
                batch.get('sfh_p16'),
                batch.get('sfh_p84'),
                epsilon=self.hparams.sfh_log_epsilon,
                w1_weight=self.hparams.sfh_reconstruction_w1_weight,
            )
            loss = (
                contrastive_loss
                + self.hparams.sfh_reconstruction_weight * reconstruction_loss
            )
            self.log('train_reconstruction_loss', reconstruction_loss, sync_dist=True)
            self.log('train_reconstruction_w1', reconstruction_parts['w1'], sync_dist=True)
        else:
            loss = contrastive_loss

        # ── within-batch rank-1 (cheap training-time diagnostic) ─────────────
        with torch.no_grad():
            logits_b = img_q @ sfh_k.T * T              # (B, B) batch-only
            r1       = ((logits_b.argmax(1) == labels).float().mean() +
                        (logits_b.T.argmax(1) == labels).float().mean()) / 2

        # ── update queue ──────────────────────────────────────────────────────
        if Q:
            self._dequeue_and_enqueue(img_k, sfh_k)

        self.log('train_loss',       loss, prog_bar=True,  sync_dist=True)
        self.log('train_contrastive_loss', contrastive_loss, sync_dist=True)
        self.log('train_exact_loss', exact_loss, prog_bar=False, sync_dist=True)
        self.log('train_rank1_batch', r1,  prog_bar=True,  sync_dist=True)
        self.log('train_logit_scale', T,   prog_bar=False, sync_dist=True)
        self.log(
            'train_loss_vs_random',
            contrastive_loss - contrastive_loss.new_tensor(np.log(B + Q)),
            sync_dist=True,
        )
        if sfh_w1 is not None:
            off_diagonal = ~torch.eye(B, device=images.device, dtype=torch.bool)
            self.log(
                'train_sfh_w1_mean', sfh_w1[off_diagonal].mean(),
                sync_dist=True,
            )
        return loss

    # ── validation step (batch-only InfoNCE, no queue) ────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images, sfhs = batch['image'], batch['sfh']
        sfh_reference = batch.get('sfh_reference', sfhs)
        T = self.log_temp.exp().clamp(min=1.0, max=100.0)

        img_e = F.normalize(self.encode_image(images), dim=-1)
        sfh_latent = self.encode_sfh_latent(sfhs)
        sfh_e = F.normalize(self.project_sfh(sfh_latent), dim=-1)

        logits  = T * (img_e @ sfh_e.T)
        labels  = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        val_exact_loss = (F.cross_entropy(logits,   labels) +
                          F.cross_entropy(logits.T, labels)) / 2.0
        if self.hparams.soft_positive_weight > 0:
            targets, _ = wasserstein_soft_targets(
                sfh_reference,
                soft_weight=self.hparams.soft_positive_weight,
                n_neighbors=self.hparams.soft_positive_k,
                epsilon=self.hparams.sfh_log_epsilon,
            )
            val_contrastive_loss = (
                soft_cross_entropy(logits, targets)
                + soft_cross_entropy(logits.T, targets)
            ) / 2.0
            self.log('val_soft_loss', val_contrastive_loss, sync_dist=True)
        else:
            val_contrastive_loss = val_exact_loss

        if self.sfh_decoder is not None:
            reconstruction = self.sfh_decoder(sfh_latent)
            reconstruction_loss, reconstruction_parts = sfh_reconstruction_loss(
                reconstruction,
                sfh_reference,
                batch.get('sfh_p16'),
                batch.get('sfh_p84'),
                epsilon=self.hparams.sfh_log_epsilon,
                w1_weight=self.hparams.sfh_reconstruction_w1_weight,
            )
            val_loss = (
                val_contrastive_loss
                + self.hparams.sfh_reconstruction_weight * reconstruction_loss
            )
            self.log('val_reconstruction_loss', reconstruction_loss, sync_dist=True)
            self.log('val_reconstruction_w1', reconstruction_parts['w1'], sync_dist=True)
        else:
            val_loss = val_contrastive_loss

        with torch.no_grad():
            r1 = ((logits.argmax(1)   == labels).float().mean() +
                  (logits.T.argmax(1) == labels).float().mean()) / 2
            k = min(5, logits.size(1))
            r5_i2s = (logits.topk(k, dim=1).indices == labels[:, None]).any(1)
            r5_s2i = (logits.T.topk(k, dim=1).indices == labels[:, None]).any(1)
            r5 = (r5_i2s.float().mean() + r5_s2i.float().mean()) / 2
            cosine = img_e @ sfh_e.T
            positive_cosine = cosine.diagonal().mean()
            if cosine.size(0) > 1:
                negative_cosine = (
                    cosine.sum() - cosine.diagonal().sum()
                ) / (cosine.numel() - cosine.size(0))
            else:
                negative_cosine = positive_cosine.new_zeros(())
            alignment_margin = positive_cosine - negative_cosine
            random_loss = val_contrastive_loss.new_tensor(np.log(logits.size(0)))

        self.log('val_loss',   val_loss, prog_bar=True,  sync_dist=True)
        self.log('val_contrastive_loss', val_contrastive_loss, sync_dist=True)
        self.log('val_exact_loss', val_exact_loss, sync_dist=True)
        self.log('val_rank1',  r1,       prog_bar=True,  sync_dist=True)
        self.log('val_rank5', r5, prog_bar=True, sync_dist=True)
        self.log(
            'val_loss_vs_random', val_contrastive_loss - random_loss, sync_dist=True,
        )
        self.log('val_positive_cosine', positive_cosine, sync_dist=True)
        self.log('val_negative_cosine', negative_cosine, sync_dist=True)
        self.log('val_alignment_margin', alignment_margin, sync_dist=True)

    # ── optimiser & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        # Split trainable parameters into backbone (unfrozen blocks, lower lr)
        # and everything else (projection head + SFH encoder, full lr).
        backbone_params = [
            p for n, p in self.image_encoder.backbone.named_parameters()
            if p.requires_grad
        ]
        other_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad
            and not n.startswith('image_encoder.backbone.')
            and not n.startswith('sfh_encoder.')
            and not n.startswith('sfh_decoder.')
            and not n.startswith('sfh_projection_m.')
            and not n.startswith('image_encoder_m.')
            and not n.startswith('sfh_encoder_m.')
        ]
        sfh_params = [p for p in self.sfh_encoder.parameters() if p.requires_grad]
        if self.sfh_decoder is not None:
            sfh_params.extend(p for p in self.sfh_decoder.parameters() if p.requires_grad)

        param_groups = [{'params': other_params, 'lr': self.hparams.lr}]
        if sfh_params:
            param_groups.append({
                'params': sfh_params,
                'lr': self.hparams.lr * self.hparams.sfh_lr_scale,
            })
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.hparams.lr * self.hparams.backbone_lr_scale,
            })

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.epochs
            if epoch < wu:
                return (epoch + 1) / max(wu, 1)
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


# ── v8: transformer SFH encoder ───────────────────────────────────────────────

class CosmosWebZooBotCLIPv8(L.LightningModule):
    """
    CosmosWebZooBotCLIP v8 — same MoCo CLIP framework as v4/v7 but with a
    Set-Transformer SFH encoder replacing the fixed-grid FC encoder.

    SFH input: (B, 9, 2) tokens — each token is (t_frac, log_sfr) for one
    CIGALE bin.  t_frac is randomly sampled within each bin's temporal extent
    at train time, breaking the plateau-width redshift artifact.
    """

    def __init__(
        self,
        zoobot_ckpt:      str,
        embed_dim:        int   = 256,
        # SFH transformer hyperparameters
        sfh_n_bins:       int   = 9,
        sfh_d_model:      int   = 64,
        sfh_n_heads:      int   = 4,
        sfh_n_layers:     int   = 3,
        # CLIP / MoCo
        temperature:      float = 0.07,
        queue_size:       int   = 1024,
        momentum:         float = 0.995,
        # Optimiser
        lr:               float = 1e-4,
        weight_decay:     float = 0.05,
        epochs:           int   = 100,
        warmup_epochs:    int   = 3,
        # Backbone fine-tuning
        unfreeze_blocks:  int   = 0,
        backbone_lr_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = ZooBotImageEncoder(
            ckpt_path=zoobot_ckpt,
            embed_dim=embed_dim,
            unfreeze_blocks=unfreeze_blocks,
        )
        self.sfh_encoder = SFHTransformerEncoder(
            n_bins=sfh_n_bins,
            d_model=sfh_d_model,
            n_heads=sfh_n_heads,
            n_layers=sfh_n_layers,
            embed_dim=embed_dim,
        )

        self.image_encoder_m = copy.deepcopy(self.image_encoder)
        self.sfh_encoder_m   = copy.deepcopy(self.sfh_encoder)
        for p in self.image_encoder_m.parameters():
            p.requires_grad_(False)
        for p in self.sfh_encoder_m.parameters():
            p.requires_grad_(False)

        self.log_temp = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))

        Q, D = queue_size, embed_dim
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

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_sfh(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 9, 2)"""
        return self.sfh_encoder(tokens)

    @staticmethod
    def _infoNCE(queries, all_keys, T):
        B = queries.shape[0]
        logits = torch.matmul(queries, all_keys.T) * T.exp()
        labels = torch.arange(B, device=queries.device)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        tokens = batch['tokens']   # (B, 9, 2)
        B = images.shape[0]

        self._momentum_update()

        img_q = F.normalize(self.image_encoder(images), dim=1)
        sfh_q = F.normalize(self.sfh_encoder(tokens),  dim=1)

        with torch.no_grad():
            img_k = F.normalize(self.image_encoder_m(images), dim=1)
            sfh_k = F.normalize(self.sfh_encoder_m(tokens),   dim=1)

        all_img = torch.cat([img_k, self.queue_img.T.clone()], dim=0)
        all_sfh = torch.cat([sfh_k, self.queue_sfh.T.clone()], dim=0)

        T = self.log_temp
        loss = (self._infoNCE(sfh_q, all_img, T) +
                self._infoNCE(img_q, all_sfh, T)) / 2.0

        self._dequeue_and_enqueue(img_k, sfh_k)

        with torch.no_grad():
            logits = torch.matmul(sfh_q, img_k.T) * T.exp()
            r1 = (logits.argmax(1) == torch.arange(B, device=self.device)).float().mean()

        self.log('train_loss',       loss, prog_bar=True,  on_step=True, on_epoch=False)
        self.log('train_rank1_batch', r1,  prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        tokens = batch['tokens']
        B = images.shape[0]

        img_q = F.normalize(self.image_encoder(images), dim=1)
        sfh_q = F.normalize(self.sfh_encoder(tokens),  dim=1)

        logits   = torch.matmul(sfh_q, img_q.T) * self.log_temp.exp()
        labels   = torch.arange(B, device=self.device)
        val_loss = (F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)) / 2.0

        with torch.no_grad():
            r1 = ((logits.argmax(1)   == labels).float().mean() +
                  (logits.T.argmax(1) == labels).float().mean()) / 2

        self.log('val_loss',  val_loss, prog_bar=True, sync_dist=True)
        self.log('val_rank1', r1,       prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        backbone_params = [p for n, p in self.image_encoder.backbone.named_parameters()
                           if p.requires_grad]
        other_params = [p for n, p in self.named_parameters()
                        if p.requires_grad
                        and not n.startswith('image_encoder.backbone.')
                        and not n.startswith('image_encoder_m.')
                        and not n.startswith('sfh_encoder_m.')]

        param_groups = [{'params': other_params, 'lr': self.hparams.lr}]
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.hparams.lr * self.hparams.backbone_lr_scale,
            })

        optimizer = torch.optim.AdamW(param_groups,
                                      weight_decay=self.hparams.weight_decay)

        def lr_lambda(epoch):
            wu    = self.hparams.warmup_epochs
            total = self.hparams.epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
