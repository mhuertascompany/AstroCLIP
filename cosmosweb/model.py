"""
CosmosWebCLIP: contrastive alignment of COSMOS-Web images and CIGALE SFHs.

Architecture mirrors AstroCLIP (astroclip/models/astroclip.py) but with:
  - No CrossAttentionHead (both encoders produce a flat vector directly)
  - Smaller default embedding dimension (256 vs 1024)
  - Full end-to-end training (no frozen backbone)

The CLIP loss maximises the cosine similarity of matched (image, SFH) pairs
while pushing non-matched pairs apart within the same batch.
"""

from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import CosmosImageEncoder
from .sfh_encoder import SFHEncoder


class CLIPLoss(nn.Module):
    """Symmetric InfoNCE / CLIP contrastive loss."""

    def forward(
        self,
        image_feats:  torch.Tensor,
        sfh_feats:    torch.Tensor,
        logit_scale:  float,
    ) -> torch.Tensor:
        image_feats = F.normalize(image_feats, dim=-1, eps=1e-3)
        sfh_feats   = F.normalize(sfh_feats,   dim=-1, eps=1e-3)

        logits = logit_scale * image_feats @ sfh_feats.T  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)

        loss = (
            F.cross_entropy(logits,   labels) +
            F.cross_entropy(logits.T, labels)
        ) / 2
        return loss


class CosmosWebCLIP(L.LightningModule):
    """
    End-to-end CLIP model for COSMOS-Web images × CIGALE SFHs.

    Parameters
    ----------
    embed_dim : int
        Shared embedding dimension for both modalities.
    sfh_input_dim : int
        Length of the SFH vector (= SFH_N_BINS, default 50).
    temperature : float
        Initial value of the learnable logit scale exp(t).
    lr : float
        Peak learning rate (cosine schedule with warm-up).
    weight_decay : float
    epochs : int
        Total training epochs (used for cosine LR schedule).
    warmup_epochs : int
        Number of linear warm-up epochs.
    pretrained_image_encoder : bool
        If True, initialise the image backbone with ImageNet weights.
    """

    def __init__(
        self,
        embed_dim:               int   = 256,
        sfh_input_dim:           int   = 50,
        temperature:             float = 0.07,
        lr:                      float = 1e-4,
        weight_decay:            float = 0.05,
        epochs:                  int   = 100,
        warmup_epochs:           int   = 5,
        pretrained_image_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = CosmosImageEncoder(
            embed_dim=embed_dim,
            pretrained=pretrained_image_encoder,
        )
        self.sfh_encoder = SFHEncoder(
            input_dim=sfh_input_dim,
            embed_dim=embed_dim,
        )

        # Learnable logit scale: logit_scale = exp(log_temp)
        self.log_temp = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))
        self.criterion = CLIPLoss()

    # ── forward ───────────────────────────────────────────────────────────────

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_sfh(self, sfh: torch.Tensor) -> torch.Tensor:
        return self.sfh_encoder(sfh)

    def forward(
        self, image: torch.Tensor, sfh: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(image), self.encode_sfh(sfh)

    # ── training / validation steps ───────────────────────────────────────────

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        img_feats = self.encode_image(batch['image'])
        sfh_feats = self.encode_sfh(batch['sfh'])

        logit_scale = self.log_temp.exp().clamp(max=np.log(100))
        loss = self.criterion(img_feats, sfh_feats, logit_scale)

        self.log(f'{stage}_loss',       loss,        prog_bar=True, sync_dist=True)
        self.log(f'{stage}_logit_scale', logit_scale, prog_bar=False, sync_dist=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, 'val')

    # ── optimiser & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Linear warm-up then cosine decay
        def lr_lambda(epoch: int) -> float:
            wu = self.hparams.warmup_epochs
            total = self.hparams.epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
