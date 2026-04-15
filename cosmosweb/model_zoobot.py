"""
CosmosWebZooBotCLIP: contrastive alignment of COSMOS-Web images and CIGALE SFHs
using a frozen ZooBOT EfficientNet backbone as the image encoder.

Mirrors model.py exactly, replacing CosmosImageEncoder with ZooBotImageEncoder.
The ResNet-18 baseline in model.py is preserved untouched.

The ZooBOT backbone is frozen — only the MLP projection head and the SFH encoder
are trained end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sfh_encoder import SFHEncoder
from .zoobot_encoder import ZooBotImageEncoder


class CLIPLoss(nn.Module):
    """Symmetric InfoNCE / CLIP contrastive loss (identical to model.py)."""

    def forward(
        self,
        image_feats: torch.Tensor,
        sfh_feats:   torch.Tensor,
        logit_scale: float,
    ) -> torch.Tensor:
        image_feats = F.normalize(image_feats, dim=-1, eps=1e-3)
        sfh_feats   = F.normalize(sfh_feats,   dim=-1, eps=1e-3)

        logits = logit_scale * image_feats @ sfh_feats.T
        labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)

        loss = (
            F.cross_entropy(logits,   labels) +
            F.cross_entropy(logits.T, labels)
        ) / 2
        return loss


class CosmosWebZooBotCLIP(L.LightningModule):
    """
    CLIP model using a frozen ZooBOT backbone as the image encoder.

    Parameters
    ----------
    zoobot_ckpt : str
        Path to the FinetuneableZoobotClassifier checkpoint.
    embed_dim : int
        Shared embedding dimension.
    sfh_input_dim : int
        Length of the SFH vector (= SFH_N_BINS, default 50).
    temperature : float
        Initial logit scale exp(t).
    lr : float
    weight_decay : float
    epochs : int
    warmup_epochs : int
    """

    def __init__(
        self,
        zoobot_ckpt:   str,
        embed_dim:     int   = 256,
        sfh_input_dim: int   = 50,
        temperature:   float = 0.07,
        lr:            float = 1e-4,
        weight_decay:  float = 0.05,
        epochs:        int   = 50,
        warmup_epochs: int   = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = ZooBotImageEncoder(
            ckpt_path=zoobot_ckpt,
            embed_dim=embed_dim,
        )
        self.sfh_encoder = SFHEncoder(
            input_dim=sfh_input_dim,
            embed_dim=embed_dim,
        )

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

        self.log(f'{stage}_loss',        loss,        prog_bar=True, sync_dist=True)
        self.log(f'{stage}_logit_scale', logit_scale, prog_bar=False, sync_dist=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, 'val')

    # ── optimiser & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        # Only optimise parameters that require gradients (backbone is frozen)
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
