"""
ZooBOT-based image encoder for COSMOS-Web CLIP training.

Uses a pretrained/finetuned ZooBOT EfficientNet backbone (frozen) as
feature extractor, with a trainable MLP projection head.

The backbone is loaded from a FinetuneableZoobotClassifier checkpoint
(e.g. the family morphology model trained on COSMOS-Web visuals).

Input:  (B, 3, 224, 224)  grayscale JPEG replicated to 3 channels, ToTensor [0,1]
Output: (B, embed_dim)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from zoobot.pytorch.training import finetune


class ZooBotImageEncoder(nn.Module):
    """
    Frozen ZooBOT EfficientNet backbone + trainable MLP projection head.

    Parameters
    ----------
    ckpt_path : str | Path
        Path to a FinetuneableZoobotClassifier Lightning checkpoint
        (e.g. family_2.ckpt trained on COSMOS-Web visual morphology).
    embed_dim : int
        Output embedding dimension.
    dropout : float
        Dropout applied inside the MLP projection head.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        embed_dim: int   = 256,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()

        # ── load ZooBOT classifier and extract the EfficientNet backbone ──────
        zoobot = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(
            str(ckpt_path), strict=False
        )
        zoobot.eval()

        backbone = getattr(zoobot, 'encoder', None)
        if backbone is None:
            raise AttributeError(
                "Cannot find 'encoder' attribute on FinetuneableZoobotClassifier. "
                "Check your ZooBOT version or inspect the checkpoint manually."
            )

        self.backbone = backbone

        # Freeze — no gradients flow through the ZooBOT backbone
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        # Probe backbone output dimension with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out   = self.backbone(dummy)
            backbone_dim = out.flatten(1).shape[1]

        # MLP projection: backbone_dim → embed_dim → embed_dim  (same as CosmosImageEncoder)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, 3, 224, 224)
            Grayscale JPEG replicated to 3 channels; pixel values in [0, 1].

        Returns
        -------
        Tensor (B, embed_dim)  — NOT L2-normalised here.
        """
        features = self.backbone(x)           # (B, backbone_dim) or (B, C, 1, 1)
        features = features.flatten(1)        # (B, backbone_dim)
        return self.projection(features)      # (B, embed_dim)
