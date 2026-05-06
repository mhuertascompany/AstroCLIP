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

        # Probe backbone output dimension with a dummy forward pass.
        # Move dummy to the same device as the backbone weights.
        with torch.no_grad():
            _device = next(self.backbone.parameters()).device
            dummy = torch.zeros(1, 3, 224, 224, device=_device)
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


def _load_backbone(ckpt_path: str | Path):
    """Load a frozen ZooBOT EfficientNet backbone from a checkpoint."""
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
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone


class MultiFilterZooBotImageEncoder(nn.Module):
    """
    Three frozen ZooBOT EfficientNet backbones (one per JWST filter) with a
    shared trainable MLP projection head.

    Each sample is routed to the backbone matching its redshift:
        z < z_low         → backbone_f150w   (rest-frame optical at z < 1)
        z_low ≤ z < z_high → backbone_f277w  (rest-frame optical at 1 ≤ z < 3)
        z ≥ z_high        → backbone_f444w   (rest-frame optical at z ≥ 3)

    All three backbones must share the same architecture and output dimension
    (which is the case for ZooBOT EfficientNet checkpoints).

    Parameters
    ----------
    ckpt_f150w, ckpt_f277w, ckpt_f444w : str | Path
        Paths to FinetuneableZoobotClassifier checkpoints for each filter.
    embed_dim : int
        Output embedding dimension for the shared projection head.
    z_low : float
        Redshift boundary between F150W and F277W (default 1.0).
    z_high : float
        Redshift boundary between F277W and F444W (default 3.0).
    dropout : float
        Dropout in the MLP projection head.
    """

    def __init__(
        self,
        ckpt_f150w:  str | Path,
        ckpt_f277w:  str | Path,
        ckpt_f444w:  str | Path,
        embed_dim:   int   = 256,
        z_low:       float = 1.0,
        z_high:      float = 3.0,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()

        self.z_low  = z_low
        self.z_high = z_high

        self.backbone_f150w = _load_backbone(ckpt_f150w)
        self.backbone_f277w = _load_backbone(ckpt_f277w)
        self.backbone_f444w = _load_backbone(ckpt_f444w)

        # Probe backbone output dimension (all three must agree)
        with torch.no_grad():
            _device = next(self.backbone_f277w.parameters()).device
            dummy = torch.zeros(1, 3, 224, 224, device=_device)
            backbone_dim = self.backbone_f277w(dummy).flatten(1).shape[1]

        # Shared MLP projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, redshifts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, 3, 224, 224)
        redshifts : Tensor (B,)  — float, redshift of each sample.

        Returns
        -------
        Tensor (B, embed_dim)  — NOT L2-normalised here.
        """
        mask_f150 = redshifts < self.z_low
        mask_f444 = redshifts >= self.z_high
        mask_f277 = ~mask_f150 & ~mask_f444

        # Allocate output buffer using the backbone dtype (may be float16 in AMP)
        feats = x.new_zeros(x.size(0), self.projection[0].in_features)

        for mask, backbone in [
            (mask_f150, self.backbone_f150w),
            (mask_f277, self.backbone_f277w),
            (mask_f444, self.backbone_f444w),
        ]:
            if mask.any():
                with torch.no_grad():
                    out = backbone(x[mask]).flatten(1)
                feats[mask] = out

        return self.projection(feats)
