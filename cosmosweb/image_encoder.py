"""
Lightweight image encoder for COSMOS-Web galaxy stamps.

Architecture: ResNet-18 (ImageNet pretrained) → global average pool → MLP projection.

Input:  (B, 3, 64, 64)  normalised galaxy stamps in F150W / F277W / F444W
Output: (B, embed_dim)  L2-normalised embedding vector

The three JWST bands map directly to the three RGB channels expected by ResNet-18.
We fine-tune the entire network end-to-end — no frozen backbone.

ImageNet pretrained weights are used for initialisation; because the bands differ
from RGB the first conv weights are used as-is (empirically still a better start
than random).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CosmosImageEncoder(nn.Module):
    """
    ResNet-18 image encoder with a two-layer MLP projection head.

    Parameters
    ----------
    embed_dim : int
        Dimension of the output embedding.
    pretrained : bool
        If True, initialise the backbone with ImageNet weights.
    dropout : float
        Dropout applied inside the MLP projection head.
    """

    def __init__(
        self,
        embed_dim:  int   = 256,
        pretrained: bool  = True,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # ResNet-18 internal feature dimension (before original FC layer)
        backbone_dim = backbone.fc.in_features  # 512

        # Remove the classification head; keep everything up to global avg pool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # MLP projection: backbone_dim → embed_dim → embed_dim
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
        x : Tensor (B, 3, H, W)

        Returns
        -------
        Tensor (B, embed_dim)  — NOT L2-normalised here; normalisation is
        applied inside the CLIP loss.
        """
        features = self.backbone(x)       # (B, 512, 1, 1)
        return self.projection(features)  # (B, embed_dim)
