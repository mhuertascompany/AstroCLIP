"""
SFH encoder for CIGALE star-formation histories.

Input:  (B, N_TIME)   log10(SFR + eps) on the common lookback-time grid
Output: (B, embed_dim)

Architecture: three-layer MLP with LayerNorm and residual connection.

The SFH is a fixed-length vector (N_TIME = 50 by default) so a shallow MLP
is well-suited and avoids overfitting on moderate-sized datasets.
Upgrade path: swap for a small Transformer (4-layer, 128-dim) if the dataset
exceeds ~50k pairs and MLP performance saturates.
"""

import torch
import torch.nn as nn


class SFHEncoder(nn.Module):
    """
    Three-layer MLP encoder for interpolated CIGALE SFH vectors.

    Parameters
    ----------
    input_dim : int
        Length of the SFH vector (= SFH_N_BINS from prepare_dataset.py, default 50).
    hidden_dim : int
        Width of the hidden layers.
    embed_dim : int
        Dimension of the output embedding.
    dropout : float
        Dropout applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim:  int   = 50,
        hidden_dim: int   = 256,
        embed_dim:  int   = 256,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        self.fc1  = nn.Linear(input_dim,  hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.fc3  = nn.Linear(hidden_dim, embed_dim)

        self.act     = nn.GELU()
        self.drop    = nn.Dropout(dropout)
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)

        # Residual projection when hidden_dim != embed_dim
        self.residual = (
            nn.Linear(hidden_dim, embed_dim, bias=False)
            if hidden_dim != embed_dim else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, input_dim)

        Returns
        -------
        Tensor (B, embed_dim)  — NOT L2-normalised here.
        """
        x = self.input_norm(x)

        h = self.act(self.norm1(self.fc1(x)))     # (B, hidden_dim)
        h = self.drop(h)
        h = self.act(self.norm2(self.fc2(h)))     # (B, hidden_dim)  residual
        h = self.drop(h)

        out = self.fc3(h) + self.residual(h)      # (B, embed_dim)
        return out
