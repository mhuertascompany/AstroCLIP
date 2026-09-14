"""
SFH Transformer Encoder for CosmosWebZooBotCLIP v8.

Instead of a fixed 50-bin interpolated grid, the SFH is represented as
a set of 9 tokens — one per CIGALE bin — each carrying two features:

    token_i = (t_frac_i,  log10(sfr_norm_i + eps))

where t_frac_i = t_sample_i / t_universe(z).

At train time t_sample_i is drawn uniformly within the bin's temporal
extent (data augmentation that prevents the encoder from reading plateau
widths).  At eval time the bin-centre value is used.

Architecture
------------
  Linear(2 → d_model)                    — per-token projection
  + learnable CLS token                  — 1 extra token prepended
  3 × TransformerEncoderLayer            — pre-LayerNorm, batch_first
  LayerNorm + Linear(d_model → embed_dim) — output projection

With d_model=64, n_heads=4, n_layers=3, embed_dim=256 the encoder has
~130 K parameters.

The module also provides a fixed-grid variant for dense Euclid SFHs. It treats
all 250 normalized lookback-time bins as tokens and accepts the same ``(B, N)``
tensor shape as the original fixed-grid MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SFHTransformerEncoder(nn.Module):
    """
    Encode 9 (t_frac, log_sfr) tokens → embed_dim embedding.

    Parameters
    ----------
    n_bins : int
        Number of CIGALE bins (default 9).
    d_model : int
        Internal transformer width (default 64).
    n_heads : int
        Number of attention heads (default 4, head_dim = d_model // n_heads = 16).
    n_layers : int
        Number of TransformerEncoderLayers (default 3).
    embed_dim : int
        Output embedding dimension (default 256).
    dropout : float
        Dropout applied inside the transformer layers (default 0.1).
    """

    def __init__(
        self,
        n_bins:    int   = 9,
        d_model:   int   = 64,
        n_heads:   int   = 4,
        n_layers:  int   = 3,
        embed_dim: int   = 256,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.n_bins    = n_bins
        self.d_model   = d_model
        self.embed_dim = embed_dim

        # Project (t_frac, log_sfr) → d_model
        self.input_proj = nn.Linear(2, d_model)

        # Learnable CLS token (pooling token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder — pre-LayerNorm for stability at small scale
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = 4 * d_model,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,   # pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output: norm + projection
        self.norm     = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, embed_dim)

        self._init_weights()

    # ── weight initialisation ──────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (B, n_bins, 2)
            Each entry is (t_frac, log_sfr) for one CIGALE bin.

        Returns
        -------
        (B, embed_dim)
        """
        B = tokens.shape[0]

        x   = self.input_proj(tokens)                        # (B, n_bins, d_model)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1,      d_model)
        x   = torch.cat([cls, x], dim=1)                     # (B, n_bins+1, d_model)

        x   = self.transformer(x)                            # (B, n_bins+1, d_model)
        cls_out = self.norm(x[:, 0])                         # (B, d_model)
        return self.out_proj(cls_out)                        # (B, embed_dim)


class FixedGridSFHTransformerEncoder(nn.Module):
    """Encode every bin of a common-grid SFH with global self-attention.

    Each scalar SFH value is paired with its fractional lookback time. A
    learned positional embedding preserves bin identity, while a CLS token
    pools information from the full history into the CLIP embedding.
    """

    def __init__(
        self,
        n_bins: int = 250,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_bins < 2:
            raise ValueError('n_bins must be at least two.')
        if d_model < 1 or n_heads < 1 or n_layers < 1:
            raise ValueError('d_model, n_heads, and n_layers must be positive.')
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads.')
        self.n_bins = n_bins
        self.input_proj = nn.Linear(2, d_model)
        self.position = nn.Parameter(torch.zeros(1, n_bins, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer(
            'time_grid', torch.linspace(0.0, 1.0, n_bins), persistent=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, embed_dim)

        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, sfh: torch.Tensor) -> torch.Tensor:
        if sfh.ndim != 2 or sfh.shape[1] != self.n_bins:
            raise ValueError(
                f'Expected SFH shape (batch, {self.n_bins}); got {tuple(sfh.shape)}.'
            )
        batch_size = sfh.shape[0]
        time = self.time_grid.to(dtype=sfh.dtype).expand(batch_size, -1)
        tokens = torch.stack((time, sfh), dim=-1)
        tokens = self.input_proj(tokens)
        tokens = tokens + self.position.to(dtype=tokens.dtype)
        cls = self.cls_token.to(dtype=tokens.dtype).expand(batch_size, -1, -1)
        encoded = self.transformer(torch.cat((cls, tokens), dim=1))
        return self.out_proj(self.norm(encoded[:, 0]))
