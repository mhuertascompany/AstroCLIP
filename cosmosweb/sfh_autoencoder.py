"""Bottleneck decoder and reconstruction losses for fixed-grid SFHs.

The existing Euclid SFH transformer remains the encoder.  The decoder receives
only its global embedding and a set of time-bin queries, which forces that
embedding to retain enough information to reconstruct the complete history.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sfh_to_mass(log_sfh: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Convert stored log10 weights to a non-negative unit-mass SFH."""
    mass = torch.clamp(torch.pow(10.0, log_sfh.float()) - epsilon, min=0.0)
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def mask_contiguous_bins(
    sfh: torch.Tensor,
    fraction: float,
    replacement: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace one random contiguous interval in each SFH.

    A single broad interval makes the encoder infer missing history from both
    earlier and later time bins instead of interpolating isolated pixels.
    """
    if sfh.ndim != 2:
        raise ValueError(f'Expected a two-dimensional SFH tensor; got {sfh.shape}.')
    if not 0.0 <= fraction < 1.0:
        raise ValueError('Mask fraction must lie in [0, 1).')
    batch_size, n_bins = sfh.shape
    mask = torch.zeros_like(sfh, dtype=torch.bool)
    if fraction == 0:
        return sfh, mask
    width = min(n_bins - 1, max(1, int(round(fraction * n_bins))))
    starts = torch.randint(0, n_bins - width + 1, (batch_size,), device=sfh.device)
    bins = torch.arange(n_bins, device=sfh.device).unsqueeze(0)
    mask = (bins >= starts[:, None]) & (bins < starts[:, None] + width)
    replacement = torch.as_tensor(replacement, dtype=sfh.dtype, device=sfh.device)
    return torch.where(mask, replacement, sfh), mask


class FixedGridSFHDecoder(nn.Module):
    """Reconstruct a unit-normalized SFH from one global latent vector."""

    def __init__(
        self,
        n_bins: int = 250,
        embed_dim: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_bins < 2:
            raise ValueError('n_bins must be at least two.')
        if min(embed_dim, d_model, n_heads, n_layers) < 1:
            raise ValueError('Decoder dimensions and layer count must be positive.')
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads.')
        self.n_bins = n_bins
        self.latent_proj = nn.Linear(embed_dim, d_model)
        self.time_proj = nn.Linear(1, d_model)
        self.query = nn.Parameter(torch.zeros(1, n_bins, d_model))
        self.register_buffer(
            'time_grid', torch.linspace(0.0, 1.0, n_bins), persistent=True,
        )
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)

        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.latent_proj.weight, std=0.02)
        nn.init.zeros_(self.latent_proj.bias)
        nn.init.trunc_normal_(self.time_proj.weight, std=0.02)
        nn.init.zeros_(self.time_proj.bias)
        nn.init.trunc_normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError(f'Expected latent shape (batch, channels); got {latent.shape}.')
        batch_size = latent.shape[0]
        memory = self.latent_proj(latent).unsqueeze(1)
        time = self.time_grid.to(dtype=latent.dtype).view(1, self.n_bins, 1)
        queries = self.query.to(dtype=latent.dtype) + self.time_proj(time)
        queries = queries.expand(batch_size, -1, -1)
        decoded = self.transformer(tgt=queries, memory=memory)
        logits = self.output(self.norm(decoded)).squeeze(-1)
        return torch.softmax(logits.float(), dim=-1).to(dtype=latent.dtype)


def sfh_reconstruction_loss(
    prediction: torch.Tensor,
    target_log: torch.Tensor,
    p16_log: torch.Tensor | None = None,
    p84_log: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-10,
    w1_weight: float = 0.5,
    huber_beta: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine cumulative W1 distance and uncertainty-weighted Huber loss."""
    if not 0.0 <= w1_weight <= 1.0:
        raise ValueError('w1_weight must lie in [0, 1].')
    target = log_sfh_to_mass(target_log, epsilon)
    prediction = prediction.float()
    if prediction.shape != target.shape:
        raise ValueError(
            f'Prediction and target shapes differ: {prediction.shape} vs {target.shape}.'
        )

    point_loss = F.smooth_l1_loss(
        prediction, target, reduction='none', beta=huber_beta,
    )
    if p16_log is not None and p84_log is not None:
        # These are pointwise posterior percentiles and need not integrate to
        # one individually; renormalizing them would distort the interval.
        lower = torch.clamp(torch.pow(10.0, p16_log.float()) - epsilon, min=0.0)
        upper = torch.clamp(torch.pow(10.0, p84_log.float()) - epsilon, min=0.0)
        width = torch.clamp(upper - lower, min=0.0)
        weights = 1.0 / (width + 1e-4)
        weights = weights / weights.mean(dim=-1, keepdim=True).clamp_min(epsilon)
        weights = weights.clamp(max=10.0)
    else:
        weights = torch.ones_like(point_loss)
    huber = (
        (point_loss * weights).sum(dim=-1)
        / weights.sum(dim=-1).clamp_min(epsilon)
        * prediction.shape[-1]
    ).mean()
    w1 = torch.abs(
        torch.cumsum(prediction, dim=-1) - torch.cumsum(target, dim=-1)
    ).mean()
    loss = w1_weight * w1 + (1.0 - w1_weight) * huber
    return loss, {'w1': w1, 'huber': huber}


def load_sfh_autoencoder_checkpoint(
    encoder: nn.Module,
    checkpoint_path: str | Path,
    decoder: nn.Module | None = None,
) -> None:
    """Load encoder and optional decoder weights from a Lightning checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 2.6
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    encoder_state = {
        key.removeprefix('encoder.'): value
        for key, value in state.items() if key.startswith('encoder.')
    }
    if not encoder_state:
        raise ValueError(f'No encoder weights found in {checkpoint_path}.')
    encoder.load_state_dict(encoder_state, strict=True)
    if decoder is not None:
        decoder_state = {
            key.removeprefix('decoder.'): value
            for key, value in state.items() if key.startswith('decoder.')
        }
        if not decoder_state:
            raise ValueError(f'No decoder weights found in {checkpoint_path}.')
        decoder.load_state_dict(decoder_state, strict=True)
