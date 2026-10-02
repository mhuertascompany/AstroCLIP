"""GalaxyTikTok transformer-tokenizer adapter for image--SFH alignment.

The pretrained tokenizer remains frozen.  Its spatial bottleneck tokens are
converted to one global alignment vector by a trainable attention pooler.  The
module deliberately exposes the same ``backbone`` / ``projection`` interface
as :class:`ZooBotImageEncoder`, so the existing alignment, momentum-encoder,
evaluation, and export code can be reused without special cases.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTokenPooler(nn.Module):
    """Pool a flattened spatial bottleneck into one trainable global vector."""

    def __init__(self, token_dim: int, n_tokens: int, output_dim: int,
                 hidden_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if min(token_dim, n_tokens, output_dim, hidden_dim, num_heads, num_layers) < 1:
            raise ValueError('Token-pooler dimensions must be positive.')
        if hidden_dim % num_heads:
            raise ValueError('Token-pooler hidden_dim must be divisible by num_heads.')
        self.token_dim = int(token_dim)
        self.n_tokens = int(n_tokens)
        self.input_norm = nn.LayerNorm(token_dim)
        self.token_projection = nn.Linear(token_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position = nn.Parameter(torch.zeros(1, n_tokens + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, flattened_tokens: torch.Tensor) -> torch.Tensor:
        if flattened_tokens.ndim != 2:
            raise ValueError('GalaxyTikTok latent must have shape (batch, features).')
        expected = self.n_tokens * self.token_dim
        if flattened_tokens.size(1) != expected:
            raise ValueError(
                f'Expected {expected} flattened token values, got '
                f'{flattened_tokens.size(1)}.'
            )
        tokens = flattened_tokens.reshape(
            flattened_tokens.size(0), self.n_tokens, self.token_dim,
        )
        tokens = self.token_projection(self.input_norm(tokens))
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        sequence = torch.cat((cls, tokens), dim=1) + self.position
        sequence = self.transformer(sequence)
        return self.output(sequence[:, 0])


class GalactikTokImageEncoder(nn.Module):
    """Frozen GalaxyTikTok image tokenizer plus trainable spatial pooling.

    The Euclid alignment dataset supplies one native-flux VIS channel using the
    same saved global percentile/asinh transform as tokenizer pretraining. If
    a replicated multi-channel tensor is supplied, only its first channel is
    used. Images are resized to the tokenizer's native resolution.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        embed_dim: int = 256,
        band: str = 'euclid-vis',
        pool_hidden_dim: int = 256,
        pool_heads: int = 4,
        pool_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from galactiktok import ImageTransformerTokenizer
            from galactiktok.models.image_transformer import Image
        except ImportError as exc:
            raise ImportError(
                'GalaxyTikTok is required for this encoder. Install it or add '
                'its src directory to PYTHONPATH.'
            ) from exc

        checkpoint = Path(checkpoint)
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f'GalaxyTikTok checkpoint directory not found: {checkpoint}'
            )
        tokenizer = ImageTransformerTokenizer.from_pretrained(str(checkpoint))
        if band not in tokenizer.canonical_bands:
            raise ValueError(
                f'Band {band!r} is absent from tokenizer bands '
                f'{tokenizer.canonical_bands!r}.'
            )
        # Alignment only calls ``encode``.  Discarding the reconstruction
        # decoder avoids storing it twice (query + momentum model) in every
        # contrastive checkpoint and substantially reduces GPU memory.
        for name in (
            'post_tok_proj', 'decoder_blocks', 'decoder_norm',
            'patch_debed_weight', 'patch_debed_bias', 'mask_token',
        ):
            if hasattr(tokenizer, name):
                delattr(tokenizer, name)
        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)
        tokenizer.eval()

        self.backbone = tokenizer
        self._image_modality_class = Image
        self.band = str(band)
        self.image_size = int(tokenizer.image_size)
        grid = self.image_size // int(tokenizer.patch_size)
        self.n_tokens = grid * grid
        self.token_dim = int(tokenizer.bottleneck_dim)
        self.source = str(checkpoint)
        self.projection = SpatialTokenPooler(
            token_dim=self.token_dim,
            n_tokens=self.n_tokens,
            output_dim=embed_dim,
            hidden_dim=pool_hidden_dim,
            num_heads=pool_heads,
            num_layers=pool_layers,
            dropout=dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode_backbone(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.size(1) < 1:
            raise ValueError('Expected image tensor with shape (batch, channels, H, W).')
        image = image[:, :1]
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode='bicubic',
                align_corners=False,
                antialias=True,
            )
        modality = self._image_modality_class(flux=image, bands=[self.band])
        tokens = self.backbone.encode(modality, mask_fraction=0.0).tokens
        return tokens.flatten(1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encode_backbone(image))


__all__ = ['GalactikTokImageEncoder', 'SpatialTokenPooler']
