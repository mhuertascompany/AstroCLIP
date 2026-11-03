"""Masked joint-embedding prediction for paired Euclid VIS images and SFHs.

The model follows the I-JEPA asymmetry: an online context encoder and predictor
learn to predict latent targets produced by a stop-gradient EMA encoder.  VIS
patches and contiguous SFH segments have separate input stems but share one
Transformer trunk and one latent coordinate system.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


TASKS = (
    "image_within",
    "sfh_within",
    "sfh_to_image",
    "image_to_sfh",
    "joint_masked",
)


@dataclass
class MaskSpec:
    name: str
    visible: torch.Tensor
    target: torch.Tensor


class JointTokenEncoder(nn.Module):
    """Modality-specific token stems followed by a shared Transformer."""

    def __init__(self, image_size=64, image_patch=8, n_sfh_bins=250,
                 sfh_patch=5, width=256, heads=8, layers=6, dropout=0.0):
        super().__init__()
        if image_size % image_patch:
            raise ValueError("image_size must be divisible by image_patch")
        if n_sfh_bins % sfh_patch:
            raise ValueError("n_sfh_bins must be divisible by sfh_patch")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.image_size = int(image_size)
        self.image_patch = int(image_patch)
        self.image_side = image_size // image_patch
        self.n_image_tokens = self.image_side**2
        self.n_sfh_bins = int(n_sfh_bins)
        self.sfh_patch = int(sfh_patch)
        self.n_sfh_tokens = n_sfh_bins // sfh_patch
        self.width = int(width)
        self.sequence_length = 1 + self.n_image_tokens + self.n_sfh_tokens

        self.image_stem = nn.Conv2d(
            1, width, kernel_size=image_patch, stride=image_patch,
        )
        self.sfh_stem = nn.Conv1d(
            1, width, kernel_size=sfh_patch, stride=sfh_patch,
        )
        self.pair_token = nn.Parameter(torch.zeros(1, 1, width))
        self.image_position = nn.Parameter(
            torch.zeros(1, self.n_image_tokens, width),
        )
        self.sfh_position = nn.Parameter(
            torch.zeros(1, self.n_sfh_tokens, width),
        )
        self.modality = nn.Parameter(torch.zeros(2, 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(width)
        self._initialize()

    @property
    def image_slice(self):
        return slice(1, 1 + self.n_image_tokens)

    @property
    def sfh_slice(self):
        return slice(1 + self.n_image_tokens, self.sequence_length)

    def _initialize(self):
        nn.init.trunc_normal_(self.pair_token, std=0.02)
        nn.init.trunc_normal_(self.image_position, std=0.02)
        nn.init.trunc_normal_(self.sfh_position, std=0.02)
        nn.init.trunc_normal_(self.modality, std=0.02)

    def tokenize(self, image, sfh):
        if image.ndim != 4 or image.shape[1:] != (
            1, self.image_size, self.image_size,
        ):
            raise ValueError(
                f"Expected images shaped (batch, 1, {self.image_size}, {self.image_size})"
            )
        if sfh.ndim != 2 or sfh.shape[1] != self.n_sfh_bins:
            raise ValueError(
                f"Expected SFHs shaped (batch, {self.n_sfh_bins})"
            )
        image_tokens = self.image_stem(image).flatten(2).transpose(1, 2)
        image_tokens = (
            image_tokens + self.image_position + self.modality[0:1]
        )
        sfh_tokens = self.sfh_stem(sfh[:, None]).transpose(1, 2)
        sfh_tokens = sfh_tokens + self.sfh_position + self.modality[1:2]
        pair = self.pair_token.expand(len(image), -1, -1)
        return torch.cat((pair, image_tokens, sfh_tokens), dim=1)

    def encode_tokens(self, tokens, visible=None):
        if visible is not None:
            if visible.ndim != 1 or len(visible) != self.sequence_length:
                raise ValueError("visible must be a sequence-length boolean mask")
            tokens = tokens[:, visible]
        return self.norm(self.transformer(tokens))

    def forward(self, image, sfh, visible=None):
        return self.encode_tokens(self.tokenize(image, sfh), visible)


class LatentPredictor(nn.Module):
    """Predict full-sequence target latents from encoded visible tokens."""

    def __init__(self, sequence_length, encoder_width=256, width=192,
                 heads=6, layers=3, dropout=0.0):
        super().__init__()
        if width % heads:
            raise ValueError("predictor width must be divisible by heads")
        self.sequence_length = int(sequence_length)
        self.context_projection = nn.Linear(encoder_width, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(
            torch.zeros(1, sequence_length, width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, encoder_width)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, encoded_context, visible):
        batch = len(encoded_context)
        projected = self.context_projection(encoded_context)
        tokens = self.mask_token.to(dtype=projected.dtype).expand(
            batch, self.sequence_length, -1,
        ).clone()
        tokens[:, visible] = projected
        tokens = tokens + self.position.to(dtype=tokens.dtype)
        return self.output(self.norm(self.transformer(tokens)))


class MultimodalJEPA(nn.Module):
    """Online encoder, EMA target encoder, and latent predictor."""

    def __init__(self, image_size=64, image_patch=8, n_sfh_bins=250,
                 sfh_patch=5, width=256, heads=8, layers=6,
                 predictor_width=192, predictor_heads=6,
                 predictor_layers=3, global_weight=1.0):
        super().__init__()
        self.global_weight = float(global_weight)
        self.online = JointTokenEncoder(
            image_size, image_patch, n_sfh_bins, sfh_patch,
            width, heads, layers,
        )
        self.target = copy.deepcopy(self.online).eval().requires_grad_(False)
        self.predictor = LatentPredictor(
            self.online.sequence_length, width, predictor_width,
            predictor_heads, predictor_layers,
        )

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum):
        momentum = float(momentum)
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must lie in [0, 1]")
        for online, target in zip(
            self.online.parameters(), self.target.parameters(), strict=True,
        ):
            target.data.mul_(momentum).add_(online.data, alpha=1 - momentum)

    def _image_blocks(self, device, generator=None, blocks=2):
        side = self.online.image_side
        mask = torch.zeros(side, side, dtype=torch.bool, device=device)
        for _ in range(blocks):
            maximum = min(4, side)
            minimum = min(2, maximum)
            height = int(torch.randint(
                minimum, maximum + 1, (), device=device, generator=generator,
            ))
            width = int(torch.randint(
                minimum, maximum + 1, (), device=device, generator=generator,
            ))
            top = int(torch.randint(0, side - height + 1, (), device=device, generator=generator))
            left = int(torch.randint(0, side - width + 1, (), device=device, generator=generator))
            mask[top:top + height, left:left + width] = True
        if mask.all():
            mask[-1, -1] = False
        return mask.flatten()

    def _sfh_windows(self, device, generator=None, windows=2):
        count = self.online.n_sfh_tokens
        mask = torch.zeros(count, dtype=torch.bool, device=device)
        minimum = max(1, round(0.1 * count))
        maximum = max(minimum, min(count - 1, round(0.2 * count)))
        for _ in range(windows):
            length = int(torch.randint(
                minimum, maximum + 1, (), device=device, generator=generator,
            ))
            start = int(torch.randint(0, count - length + 1, (), device=device, generator=generator))
            mask[start:start + length] = True
        if mask.all():
            mask[-1] = False
        return mask

    def make_mask(self, task, device, generator=None):
        if task not in TASKS:
            raise ValueError(f"Unknown JEPA task: {task}")
        length = self.online.sequence_length
        visible = torch.zeros(length, dtype=torch.bool, device=device)
        target = torch.zeros_like(visible)
        visible[0] = True
        image = torch.arange(
            self.online.image_slice.start, self.online.image_slice.stop,
            device=device,
        )
        sfh = torch.arange(
            self.online.sfh_slice.start, self.online.sfh_slice.stop,
            device=device,
        )
        if task in ("image_within", "joint_masked"):
            selected = self._image_blocks(device, generator)
            visible[image] = ~selected
            target[image] = selected
        elif task == "sfh_to_image":
            visible[sfh] = True
            target[image] = True
        elif task == "image_to_sfh":
            visible[image] = True
            target[sfh] = True
        if task in ("sfh_within", "joint_masked"):
            selected = self._sfh_windows(device, generator)
            visible[sfh] = ~selected
            target[sfh] = selected
        if not target.any() or not visible[1:].any():
            raise RuntimeError("Invalid JEPA mask without context or targets")
        return MaskSpec(task, visible, target)

    def predict(self, image, sfh, task, generator=None, shuffle_source=False):
        spec = self.make_mask(task, image.device, generator)
        context_image, context_sfh = image, sfh
        if shuffle_source:
            if task == "sfh_to_image":
                context_sfh = sfh.roll(1, 0)
            elif task == "image_to_sfh":
                context_image = image.roll(1, 0)
            else:
                raise ValueError("shuffle_source is defined only for cross-modal tasks")
        context = self.online(
            context_image, context_sfh, spec.visible,
        )
        prediction = self.predictor(context, spec.visible)
        with torch.no_grad():
            target = self.target(image, sfh)
        return prediction, target, spec

    def loss(self, image, sfh, task, generator=None, shuffle_source=False):
        prediction, target, spec = self.predict(
            image, sfh, task, generator, shuffle_source,
        )
        token_loss = F.smooth_l1_loss(
            prediction[:, spec.target].float(),
            target[:, spec.target].float(),
        )
        global_loss = F.smooth_l1_loss(
            prediction[:, 0].float(), target[:, 0].float(),
        )
        predicted_tokens = F.normalize(
            prediction[:, spec.target].float(), dim=-1,
        )
        target_tokens = F.normalize(
            target[:, spec.target].float(), dim=-1,
        )
        token_cosine = (predicted_tokens * target_tokens).sum(-1).mean()
        global_cosine = F.cosine_similarity(
            prediction[:, 0].float(), target[:, 0].float(), dim=-1,
        ).mean()
        total = token_loss + self.global_weight * global_loss
        return total, {
            "token_loss": token_loss.detach(),
            "global_loss": global_loss.detach(),
            "token_cosine": token_cosine.detach(),
            "global_cosine": global_cosine.detach(),
            "target_fraction": spec.target.float().mean().detach(),
        }

    @torch.inference_mode()
    def embed(self, image, sfh, mode="joint"):
        visible = torch.zeros(
            self.online.sequence_length, dtype=torch.bool,
            device=image.device,
        )
        visible[0] = True
        if mode in ("joint", "image"):
            visible[self.online.image_slice] = True
        if mode in ("joint", "sfh"):
            visible[self.online.sfh_slice] = True
        if mode not in ("joint", "image", "sfh"):
            raise ValueError("mode must be joint, image, or sfh")
        return self.online(image, sfh, visible)[:, 0]

    @torch.inference_mode()
    def cross_modal_errors(self, image, sfh, task, shuffle_source=False):
        if task not in ("sfh_to_image", "image_to_sfh"):
            raise ValueError("Cross-modal error maps require a cross-modal task")
        prediction, target, spec = self.predict(
            image, sfh, task, shuffle_source=shuffle_source,
        )
        cosine = F.cosine_similarity(
            prediction[:, spec.target].float(),
            target[:, spec.target].float(), dim=-1,
        )
        return 1 - cosine
