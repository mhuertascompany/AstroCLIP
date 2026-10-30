"""SFH-conditioned VIS flow using the published ``galaxy-counter`` decoder.

This is deliberately a thin cross-modal adapter around galaxy-counter commit
19679fa1c40f86cc4e2ce485660076e8add486d5.  The ResNet token encoder,
Diffusers cross-attention U-Net, rectified-flow path, Euler sampler, and
Sinkhorn geometry loss follow that implementation.  Only the paired
same-galaxy image encoder is replaced by a one-dimensional SFH token encoder.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


UPSTREAM_COMMIT = "19679fa1c40f86cc4e2ce485660076e8add486d5"
UPSTREAM_URL = "https://github.com/pablomerc/galaxy-counter.git"


def _upstream_components():
    try:
        from diffusers import UNet2DConditionModel
        import timm
        import geomloss
    except ImportError as exc:
        raise ImportError(
            "Install the pinned galaxy-counter source and its diffusers/geomloss "
            "dependencies before using the paper-faithful flow."
        ) from exc
    return UNet2DConditionModel, timm, geomloss


class ResNetTokenEncoder(nn.Module):
    """Exact ResNet token wrapper from galaxy-counter at ``UPSTREAM_COMMIT``.

    It is kept here because importing the upstream training module also imports
    W&B, UMAP and legacy Lightning, which initializes conflicting CUDA/XLA
    libraries on Rusty before the model can be constructed.
    """

    def __init__(self, in_channels=1, token_dim=16):
        super().__init__()
        _, timm, _ = _upstream_components()
        self.backbone = timm.create_model(
            "resnet18", pretrained=False, features_only=True,
            out_indices=(2, 3, 4),
        )
        if in_channels != 3:
            old = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding,
                bias=old.bias is not None,
            )
        self.proj = nn.Conv2d(512, token_dim, kernel_size=1)

    def forward(self, image):
        feature = self.proj(self.backbone(image)[-1])
        batch, channels, height, width = feature.shape
        return feature.view(batch, channels, height * width).permute(0, 2, 1)


class SFHTokenEncoder(nn.Module):
    """Map a normalized SFH curve to the paper's four 16-D context tokens."""

    def __init__(self, n_bins=250, token_dim=16, d_model=128, layers=4, heads=8):
        super().__init__()
        self.n_bins = int(n_bins)
        self.input = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.randn(1, self.n_bins, d_model) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, token_dim)

    def forward(self, sfh):
        if sfh.ndim != 2 or sfh.shape[1] != self.n_bins:
            raise ValueError(f"Expected SFHs shaped (B, {self.n_bins}); got {tuple(sfh.shape)}")
        hidden = self.transformer(self.input(sfh[..., None]) + self.position)
        # Preserve coarse temporal ordering while matching galaxy-counter's 2x2 token count.
        hidden = F.adaptive_avg_pool1d(hidden.transpose(1, 2), 4).transpose(1, 2)
        return self.projection(self.norm(hidden))


class GalaxyCounterSFHToVIS(nn.Module):
    """Generate a VIS anchor from paired SFH tokens and one donor VIS prior."""

    def __init__(
        self, n_sfh_bins=250, image_size=64, token_dim=16,
        model_channels=128, channel_mult=(1, 2, 4, 4), layers_per_block=2,
        attention_head_dim=8, sfh_d_model=128, sfh_layers=4, sfh_heads=8,
        geometric_weight=0.0, sinkhorn_blur=0.01,
    ):
        super().__init__()
        UNet2DConditionModel, _, geomloss = _upstream_components()
        self.image_size = int(image_size)
        self.geometric_weight = float(geometric_weight)
        self.sfh_common_encoder = SFHTokenEncoder(
            n_sfh_bins, token_dim, sfh_d_model, sfh_layers, sfh_heads,
        )
        # Separate encoders mirror the paper's physics and instrument encoders.
        self.image_common_encoder = ResNetTokenEncoder(1, token_dim)
        self.image_private_encoder = ResNetTokenEncoder(1, token_dim)
        widths = tuple(model_channels * value for value in channel_mult)
        attention_down = tuple("CrossAttnDownBlock2D" for _ in widths)
        attention_up = tuple("CrossAttnUpBlock2D" for _ in widths)
        self.velocity_model = UNet2DConditionModel(
            sample_size=self.image_size, in_channels=1, out_channels=1,
            layers_per_block=layers_per_block, block_out_channels=widths,
            down_block_types=attention_down, up_block_types=attention_up,
            cross_attention_dim=token_dim, attention_head_dim=attention_head_dim,
        )
        self.geom_loss_fn = None
        if self.geometric_weight > 0:
            self.geom_loss_fn = geomloss.SamplesLoss(
                loss="sinkhorn", p=2, blur=float(sinkhorn_blur),
                backend="tensorized", debias=True,
            )

    @staticmethod
    def _control(value, mode):
        if mode == "paired":
            return value
        if mode == "shuffled":
            return value.roll(1, 0)
        if mode == "zero":
            return torch.zeros_like(value)
        raise ValueError("mode must be paired, shuffled, or zero")

    def encode(self, image, sfh, donor_image, common_mode="paired", private_mode="paired"):
        sfh_common = self._control(self.sfh_common_encoder(sfh), common_mode)
        image_private = self._control(self.image_private_encoder(donor_image), private_mode)
        return torch.cat((sfh_common, image_private), dim=1)

    def forward(self, state, time, context):
        return self.velocity_model(
            state, time * 1000.0, encoder_hidden_states=context,
        ).sample

    def loss(self, batch, common_mode="paired", private_mode="paired",
             time=None, noise=None):
        image = batch["image"]
        context = self.encode(
            image, batch["sfh"], batch["donor_image"], common_mode, private_mode,
        )
        if time is None:
            time = torch.rand(len(image), device=image.device)
        if noise is None:
            noise = torch.randn_like(image)
        expanded = time[:, None, None, None]
        state = (1 - expanded) * noise + expanded * image
        target_velocity = image - noise
        prediction = self(state, time, context)
        generative = F.mse_loss(prediction.float(), target_velocity.float())
        geometric = generative.new_zeros(())
        if self.geom_loss_fn is not None:
            image_common = self.image_common_encoder(image).flatten(1).float().contiguous()
            sfh_common = self.sfh_common_encoder(batch["sfh"]).flatten(1).float().contiguous()
            geometric = self.geom_loss_fn(image_common, sfh_common)
        total = generative + self.geometric_weight * geometric
        return total, {"generative_loss": generative.detach(), "geometric_loss": geometric.detach()}

    @torch.inference_mode()
    def sample(self, batch, common_mode="paired", private_mode="paired",
               steps=250, noise=None):
        context = self.encode(
            batch["image"], batch["sfh"], batch["donor_image"],
            common_mode, private_mode,
        )
        state = torch.randn_like(batch["image"]) if noise is None else noise.clone()
        delta = 1.0 / int(steps)
        for index in range(int(steps)):
            time = torch.full(
                (len(state),), index * delta, device=state.device,
                dtype=torch.float32,
            )
            state.add_(delta * self(state, time, context))
        return state
