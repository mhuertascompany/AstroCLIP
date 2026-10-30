"""Bidirectional SFH/VIS flow using the published ``galaxy-counter`` design.

This is deliberately a thin cross-modal adapter around galaxy-counter commit
19679fa1c40f86cc4e2ce485660076e8add486d5.  The ResNet token encoder,
Diffusers cross-attention U-Net, rectified-flow path, Euler sampler, and
Sinkhorn geometry loss follow that implementation.  The image-to-image
same-galaxy branch is generalized to a genuinely multimodal pair: each raw
modality is reconstructed from the other modality's common tokens and one
same-modality donor's private tokens.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .sfh_conditional_diffusion import sinusoidal_time_embedding


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


class SharedCommonTrunk(nn.Module):
    """A single common encoder applied after both modality-specific stems."""

    def __init__(self, input_dim=128, output_dim=16, layers=2, heads=8,
                 tokens=4, dropout=0.1):
        super().__init__()
        if input_dim % heads:
            raise ValueError("Shared common input_dim must be divisible by heads")
        self.tokens = int(tokens)
        self.position = nn.Parameter(
            torch.randn(1, self.tokens, input_dim) * 0.02,
        )
        block = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=heads, dim_feedforward=4 * input_dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            block, num_layers=layers, norm=nn.LayerNorm(input_dim),
        )
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] != self.tokens:
            raise ValueError(
                f"Expected common stem tokens shaped (B, {self.tokens}, D); "
                f"got {tuple(tokens.shape)}"
            )
        return self.output(self.transformer(tokens + self.position))


class TokenConditionedSFHVelocity(nn.Module):
    """Predict SFH rectified-flow velocity using token cross-attention."""

    def __init__(self, n_bins=250, context_dim=16, d_model=128, layers=4,
                 heads=8, dropout=0.1):
        super().__init__()
        self.n_bins = int(n_bins)
        self.input = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.randn(1, self.n_bins, d_model) * 0.02)
        self.time = nn.Sequential(
            nn.Linear(64, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.context = nn.Linear(context_dim, d_model)
        block = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            block, num_layers=layers, norm=nn.LayerNorm(d_model),
        )
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, state, time, context):
        if state.ndim != 2 or state.shape[1] != self.n_bins:
            raise ValueError(
                f"Expected SFH state shaped (B, {self.n_bins}); got {tuple(state.shape)}"
            )
        query = self.input(state[..., None]) + self.position
        query = query + self.time(sinusoidal_time_embedding(time))[:, None]
        hidden = self.transformer(query, self.context(context))
        return self.output(hidden).squeeze(-1)


class GalaxyCounterSFHToVIS(nn.Module):
    """Reconstruct both raw modalities from common and private information."""

    def __init__(
        self, n_sfh_bins=250, image_size=64, token_dim=16,
        model_channels=128, channel_mult=(1, 2, 4, 4), layers_per_block=2,
        attention_head_dim=8, sfh_d_model=128, sfh_layers=4, sfh_heads=8,
        common_stem_dim=128, common_trunk_layers=2, common_trunk_heads=8,
        geometric_weight=0.0, sinkhorn_blur=0.01,
        geometry_objective="sinkhorn", geometry_variance_weight=1.0,
        sfh_loss_weight=1.0,
    ):
        super().__init__()
        UNet2DConditionModel, _, geomloss = _upstream_components()
        self.image_size = int(image_size)
        self.geometric_weight = float(geometric_weight)
        self.geometry_objective = str(geometry_objective)
        self.geometry_variance_weight = float(geometry_variance_weight)
        self.sfh_loss_weight = float(sfh_loss_weight)
        if self.geometry_objective not in {"sinkhorn", "paired_cosine_variance"}:
            raise ValueError(
                "geometry_objective must be sinkhorn or paired_cosine_variance"
            )
        # The raw modalities need different stems, but the representation called
        # "common" is produced by one weight-shared trunk, matching the paper's
        # use of a single encoder_1 for either survey.
        self.sfh_common_stem = SFHTokenEncoder(
            n_sfh_bins, common_stem_dim, sfh_d_model, sfh_layers, sfh_heads,
        )
        self.image_common_stem = ResNetTokenEncoder(1, common_stem_dim)
        self.common_trunk = SharedCommonTrunk(
            input_dim=common_stem_dim, output_dim=token_dim,
            layers=common_trunk_layers, heads=common_trunk_heads,
        )
        # Private encoders remain modality-specific.
        self.image_private_encoder = ResNetTokenEncoder(1, token_dim)
        self.sfh_private_encoder = SFHTokenEncoder(
            n_sfh_bins, token_dim, sfh_d_model, sfh_layers, sfh_heads,
        )
        widths = tuple(model_channels * value for value in channel_mult)
        attention_down = tuple("CrossAttnDownBlock2D" for _ in widths)
        attention_up = tuple("CrossAttnUpBlock2D" for _ in widths)
        self.velocity_model = UNet2DConditionModel(
            sample_size=self.image_size, in_channels=1, out_channels=1,
            layers_per_block=layers_per_block, block_out_channels=widths,
            down_block_types=attention_down, up_block_types=attention_up,
            cross_attention_dim=token_dim, attention_head_dim=attention_head_dim,
        )
        self.sfh_velocity_model = TokenConditionedSFHVelocity(
            n_bins=n_sfh_bins, context_dim=token_dim, d_model=sfh_d_model,
            layers=sfh_layers, heads=sfh_heads,
        )
        self.geom_loss_fn = None
        if self.geometric_weight > 0 and self.geometry_objective == "sinkhorn":
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

    def target_embeddings(self, image, sfh):
        return {
            "image_common": self.common_trunk(self.image_common_stem(image)),
            "sfh_common": self.common_trunk(self.sfh_common_stem(sfh)),
            "image_private": self.image_private_encoder(image),
            "sfh_private": self.sfh_private_encoder(sfh),
        }

    def encode(self, image, sfh, donor_image, donor_sfh,
               common_mode="paired", private_mode="paired"):
        embeddings = self.target_embeddings(image, sfh)
        sfh_common = self._control(embeddings["sfh_common"], common_mode)
        image_common = self._control(embeddings["image_common"], common_mode)
        image_private = self._control(
            self.image_private_encoder(donor_image), private_mode,
        )
        sfh_private = self._control(
            self.sfh_private_encoder(donor_sfh), private_mode,
        )
        image_context = torch.cat((sfh_common, image_private), dim=1)
        sfh_context = torch.cat((image_common, sfh_private), dim=1)
        return image_context, sfh_context, embeddings

    def forward(self, state, time, context):
        return self.velocity_model(
            state, time * 1000.0, encoder_hidden_states=context,
        ).sample

    def loss(self, batch, anchor, common_mode="paired", private_mode="paired",
             time=None, image_noise=None, sfh_noise=None):
        if anchor not in {"image", "sfh"}:
            raise ValueError("anchor must be 'image' or 'sfh'")
        image = batch["image"]
        sfh = batch["sfh"]
        image_context, sfh_context, embeddings = self.encode(
            image, sfh, batch["donor_image"], batch["donor_sfh"],
            common_mode, private_mode,
        )
        if time is None:
            time = torch.rand(len(image), device=image.device)
        image_noise = torch.randn_like(image) if image_noise is None else image_noise
        sfh_noise = torch.randn_like(sfh) if sfh_noise is None else sfh_noise
        zero = image.new_zeros((), dtype=torch.float32)
        image_loss = zero
        sfh_loss = zero
        if anchor == "image":
            image_time = time[:, None, None, None]
            image_state = (1 - image_time) * image_noise + image_time * image
            image_prediction = self(image_state, time, image_context)
            image_loss = F.mse_loss(
                image_prediction.float(), (image - image_noise).float(),
            )
            generative = image_loss
        else:
            sfh_time = time[:, None]
            sfh_state = (1 - sfh_time) * sfh_noise + sfh_time * sfh
            sfh_prediction = self.sfh_velocity_model(sfh_state, time, sfh_context)
            sfh_loss = F.mse_loss(
                sfh_prediction.float(), (sfh - sfh_noise).float(),
            )
            generative = self.sfh_loss_weight * sfh_loss
        geometric = generative.new_zeros(())
        geometry_pair = generative.new_zeros(())
        geometry_variance = generative.new_zeros(())
        if self.geometric_weight > 0:
            image_common = embeddings["image_common"].flatten(1).float().contiguous()
            sfh_common = embeddings["sfh_common"].flatten(1).float().contiguous()
            if self.geometry_objective == "sinkhorn":
                geometric = self.geom_loss_fn(image_common, sfh_common)
            else:
                geometry_pair = (
                    1 - F.cosine_similarity(image_common, sfh_common, dim=1)
                ).mean()
                image_std = torch.sqrt(image_common.var(dim=0) + 1e-4)
                sfh_std = torch.sqrt(sfh_common.var(dim=0) + 1e-4)
                geometry_variance = 0.5 * (
                    F.relu(1 - image_std).mean()
                    + F.relu(1 - sfh_std).mean()
                )
                geometric = (
                    geometry_pair
                    + self.geometry_variance_weight * geometry_variance
                )
        total = generative + self.geometric_weight * geometric
        return total, {
            "image_flow_loss": image_loss.detach(),
            "sfh_flow_loss": sfh_loss.detach(),
            "geometric_loss": geometric.detach(),
            "geometry_pair_loss": geometry_pair.detach(),
            "geometry_variance_loss": geometry_variance.detach(),
        }

    @torch.inference_mode()
    def sample(self, batch, common_mode="paired", private_mode="paired",
               steps=250, image_noise=None, sfh_noise=None):
        image_context, sfh_context, _ = self.encode(
            batch["image"], batch["sfh"], batch["donor_image"],
            batch["donor_sfh"], common_mode, private_mode,
        )
        image = (
            torch.randn_like(batch["image"]) if image_noise is None
            else image_noise.clone()
        )
        sfh = (
            torch.randn_like(batch["sfh"]) if sfh_noise is None
            else sfh_noise.clone()
        )
        delta = 1.0 / int(steps)
        for index in range(int(steps)):
            time = torch.full(
                (len(image),), (index + 0.5) * delta, device=image.device,
                dtype=torch.float32,
            )
            image.add_(delta * self(image, time, image_context))
            sfh.add_(delta * self.sfh_velocity_model(sfh, time, sfh_context))
        return image, sfh
