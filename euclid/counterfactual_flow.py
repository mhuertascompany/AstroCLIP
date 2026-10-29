"""Bidirectional counterfactual flow matching for Euclid VIS images and SFHs.

The architecture follows the conditioning pattern of ``galaxy-counter``:
target-object information comes from the paired modality, while a set of
different galaxies in the target modality supplies a conditional population
prior.  Neither target is passed to its same-modality prior encoder.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cosmosweb.sfh_transformer import FixedGridSFHTransformerEncoder

from .pixel_diffusion import ConditionalUNet
from .sfh_conditional_diffusion import ConditionalSFHTransformer


class ImageTokenEncoder(nn.Module):
    """Small image encoder used for common and prior representations."""

    def __init__(self, output_dim=64, base=16):
        super().__init__()
        if base < 8 or base % 8:
            raise ValueError("base must be a positive multiple of eight")
        widths = (base, 2 * base, 4 * base, 8 * base)
        blocks = []
        in_channels = 1
        for width in widths:
            blocks.extend([
                nn.Conv2d(in_channels, width, 3, stride=2, padding=1),
                nn.GroupNorm(8, width),
                nn.SiLU(),
            ])
            in_channels = width
        self.features = nn.Sequential(*blocks)
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(widths[-1], output_dim), nn.LayerNorm(output_dim),
        )

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Expected grayscale images shaped (batch, 1, H, W)")
        return self.output(self.features(image))


class DonorSetPool(nn.Module):
    """Attention-pool a fixed-size, order-invariant donor set."""

    def __init__(self, width=64, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, donors):
        if donors.ndim != 3 or donors.shape[1] < 1:
            raise ValueError("Expected donor tokens shaped (batch, donors, width)")
        normalized = self.norm(donors)
        query = self.query.expand(len(donors), -1, -1)
        return self.attention(
            query, normalized, normalized, need_weights=False,
        )[0][:, 0]


def _standardize_batch(values, epsilon=1e-4):
    return (values - values.mean(0, keepdim=True)) / (
        values.std(0, unbiased=False, keepdim=True) + epsilon
    )


class EuclidCounterfactualFlow(nn.Module):
    """Two rectified flows with common and matched-population conditioning."""

    def __init__(
        self,
        n_sfh_bins=250,
        common_dim=64,
        prior_dim=64,
        condition_dim=128,
        image_encoder_base=16,
        image_flow_base=16,
        sfh_d_model=96,
        sfh_layers=3,
        sfh_heads=4,
        common_loss_weight=0.05,
        sfh_loss_weight=1.0,
        common_dropout=0.1,
    ):
        super().__init__()
        if not 0 <= common_dropout < 1:
            raise ValueError("common_dropout must lie in [0, 1)")
        self.n_sfh_bins = int(n_sfh_bins)
        self.common_loss_weight = float(common_loss_weight)
        self.sfh_loss_weight = float(sfh_loss_weight)
        self.common_dropout = float(common_dropout)

        self.image_common_encoder = ImageTokenEncoder(
            common_dim, image_encoder_base,
        )
        self.sfh_common_encoder = FixedGridSFHTransformerEncoder(
            n_bins=n_sfh_bins, d_model=sfh_d_model, n_heads=sfh_heads,
            n_layers=sfh_layers, embed_dim=common_dim,
        )
        self.image_prior_encoder = ImageTokenEncoder(
            prior_dim, image_encoder_base,
        )
        self.sfh_prior_encoder = FixedGridSFHTransformerEncoder(
            n_bins=n_sfh_bins, d_model=sfh_d_model, n_heads=sfh_heads,
            n_layers=sfh_layers, embed_dim=prior_dim,
        )
        self.image_prior_pool = DonorSetPool(prior_dim, sfh_heads)
        self.sfh_prior_pool = DonorSetPool(prior_dim, sfh_heads)

        # q = standardized log mass, standardized z, one-hot activity class.
        covariate_dim = 5
        combined = common_dim + prior_dim + covariate_dim
        self.image_condition = nn.Sequential(
            nn.Linear(combined, condition_dim), nn.SiLU(),
            nn.Linear(condition_dim, condition_dim), nn.LayerNorm(condition_dim),
        )
        self.sfh_condition = nn.Sequential(
            nn.Linear(combined, condition_dim), nn.SiLU(),
            nn.Linear(condition_dim, condition_dim), nn.LayerNorm(condition_dim),
        )
        self.image_velocity = ConditionalUNet(condition_dim, image_flow_base)
        self.sfh_velocity = ConditionalSFHTransformer(
            n_bins=n_sfh_bins, condition_dim=condition_dim,
            d_model=sfh_d_model, n_heads=sfh_heads,
            n_layers=sfh_layers, dropout=0.1,
        )

    def encode_conditions(self, image, sfh, donor_images, donor_sfhs, covariates,
                          common_mode="paired", common_drop=None):
        """Return flow conditions and paired common representations.

        ``common_mode`` can remove or batch-shuffle target-specific information
        for the validation controls without changing the matched donor prior.
        """
        image_common = self.image_common_encoder(image)
        sfh_common = self.sfh_common_encoder(sfh)
        batch, donors = donor_images.shape[:2]
        image_prior = self.image_prior_encoder(
            donor_images.flatten(0, 1),
        ).unflatten(0, (batch, donors))
        sfh_prior = self.sfh_prior_encoder(
            donor_sfhs.flatten(0, 1),
        ).unflatten(0, (batch, donors))
        image_prior = self.image_prior_pool(image_prior)
        sfh_prior = self.sfh_prior_pool(sfh_prior)

        if common_mode == "shuffled":
            image_for_sfh = image_common.roll(1, 0)
            sfh_for_image = sfh_common.roll(1, 0)
        elif common_mode == "prior_only":
            image_for_sfh = torch.zeros_like(image_common)
            sfh_for_image = torch.zeros_like(sfh_common)
        elif common_mode == "paired":
            image_for_sfh, sfh_for_image = image_common, sfh_common
        else:
            raise ValueError("common_mode must be paired, shuffled, or prior_only")
        if common_drop is not None:
            image_for_sfh = torch.where(
                common_drop[:, None], torch.zeros_like(image_for_sfh),
                image_for_sfh,
            )
            sfh_for_image = torch.where(
                common_drop[:, None], torch.zeros_like(sfh_for_image),
                sfh_for_image,
            )

        image_condition = self.image_condition(torch.cat(
            [sfh_for_image, image_prior, covariates], dim=1,
        ))
        sfh_condition = self.sfh_condition(torch.cat(
            [image_for_sfh, sfh_prior, covariates], dim=1,
        ))
        return image_condition, sfh_condition, image_common, sfh_common

    @staticmethod
    def flow_pair(target, time, noise):
        """Straight probability path from Gaussian noise to target data."""
        expand = (len(target),) + (1,) * (target.ndim - 1)
        time_view = time.view(expand)
        state = (1 - time_view) * noise + time_view * target
        return state, target - noise

    def loss(self, batch, common_mode="paired", training=True, time=None,
             image_noise=None, sfh_noise=None):
        image = batch["image"]
        sfh = batch["sfh"]
        common_drop = None
        if training and self.common_dropout:
            common_drop = (
                torch.rand(len(image), device=image.device) < self.common_dropout
            )
        conditions = self.encode_conditions(
            image, sfh, batch["donor_images"], batch["donor_sfhs"],
            batch["covariates"], common_mode=common_mode,
            common_drop=common_drop,
        )
        image_condition, sfh_condition, image_common, sfh_common = conditions
        if time is None:
            time = torch.rand(len(image), device=image.device)
        if image_noise is None:
            image_noise = torch.randn_like(image)
        if sfh_noise is None:
            sfh_noise = torch.randn_like(sfh)
        image_state, image_velocity = self.flow_pair(
            image, time, image_noise,
        )
        sfh_state, sfh_velocity = self.flow_pair(
            sfh, time, sfh_noise,
        )
        time_embedding = time * 1000.0
        predicted_image = self.image_velocity(
            image_state, time_embedding, image_condition,
        )
        predicted_sfh = self.sfh_velocity(
            sfh_state, time_embedding, sfh_condition,
        )
        image_loss = F.mse_loss(predicted_image.float(), image_velocity.float())
        sfh_loss = F.mse_loss(predicted_sfh.float(), sfh_velocity.float())
        common_loss = F.mse_loss(
            _standardize_batch(image_common.float()),
            _standardize_batch(sfh_common.float()),
        )
        total = (
            image_loss + self.sfh_loss_weight * sfh_loss
            + self.common_loss_weight * common_loss
        )
        return total, {
            "image_flow_loss": image_loss.detach(),
            "sfh_flow_loss": sfh_loss.detach(),
            "common_loss": common_loss.detach(),
        }
