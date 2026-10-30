"""Paper-inspired bidirectional flow with one modality-private neighbour."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cosmosweb.sfh_transformer import FixedGridSFHTransformerEncoder

from .counterfactual_flow import ImageTokenEncoder
from .pixel_diffusion import ConditionalUNet
from .sfh_conditional_diffusion import ConditionalSFHTransformer


class EuclidSingleNeighborFlow(nn.Module):
    """Generate each raw modality from paired-common and one private neighbour."""

    def __init__(
        self,
        n_sfh_bins=250,
        common_dim=64,
        private_dim=64,
        condition_dim=128,
        image_encoder_base=16,
        image_flow_base=16,
        sfh_d_model=96,
        sfh_layers=3,
        sfh_heads=4,
        geometric_weight=0.1,
        sinkhorn_blur=0.01,
        sfh_loss_weight=1.0,
    ):
        super().__init__()
        try:
            import geomloss
        except ImportError as exc:
            raise ImportError(
                "The single-neighbour flow requires geomloss. Install it with "
                "`python -m pip install geomloss`."
            ) from exc
        self.n_sfh_bins = int(n_sfh_bins)
        self.geometric_weight = float(geometric_weight)
        self.sfh_loss_weight = float(sfh_loss_weight)
        self.image_common_encoder = ImageTokenEncoder(common_dim, image_encoder_base)
        self.sfh_common_encoder = FixedGridSFHTransformerEncoder(
            n_bins=n_sfh_bins, d_model=sfh_d_model, n_heads=sfh_heads,
            n_layers=sfh_layers, embed_dim=common_dim,
        )
        self.image_private_encoder = ImageTokenEncoder(private_dim, image_encoder_base)
        self.sfh_private_encoder = FixedGridSFHTransformerEncoder(
            n_bins=n_sfh_bins, d_model=sfh_d_model, n_heads=sfh_heads,
            n_layers=sfh_layers, embed_dim=private_dim,
        )
        combined = common_dim + private_dim
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
        self.geometric_loss = geomloss.SamplesLoss(
            loss="sinkhorn", p=2, blur=float(sinkhorn_blur),
            backend="tensorized", debias=True,
        )

    def target_embeddings(self, image, sfh):
        """Return the four target embeddings used by geometry diagnostics."""
        return {
            "image_common": self.image_common_encoder(image),
            "sfh_common": self.sfh_common_encoder(sfh),
            "image_private": self.image_private_encoder(image),
            "sfh_private": self.sfh_private_encoder(sfh),
        }

    @staticmethod
    def _control(values, mode):
        if mode == "paired":
            return values
        if mode == "shuffled":
            return values.roll(1, 0)
        if mode == "zero":
            return torch.zeros_like(values)
        raise ValueError("condition controls must be paired, shuffled, or zero")

    def encode_conditions(self, image, sfh, donor_image, donor_sfh,
                          common_mode="paired", private_mode="paired"):
        embeddings = self.target_embeddings(image, sfh)
        image_private = self.image_private_encoder(donor_image)
        sfh_private = self.sfh_private_encoder(donor_sfh)
        image_common = self._control(embeddings["sfh_common"], common_mode)
        sfh_common = self._control(embeddings["image_common"], common_mode)
        image_private = self._control(image_private, private_mode)
        sfh_private = self._control(sfh_private, private_mode)
        image_condition = self.image_condition(torch.cat(
            (image_common, image_private), dim=1,
        ))
        sfh_condition = self.sfh_condition(torch.cat(
            (sfh_common, sfh_private), dim=1,
        ))
        return image_condition, sfh_condition, embeddings

    @staticmethod
    def flow_pair(target, time, noise):
        shape = (len(target),) + (1,) * (target.ndim - 1)
        time = time.view(shape)
        return (1 - time) * noise + time * target, target - noise

    def loss(self, batch, common_mode="paired", private_mode="paired",
             time=None, image_noise=None, sfh_noise=None):
        image, sfh = batch["image"], batch["sfh"]
        image_condition, sfh_condition, embeddings = self.encode_conditions(
            image, sfh, batch["donor_image"], batch["donor_sfh"],
            common_mode=common_mode, private_mode=private_mode,
        )
        if time is None:
            time = torch.rand(len(image), device=image.device)
        image_noise = torch.randn_like(image) if image_noise is None else image_noise
        sfh_noise = torch.randn_like(sfh) if sfh_noise is None else sfh_noise
        image_state, image_target = self.flow_pair(image, time, image_noise)
        sfh_state, sfh_target = self.flow_pair(sfh, time, sfh_noise)
        time_embedding = time * 1000.0
        image_prediction = self.image_velocity(
            image_state, time_embedding, image_condition,
        )
        sfh_prediction = self.sfh_velocity(
            sfh_state, time_embedding, sfh_condition,
        )
        image_loss = F.mse_loss(image_prediction.float(), image_target.float())
        sfh_loss = F.mse_loss(sfh_prediction.float(), sfh_target.float())
        geometric = self.geometric_loss(
            embeddings["image_common"].float().contiguous(),
            embeddings["sfh_common"].float().contiguous(),
        )
        total = (
            image_loss + self.sfh_loss_weight * sfh_loss
            + self.geometric_weight * geometric
        )
        return total, {
            "image_flow_loss": image_loss.detach(),
            "sfh_flow_loss": sfh_loss.detach(),
            "geometric_loss": geometric.detach(),
        }

    @torch.inference_mode()
    def sample(self, batch, common_mode="paired", private_mode="paired",
               steps=32, image_noise=None, sfh_noise=None):
        image_condition, sfh_condition, _ = self.encode_conditions(
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
        delta = 1.0 / steps
        for step in range(steps):
            time = torch.full(
                (len(image),), (step + 0.5) * delta,
                device=image.device, dtype=torch.float32,
            )
            image.add_(delta * self.image_velocity(
                image, time * 1000.0, image_condition,
            ))
            sfh.add_(delta * self.sfh_velocity(
                sfh, time * 1000.0, sfh_condition,
            ))
        return image, sfh
