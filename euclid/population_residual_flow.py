"""Population and cross-modal residual rectified flows for VIS and SFHs.

The population vector field is conditioned only on log mass, redshift, and
three-class star-formation activity.  Once trained it is frozen.  A second
model learns a paired cross-modal correction through an explicit bottleneck:

    v(x, t | C, z) = v_population(x, t | C) + R(x, t, z) - R(x, t, 0).

The subtraction makes the null cross-modal condition exactly equal to the
population model and prevents the residual network becoming a second
unconditional generator.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cosmosweb.sfh_transformer import FixedGridSFHTransformerEncoder

from .counterfactual_flow import ImageTokenEncoder
from .pixel_diffusion import ConditionalUNet
from .sfh_conditional_diffusion import ConditionalSFHTransformer


def flow_pair(target, time, noise):
    shape = (len(target),) + (1,) * (target.ndim - 1)
    value = time.view(shape)
    return (1 - value) * noise + value * target, target - noise


def paired_alignment(image_embedding, sfh_embedding, variance_weight=1.0):
    """Paired cosine alignment with a VICReg-style anti-collapse floor."""
    pair = (1 - F.cosine_similarity(
        image_embedding.float(), sfh_embedding.float(), dim=1,
    )).mean()
    image_std = torch.sqrt(image_embedding.float().var(dim=0) + 1e-4)
    sfh_std = torch.sqrt(sfh_embedding.float().var(dim=0) + 1e-4)
    variance = 0.5 * (
        F.relu(1 - image_std).mean() + F.relu(1 - sfh_std).mean()
    )
    return pair + float(variance_weight) * variance, pair, variance


class PopulationVelocity(nn.Module):
    """Population vector fields conditioned only on physical covariates."""

    def __init__(self, n_sfh_bins=250, covariate_dim=5, image_base=16,
                 sfh_d_model=96, sfh_heads=4, sfh_layers=3):
        super().__init__()
        self.image = ConditionalUNet(covariate_dim, image_base)
        self.sfh = ConditionalSFHTransformer(
            n_bins=n_sfh_bins, condition_dim=covariate_dim,
            d_model=sfh_d_model, n_heads=sfh_heads,
            n_layers=sfh_layers, dropout=0.1,
        )

    def predict(self, anchor, state, time, covariates):
        timestep = time * 1000.0
        if anchor == "image":
            return self.image(state, timestep, covariates)
        if anchor == "sfh":
            return self.sfh(state, timestep, covariates)
        raise ValueError("anchor must be image or sfh")


class PopulationFlow(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.velocity = PopulationVelocity(**kwargs)

    @staticmethod
    def _covariates(batch, mode):
        covariates = batch["covariates"]
        if mode == "shuffled":
            return covariates.roll(1, 0)
        if mode == "counterfactual":
            # Choose a deliberately different activity class and, within that
            # class, the most distant mass/redshift condition in this batch.
            # A simple roll often preserves class and nearby physical values.
            labels = covariates[:, 2:].argmax(1)
            physical = covariates[:, :2]
            class_mismatch = labels[:, None] != labels[None, :]
            distance = torch.cdist(physical.float(), physical.float())
            distance = distance.masked_fill(~class_mismatch, -torch.inf)
            available = class_mismatch.any(1)
            result = covariates[distance.argmax(1)].clone()
            # Very small final validation batches can contain one class only.
            # Retain their physical values and rotate the one-hot class rather
            # than dropping the control or failing validation.
            if not torch.all(available):
                result[~available] = covariates[~available]
                result[~available, 2:] = 0
                replacement = (labels[~available] + 1) % 3
                result[~available, 2 + replacement] = 1
            return result
        if mode == "zero":
            return torch.zeros_like(covariates)
        if mode != "paired":
            raise ValueError(
                "mode must be paired, shuffled, counterfactual, or zero"
            )
        return covariates

    def loss(self, batch, anchor, time=None, noise=None,
             condition_dropout=0.0, mode="paired", reduction="mean"):
        target = batch[anchor]
        if time is None:
            time = torch.rand(len(target), device=target.device)
        if noise is None:
            noise = torch.randn_like(target)
        covariates = self._covariates(batch, mode)
        if condition_dropout:
            drop = torch.rand(len(target), device=target.device) < condition_dropout
            covariates = torch.where(drop[:, None], torch.zeros_like(covariates), covariates)
        state, velocity = flow_pair(target, time, noise)
        prediction = self.velocity.predict(anchor, state, time, covariates)
        per_element = F.mse_loss(
            prediction.float(), velocity.float(), reduction="none",
        )
        per_example = per_element.flatten(1).mean(1)
        if reduction == "none":
            return per_example
        if reduction != "mean":
            raise ValueError("reduction must be mean or none")
        return per_example.mean()

    @torch.inference_mode()
    def sample(self, batch, anchor, steps=64, noise=None, mode="paired"):
        target = batch[anchor]
        state = torch.randn_like(target) if noise is None else noise.clone()
        delta = 1.0 / int(steps)
        for index in range(int(steps)):
            time = torch.full(
                (len(state),), (index + 0.5) * delta,
                device=state.device, dtype=torch.float32,
            )
            state.add_(delta * self.velocity.predict(
                anchor, state, time, self._covariates(batch, mode),
            ).float())
        return state


class ResidualVelocity(nn.Module):
    """Frozen population fields plus paired cross-modal residual fields."""

    def __init__(self, population, n_sfh_bins=250, embedding_dim=64,
                 image_encoder_base=16, image_flow_base=16,
                 sfh_d_model=96, sfh_heads=4, sfh_layers=3,
                 alignment_weight=0.1, variance_weight=1.0):
        super().__init__()
        self.population = population.eval().requires_grad_(False)
        self.alignment_weight = float(alignment_weight)
        self.variance_weight = float(variance_weight)
        self.image_encoder = ImageTokenEncoder(embedding_dim, image_encoder_base)
        self.sfh_encoder = FixedGridSFHTransformerEncoder(
            n_bins=n_sfh_bins, d_model=sfh_d_model, n_heads=sfh_heads,
            n_layers=sfh_layers, embed_dim=embedding_dim,
        )
        self.image_residual = ConditionalUNet(embedding_dim, image_flow_base)
        self.sfh_residual = ConditionalSFHTransformer(
            n_bins=n_sfh_bins, condition_dim=embedding_dim,
            d_model=sfh_d_model, n_heads=sfh_heads,
            # R(z)-R(0) must use the same deterministic vector field twice.
            # Independent dropout masks would break the exact null identity.
            n_layers=sfh_layers, dropout=0.0,
        )

    def train(self, mode=True):
        super().train(mode)
        self.population.eval()
        return self

    def embeddings(self, image, sfh):
        return self.image_encoder(image), self.sfh_encoder(sfh)

    def _residual(self, anchor, state, time, embedding):
        timestep = time * 1000.0
        zero = torch.zeros_like(embedding)
        if anchor == "image":
            paired = self.image_residual(state, timestep, embedding)
            null = self.image_residual(state, timestep, zero)
        elif anchor == "sfh":
            paired = self.sfh_residual(state, timestep, embedding)
            null = self.sfh_residual(state, timestep, zero)
        else:
            raise ValueError("anchor must be image or sfh")
        return paired - null

    def predict(self, batch, anchor, state, time, mode="paired"):
        image_embedding, sfh_embedding = self.embeddings(
            batch["image"], batch["sfh"],
        )
        embedding = sfh_embedding if anchor == "image" else image_embedding
        if mode == "shuffled":
            embedding = embedding.roll(1, 0)
        elif mode == "zero":
            embedding = torch.zeros_like(embedding)
        elif mode != "paired":
            raise ValueError("mode must be paired, shuffled, or zero")
        with torch.no_grad():
            baseline = self.population.predict(
                anchor, state, time, batch["covariates"],
            )
        return baseline + self._residual(anchor, state, time, embedding)

    def loss(self, batch, anchor, time=None, noise=None, mode="paired"):
        target = batch[anchor]
        if time is None:
            time = torch.rand(len(target), device=target.device)
        if noise is None:
            noise = torch.randn_like(target)
        state, velocity = flow_pair(target, time, noise)
        prediction = self.predict(batch, anchor, state, time, mode)
        generative = F.mse_loss(prediction.float(), velocity.float())
        image_embedding, sfh_embedding = self.embeddings(
            batch["image"], batch["sfh"],
        )
        alignment, pair, variance = paired_alignment(
            image_embedding, sfh_embedding, self.variance_weight,
        )
        total = generative + self.alignment_weight * alignment
        return total, {
            "generative_loss": generative.detach(),
            "alignment_loss": alignment.detach(),
            "pair_loss": pair.detach(),
            "variance_loss": variance.detach(),
        }

    @torch.inference_mode()
    def sample(self, batch, anchor, mode="paired", steps=64, noise=None):
        target = batch[anchor]
        state = torch.randn_like(target) if noise is None else noise.clone()
        delta = 1.0 / int(steps)
        for index in range(int(steps)):
            time = torch.full(
                (len(state),), (index + 0.5) * delta,
                device=state.device, dtype=torch.float32,
            )
            state.add_(delta * self.predict(batch, anchor, state, time, mode).float())
        return state
