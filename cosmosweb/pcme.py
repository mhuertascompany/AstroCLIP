"""Probabilistic cross-modal matching losses used by PCME++.

The implementation follows the closed-form sampled distance (CSD) and
pseudo-positive objective from Chun et al., ICLR 2024.  Each modality is a
diagonal Gaussian with an L2-normalized mean and a learned log variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PCMEPPOutput:
    """Components returned by :func:`pcmepp_loss`."""

    loss: torch.Tensor
    match_loss: torch.Tensor
    pseudo_positive_loss: torch.Tensor
    vib_loss: torch.Tensor
    distance: torch.Tensor
    logits: torch.Tensor
    image_to_sfh_targets: torch.Tensor
    sfh_to_image_targets: torch.Tensor
    scale: torch.Tensor


def pairwise_closed_form_sampled_distance(
    mean_a: torch.Tensor,
    log_variance_a: torch.Tensor,
    mean_b: torch.Tensor,
    log_variance_b: torch.Tensor,
    *,
    log_variance_min: float = -12.0,
    log_variance_max: float = 8.0,
) -> torch.Tensor:
    """Expected pairwise squared distance between diagonal Gaussians.

    ``E[||z_a-z_b||^2]`` is the squared distance between the means plus the
    sum of both diagonal variances.  Computation is promoted to float32 so
    exponential variances remain stable under mixed-precision training.
    """
    if mean_a.ndim != 2 or mean_b.ndim != 2:
        raise ValueError('PCME++ means must be rank-2 tensors.')
    if mean_a.shape != log_variance_a.shape:
        raise ValueError('mean_a and log_variance_a must have the same shape.')
    if mean_b.shape != log_variance_b.shape:
        raise ValueError('mean_b and log_variance_b must have the same shape.')
    if mean_a.shape[1] != mean_b.shape[1]:
        raise ValueError('The two PCME++ distributions need equal dimensions.')
    if log_variance_min >= log_variance_max:
        raise ValueError('log_variance_min must be below log_variance_max.')

    mean_a_f = mean_a.float()
    mean_b_f = mean_b.float()
    variance_a = log_variance_a.float().clamp(
        log_variance_min, log_variance_max,
    ).exp()
    variance_b = log_variance_b.float().clamp(
        log_variance_min, log_variance_max,
    ).exp()
    squared_mean_distance = (
        mean_a_f.square().sum(dim=1, keepdim=True)
        + mean_b_f.square().sum(dim=1).unsqueeze(0)
        - 2.0 * mean_a_f @ mean_b_f.T
    ).clamp_min(0.0)
    uncertainty = (
        variance_a.sum(dim=1, keepdim=True)
        + variance_b.sum(dim=1).unsqueeze(0)
    )
    return squared_mean_distance + uncertainty


def gaussian_vib_loss(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    *,
    log_variance_min: float = -12.0,
    log_variance_max: float = 8.0,
) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) averaged over examples and dimensions."""
    log_variance_f = log_variance.float().clamp(
        log_variance_min, log_variance_max,
    )
    return -0.5 * (
        1.0 + log_variance_f - mean.float().square()
        - log_variance_f.exp()
    ).mean()


def pseudo_positive_targets(logits: torch.Tensor) -> torch.Tensor:
    """Promote candidates scoring at least as highly as the paired example.

    The comparison is detached: pseudo-label selection itself is not a route
    for gradients.  The diagonal is always included by construction.
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError('Pseudo-positive logits must be a square matrix.')
    detached = logits.detach()
    return (detached >= detached.diagonal().unsqueeze(1)).to(logits.dtype)


def pcmepp_loss(
    image_mean: torch.Tensor,
    image_log_variance: torch.Tensor,
    sfh_mean: torch.Tensor,
    sfh_log_variance: torch.Tensor,
    *,
    scale: torch.Tensor,
    bias: torch.Tensor,
    pseudo_positive_weight: float = 0.1,
    vib_weight: float = 1e-4,
) -> PCMEPPOutput:
    """Compute the symmetric PCME++ objective for one paired batch."""
    if image_mean.shape[0] != sfh_mean.shape[0]:
        raise ValueError('PCME++ requires equally sized paired modalities.')
    if pseudo_positive_weight < 0 or vib_weight < 0:
        raise ValueError('PCME++ loss weights cannot be negative.')

    distance = pairwise_closed_form_sampled_distance(
        image_mean, image_log_variance, sfh_mean, sfh_log_variance,
    )
    logits = bias.float() - scale.float() * distance
    exact_targets = torch.eye(
        logits.shape[0], device=logits.device, dtype=logits.dtype,
    )
    match_loss = F.binary_cross_entropy_with_logits(logits, exact_targets)

    image_to_sfh_targets = pseudo_positive_targets(logits)
    sfh_to_image_targets = pseudo_positive_targets(logits.T)
    image_to_sfh_pseudo_loss = F.binary_cross_entropy_with_logits(
        logits, image_to_sfh_targets,
    )
    sfh_to_image_pseudo_loss = F.binary_cross_entropy_with_logits(
        logits.T, sfh_to_image_targets,
    )
    pseudo_positive_loss = 0.5 * (
        image_to_sfh_pseudo_loss + sfh_to_image_pseudo_loss
    )
    vib_loss = (
        gaussian_vib_loss(image_mean, image_log_variance)
        + gaussian_vib_loss(sfh_mean, sfh_log_variance)
    )
    loss = (
        2.0 * match_loss
        + pseudo_positive_weight * (
            image_to_sfh_pseudo_loss + sfh_to_image_pseudo_loss
        )
        + vib_weight * vib_loss
    )
    return PCMEPPOutput(
        loss=loss,
        match_loss=match_loss,
        pseudo_positive_loss=pseudo_positive_loss,
        vib_loss=vib_loss,
        distance=distance,
        logits=logits,
        image_to_sfh_targets=image_to_sfh_targets,
        sfh_to_image_targets=sfh_to_image_targets,
        scale=scale,
    )
