"""Deterministic cross-modal alignment losses.

These losses operate on L2-normalized embeddings.  They are kept separate
from the Lightning module so their targets and scaling can be tested without
constructing either survey encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CWCLResult:
    """Components of continuously weighted contrastive learning."""

    loss: torch.Tensor
    weighted_image_to_sfh_loss: torch.Tensor
    exact_sfh_to_image_loss: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor
    target_entropy: torch.Tensor
    effective_positives: torch.Tensor


@dataclass
class CyCLIPResult:
    """Components of the CyCLIP objective."""

    loss: torch.Tensor
    clip_loss: torch.Tensor
    inmodal_cyclic_loss: torch.Tensor
    crossmodal_cyclic_loss: torch.Tensor
    logits: torch.Tensor


def cwcl_targets(
    reference_embeddings: torch.Tensor,
    similarity_temperature: float = 0.1,
) -> torch.Tensor:
    """Return row-normalized CWCL targets from fixed reference geometry.

    Cosine similarity is converted to the continuous kernel
    ``exp((cosine - 1) / temperature)``.  It has unit self-similarity and
    remains strictly continuous without introducing a nearest-neighbour
    cutoff.  Lower temperatures sharpen the target distribution.
    """
    if reference_embeddings.ndim != 2:
        raise ValueError('reference_embeddings must have shape (batch, dim).')
    if similarity_temperature <= 0:
        raise ValueError('similarity_temperature must be positive.')

    reference = F.normalize(reference_embeddings.detach().float(), dim=-1)
    cosine = (reference @ reference.T).clamp(-1.0, 1.0)
    weights = torch.exp((cosine - 1.0) / float(similarity_temperature))
    weights.fill_diagonal_(1.0)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


def cwcl_loss(
    image_embeddings: torch.Tensor,
    sfh_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    logit_scale: torch.Tensor | float,
    similarity_temperature: float = 0.1,
    reverse_exact_weight: float = 1.0,
) -> CWCLResult:
    """CWCL transfer from fixed SFH geometry to the image representation.

    The image-to-SFH direction uses continuous SFH-AE targets.  The reverse
    direction is ordinary paired CLIP, matching the cross-modal transfer loss
    in the CWCL paper.  Setting ``reverse_exact_weight=0`` gives a purely
    continuously weighted objective.
    """
    if image_embeddings.shape != sfh_embeddings.shape:
        raise ValueError('image_embeddings and sfh_embeddings must match.')
    if image_embeddings.ndim != 2:
        raise ValueError('embeddings must have shape (batch, dim).')
    if reference_embeddings.size(0) != image_embeddings.size(0):
        raise ValueError('reference and aligned batches must have equal size.')
    if reverse_exact_weight < 0:
        raise ValueError('reverse_exact_weight cannot be negative.')

    image = F.normalize(image_embeddings, dim=-1)
    sfh = F.normalize(sfh_embeddings, dim=-1)
    logits = (image @ sfh.T) * logit_scale
    with torch.no_grad():
        targets = cwcl_targets(reference_embeddings, similarity_temperature).to(
            device=logits.device, dtype=logits.dtype,
        )
    weighted = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    labels = torch.arange(logits.size(0), device=logits.device)
    reverse_exact = F.cross_entropy(logits.T, labels)
    denominator = 1.0 + float(reverse_exact_weight)
    loss = (weighted + reverse_exact_weight * reverse_exact) / denominator
    entropy = -(targets * targets.clamp_min(1e-12).log()).sum(1).mean()
    effective_positives = entropy.exp()
    return CWCLResult(
        loss=loss,
        weighted_image_to_sfh_loss=weighted,
        exact_sfh_to_image_loss=reverse_exact,
        logits=logits,
        targets=targets,
        target_entropy=entropy,
        effective_positives=effective_positives,
    )


def cyclip_loss(
    image_embeddings: torch.Tensor,
    sfh_embeddings: torch.Tensor,
    logit_scale: torch.Tensor | float,
    clip_weight: float = 0.25,
    inmodal_weight: float = 1.0,
    crossmodal_weight: float = 0.25,
) -> CyCLIPResult:
    """CyCLIP loss with independently weighted paired and cyclic terms.

    The cyclic losses use the original implementation's batch-size scaling.
    This makes each term an average per anchor rather than an average over all
    ``batch_size**2`` matrix entries.
    """
    if image_embeddings.shape != sfh_embeddings.shape:
        raise ValueError('image_embeddings and sfh_embeddings must match.')
    if image_embeddings.ndim != 2:
        raise ValueError('embeddings must have shape (batch, dim).')
    if min(clip_weight, inmodal_weight, crossmodal_weight) < 0:
        raise ValueError('CyCLIP weights cannot be negative.')
    if clip_weight + inmodal_weight + crossmodal_weight == 0:
        raise ValueError('At least one CyCLIP weight must be positive.')

    image = F.normalize(image_embeddings, dim=-1)
    sfh = F.normalize(sfh_embeddings, dim=-1)
    cosine_cross = image @ sfh.T
    logits = cosine_cross * logit_scale
    labels = torch.arange(logits.size(0), device=logits.device)
    clip = 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )

    batch_size = image.size(0)
    inmodal = batch_size * F.mse_loss(image @ image.T, sfh @ sfh.T)
    crossmodal = batch_size * F.mse_loss(cosine_cross, cosine_cross.T)
    loss = (
        clip_weight * clip
        + inmodal_weight * inmodal
        + crossmodal_weight * crossmodal
    )
    return CyCLIPResult(
        loss=loss,
        clip_loss=clip,
        inmodal_cyclic_loss=inmodal,
        crossmodal_cyclic_loss=crossmodal,
        logits=logits,
    )


__all__ = [
    'CWCLResult',
    'CyCLIPResult',
    'cwcl_loss',
    'cwcl_targets',
    'cyclip_loss',
]
