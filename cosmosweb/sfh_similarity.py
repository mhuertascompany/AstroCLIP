"""Distances and soft contrastive targets for normalized SFH shapes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_sfh_weights(
    log_sfhs: torch.Tensor,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Convert log10 SFH weights to non-negative, unit-normalized weights."""
    if log_sfhs.ndim != 2 or log_sfhs.shape[1] < 2:
        raise ValueError('log_sfhs must have shape (batch, at least two time bins).')
    if epsilon <= 0:
        raise ValueError('epsilon must be positive.')

    # Keep the label geometry in float32 under mixed-precision training.
    values = log_sfhs.detach().to(dtype=torch.float32)
    weights = torch.clamp(torch.pow(10.0, values) - epsilon, min=0.0)
    totals = weights.sum(dim=1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError('Every SFH must have positive total weight.')
    return weights / totals


def pairwise_sfh_wasserstein1(
    log_sfhs: torch.Tensor,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Pairwise 1D Wasserstein-1 distances on a uniform [0, 1] time grid.

    The input SFHs are interpreted as discrete mass distributions over the
    common fractional-lookback-time grid. In one dimension W1 is the integral
    of the absolute difference between their cumulative distributions.
    """
    weights = linear_sfh_weights(log_sfhs, epsilon=epsilon)
    cdf = weights.cumsum(dim=1)
    delta_t = 1.0 / (weights.shape[1] - 1)
    return torch.cdist(cdf, cdf, p=1) * delta_t


def wasserstein_soft_targets(
    log_sfhs: torch.Tensor,
    soft_weight: float = 0.25,
    n_neighbors: int = 8,
    epsilon: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact-pair plus adaptive W1-neighbour target distributions.

    Each row assigns ``1 - soft_weight`` to the exact pair. The remaining
    probability is distributed across the nearest SFHs using a self-tuned
    Gaussian affinity whose bandwidth is the row's kth-neighbour distance.
    """
    if not 0.0 <= soft_weight <= 1.0:
        raise ValueError('soft_weight must lie in [0, 1].')
    if n_neighbors < 1:
        raise ValueError('n_neighbors must be positive.')

    distances = pairwise_sfh_wasserstein1(log_sfhs, epsilon=epsilon)
    batch_size = distances.shape[0]
    identity = torch.eye(
        batch_size, device=distances.device, dtype=distances.dtype,
    )
    if soft_weight == 0.0 or batch_size == 1:
        return identity, distances

    k = min(n_neighbors, batch_size - 1)
    without_self = distances.masked_fill(identity.bool(), float('inf'))
    nearest_distance, nearest_index = torch.topk(
        without_self, k=k, dim=1, largest=False, sorted=True,
    )
    bandwidth = nearest_distance[:, -1].clamp_min(torch.finfo(distances.dtype).eps)
    # Row-wise scaling guarantees that the kth neighbour has affinity exp(-1)
    # and prevents an unusually dense neighbouring row from underflowing every
    # candidate affinity to zero.
    denominator = bandwidth[:, None].square().clamp_min(
        torch.finfo(distances.dtype).eps,
    )
    affinity = torch.exp(-distances.square() / denominator)

    neighbor_mask = torch.zeros_like(affinity, dtype=torch.bool)
    neighbor_mask.scatter_(1, nearest_index, True)
    affinity = affinity.masked_fill(~neighbor_mask, 0.0)
    affinity.fill_diagonal_(0.0)
    neighbor_probability = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(affinity.dtype).eps,
    )
    targets = (1.0 - soft_weight) * identity + soft_weight * neighbor_probability
    return targets, distances


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy with a probability distribution for each target row."""
    if logits.shape != targets.shape:
        raise ValueError('logits and targets must have identical shapes.')
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def ae_false_negative_mask(
    latents: torch.Tensor,
    n_neighbors: int = 0,
    max_cosine_distance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return valid contrastive pairs after removing nearby AE latents.

    The diagonal (the exact image--SFH pair) is always retained. Off-diagonal
    pairs are removed when either object selects the other among its nearest
    neighbours in the fixed autoencoder space. When ``max_cosine_distance`` is
    supplied, selected neighbours are removed only if they also lie within
    that distance. With ``n_neighbors=0``, the distance threshold alone selects
    false negatives.

    Symmetrising the neighbour graph makes the same physical pair valid or
    invalid in both directions of the CLIP objective.
    """
    if latents.ndim != 2:
        raise ValueError('latents must have shape (batch, features).')
    if n_neighbors < 0:
        raise ValueError('n_neighbors cannot be negative.')
    if max_cosine_distance is not None and not (
        0.0 <= max_cosine_distance <= 2.0
    ):
        raise ValueError('max_cosine_distance must lie in [0, 2].')

    values = F.normalize(latents.detach().to(dtype=torch.float32), dim=1)
    distances = (1.0 - values @ values.T).clamp(min=0.0, max=2.0)
    batch_size = distances.shape[0]
    identity = torch.eye(batch_size, device=distances.device, dtype=torch.bool)
    excluded = torch.zeros_like(identity)

    if batch_size > 1 and n_neighbors > 0:
        k = min(n_neighbors, batch_size - 1)
        candidates = distances.masked_fill(identity, float('inf'))
        nearest_index = torch.topk(
            candidates, k=k, dim=1, largest=False, sorted=False,
        ).indices
        excluded.scatter_(1, nearest_index, True)
    elif batch_size > 1 and max_cosine_distance is not None:
        excluded = distances <= max_cosine_distance

    if max_cosine_distance is not None:
        excluded &= distances <= max_cosine_distance
    excluded &= ~identity
    excluded = excluded | excluded.T
    valid = ~excluded
    valid.fill_diagonal_(True)
    return valid, distances


def masked_cross_entropy(
    logits: torch.Tensor,
    valid_pairs: torch.Tensor,
) -> torch.Tensor:
    """Exact-pair cross entropy after excluding selected negatives."""
    if logits.ndim != 2 or logits.shape != valid_pairs.shape:
        raise ValueError('logits and valid_pairs must have identical 2D shapes.')
    if valid_pairs.dtype != torch.bool:
        raise ValueError('valid_pairs must be boolean.')
    n_queries = logits.shape[0]
    if logits.shape[1] < n_queries:
        raise ValueError('Each exact target must exist in the key dimension.')
    labels = torch.arange(n_queries, device=logits.device)
    if not torch.all(valid_pairs[labels, labels]):
        raise ValueError('The exact-pair diagonal cannot be masked.')
    masked_logits = logits.masked_fill(~valid_pairs, -torch.inf)
    return F.cross_entropy(masked_logits, labels)


def adjacency_consistency_loss(
    cross_modal_logits: torch.Tensor,
    reference_latents: torch.Tensor,
    logit_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Match cross-modal probabilities to fixed unimodal AE adjacency.

    This is the SFH-side analogue of the False Negative Suppression term in
    Sun et al. (CVPR 2023): all pairwise relations contribute continuously,
    avoiding a hard nearest-neighbour count or distance threshold.
    """
    if cross_modal_logits.ndim != 2:
        raise ValueError('cross_modal_logits must be a matrix.')
    if cross_modal_logits.shape[0] != cross_modal_logits.shape[1]:
        raise ValueError('Adjacency consistency requires square batch logits.')
    if reference_latents.ndim != 2:
        raise ValueError('reference_latents must have shape (batch, features).')
    if reference_latents.shape[0] != cross_modal_logits.shape[0]:
        raise ValueError('Logits and reference latents must share a batch size.')

    reference = F.normalize(
        reference_latents.detach().to(dtype=torch.float32), dim=1,
    )
    scale = torch.as_tensor(
        logit_scale, device=reference.device, dtype=reference.dtype,
    ).detach()
    reference_probability = F.softmax(
        scale * (reference @ reference.T), dim=1,
    )
    cross_modal_probability = F.softmax(
        cross_modal_logits.to(dtype=torch.float32), dim=1,
    )
    return F.l1_loss(cross_modal_probability, reference_probability)
