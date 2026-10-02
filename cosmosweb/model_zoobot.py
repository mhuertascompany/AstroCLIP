"""Shared ZooBot--SFH alignment models.

``CosmosWebZooBotCLIP`` was introduced for COSMOS-Web images and CIGALE SFHs,
then reused as the survey-agnostic alignment engine for Euclid VIS. New Euclid
code should use :class:`euclid.model_zoobot.EuclidZooBotCLIP`; the historical
class remains here for COSMOS-Web scripts and checkpoint compatibility.

The primary model uses a frozen ZooBOT ConvNeXt backbone as the image encoder.

v2 adds a MoCo-style momentum encoder queue for both modalities.  The queue
extends the effective number of negatives per step from batch_size-1 to
batch_size + queue_size - 1, dramatically sharpening the contrastive signal.

Architecture
------------
  Main image encoder  : frozen ZooBOT backbone + trainable MLP projection
  Main SFH encoder    : trainable transformer/MLP (SFHEncoder)
  Momentum encoders   : EMA copies of both main encoders (no gradient)
  Queue               : two circular buffers (img, sfh) of size Q

Loss (per step)
---------------
  Query    : main-encoder embeddings (get gradients)
  Keys     : momentum-encoder embeddings for current batch + queue
  Logits   : (B, B+Q) for each direction
  Labels   : arange(B)  (positive is always the diagonal of the batch block)
  Loss     : mean of cross_entropy(logits_i2s, labels) and cross_entropy(logits_s2i, labels)

Validation loss uses only the current batch (no queue) so it stays comparable.
An opt-in SFH-aware mode replaces one-hot targets with a mixture of the exact
pair and Wasserstein-nearest SFHs. Its default weight is zero, preserving the
original objective and existing checkpoints.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sfh_encoder import SFHEncoder
from .sfh_autoencoder import FixedGridSFHDecoder, sfh_reconstruction_loss
from .sfh_transformer import (
    FixedGridSFHTransformerEncoder,
    SFHTransformerEncoder,
)
from .sfh_similarity import (
    adjacency_consistency_loss,
    ae_false_negative_mask,
    masked_cross_entropy,
    soft_cross_entropy,
    wasserstein_soft_targets,
)
from .pcme import pcmepp_loss
from .alignment_losses import cwcl_loss, cyclip_loss
from .zoobot_encoder import (
    FlexibleProjection,
    MultiFilterZooBotImageEncoder,
    ZooBotImageEncoder,
)
from .galactiktok_encoder import GalactikTokImageEncoder


class ResidualSFHProjection(nn.Module):
    """Near-identity nonlinear map from a fixed SFH latent to CLIP space."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None,
                 residual_scale: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 2 * embed_dim
        if embed_dim < 1 or hidden_dim < 1:
            raise ValueError('Projection dimensions must be positive.')
        if residual_scale <= 0:
            raise ValueError('residual_scale must be positive.')
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.residual_scale = float(residual_scale)

        # The branch starts at zero, making the complete projection exactly
        # the identity before CLIP training. Gradients reach fc2 immediately;
        # after its first update they also reach the earlier branch layers.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(self.activation(self.fc1(self.norm(latent))))
        return latent + self.residual_scale * residual


def make_sfh_projection(kind: str, embed_dim: int,
                        hidden_dim: int | None = None,
                        residual_scale: float = 0.1,
                        hidden_layers: int = 2) -> nn.Module:
    """Build the optional map from the SFH latent into the CLIP space.

    The linear option starts as the identity, so an autoencoder latent is
    unchanged before contrastive training.  Keeping this map outside the SFH
    encoder lets the latter remain a fixed, reconstructive representation.
    """
    if kind == 'identity':
        return nn.Identity()
    if kind == 'linear':
        projection = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(projection.weight)
        return projection
    if kind == 'residual_mlp':
        return ResidualSFHProjection(
            embed_dim, hidden_dim=hidden_dim,
            residual_scale=residual_scale,
        )
    if kind == 'mlp':
        return FlexibleProjection(
            embed_dim, embed_dim,
            hidden_dim=hidden_dim or 2 * embed_dim,
            hidden_layers=hidden_layers,
        )
    raise ValueError(
        f'Unknown sfh_projection_type={kind!r}; choose identity, linear, '
        'residual_mlp, or mlp.'
    )


class CosmosWebZooBotCLIP(L.LightningModule):

    def __init__(
        self,
        zoobot_ckpt:      str | None = None,
        zoobot_model_name: str | None = None,
        embed_dim:        int   = 256,
        sfh_input_dim:    int   = 50,
        temperature:      float = 0.07,
        queue_size:       int   = 4096,
        momentum:         float = 0.995,
        lr:               float = 1e-4,
        weight_decay:     float = 0.05,
        epochs:           int   = 50,
        warmup_epochs:    int   = 5,
        unfreeze_blocks:  int   = 0,
        backbone_lr_scale: float = 0.1,
        image_projection_type: str = 'legacy',
        image_projection_hidden_dim: int | None = None,
        image_projection_hidden_layers: int = 2,
        sfh_encoder_type: str = 'mlp',
        sfh_d_model:      int = 128,
        sfh_n_heads:      int = 4,
        sfh_n_layers:     int = 4,
        sfh_lr_scale:     float = 1.0,
        soft_positive_weight: float = 0.0,
        soft_positive_k:  int = 8,
        ae_false_negative_k: int = 0,
        ae_false_negative_max_distance: float | None = None,
        ae_adjacency_weight: float = 0.0,
        ae_adjacency_warmup_epochs: int = 3,
        ae_adjacency_temperature: float = 0.07,
        sfh_log_epsilon:  float = 1e-10,
        sfh_reconstruction_weight: float = 0.0,
        sfh_reconstruction_w1_weight: float = 0.5,
        sfh_decoder_layers: int = 2,
        sfh_projection_type: str = 'identity',
        sfh_projection_hidden_dim: int | None = None,
        sfh_projection_residual_scale: float = 0.1,
        sfh_projection_hidden_layers: int = 2,
        freeze_sfh_encoder: bool = False,
        freeze_sfh_decoder: bool = False,
        alignment_objective: str = 'clip',
        pcme_pseudo_positive_weight: float = 0.1,
        pcme_vib_weight: float = 1e-4,
        pcme_initial_scale: float = 5.0,
        pcme_initial_bias: float = 5.0,
        pcme_initial_uncertainty: float = 0.01,
        cwcl_similarity_temperature: float = 0.1,
        cwcl_reverse_exact_weight: float = 1.0,
        cyclip_clip_weight: float = 0.25,
        cyclip_inmodal_weight: float = 1.0,
        cyclip_crossmodal_weight: float = 0.25,
        image_encoder_type: str = 'zoobot',
        galactiktok_checkpoint: str | None = None,
        galactiktok_band: str = 'euclid-vis',
        token_pool_hidden_dim: int = 256,
        token_pool_heads: int = 4,
        token_pool_layers: int = 2,
    ) -> None:
        super().__init__()
        if alignment_objective not in {'clip', 'pcmepp', 'cwcl', 'cyclip'}:
            raise ValueError(
                'alignment_objective must be clip, pcmepp, cwcl, or cyclip.'
            )
        if image_encoder_type not in {'zoobot', 'galactiktok'}:
            raise ValueError('image_encoder_type must be zoobot or galactiktok.')
        if image_encoder_type == 'zoobot':
            if (zoobot_ckpt is None) == (zoobot_model_name is None):
                raise ValueError(
                    'ZooBot requires exactly one of zoobot_ckpt or '
                    'zoobot_model_name.'
                )
            if galactiktok_checkpoint is not None:
                raise ValueError('Do not combine ZooBot and GalaxyTikTok checkpoints.')
        else:
            if galactiktok_checkpoint is None:
                raise ValueError('GalaxyTikTok requires galactiktok_checkpoint.')
            if zoobot_ckpt is not None or zoobot_model_name is not None:
                raise ValueError('Do not combine GalaxyTikTok and ZooBot checkpoints.')
            if unfreeze_blocks:
                raise ValueError(
                    'GalaxyTikTok is frozen in this experiment; use unfreeze_blocks=0.'
                )
            if min(token_pool_hidden_dim, token_pool_heads, token_pool_layers) < 1:
                raise ValueError('Token-pooler dimensions must be positive.')
            if token_pool_hidden_dim % token_pool_heads:
                raise ValueError(
                    'token_pool_hidden_dim must be divisible by token_pool_heads.'
                )
        if pcme_pseudo_positive_weight < 0:
            raise ValueError('pcme_pseudo_positive_weight cannot be negative.')
        if pcme_vib_weight < 0:
            raise ValueError('pcme_vib_weight cannot be negative.')
        if pcme_initial_scale <= 0:
            raise ValueError('pcme_initial_scale must be positive.')
        if pcme_initial_uncertainty <= 0:
            raise ValueError('pcme_initial_uncertainty must be positive.')
        if cwcl_similarity_temperature <= 0:
            raise ValueError('cwcl_similarity_temperature must be positive.')
        if cwcl_reverse_exact_weight < 0:
            raise ValueError('cwcl_reverse_exact_weight cannot be negative.')
        if min(
            cyclip_clip_weight,
            cyclip_inmodal_weight,
            cyclip_crossmodal_weight,
        ) < 0:
            raise ValueError('CyCLIP weights cannot be negative.')
        if (
            cyclip_clip_weight
            + cyclip_inmodal_weight
            + cyclip_crossmodal_weight
        ) == 0:
            raise ValueError('At least one CyCLIP weight must be positive.')
        if not 0.0 <= soft_positive_weight <= 1.0:
            raise ValueError('soft_positive_weight must lie in [0, 1].')
        if soft_positive_k < 1:
            raise ValueError('soft_positive_k must be positive.')
        if ae_false_negative_k < 0:
            raise ValueError('ae_false_negative_k cannot be negative.')
        if ae_false_negative_max_distance is not None and not (
            0.0 <= ae_false_negative_max_distance <= 2.0
        ):
            raise ValueError(
                'ae_false_negative_max_distance must lie in [0, 2].'
            )
        if ae_adjacency_weight < 0:
            raise ValueError('ae_adjacency_weight cannot be negative.')
        if ae_adjacency_warmup_epochs < 0:
            raise ValueError('ae_adjacency_warmup_epochs cannot be negative.')
        if ae_adjacency_temperature <= 0:
            raise ValueError('ae_adjacency_temperature must be positive.')
        if sfh_log_epsilon <= 0:
            raise ValueError('sfh_log_epsilon must be positive.')
        if sfh_reconstruction_weight < 0:
            raise ValueError('sfh_reconstruction_weight cannot be negative.')
        if not 0 <= sfh_reconstruction_w1_weight <= 1:
            raise ValueError('sfh_reconstruction_w1_weight must lie in [0, 1].')
        if sfh_projection_type not in {'identity', 'linear', 'residual_mlp', 'mlp'}:
            raise ValueError(
                'sfh_projection_type must be identity, linear, residual_mlp, or mlp.'
            )
        if image_projection_type not in {'legacy', 'mlp'}:
            raise ValueError('image_projection_type must be legacy or mlp.')
        if (
            sfh_projection_hidden_dim is not None
            and sfh_projection_hidden_dim < 1
        ):
            raise ValueError('sfh_projection_hidden_dim must be positive.')
        if (
            image_projection_hidden_dim is not None
            and image_projection_hidden_dim < 1
        ):
            raise ValueError('image_projection_hidden_dim must be positive.')
        if sfh_projection_residual_scale <= 0:
            raise ValueError('sfh_projection_residual_scale must be positive.')
        if min(image_projection_hidden_layers, sfh_projection_hidden_layers) < 1:
            raise ValueError('Projection hidden-layer counts must be positive.')
        if freeze_sfh_decoder and sfh_reconstruction_weight <= 0:
            raise ValueError(
                'freeze_sfh_decoder requires a positive reconstruction weight '
                'so that a decoder is constructed.'
            )
        if soft_positive_weight > 0 and queue_size:
            raise ValueError(
                'SFH soft positives currently require queue_size=0 because '
                'the embedding queue does not store reference SFHs.'
            )
        ae_filtering = (
            ae_false_negative_k > 0
            or ae_false_negative_max_distance is not None
        )
        if ae_filtering and queue_size:
            raise ValueError(
                'AE false-negative filtering requires queue_size=0 because '
                'the embedding queue does not store reference AE latents.'
            )
        if ae_filtering and soft_positive_weight > 0:
            raise ValueError(
                'Choose either AE false-negative filtering or SFH soft positives.'
            )
        if ae_filtering and not freeze_sfh_encoder:
            raise ValueError(
                'AE false-negative filtering requires freeze_sfh_encoder=True '
                'so its reference geometry remains fixed.'
            )
        if ae_adjacency_weight > 0 and queue_size:
            raise ValueError(
                'AE adjacency regularization requires queue_size=0.'
            )
        if ae_adjacency_weight > 0 and soft_positive_weight > 0:
            raise ValueError(
                'Choose either AE adjacency regularization or SFH soft positives.'
            )
        if ae_adjacency_weight > 0 and ae_filtering:
            raise ValueError(
                'Choose either continuous AE adjacency regularization or hard '
                'AE false-negative filtering.'
            )
        if ae_adjacency_weight > 0 and not freeze_sfh_encoder:
            raise ValueError(
                'AE adjacency regularization requires freeze_sfh_encoder=True '
                'so its reference geometry remains fixed.'
            )
        if alignment_objective in {'pcmepp', 'cwcl', 'cyclip'}:
            incompatible = []
            if queue_size:
                incompatible.append('queue_size')
            if soft_positive_weight > 0:
                incompatible.append('soft_positive_weight')
            if ae_filtering:
                incompatible.append('AE false-negative filtering')
            if ae_adjacency_weight > 0:
                incompatible.append('AE adjacency regularization')
            if incompatible:
                raise ValueError(
                    f'{alignment_objective} uses its own in-batch objective; '
                    'disable ' + ', '.join(incompatible) + '.'
                )
        if alignment_objective == 'cwcl' and not freeze_sfh_encoder:
            raise ValueError(
                'CWCL requires freeze_sfh_encoder=True so the SFH-AE target '
                'geometry remains fixed.'
            )
        self.save_hyperparameters()

        # ── main encoders (receive gradients) ────────────────────────────────
        if image_encoder_type == 'zoobot':
            self.image_encoder = ZooBotImageEncoder(
                ckpt_path=zoobot_ckpt,
                model_name=zoobot_model_name,
                embed_dim=embed_dim,
                unfreeze_blocks=unfreeze_blocks,
                projection_type=image_projection_type,
                projection_hidden_dim=image_projection_hidden_dim,
                projection_hidden_layers=image_projection_hidden_layers,
            )
        else:
            self.image_encoder = GalactikTokImageEncoder(
                checkpoint=galactiktok_checkpoint,
                embed_dim=embed_dim,
                band=galactiktok_band,
                pool_hidden_dim=token_pool_hidden_dim,
                pool_heads=token_pool_heads,
                pool_layers=token_pool_layers,
            )
        if sfh_encoder_type == 'mlp':
            self.sfh_encoder = SFHEncoder(
                input_dim=sfh_input_dim,
                embed_dim=embed_dim,
            )
        elif sfh_encoder_type == 'transformer':
            self.sfh_encoder = FixedGridSFHTransformerEncoder(
                n_bins=sfh_input_dim,
                d_model=sfh_d_model,
                n_heads=sfh_n_heads,
                n_layers=sfh_n_layers,
                embed_dim=embed_dim,
            )
        else:
            raise ValueError(
                f'Unknown sfh_encoder_type={sfh_encoder_type!r}; '
                "choose 'mlp' or 'transformer'."
            )
        self.sfh_projection = make_sfh_projection(
            sfh_projection_type, embed_dim,
            hidden_dim=sfh_projection_hidden_dim,
            residual_scale=sfh_projection_residual_scale,
            hidden_layers=sfh_projection_hidden_layers,
        )
        if sfh_reconstruction_weight > 0:
            if sfh_encoder_type != 'transformer':
                raise ValueError('SFH reconstruction requires the transformer encoder.')
            self.sfh_decoder = FixedGridSFHDecoder(
                n_bins=sfh_input_dim,
                embed_dim=embed_dim,
                d_model=sfh_d_model,
                n_heads=sfh_n_heads,
                n_layers=sfh_decoder_layers,
            )
        else:
            self.sfh_decoder = None

        if freeze_sfh_encoder:
            for parameter in self.sfh_encoder.parameters():
                parameter.requires_grad_(False)
            self.sfh_encoder.eval()
        if freeze_sfh_decoder:
            for parameter in self.sfh_decoder.parameters():
                parameter.requires_grad_(False)
            self.sfh_decoder.eval()

        # ── momentum encoders (EMA, no gradient) ─────────────────────────────
        self.image_encoder_m = copy.deepcopy(self.image_encoder)
        self.sfh_encoder_m   = copy.deepcopy(self.sfh_encoder)
        self.sfh_projection_m = copy.deepcopy(self.sfh_projection)
        for p in self.image_encoder_m.parameters():
            p.requires_grad_(False)
        for p in self.sfh_encoder_m.parameters():
            p.requires_grad_(False)
        for p in self.sfh_projection_m.parameters():
            p.requires_grad_(False)
        self.image_encoder_m.eval()
        self.sfh_encoder_m.eval()
        self.sfh_projection_m.eval()

        # ── learnable temperature ─────────────────────────────────────────────
        self.log_temp = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))
        if alignment_objective == 'pcmepp':
            # The existing projections define the probabilistic means.  A
            # separate head estimates diagonal log variance for each modality.
            self.image_log_variance = nn.Linear(embed_dim, embed_dim)
            self.sfh_log_variance = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(self.image_log_variance.weight)
            nn.init.zeros_(self.sfh_log_variance.weight)
            initial_log_variance = float(np.log(
                pcme_initial_uncertainty / embed_dim,
            ))
            nn.init.constant_(self.image_log_variance.bias, initial_log_variance)
            nn.init.constant_(self.sfh_log_variance.bias, initial_log_variance)
            self.pcme_scale = nn.Parameter(
                torch.tensor(float(pcme_initial_scale)),
            )
            self.pcme_bias = nn.Parameter(torch.tensor(float(pcme_initial_bias)))
            self.log_temp.requires_grad_(False)
        else:
            self.image_log_variance = None
            self.sfh_log_variance = None
            self.pcme_scale = None
            self.pcme_bias = None

        # ── circular queues (registered as buffers → saved in checkpoint) ────
        Q = queue_size
        D = embed_dim
        if Q < 0:
            raise ValueError('queue_size cannot be negative.')
        queue_img = F.normalize(torch.randn(D, Q), dim=0) if Q else torch.empty(D, 0)
        queue_sfh = F.normalize(torch.randn(D, Q), dim=0) if Q else torch.empty(D, 0)
        self.register_buffer('queue_img', queue_img)
        self.register_buffer('queue_sfh', queue_sfh)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    def train(self, mode: bool = True):
        """Keep momentum targets deterministic while training query encoders."""
        super().train(mode)
        self.image_encoder_m.eval()
        self.sfh_encoder_m.eval()
        self.sfh_projection_m.eval()
        if self.hparams.freeze_sfh_encoder:
            self.sfh_encoder.eval()
        if self.sfh_decoder is not None and self.hparams.freeze_sfh_decoder:
            self.sfh_decoder.eval()
        return self

    # ── momentum update ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _momentum_update(self) -> None:
        m = self.hparams.momentum
        for p, pm in zip(self.image_encoder.parameters(),
                         self.image_encoder_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)
        for p, pm in zip(self.sfh_encoder.parameters(),
                         self.sfh_encoder_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)
        for p, pm in zip(self.sfh_projection.parameters(),
                         self.sfh_projection_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)

    # ── queue management ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self,
        img_keys: torch.Tensor,   # (B, D)  from momentum image encoder
        sfh_keys: torch.Tensor,   # (B, D)  from momentum SFH encoder
    ) -> None:
        B   = img_keys.shape[0]
        Q   = self.hparams.queue_size
        ptr = int(self.queue_ptr)
        # queue_size must be divisible by batch_size; checked in training_step
        self.queue_img[:, ptr:ptr + B] = img_keys.T
        self.queue_sfh[:, ptr:ptr + B] = sfh_keys.T
        self.queue_ptr[0] = (ptr + B) % Q

    # ── encoders (public API, used by evaluate / umap scripts) ───────────────

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.project_image(self.encode_image_latent(image))

    def encode_image_latent(self, image: torch.Tensor) -> torch.Tensor:
        """Return frozen image features before the trainable alignment adapter."""
        return self.image_encoder.encode_backbone(image)

    def project_image(self, latent: torch.Tensor) -> torch.Tensor:
        """Map frozen ZooBot features into the aligned CLIP space."""
        return self.image_encoder.projection(latent)

    def encode_sfh_latent(self, sfh: torch.Tensor) -> torch.Tensor:
        return self.sfh_encoder(sfh)

    def project_sfh(self, latent: torch.Tensor) -> torch.Tensor:
        return self.sfh_projection(latent)

    def encode_sfh(self, sfh: torch.Tensor) -> torch.Tensor:
        return self.project_sfh(self.encode_sfh_latent(sfh))

    def encode_image_distribution(
        self, image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized PCME++ image mean and diagonal log variance."""
        if self.hparams.alignment_objective != 'pcmepp':
            raise RuntimeError('Image distributions require alignment_objective=pcmepp.')
        projected = self.encode_image(image)
        return F.normalize(projected, dim=-1), self.image_log_variance(projected)

    def encode_sfh_distribution(
        self, sfh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized PCME++ SFH mean and diagonal log variance."""
        if self.hparams.alignment_objective != 'pcmepp':
            raise RuntimeError('SFH distributions require alignment_objective=pcmepp.')
        projected = self.encode_sfh(sfh)
        return F.normalize(projected, dim=-1), self.sfh_log_variance(projected)

    def _pcme_scale(self) -> torch.Tensor:
        return self.pcme_scale

    def _pcme_step(self, images: torch.Tensor, sfhs: torch.Tensor):
        image_mean, image_log_variance = self.encode_image_distribution(images)
        sfh_latent = self.encode_sfh_latent(sfhs)
        sfh_projected = self.project_sfh(sfh_latent)
        sfh_mean = F.normalize(sfh_projected, dim=-1)
        sfh_log_variance = self.sfh_log_variance(sfh_projected)
        result = pcmepp_loss(
            image_mean,
            image_log_variance,
            sfh_mean,
            sfh_log_variance,
            scale=self._pcme_scale(),
            bias=self.pcme_bias,
            pseudo_positive_weight=self.hparams.pcme_pseudo_positive_weight,
            vib_weight=self.hparams.pcme_vib_weight,
        )
        return result, sfh_latent, image_mean, sfh_mean, image_log_variance, sfh_log_variance

    def _alternative_alignment_step(
        self,
        images: torch.Tensor,
        sfhs: torch.Tensor,
        sfh_reference: torch.Tensor,
    ):
        """Run CWCL or CyCLIP and return common deterministic embeddings."""
        image_embedding = F.normalize(self.encode_image(images), dim=-1)
        sfh_latent = self.encode_sfh_latent(sfhs)
        sfh_embedding = F.normalize(self.project_sfh(sfh_latent), dim=-1)
        logit_scale = self.log_temp.exp().clamp(min=1.0, max=100.0)
        if self.hparams.alignment_objective == 'cwcl':
            with torch.no_grad():
                reference_latent = self.encode_sfh_latent(sfh_reference)
            result = cwcl_loss(
                image_embedding,
                sfh_embedding,
                reference_latent,
                logit_scale=logit_scale,
                similarity_temperature=self.hparams.cwcl_similarity_temperature,
                reverse_exact_weight=self.hparams.cwcl_reverse_exact_weight,
            )
        elif self.hparams.alignment_objective == 'cyclip':
            result = cyclip_loss(
                image_embedding,
                sfh_embedding,
                logit_scale=logit_scale,
                clip_weight=self.hparams.cyclip_clip_weight,
                inmodal_weight=self.hparams.cyclip_inmodal_weight,
                crossmodal_weight=self.hparams.cyclip_crossmodal_weight,
            )
        else:
            raise RuntimeError('Alternative step requires CWCL or CyCLIP.')
        return result, sfh_latent, image_embedding, sfh_embedding, logit_scale

    def forward(
        self, image: torch.Tensor, sfh: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(image), self.encode_sfh(sfh)

    # ── loss helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _infoNCE(
        queries:  torch.Tensor,   # (B, D)  L2-normalised
        all_keys: torch.Tensor,   # (B+Q, D) L2-normalised
        T:        torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy over logits (B, B+Q); positives at indices 0..B-1."""
        logits = queries @ all_keys.T * T          # (B, B+Q)  T = logit_scale = 1/temp
        labels = torch.arange(queries.size(0),
                              device=queries.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

    # ── training step ─────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, sfhs = batch['image'], batch['sfh']
        sfh_reference = batch.get('sfh_reference', sfhs)
        B = images.size(0)
        Q = self.hparams.queue_size

        if self.hparams.alignment_objective == 'pcmepp':
            result, sfh_latent, image_mean, sfh_mean, image_logvar, sfh_logvar = (
                self._pcme_step(images, sfhs)
            )
            loss = result.loss
            if self.sfh_decoder is not None:
                reconstruction = self.sfh_decoder(sfh_latent)
                reconstruction_loss, reconstruction_parts = sfh_reconstruction_loss(
                    reconstruction,
                    sfh_reference,
                    batch.get('sfh_p16'),
                    batch.get('sfh_p84'),
                    epsilon=self.hparams.sfh_log_epsilon,
                    w1_weight=self.hparams.sfh_reconstruction_w1_weight,
                )
                loss = loss + self.hparams.sfh_reconstruction_weight * reconstruction_loss
                self.log('train_reconstruction_loss', reconstruction_loss, sync_dist=True)
                self.log('train_reconstruction_w1', reconstruction_parts['w1'], sync_dist=True)
            labels = torch.arange(B, device=images.device)
            with torch.no_grad():
                rank1 = (
                    (result.logits.argmax(1) == labels).float().mean()
                    + (result.logits.T.argmax(1) == labels).float().mean()
                ) / 2.0
                pseudo_fraction = 0.5 * (
                    result.image_to_sfh_targets.mean()
                    + result.sfh_to_image_targets.mean()
                )
            self.log('train_loss', loss, prog_bar=True, sync_dist=True)
            self.log('train_pcme_match_loss', result.match_loss, sync_dist=True)
            self.log('train_pcme_pseudo_positive_loss', result.pseudo_positive_loss, sync_dist=True)
            self.log('train_pcme_vib_loss', result.vib_loss, sync_dist=True)
            self.log('train_pcme_scale', result.scale, sync_dist=True)
            self.log('train_pcme_bias', self.pcme_bias, sync_dist=True)
            self.log('train_pcme_pseudo_positive_fraction', pseudo_fraction, sync_dist=True)
            self.log(
                'train_image_uncertainty',
                image_logvar.float().clamp(-12, 8).exp().sum(1).mean(),
                sync_dist=True,
            )
            self.log(
                'train_sfh_uncertainty',
                sfh_logvar.float().clamp(-12, 8).exp().sum(1).mean(),
                sync_dist=True,
            )
            self.log('train_rank1_batch', rank1, prog_bar=True, sync_dist=True)
            return loss

        if self.hparams.alignment_objective in {'cwcl', 'cyclip'}:
            result, sfh_latent, img_q, sfh_q, T = self._alternative_alignment_step(
                images, sfhs, sfh_reference,
            )
            alignment_loss = result.loss
            loss = alignment_loss
            if self.sfh_decoder is not None:
                reconstruction = self.sfh_decoder(sfh_latent)
                reconstruction_loss, reconstruction_parts = sfh_reconstruction_loss(
                    reconstruction,
                    sfh_reference,
                    batch.get('sfh_p16'),
                    batch.get('sfh_p84'),
                    epsilon=self.hparams.sfh_log_epsilon,
                    w1_weight=self.hparams.sfh_reconstruction_w1_weight,
                )
                loss = loss + self.hparams.sfh_reconstruction_weight * reconstruction_loss
                self.log('train_reconstruction_loss', reconstruction_loss, sync_dist=True)
                self.log('train_reconstruction_w1', reconstruction_parts['w1'], sync_dist=True)
            labels = torch.arange(B, device=images.device)
            with torch.no_grad():
                rank1 = 0.5 * (
                    (result.logits.argmax(1) == labels).float().mean()
                    + (result.logits.T.argmax(1) == labels).float().mean()
                )
            self.log('train_loss', loss, prog_bar=True, sync_dist=True)
            self.log('train_alignment_loss', alignment_loss, sync_dist=True)
            self.log('train_rank1_batch', rank1, prog_bar=True, sync_dist=True)
            self.log('train_logit_scale', T, sync_dist=True)
            if self.hparams.alignment_objective == 'cwcl':
                self.log(
                    'train_cwcl_weighted_i2s_loss',
                    result.weighted_image_to_sfh_loss, sync_dist=True,
                )
                self.log(
                    'train_cwcl_exact_s2i_loss',
                    result.exact_sfh_to_image_loss, sync_dist=True,
                )
                self.log(
                    'train_cwcl_target_entropy', result.target_entropy,
                    sync_dist=True,
                )
                self.log(
                    'train_cwcl_effective_positives', result.effective_positives,
                    sync_dist=True,
                )
            else:
                self.log('train_cyclip_clip_loss', result.clip_loss, sync_dist=True)
                self.log(
                    'train_cyclip_inmodal_loss', result.inmodal_cyclic_loss,
                    sync_dist=True,
                )
                self.log(
                    'train_cyclip_crossmodal_loss',
                    result.crossmodal_cyclic_loss, sync_dist=True,
                )
            return loss

        if Q and B > Q:
            raise ValueError(
                f'batch_size ({B}) > queue_size ({Q}). '
                'Reduce batch_size or increase queue_size.')

        T = self.log_temp.exp().clamp(min=1.0, max=100.0)

        # ── query embeddings (main encoders, receive gradients) ───────────────
        img_q = F.normalize(self.encode_image(images), dim=-1)   # (B, D)
        sfh_latent = self.encode_sfh_latent(sfhs)
        sfh_q = F.normalize(self.project_sfh(sfh_latent), dim=-1)  # (B, D)

        # ── key embeddings (momentum encoders, no gradient) ───────────────────
        with torch.no_grad():
            self._momentum_update()
            img_k = F.normalize(self.image_encoder_m(images), dim=-1)  # (B, D)
            sfh_k = F.normalize(
                self.sfh_projection_m(self.sfh_encoder_m(sfhs)), dim=-1,
            )

        # ── all keys = current batch (momentum) + queue ───────────────────────
        # Shape: (B + Q, D)
        all_sfh = torch.cat([sfh_k, self.queue_sfh.T.clone().detach()], dim=0)
        all_img = torch.cat([img_k, self.queue_img.T.clone().detach()], dim=0)

        # ── contrastive loss (symmetric) ──────────────────────────────────────
        logits_i2s = img_q @ all_sfh.T * T
        logits_s2i = sfh_q @ all_img.T * T
        labels = torch.arange(B, device=images.device, dtype=torch.long)
        exact_loss = (
            F.cross_entropy(logits_i2s, labels)
            + F.cross_entropy(logits_s2i, labels)
        ) / 2.0
        ae_filtering = (
            self.hparams.ae_false_negative_k > 0
            or self.hparams.ae_false_negative_max_distance is not None
        )
        if ae_filtering:
            with torch.no_grad():
                reference_latent = self.encode_sfh_latent(sfh_reference)
                valid_pairs, ae_distances = ae_false_negative_mask(
                    reference_latent,
                    n_neighbors=self.hparams.ae_false_negative_k,
                    max_cosine_distance=(
                        self.hparams.ae_false_negative_max_distance
                    ),
                )
            contrastive_loss = (
                masked_cross_entropy(logits_i2s, valid_pairs)
                + masked_cross_entropy(logits_s2i, valid_pairs)
            ) / 2.0
            sfh_w1 = None
        elif self.hparams.soft_positive_weight > 0:
            targets, sfh_w1 = wasserstein_soft_targets(
                sfh_reference,
                soft_weight=self.hparams.soft_positive_weight,
                n_neighbors=self.hparams.soft_positive_k,
                epsilon=self.hparams.sfh_log_epsilon,
            )
            contrastive_loss = (
                soft_cross_entropy(logits_i2s, targets)
                + soft_cross_entropy(logits_s2i, targets)
            ) / 2.0
        else:
            sfh_w1 = None
            contrastive_loss = exact_loss

        ae_adjacency_active = (
            self.hparams.ae_adjacency_weight > 0
            and self.current_epoch >= self.hparams.ae_adjacency_warmup_epochs
        )
        if self.hparams.ae_adjacency_weight > 0:
            with torch.no_grad():
                adjacency_latent = self.encode_sfh_latent(sfh_reference)
            adjacency_scale = 1.0 / self.hparams.ae_adjacency_temperature
            ae_adjacency_loss = (
                adjacency_consistency_loss(
                    adjacency_scale * (img_q @ all_sfh.T),
                    adjacency_latent, adjacency_scale,
                )
                + adjacency_consistency_loss(
                    adjacency_scale * (sfh_q @ all_img.T),
                    adjacency_latent, adjacency_scale,
                )
            ) / 2.0
            if ae_adjacency_active:
                contrastive_objective = (
                    contrastive_loss
                    + self.hparams.ae_adjacency_weight * ae_adjacency_loss
                )
            else:
                contrastive_objective = contrastive_loss
            self.log(
                'train_ae_adjacency_loss', ae_adjacency_loss, sync_dist=True,
            )
            self.log(
                'train_ae_adjacency_active',
                contrastive_loss.new_tensor(float(ae_adjacency_active)),
                sync_dist=True,
            )
        else:
            contrastive_objective = contrastive_loss

        if self.sfh_decoder is not None:
            reconstruction = self.sfh_decoder(sfh_latent)
            reconstruction_loss, reconstruction_parts = sfh_reconstruction_loss(
                reconstruction,
                sfh_reference,
                batch.get('sfh_p16'),
                batch.get('sfh_p84'),
                epsilon=self.hparams.sfh_log_epsilon,
                w1_weight=self.hparams.sfh_reconstruction_w1_weight,
            )
            loss = (
                contrastive_objective
                + self.hparams.sfh_reconstruction_weight * reconstruction_loss
            )
            self.log('train_reconstruction_loss', reconstruction_loss, sync_dist=True)
            self.log('train_reconstruction_w1', reconstruction_parts['w1'], sync_dist=True)
        else:
            loss = contrastive_objective

        # ── within-batch rank-1 (cheap training-time diagnostic) ─────────────
        with torch.no_grad():
            logits_b = img_q @ sfh_k.T * T              # (B, B) batch-only
            r1       = ((logits_b.argmax(1) == labels).float().mean() +
                        (logits_b.T.argmax(1) == labels).float().mean()) / 2

        # ── update queue ──────────────────────────────────────────────────────
        if Q:
            self._dequeue_and_enqueue(img_k, sfh_k)

        self.log('train_loss',       loss, prog_bar=True,  sync_dist=True)
        self.log('train_contrastive_loss', contrastive_loss, sync_dist=True)
        self.log('train_exact_loss', exact_loss, prog_bar=False, sync_dist=True)
        self.log('train_rank1_batch', r1,  prog_bar=True,  sync_dist=True)
        self.log('train_logit_scale', T,   prog_bar=False, sync_dist=True)
        self.log(
            'train_loss_vs_random',
            contrastive_loss - (
                valid_pairs.sum(dim=1).float().log().mean()
                if ae_filtering else contrastive_loss.new_tensor(np.log(B + Q))
            ),
            sync_dist=True,
        )
        if sfh_w1 is not None:
            off_diagonal = ~torch.eye(B, device=images.device, dtype=torch.bool)
            self.log(
                'train_sfh_w1_mean', sfh_w1[off_diagonal].mean(),
                sync_dist=True,
            )
        if ae_filtering:
            excluded = ~valid_pairs
            excluded_per_anchor = excluded.sum(dim=1).float().mean()
            self.log(
                'train_ae_negatives_excluded', excluded_per_anchor,
                sync_dist=True,
            )
            if excluded.any():
                self.log(
                    'train_ae_excluded_distance',
                    ae_distances[excluded].mean(), sync_dist=True,
                )
        return loss

    # ── validation step (batch-only InfoNCE, no queue) ────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images, sfhs = batch['image'], batch['sfh']
        sfh_reference = batch.get('sfh_reference', sfhs)
        if self.hparams.alignment_objective == 'pcmepp':
            result, sfh_latent, image_mean, sfh_mean, image_logvar, sfh_logvar = (
                self._pcme_step(images, sfhs)
            )
            val_loss = result.loss
            if self.sfh_decoder is not None:
                reconstruction = self.sfh_decoder(sfh_latent)
                reconstruction_loss, reconstruction_parts = sfh_reconstruction_loss(
                    reconstruction,
                    sfh_reference,
                    batch.get('sfh_p16'),
                    batch.get('sfh_p84'),
                    epsilon=self.hparams.sfh_log_epsilon,
                    w1_weight=self.hparams.sfh_reconstruction_w1_weight,
                )
                val_loss = val_loss + self.hparams.sfh_reconstruction_weight * reconstruction_loss
                self.log('val_reconstruction_loss', reconstruction_loss, sync_dist=True)
                self.log('val_reconstruction_w1', reconstruction_parts['w1'], sync_dist=True)
            labels = torch.arange(result.logits.size(0), device=result.logits.device)
            with torch.no_grad():
                rank1 = (
                    (result.logits.argmax(1) == labels).float().mean()
                    + (result.logits.T.argmax(1) == labels).float().mean()
                ) / 2.0
                k = min(5, result.logits.size(1))
                rank5 = 0.5 * (
                    (result.logits.topk(k, dim=1).indices == labels[:, None]).any(1).float().mean()
                    + (result.logits.T.topk(k, dim=1).indices == labels[:, None]).any(1).float().mean()
                )
                probabilities = result.logits.sigmoid()
                positive_probability = probabilities.diagonal().mean()
                if probabilities.size(0) > 1:
                    negative_probability = (
                        probabilities.sum() - probabilities.diagonal().sum()
                    ) / (probabilities.numel() - probabilities.size(0))
                else:
                    negative_probability = positive_probability.new_zeros(())
                cosine = image_mean @ sfh_mean.T
                positive_cosine = cosine.diagonal().mean()
                negative_cosine = (
                    (cosine.sum() - cosine.diagonal().sum())
                    / max(cosine.numel() - cosine.size(0), 1)
                )
                pseudo_fraction = 0.5 * (
                    result.image_to_sfh_targets.mean()
                    + result.sfh_to_image_targets.mean()
                )
            self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
            self.log('val_pcme_match_loss', result.match_loss, sync_dist=True)
            self.log('val_pcme_pseudo_positive_loss', result.pseudo_positive_loss, sync_dist=True)
            self.log('val_pcme_vib_loss', result.vib_loss, sync_dist=True)
            self.log('val_pcme_scale', result.scale, sync_dist=True)
            self.log('val_pcme_bias', self.pcme_bias, sync_dist=True)
            self.log('val_pcme_pseudo_positive_fraction', pseudo_fraction, sync_dist=True)
            self.log(
                'val_image_uncertainty',
                image_logvar.float().clamp(-12, 8).exp().sum(1).mean(),
                sync_dist=True,
            )
            self.log(
                'val_sfh_uncertainty',
                sfh_logvar.float().clamp(-12, 8).exp().sum(1).mean(),
                sync_dist=True,
            )
            self.log('val_positive_match_probability', positive_probability, sync_dist=True)
            self.log('val_negative_match_probability', negative_probability, sync_dist=True)
            self.log('val_positive_cosine', positive_cosine, sync_dist=True)
            self.log('val_negative_cosine', negative_cosine, sync_dist=True)
            self.log('val_alignment_margin', positive_cosine - negative_cosine, sync_dist=True)
            self.log('val_rank1', rank1, prog_bar=True, sync_dist=True)
            self.log('val_rank5', rank5, prog_bar=True, sync_dist=True)
            return
        if self.hparams.alignment_objective in {'cwcl', 'cyclip'}:
            result, sfh_latent, img_e, sfh_e, T = self._alternative_alignment_step(
                images, sfhs, sfh_reference,
            )
            val_loss = result.loss
            if self.sfh_decoder is not None:
                reconstruction = self.sfh_decoder(sfh_latent)
                reconstruction_loss, reconstruction_parts = sfh_reconstruction_loss(
                    reconstruction,
                    sfh_reference,
                    batch.get('sfh_p16'),
                    batch.get('sfh_p84'),
                    epsilon=self.hparams.sfh_log_epsilon,
                    w1_weight=self.hparams.sfh_reconstruction_w1_weight,
                )
                val_loss = val_loss + self.hparams.sfh_reconstruction_weight * reconstruction_loss
                self.log('val_reconstruction_loss', reconstruction_loss, sync_dist=True)
                self.log('val_reconstruction_w1', reconstruction_parts['w1'], sync_dist=True)
            labels = torch.arange(result.logits.size(0), device=result.logits.device)
            with torch.no_grad():
                rank1 = 0.5 * (
                    (result.logits.argmax(1) == labels).float().mean()
                    + (result.logits.T.argmax(1) == labels).float().mean()
                )
                k = min(5, result.logits.size(1))
                rank5 = 0.5 * (
                    (result.logits.topk(k, 1).indices == labels[:, None]).any(1).float().mean()
                    + (result.logits.T.topk(k, 1).indices == labels[:, None]).any(1).float().mean()
                )
                cosine = img_e @ sfh_e.T
                positive_cosine = cosine.diagonal().mean()
                if cosine.size(0) > 1:
                    negative_cosine = (
                        cosine.sum() - cosine.diagonal().sum()
                    ) / (cosine.numel() - cosine.size(0))
                else:
                    negative_cosine = positive_cosine.new_zeros(())
            self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
            self.log('val_alignment_loss', result.loss, sync_dist=True)
            self.log('val_rank1', rank1, prog_bar=True, sync_dist=True)
            self.log('val_rank5', rank5, prog_bar=True, sync_dist=True)
            self.log('val_positive_cosine', positive_cosine, sync_dist=True)
            self.log('val_negative_cosine', negative_cosine, sync_dist=True)
            self.log(
                'val_alignment_margin', positive_cosine - negative_cosine,
                sync_dist=True,
            )
            self.log('val_logit_scale', T, sync_dist=True)
            if self.hparams.alignment_objective == 'cwcl':
                self.log(
                    'val_cwcl_weighted_i2s_loss',
                    result.weighted_image_to_sfh_loss, sync_dist=True,
                )
                self.log(
                    'val_cwcl_exact_s2i_loss',
                    result.exact_sfh_to_image_loss, sync_dist=True,
                )
                self.log(
                    'val_cwcl_target_entropy', result.target_entropy,
                    sync_dist=True,
                )
                self.log(
                    'val_cwcl_effective_positives', result.effective_positives,
                    sync_dist=True,
                )
            else:
                self.log('val_cyclip_clip_loss', result.clip_loss, sync_dist=True)
                self.log(
                    'val_cyclip_inmodal_loss', result.inmodal_cyclic_loss,
                    sync_dist=True,
                )
                self.log(
                    'val_cyclip_crossmodal_loss',
                    result.crossmodal_cyclic_loss, sync_dist=True,
                )
            return
        T = self.log_temp.exp().clamp(min=1.0, max=100.0)

        img_e = F.normalize(self.encode_image(images), dim=-1)
        sfh_latent = self.encode_sfh_latent(sfhs)
        sfh_e = F.normalize(self.project_sfh(sfh_latent), dim=-1)

        logits  = T * (img_e @ sfh_e.T)
        labels  = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        val_exact_loss = (F.cross_entropy(logits,   labels) +
                          F.cross_entropy(logits.T, labels)) / 2.0
        ae_filtering = (
            self.hparams.ae_false_negative_k > 0
            or self.hparams.ae_false_negative_max_distance is not None
        )
        if ae_filtering:
            with torch.no_grad():
                reference_latent = self.encode_sfh_latent(sfh_reference)
                valid_pairs, ae_distances = ae_false_negative_mask(
                    reference_latent,
                    n_neighbors=self.hparams.ae_false_negative_k,
                    max_cosine_distance=(
                        self.hparams.ae_false_negative_max_distance
                    ),
                )
            val_contrastive_loss = (
                masked_cross_entropy(logits, valid_pairs)
                + masked_cross_entropy(logits.T, valid_pairs)
            ) / 2.0
            self.log('val_ae_filtered_loss', val_contrastive_loss, sync_dist=True)
            self.log(
                'val_ae_negatives_excluded',
                (~valid_pairs).sum(dim=1).float().mean(), sync_dist=True,
            )
            if (~valid_pairs).any():
                self.log(
                    'val_ae_excluded_distance',
                    ae_distances[~valid_pairs].mean(), sync_dist=True,
                )
        elif self.hparams.soft_positive_weight > 0:
            targets, _ = wasserstein_soft_targets(
                sfh_reference,
                soft_weight=self.hparams.soft_positive_weight,
                n_neighbors=self.hparams.soft_positive_k,
                epsilon=self.hparams.sfh_log_epsilon,
            )
            val_contrastive_loss = (
                soft_cross_entropy(logits, targets)
                + soft_cross_entropy(logits.T, targets)
            ) / 2.0
            self.log('val_soft_loss', val_contrastive_loss, sync_dist=True)
        else:
            val_contrastive_loss = val_exact_loss

        ae_adjacency_active = (
            self.hparams.ae_adjacency_weight > 0
            and self.current_epoch >= self.hparams.ae_adjacency_warmup_epochs
        )
        if self.hparams.ae_adjacency_weight > 0:
            with torch.no_grad():
                adjacency_latent = self.encode_sfh_latent(sfh_reference)
            adjacency_scale = 1.0 / self.hparams.ae_adjacency_temperature
            adjacency_cross = adjacency_scale * (img_e @ sfh_e.T)
            val_ae_adjacency_loss = (
                adjacency_consistency_loss(
                    adjacency_cross, adjacency_latent, adjacency_scale,
                )
                + adjacency_consistency_loss(
                    adjacency_cross.T, adjacency_latent, adjacency_scale,
                )
            ) / 2.0
            self.log(
                'val_ae_adjacency_loss', val_ae_adjacency_loss, sync_dist=True,
            )
            if ae_adjacency_active:
                val_contrastive_objective = (
                    val_contrastive_loss
                    + self.hparams.ae_adjacency_weight * val_ae_adjacency_loss
                )
            else:
                val_contrastive_objective = val_contrastive_loss
        else:
            val_contrastive_objective = val_contrastive_loss

        if self.sfh_decoder is not None:
            reconstruction = self.sfh_decoder(sfh_latent)
            reconstruction_loss, reconstruction_parts = sfh_reconstruction_loss(
                reconstruction,
                sfh_reference,
                batch.get('sfh_p16'),
                batch.get('sfh_p84'),
                epsilon=self.hparams.sfh_log_epsilon,
                w1_weight=self.hparams.sfh_reconstruction_w1_weight,
            )
            val_loss = (
                val_contrastive_objective
                + self.hparams.sfh_reconstruction_weight * reconstruction_loss
            )
            self.log('val_reconstruction_loss', reconstruction_loss, sync_dist=True)
            self.log('val_reconstruction_w1', reconstruction_parts['w1'], sync_dist=True)
        else:
            val_loss = val_contrastive_objective

        with torch.no_grad():
            r1 = ((logits.argmax(1)   == labels).float().mean() +
                  (logits.T.argmax(1) == labels).float().mean()) / 2
            k = min(5, logits.size(1))
            r5_i2s = (logits.topk(k, dim=1).indices == labels[:, None]).any(1)
            r5_s2i = (logits.T.topk(k, dim=1).indices == labels[:, None]).any(1)
            r5 = (r5_i2s.float().mean() + r5_s2i.float().mean()) / 2
            cosine = img_e @ sfh_e.T
            positive_cosine = cosine.diagonal().mean()
            if cosine.size(0) > 1:
                negative_cosine = (
                    cosine.sum() - cosine.diagonal().sum()
                ) / (cosine.numel() - cosine.size(0))
            else:
                negative_cosine = positive_cosine.new_zeros(())
            alignment_margin = positive_cosine - negative_cosine
            if ae_filtering:
                random_loss = valid_pairs.sum(dim=1).float().log().mean()
            else:
                random_loss = val_contrastive_loss.new_tensor(np.log(logits.size(0)))

        self.log('val_loss',   val_loss, prog_bar=True,  sync_dist=True)
        self.log('val_contrastive_loss', val_contrastive_loss, sync_dist=True)
        self.log('val_exact_loss', val_exact_loss, sync_dist=True)
        self.log('val_rank1',  r1,       prog_bar=True,  sync_dist=True)
        self.log('val_rank5', r5, prog_bar=True, sync_dist=True)
        self.log(
            'val_loss_vs_random', val_contrastive_loss - random_loss, sync_dist=True,
        )
        self.log('val_positive_cosine', positive_cosine, sync_dist=True)
        self.log('val_negative_cosine', negative_cosine, sync_dist=True)
        self.log('val_alignment_margin', alignment_margin, sync_dist=True)

    # ── optimiser & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        # Split trainable parameters into backbone (unfrozen blocks, lower lr)
        # and everything else (projection head + SFH encoder, full lr).
        backbone_params = [
            p for n, p in self.image_encoder.backbone.named_parameters()
            if p.requires_grad
        ]
        other_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad
            and not n.startswith('image_encoder.backbone.')
            and not n.startswith('sfh_encoder.')
            and not n.startswith('sfh_decoder.')
            and not n.startswith('sfh_projection_m.')
            and not n.startswith('image_encoder_m.')
            and not n.startswith('sfh_encoder_m.')
        ]
        sfh_params = [p for p in self.sfh_encoder.parameters() if p.requires_grad]
        if self.sfh_decoder is not None:
            sfh_params.extend(p for p in self.sfh_decoder.parameters() if p.requires_grad)

        param_groups = [{'params': other_params, 'lr': self.hparams.lr}]
        if sfh_params:
            param_groups.append({
                'params': sfh_params,
                'lr': self.hparams.lr * self.hparams.sfh_lr_scale,
            })
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.hparams.lr * self.hparams.backbone_lr_scale,
            })

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.epochs
            if epoch < wu:
                return (epoch + 1) / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer':    optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


class MultiFilterCosmosWebZooBotCLIP(L.LightningModule):
    """
    MoCo-style CLIP model using three frozen ZooBOT backbones (F150W / F277W / F444W)
    routed by redshift, sharing a single MLP projection head.

    Identical training dynamics to CosmosWebZooBotCLIP; the only difference is
    that each batch also carries a 'redshift' tensor used to route images to the
    correct backbone inside MultiFilterZooBotImageEncoder.

    Parameters
    ----------
    zoobot_ckpt_f150w, zoobot_ckpt_f277w, zoobot_ckpt_f444w : str
        Paths to FinetuneableZoobotClassifier checkpoints for each filter.
    z_low, z_high : float
        Redshift boundaries for filter routing (default 1.0 and 3.0).
    All other parameters same as CosmosWebZooBotCLIP.
    """

    def __init__(
        self,
        zoobot_ckpt_f150w: str,
        zoobot_ckpt_f277w: str,
        zoobot_ckpt_f444w: str,
        z_low:         float = 1.0,
        z_high:        float = 3.0,
        embed_dim:     int   = 256,
        sfh_input_dim: int   = 50,
        temperature:   float = 0.07,
        queue_size:    int   = 4096,
        momentum:      float = 0.995,
        lr:            float = 1e-4,
        weight_decay:  float = 0.05,
        epochs:        int   = 50,
        warmup_epochs: int   = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = MultiFilterZooBotImageEncoder(
            ckpt_f150w=zoobot_ckpt_f150w,
            ckpt_f277w=zoobot_ckpt_f277w,
            ckpt_f444w=zoobot_ckpt_f444w,
            embed_dim=embed_dim,
            z_low=z_low,
            z_high=z_high,
        )
        self.sfh_encoder = SFHEncoder(
            input_dim=sfh_input_dim,
            embed_dim=embed_dim,
        )

        self.image_encoder_m = copy.deepcopy(self.image_encoder)
        self.sfh_encoder_m   = copy.deepcopy(self.sfh_encoder)
        for p in self.image_encoder_m.parameters():
            p.requires_grad_(False)
        for p in self.sfh_encoder_m.parameters():
            p.requires_grad_(False)

        self.log_temp = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))

        Q = queue_size
        D = embed_dim
        self.register_buffer('queue_img', F.normalize(torch.randn(D, Q), dim=0))
        self.register_buffer('queue_sfh', F.normalize(torch.randn(D, Q), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self) -> None:
        m = self.hparams.momentum
        for p, pm in zip(self.image_encoder.parameters(),
                         self.image_encoder_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)
        for p, pm in zip(self.sfh_encoder.parameters(),
                         self.sfh_encoder_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_keys, sfh_keys) -> None:
        B   = img_keys.shape[0]
        Q   = self.hparams.queue_size
        ptr = int(self.queue_ptr)
        self.queue_img[:, ptr:ptr + B] = img_keys.T
        self.queue_sfh[:, ptr:ptr + B] = sfh_keys.T
        self.queue_ptr[0] = (ptr + B) % Q

    def encode_image(self, image: torch.Tensor, redshifts: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image, redshifts)

    def encode_sfh(self, sfh: torch.Tensor) -> torch.Tensor:
        return self.sfh_encoder(sfh)

    @staticmethod
    def _infoNCE(queries, all_keys, T) -> torch.Tensor:
        logits = queries @ all_keys.T * T
        labels = torch.arange(queries.size(0), device=queries.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images    = batch['image']
        sfhs      = batch['sfh']
        redshifts = batch['redshift']
        B = images.size(0)
        Q = self.hparams.queue_size

        if B > Q:
            raise ValueError(f'batch_size ({B}) > queue_size ({Q}).')

        T = self.log_temp.exp().clamp(min=1.0, max=100.0)

        img_q = F.normalize(self.encode_image(images, redshifts), dim=-1)
        sfh_q = F.normalize(self.encode_sfh(sfhs),                dim=-1)

        with torch.no_grad():
            self._momentum_update()
            img_k = F.normalize(self.image_encoder_m(images, redshifts), dim=-1)
            sfh_k = F.normalize(self.sfh_encoder_m(sfhs),                dim=-1)

        all_sfh = torch.cat([sfh_k, self.queue_sfh.T.clone().detach()], dim=0)
        all_img = torch.cat([img_k, self.queue_img.T.clone().detach()], dim=0)

        loss_i2s = self._infoNCE(img_q, all_sfh, T)
        loss_s2i = self._infoNCE(sfh_q, all_img, T)
        loss     = (loss_i2s + loss_s2i) / 2.0

        with torch.no_grad():
            labels   = torch.arange(B, device=images.device)
            logits_b = img_q @ sfh_k.T * T
            r1       = ((logits_b.argmax(1) == labels).float().mean() +
                        (logits_b.T.argmax(1) == labels).float().mean()) / 2

        self._dequeue_and_enqueue(img_k, sfh_k)

        self.log('train_loss',        loss, prog_bar=True,  sync_dist=True)
        self.log('train_rank1_batch', r1,   prog_bar=True,  sync_dist=True)
        self.log('train_logit_scale', T,    prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images    = batch['image']
        sfhs      = batch['sfh']
        redshifts = batch['redshift']
        T = self.log_temp.exp().clamp(min=1.0, max=100.0)

        img_e = F.normalize(self.encode_image(images, redshifts), dim=-1)
        sfh_e = F.normalize(self.encode_sfh(sfhs),                dim=-1)

        logits   = T * (img_e @ sfh_e.T)
        labels   = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        val_loss = (F.cross_entropy(logits,   labels) +
                    F.cross_entropy(logits.T, labels)) / 2.0

        with torch.no_grad():
            r1 = ((logits.argmax(1)   == labels).float().mean() +
                  (logits.T.argmax(1) == labels).float().mean()) / 2

        self.log('val_loss',  val_loss, prog_bar=True,  sync_dist=True)
        self.log('val_rank1', r1,       prog_bar=True,  sync_dist=True)

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer':    optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


# ── v8: transformer SFH encoder ───────────────────────────────────────────────

class CosmosWebZooBotCLIPv8(L.LightningModule):
    """
    CosmosWebZooBotCLIP v8 — same MoCo CLIP framework as v4/v7 but with a
    Set-Transformer SFH encoder replacing the fixed-grid FC encoder.

    SFH input: (B, 9, 2) tokens — each token is (t_frac, log_sfr) for one
    CIGALE bin.  t_frac is randomly sampled within each bin's temporal extent
    at train time, breaking the plateau-width redshift artifact.
    """

    def __init__(
        self,
        zoobot_ckpt:      str,
        embed_dim:        int   = 256,
        # SFH transformer hyperparameters
        sfh_n_bins:       int   = 9,
        sfh_d_model:      int   = 64,
        sfh_n_heads:      int   = 4,
        sfh_n_layers:     int   = 3,
        # CLIP / MoCo
        temperature:      float = 0.07,
        queue_size:       int   = 1024,
        momentum:         float = 0.995,
        # Optimiser
        lr:               float = 1e-4,
        weight_decay:     float = 0.05,
        epochs:           int   = 100,
        warmup_epochs:    int   = 3,
        # Backbone fine-tuning
        unfreeze_blocks:  int   = 0,
        backbone_lr_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = ZooBotImageEncoder(
            ckpt_path=zoobot_ckpt,
            embed_dim=embed_dim,
            unfreeze_blocks=unfreeze_blocks,
        )
        self.sfh_encoder = SFHTransformerEncoder(
            n_bins=sfh_n_bins,
            d_model=sfh_d_model,
            n_heads=sfh_n_heads,
            n_layers=sfh_n_layers,
            embed_dim=embed_dim,
        )

        self.image_encoder_m = copy.deepcopy(self.image_encoder)
        self.sfh_encoder_m   = copy.deepcopy(self.sfh_encoder)
        for p in self.image_encoder_m.parameters():
            p.requires_grad_(False)
        for p in self.sfh_encoder_m.parameters():
            p.requires_grad_(False)

        self.log_temp = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))

        Q, D = queue_size, embed_dim
        self.register_buffer('queue_img', F.normalize(torch.randn(D, Q), dim=0))
        self.register_buffer('queue_sfh', F.normalize(torch.randn(D, Q), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self) -> None:
        m = self.hparams.momentum
        for p, pm in zip(self.image_encoder.parameters(),
                         self.image_encoder_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)
        for p, pm in zip(self.sfh_encoder.parameters(),
                         self.sfh_encoder_m.parameters()):
            pm.data.mul_(m).add_((1.0 - m) * p.data)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_keys, sfh_keys) -> None:
        B   = img_keys.shape[0]
        Q   = self.hparams.queue_size
        ptr = int(self.queue_ptr)
        self.queue_img[:, ptr:ptr + B] = img_keys.T
        self.queue_sfh[:, ptr:ptr + B] = sfh_keys.T
        self.queue_ptr[0] = (ptr + B) % Q

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_sfh(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 9, 2)"""
        return self.sfh_encoder(tokens)

    @staticmethod
    def _infoNCE(queries, all_keys, T):
        B = queries.shape[0]
        logits = torch.matmul(queries, all_keys.T) * T.exp()
        labels = torch.arange(B, device=queries.device)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        tokens = batch['tokens']   # (B, 9, 2)
        B = images.shape[0]

        self._momentum_update()

        img_q = F.normalize(self.image_encoder(images), dim=1)
        sfh_q = F.normalize(self.sfh_encoder(tokens),  dim=1)

        with torch.no_grad():
            img_k = F.normalize(self.image_encoder_m(images), dim=1)
            sfh_k = F.normalize(self.sfh_encoder_m(tokens),   dim=1)

        all_img = torch.cat([img_k, self.queue_img.T.clone()], dim=0)
        all_sfh = torch.cat([sfh_k, self.queue_sfh.T.clone()], dim=0)

        T = self.log_temp
        loss = (self._infoNCE(sfh_q, all_img, T) +
                self._infoNCE(img_q, all_sfh, T)) / 2.0

        self._dequeue_and_enqueue(img_k, sfh_k)

        with torch.no_grad():
            logits = torch.matmul(sfh_q, img_k.T) * T.exp()
            r1 = (logits.argmax(1) == torch.arange(B, device=self.device)).float().mean()

        self.log('train_loss',       loss, prog_bar=True,  on_step=True, on_epoch=False)
        self.log('train_rank1_batch', r1,  prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        tokens = batch['tokens']
        B = images.shape[0]

        img_q = F.normalize(self.image_encoder(images), dim=1)
        sfh_q = F.normalize(self.sfh_encoder(tokens),  dim=1)

        logits   = torch.matmul(sfh_q, img_q.T) * self.log_temp.exp()
        labels   = torch.arange(B, device=self.device)
        val_loss = (F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)) / 2.0

        with torch.no_grad():
            r1 = ((logits.argmax(1)   == labels).float().mean() +
                  (logits.T.argmax(1) == labels).float().mean()) / 2

        self.log('val_loss',  val_loss, prog_bar=True, sync_dist=True)
        self.log('val_rank1', r1,       prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        backbone_params = [p for n, p in self.image_encoder.backbone.named_parameters()
                           if p.requires_grad]
        other_params = [p for n, p in self.named_parameters()
                        if p.requires_grad
                        and not n.startswith('image_encoder.backbone.')
                        and not n.startswith('image_encoder_m.')
                        and not n.startswith('sfh_encoder_m.')]

        param_groups = [{'params': other_params, 'lr': self.hparams.lr}]
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.hparams.lr * self.hparams.backbone_lr_scale,
            })

        optimizer = torch.optim.AdamW(param_groups,
                                      weight_decay=self.hparams.weight_decay)

        def lr_lambda(epoch):
            wu    = self.hparams.warmup_epochs
            total = self.hparams.epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
