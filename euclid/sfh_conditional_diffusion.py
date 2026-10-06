"""Conditional diffusion for normalized one-dimensional Euclid SFHs."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def log_sfh_to_weights(log_sfh, epsilon=1e-10):
    """Convert stored log SFHs to strictly normalized nonnegative weights."""
    values = np.asarray(log_sfh, dtype=np.float64)
    weights = np.maximum(np.power(10.0, values) - epsilon, 0.0)
    total = weights.sum(axis=-1, keepdims=True)
    if not np.all(np.isfinite(weights)) or np.any(total <= 0):
        raise ValueError('SFHs must contain finite positive total mass.')
    return (weights / total).astype(np.float32)


def weights_to_clr(weights, floor=1e-8):
    """Map simplex-valued SFHs to centered log-ratio coordinates."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim < 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('SFH weights must be finite and nonnegative.')
    total = values.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError('SFH weights must have positive total mass.')
    values = values / total
    logged = np.log(np.maximum(values, floor))
    return (logged - logged.mean(axis=-1, keepdims=True)).astype(np.float32)


def clr_to_weights(clr):
    """Invert CLR coordinates with a stable softmax; sums are exactly one."""
    values = np.asarray(clr, dtype=np.float64)
    if values.ndim < 1 or not np.all(np.isfinite(values)):
        raise ValueError('CLR values must be finite.')
    shifted = values - values.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights.astype(np.float32)


def sinusoidal_time_embedding(timestep, width=64):
    if width < 4 or width % 2:
        raise ValueError('Time embedding width must be an even integer >= 4.')
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(width // 2, device=timestep.device)
        / (width // 2 - 1)
    )
    phases = timestep.float()[:, None] * frequencies[None]
    return torch.cat([phases.sin(), phases.cos()], dim=1)


class ConditionalSFHTransformer(nn.Module):
    """Transformer velocity predictor conditioned through additive FiLM context."""

    def __init__(self, n_bins=250, condition_dim=3, d_model=128, n_heads=4,
                 n_layers=4, dropout=0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads.')
        self.n_bins = n_bins
        self.input_projection = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.randn(1, n_bins, d_model) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(64, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.null_context = nn.Parameter(torch.zeros(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers, nn.LayerNorm(d_model))
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, noisy, timestep, condition, drop=None):
        if noisy.ndim != 2 or noisy.shape[1] != self.n_bins:
            raise ValueError(f'Expected noisy SFHs with shape (batch, {self.n_bins}).')
        context = self.condition_mlp(condition)
        if drop is not None:
            context = torch.where(drop[:, None], self.null_context[None], context)
        context = context + self.time_mlp(sinusoidal_time_embedding(timestep))
        tokens = self.input_projection(noisy[:, :, None]) + self.position
        tokens = tokens + context[:, None, :]
        return self.output(self.transformer(tokens)).squeeze(-1)


class SFHDiffusionSchedule(nn.Module):
    def __init__(self, steps=1000):
        super().__init__()
        if steps < 2:
            raise ValueError('At least two diffusion steps are required.')
        grid = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        alpha = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
        alpha = (alpha / alpha[0])[1:].float()
        alpha[-1] = 0
        self.register_buffer('alpha', alpha)

    def coefficients(self, timestep):
        alpha = self.alpha[timestep][:, None]
        return alpha.sqrt(), (1 - alpha).sqrt()

    def noisy_target(self, clean, noise, timestep):
        alpha, sigma = self.coefficients(timestep)
        return alpha * clean + sigma * noise, alpha * noise - sigma * clean

    def clean_noise(self, noisy, velocity, timestep):
        alpha, sigma = self.coefficients(timestep)
        return alpha * noisy - sigma * velocity, sigma * noisy + alpha * velocity

    @torch.inference_mode()
    def sample(self, model, condition, n_bins, steps=100, guidance=1.0, seed=42):
        if not 2 <= steps <= len(self.alpha):
            raise ValueError('Sampling steps must be between 2 and training steps.')
        generator = torch.Generator(device=condition.device).manual_seed(int(seed))
        sample = torch.randn(
            (len(condition), n_bins), device=condition.device, generator=generator,
        )
        times = torch.linspace(
            len(self.alpha) - 1, 0, steps, device=sample.device,
        ).round().long()
        null = torch.ones(len(sample), device=sample.device, dtype=torch.bool)
        for index, scalar_t in enumerate(times):
            timestep = scalar_t.expand(len(sample))
            conditional = model(sample, timestep, condition)
            if guidance == 1:
                velocity = conditional
            else:
                unconditional = model(sample, timestep, condition, null)
                velocity = unconditional + guidance * (conditional - unconditional)
            clean, noise = self.clean_noise(sample, velocity.float(), timestep)
            if index + 1 == len(times):
                sample = clean
            else:
                alpha_next = self.alpha[times[index + 1]]
                sample = alpha_next.sqrt() * clean + (1 - alpha_next).sqrt() * noise
        return sample


class ConditionalSFHDiffusionModule(nn.Module):
    """Network, EMA copy, schedule, and fixed data transforms."""

    def __init__(self, n_bins, condition_mean, condition_scale, clr_mean,
                 clr_scale, d_model=128, n_heads=4, n_layers=4,
                 dropout=0.1, diffusion_steps=1000):
        super().__init__()
        condition_mean = torch.as_tensor(condition_mean, dtype=torch.float32)
        condition_scale = torch.as_tensor(condition_scale, dtype=torch.float32)
        clr_mean = torch.as_tensor(clr_mean, dtype=torch.float32)
        clr_scale = torch.as_tensor(clr_scale, dtype=torch.float32)
        if condition_mean.shape != condition_scale.shape:
            raise ValueError('Condition mean and scale shapes differ.')
        if clr_mean.shape != (n_bins,) or clr_scale.shape != (n_bins,):
            raise ValueError('CLR statistics do not match n_bins.')
        if torch.any(condition_scale <= 0) or torch.any(clr_scale <= 0):
            raise ValueError('All normalization scales must be positive.')
        self.register_buffer('condition_mean', condition_mean)
        self.register_buffer('condition_scale', condition_scale)
        self.register_buffer('clr_mean', clr_mean)
        self.register_buffer('clr_scale', clr_scale)
        self.net = ConditionalSFHTransformer(
            n_bins, len(condition_mean), d_model, n_heads, n_layers, dropout,
        )
        self.ema = copy.deepcopy(self.net).eval().requires_grad_(False)
        self.schedule = SFHDiffusionSchedule(diffusion_steps)

    def standardize_condition(self, condition):
        return (condition - self.condition_mean) / self.condition_scale

    def standardize_clr(self, clr):
        return (clr - self.clr_mean) / self.clr_scale

    def unstandardize_clr(self, standardized):
        return standardized * self.clr_scale + self.clr_mean

    @torch.inference_mode()
    def sample_weights(self, condition, steps=100, guidance=1.0, seed=42):
        standardized_condition = self.standardize_condition(condition)
        standardized_clr = self.schedule.sample(
            self.ema, standardized_condition, len(self.clr_mean),
            steps=steps, guidance=guidance, seed=seed,
        )
        clr = self.unstandardize_clr(standardized_clr)
        # The CLR representation is defined only up to a shared additive
        # constant. Re-centering makes exported residuals identifiable without
        # changing the inverse-softmax SFH.
        clr = clr - clr.mean(dim=-1, keepdim=True)
        return F.softmax(clr.double(), dim=-1).float(), clr
