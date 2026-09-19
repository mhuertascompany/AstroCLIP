"""Small SFH-conditioned pixel U-Net; cosine schedule, velocity prediction, DDIM."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class FiLMBlock(nn.Module):
    def __init__(self, cin, cout, context):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.film = nn.Linear(context, 2 * cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, context):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(F.silu(context)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return (self.skip(x) + self.conv2(F.silu(h))) / math.sqrt(2)


class ConditionalUNet(nn.Module):
    def __init__(self, condition_dim=256, base=32):
        super().__init__()
        if base < 8 or base % 8:
            raise ValueError('base must be a positive multiple of 8.')
        context = base * 4
        self.null_condition = nn.Parameter(torch.zeros(condition_dim))
        self.time_mlp = nn.Sequential(nn.Linear(64, context), nn.SiLU(), nn.Linear(context, context))
        self.condition_mlp = nn.Sequential(nn.Linear(condition_dim, context), nn.SiLU(), nn.Linear(context, context))
        widths = [base, base * 2, base * 3, base * 4, base * 4]
        self.input = nn.Conv2d(1, base, 3, padding=1)
        self.down = nn.ModuleList([FiLMBlock(c, c, context) for c in widths])
        self.reduce = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(widths, widths[1:])])
        self.middle = FiLMBlock(widths[-1], widths[-1], context)
        self.attn_norm = nn.LayerNorm(widths[-1])
        self.attn = nn.MultiheadAttention(widths[-1], 4, batch_first=True)
        self.up = nn.ModuleList([FiLMBlock(widths[i+1] + widths[i], widths[i], context) for i in reversed(range(4))])
        self.output = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, x, t, condition, drop=None):
        if x.shape[-1] % 16 or x.shape[-2] % 16:
            raise ValueError('Spatial size must be divisible by 16.')
        if drop is not None:
            condition = torch.where(drop[:, None], self.null_condition[None], condition)
        frequencies = torch.exp(-math.log(10000) * torch.arange(32, device=x.device) / 31)
        phases = t.float()[:, None] * frequencies[None]
        context = self.time_mlp(torch.cat([phases.sin(), phases.cos()], dim=1)) + self.condition_mlp(condition)
        h = self.input(x)
        skips = []
        for i, block in enumerate(self.down):
            h = block(h, context)
            if i < 4:
                skips.append(h)
                h = self.reduce[i](h)
        h = self.middle(h, context)
        tokens = h.flatten(2).transpose(1, 2)
        normalized = self.attn_norm(tokens)
        tokens = tokens + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        h = tokens.transpose(1, 2).reshape_as(h)
        for block, skip in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = block(torch.cat([h, skip], dim=1), context)
        return self.output(h)


class DiffusionSchedule(nn.Module):
    def __init__(self, steps=1000):
        super().__init__()
        if steps < 2:
            raise ValueError('At least two diffusion steps required.')
        # Continuous cosine cumulative alpha, sampled through zero terminal SNR.
        grid = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        alpha = torch.cos((grid + .008) / 1.008 * math.pi / 2).square()
        alpha = (alpha / alpha[0])[1:].float()
        alpha[-1] = 0
        self.register_buffer('alpha', alpha)

    def coefficients(self, t):
        a = self.alpha[t][:, None, None, None]
        return a.sqrt(), (1 - a).sqrt()

    def noisy_target(self, clean, noise, t):
        a, s = self.coefficients(t)
        return a * clean + s * noise, a * noise - s * clean

    def clean_noise(self, noisy, velocity, t):
        a, s = self.coefficients(t)
        return a * noisy - s * velocity, s * noisy + a * velocity

    @torch.inference_mode()
    def sample(self, model, condition, size=224, steps=100, guidance=2., seed=42):
        if not 2 <= steps <= len(self.alpha):
            raise ValueError('Sampling steps must be between 2 and training steps.')
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        x = torch.randn((len(condition), 1, size, size), device=condition.device, generator=generator)
        times = torch.linspace(len(self.alpha)-1, 0, steps, device=x.device).round().long()
        for index, timestep in enumerate(times):
            t = timestep.expand(len(x))
            conditional = model(x, t, condition)
            unconditional = model(x, t, condition, torch.ones(len(x), device=x.device, dtype=torch.bool))
            velocity = unconditional + guidance * (conditional - unconditional)
            clean, noise = self.clean_noise(x, velocity.float(), t)
            clean = clean.clamp(-1, 1)
            # Recompute noise after clipping for a consistent DDIM update.
            a, s = self.coefficients(t)
            noise = (x - a * clean) / s.clamp_min(1e-8)
            if index + 1 == len(times):
                x = clean
            else:
                ap = self.alpha[times[index+1]]
                x = ap.sqrt() * clean + (1-ap).sqrt() * noise
        return x
