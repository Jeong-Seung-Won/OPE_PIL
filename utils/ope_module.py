"""
Adaptive Orbital Period Embedding (OPE)
========================================
Section 4.2 of the paper.

Replaces the standard sinusoidal positional embedding with a harmonic
basis tuned to the satellite's orbital period T_orb. T_orb is either
estimated from training data via Kepler's third law (--ope_auto_period 1)
or passed as a hyperparameter (--ope_period).
"""

import math
import torch
import torch.nn as nn


class OrbitalPeriodEmbedding(nn.Module):
    """sin(2 pi k t / T_orb), cos(2 pi k t / T_orb)  for k = 1..K."""

    def __init__(self, d_model, max_len=5000, n_harmonics=4,
                 period=97.0, dt_minutes=1.0):
        super().__init__()
        self.n_harmonics = max(1, min(n_harmonics, d_model // 2))
        self.period = float(period)
        self.dt_minutes = float(dt_minutes)

        t = torch.arange(max_len, dtype=torch.float32) * self.dt_minutes
        feats = []
        for k in range(1, self.n_harmonics + 1):
            phase = 2 * math.pi * k * t / self.period
            feats.append(torch.sin(phase))
            feats.append(torch.cos(phase))
        emb = torch.stack(feats, dim=-1)

        self.proj = nn.Linear(2 * self.n_harmonics, d_model)
        self.register_buffer('base_emb', emb.unsqueeze(0))

    def forward(self, x):
        return self.proj(self.base_emb[:, :x.size(1)])


class SinusoidalPositionalEmbedding(nn.Module):
    """Standard sinusoidal PE (ablation against OPE)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]
