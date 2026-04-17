"""Temporal encoders for continuous-time fields."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Encode scalar time t in [0, 1] to a feature vector."""

    def __init__(self, method: str = "pe", n_freqs: int = 8, embed_dim: int = 16):
        super().__init__()
        self.method = str(method).lower()
        self.n_freqs = int(n_freqs)
        self.embed_dim = int(embed_dim)
        if self.method not in {"pe", "learned"}:
            raise ValueError(f"Unsupported temporal encoding: {method}")
        if self.method == "learned":
            self.mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, self.embed_dim),
            )
            self.out_dim = self.embed_dim
        else:
            self.mlp = None
            self.out_dim = self.n_freqs * 2

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError(f"t must be [N,1], got {tuple(t.shape)}")
        if self.method == "learned":
            return self.mlp(t)
        freqs = torch.arange(self.n_freqs, device=t.device, dtype=t.dtype)
        freqs = (2.0**freqs)[None, :] * math.pi
        x = t * freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

