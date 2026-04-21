"""Motion manifold regularizer."""

from __future__ import annotations

import torch
import torch.nn as nn


class MotionManifoldAE(nn.Module):
    def __init__(self, in_dim: int = 3, latent_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec, z


def manifold_regularization_loss(ae: MotionManifoldAE, v: torch.Tensor) -> torch.Tensor:
    rec, z = ae(v)
    return torch.mean((v - rec) ** 2) + torch.mean(z**2)

