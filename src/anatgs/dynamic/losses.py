"""Losses for 4D continuous-time training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def projection_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def temporal_smoothness_loss(
    model,
    xyz: torch.Tensor,
    t: torch.Tensor,
    dt: float = 0.01,
) -> torch.Tensor:
    t2 = (t + float(dt)).clamp(0.0, 1.0)
    mu1 = model(xyz, t)
    mu2 = model(xyz, t2)
    return torch.mean((mu1 - mu2) ** 2)


def reference_loss(
    model,
    xyz: torch.Tensor,
    t_ref: float,
    mu_ref: torch.Tensor,
    weight_map: torch.Tensor | None = None,
) -> torch.Tensor:
    t = torch.full((xyz.shape[0], 1), float(t_ref), device=xyz.device, dtype=xyz.dtype)
    mu = model(xyz, t)
    diff = (mu - mu_ref) ** 2
    if weight_map is not None:
        diff = diff * weight_map
    return diff.mean()

