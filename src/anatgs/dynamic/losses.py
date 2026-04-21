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


def signal_corr_loss(s_pred: torch.Tensor, s_meas: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if s_pred.ndim != 1:
        s_pred = s_pred.reshape(-1)
    if s_meas.ndim != 1:
        s_meas = s_meas.reshape(-1)
    if s_pred.shape[0] != s_meas.shape[0]:
        raise ValueError(f"Shape mismatch: pred={tuple(s_pred.shape)} meas={tuple(s_meas.shape)}")
    p = s_pred - s_pred.mean()
    m = s_meas - s_meas.mean()
    denom = torch.sqrt(torch.sum(p**2) * torch.sum(m**2) + eps)
    corr = torch.sum(p * m) / denom
    return 1.0 - corr.clamp(-1.0, 1.0)


def velocity_tv_smoothness_loss(model, xyz: torch.Tensor, cond: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Finite-difference TV penalty on velocity field samples."""
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be [N,3], got {tuple(xyz.shape)}")
    if cond.ndim != 2:
        raise ValueError(f"cond must be [N,C], got {tuple(cond.shape)}")
    n = xyz.shape[0]
    if cond.shape[0] != n:
        raise ValueError(f"Batch mismatch: xyz={n}, cond={cond.shape[0]}")
    v0 = model.velocity(xyz, cond)
    e = float(eps)
    ex = torch.tensor([e, 0.0, 0.0], device=xyz.device, dtype=xyz.dtype)[None, :]
    ey = torch.tensor([0.0, e, 0.0], device=xyz.device, dtype=xyz.dtype)[None, :]
    ez = torch.tensor([0.0, 0.0, e], device=xyz.device, dtype=xyz.dtype)[None, :]
    vx = model.velocity((xyz + ex).clamp(0.0, 1.0), cond)
    vy = model.velocity((xyz + ey).clamp(0.0, 1.0), cond)
    vz = model.velocity((xyz + ez).clamp(0.0, 1.0), cond)
    tv = torch.mean(torch.abs(vx - v0) + torch.abs(vy - v0) + torch.abs(vz - v0))
    return tv
