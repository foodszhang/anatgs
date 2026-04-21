"""Differentiable Amsterdam-shroud-style surrogate extraction."""

from __future__ import annotations

import torch


def _make_grid(resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r = int(resolution)
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, r, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, r, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, r, device=device, dtype=dtype),
            indexing="ij",
        ),
        dim=-1,
    )
    return grid.reshape(-1, 3)


def query_volume_conditioned(
    model,
    condition: torch.Tensor,
    resolution: int = 32,
    query_batch: int = 131072,
) -> torch.Tensor:
    """Query μ(x|condition) to a dense [R,R,R] tensor with gradients enabled."""
    if condition.ndim == 1:
        condition = condition[None, :]
    if condition.ndim != 2 or condition.shape[0] != 1:
        raise ValueError(f"condition must be [1,C], got {tuple(condition.shape)}")
    grid = _make_grid(
        resolution=resolution,
        device=condition.device,
        dtype=next(model.parameters()).dtype,
    )
    out = []
    cdim = condition.shape[-1]
    for i in range(0, grid.shape[0], int(query_batch)):
        xyz = grid[i : i + int(query_batch)]
        cond = condition.expand(xyz.shape[0], cdim)
        out.append(model(xyz, cond).squeeze(-1))
    r = int(resolution)
    return torch.cat(out, dim=0).reshape(r, r, r)


def torch_hilbert_1d(x: torch.Tensor) -> torch.Tensor:
    """Return analytic signal via FFT-based Hilbert transform."""
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {tuple(x.shape)}")
    n = x.shape[0]
    Xf = torch.fft.fft(x)
    h = torch.zeros(n, device=x.device, dtype=x.dtype)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return torch.fft.ifft(Xf * h.to(dtype=Xf.dtype))


def amsterdam_shroud(projs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute one scalar respiratory surrogate per view.
    projs: [N,H,W] (N views)
    """
    if projs.ndim != 3:
        raise ValueError(f"projs must be [N,H,W], got {tuple(projs.shape)}")
    # 1) lateral integration: [N,H]
    s_proj = projs.sum(dim=-1)
    # 2) temporal derivative and cumulative motion band: [N,H]
    d = torch.diff(s_proj, dim=0, prepend=s_proj[:1])
    motion = torch.cumsum(d, dim=0)
    motion = motion - motion.min(dim=1, keepdim=True).values + eps
    # 3) Hilbert envelope (torch FFT implementation) per view profile
    env = torch.stack([torch.abs(torch_hilbert_1d(motion[i])) for i in range(motion.shape[0])], dim=0)
    # weighted centroid along cranio-caudal y
    y = torch.linspace(0.0, 1.0, env.shape[1], device=env.device, dtype=env.dtype)
    num = (env * y[None, :]).sum(dim=1)
    den = env.sum(dim=1).clamp_min(eps)
    return num / den


def surrogate_from_volume(volume: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Lateral-integral then cranio-caudal weighted centroid."""
    if volume.ndim != 3:
        raise ValueError(f"volume must be [D,H,W], got {tuple(volume.shape)}")
    shadow = volume.sum(dim=-1)  # [D,H]
    profile = shadow.mean(dim=-1)  # [D]
    # Use descending cranio-caudal coordinate so larger superior expansion
    # maps to larger surrogate values, matching RPM-like positive direction.
    z = torch.linspace(1.0, 0.0, profile.shape[0], device=profile.device, dtype=profile.dtype)
    return (profile * z).sum() / profile.sum().clamp_min(eps)


def predict_surrogate_from_model(
    model,
    conditions: torch.Tensor,
    resolution: int = 32,
    query_batch: int = 131072,
) -> torch.Tensor:
    """Predict one scalar surrogate value for each condition row."""
    if conditions.ndim != 2:
        raise ValueError(f"conditions must be [N,C], got {tuple(conditions.shape)}")
    vals = []
    for i in range(conditions.shape[0]):
        vol = query_volume_conditioned(
            model=model,
            condition=conditions[i : i + 1],
            resolution=resolution,
            query_batch=query_batch,
        )
        vals.append(surrogate_from_volume(vol))
    return torch.stack(vals, dim=0)


def predict_shroud_surrogate_from_model(
    model,
    conditions: torch.Tensor,
    resolution: int = 32,
    query_batch: int = 131072,
) -> torch.Tensor:
    """Predict surrogate by pseudo-lateral projections + Amsterdam shroud."""
    if conditions.ndim != 2:
        raise ValueError(f"conditions must be [N,C], got {tuple(conditions.shape)}")
    projs = []
    for i in range(conditions.shape[0]):
        vol = query_volume_conditioned(
            model=model,
            condition=conditions[i : i + 1],
            resolution=resolution,
            query_batch=query_batch,
        )
        projs.append(vol.sum(dim=-1))
    stack = torch.stack(projs, dim=0)
    return amsterdam_shroud(stack)
