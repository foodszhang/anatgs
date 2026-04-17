"""Volume rendering helpers for continuous-time attenuation fields."""

from __future__ import annotations

import torch


def render_ray_batch(
    model,
    points_unit: torch.Tensor,
    timestamps: torch.Tensor,
    delta: torch.Tensor,
    projection_mode: str = "line_integral",
) -> torch.Tensor:
    """Beer-Lambert rendering for a ray batch. Returns intensity [B,1]."""
    if points_unit.ndim != 3 or points_unit.shape[-1] != 3:
        raise ValueError(f"points_unit must be [B,S,3], got {tuple(points_unit.shape)}")
    b, s, _ = points_unit.shape
    if timestamps.ndim == 1:
        timestamps = timestamps[:, None]
    t = timestamps[:, None, :].expand(b, s, 1).reshape(b * s, 1)
    xyz = points_unit.reshape(b * s, 3)
    mu = model(xyz, t).reshape(b, s)
    integral = (mu.sum(dim=1) * delta).clamp_min(0.0)
    mode = str(projection_mode).lower()
    if mode == "line_integral":
        return integral[:, None]
    if mode == "beer_lambert":
        return torch.exp(-integral)[:, None]
    raise ValueError(f"Unsupported projection_mode={projection_mode}")


@torch.no_grad()
def query_volume(model, t_value: float, resolution: int = 128, batch_size: int = 262144) -> torch.Tensor:
    """Query the full 3D volume at fixed time t, shape [R,R,R]."""
    device = next(model.parameters()).device
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, resolution, device=device),
            torch.linspace(0.0, 1.0, resolution, device=device),
            torch.linspace(0.0, 1.0, resolution, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    out = []
    t = torch.full((batch_size, 1), float(t_value), device=device)
    for i in range(0, coords.shape[0], batch_size):
        c = coords[i : i + batch_size]
        tt = t[: c.shape[0]]
        out.append(model(c, tt).squeeze(-1))
    return torch.cat(out, dim=0).reshape(resolution, resolution, resolution)
