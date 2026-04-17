"""Simple cone-beam ray generation with timestamps."""

from __future__ import annotations

import torch


def _source_and_detector(
    angles: torch.Tensor,
    sod: float,
    sdd: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ca = torch.cos(angles)
    sa = torch.sin(angles)
    src = torch.stack([sod * ca, sod * sa, torch.zeros_like(ca)], dim=-1)
    u = torch.stack([ca, sa, torch.zeros_like(ca)], dim=-1)
    det_center = src - u * sdd
    det_x = torch.stack([-sa, ca, torch.zeros_like(ca)], dim=-1)
    det_y = torch.tensor([0.0, 0.0, 1.0], device=angles.device, dtype=angles.dtype)[None, :].expand_as(det_x)
    return src, det_center, det_x, det_y


def build_ray_batch(
    angles: torch.Tensor,
    u_idx: torch.Tensor,
    v_idx: torch.Tensor,
    det_h: int,
    det_w: int,
    det_spacing_h: float,
    det_spacing_w: float,
    sod: float,
    sdd: float,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sampled points [B,S,3] in world coordinates and delta [B]."""
    src, det_center, det_x, det_y = _source_and_detector(angles, sod=sod, sdd=sdd)
    uu = (u_idx.float() - (det_w - 1) * 0.5) * float(det_spacing_w)
    vv = (v_idx.float() - (det_h - 1) * 0.5) * float(det_spacing_h)
    det = det_center + uu[:, None] * det_x + vv[:, None] * det_y
    ray = det - src
    tau = torch.linspace(0.0, 1.0, int(n_samples), device=angles.device, dtype=angles.dtype)[None, :, None]
    pts = src[:, None, :] + tau * ray[:, None, :]
    seg_len = torch.linalg.norm(ray, dim=-1).clamp_min(1e-8)
    delta = seg_len / max(int(n_samples) - 1, 1)
    return pts, delta


def world_to_unit(points_world: torch.Tensor, volume_size_mm: float) -> torch.Tensor:
    """Map world mm coordinates from [-size/2, size/2] to [0,1]."""
    return points_world / float(volume_size_mm) + 0.5

