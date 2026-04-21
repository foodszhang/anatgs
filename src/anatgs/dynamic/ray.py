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


def ray_aabb_intersect(src: torch.Tensor, ray_dir: torch.Tensor, box_half: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return near/far ray params for intersection with axis-aligned box [-h,h]^3."""
    if isinstance(box_half, (tuple, list)):
        h = torch.tensor([float(box_half[0]), float(box_half[1]), float(box_half[2])], device=src.device, dtype=src.dtype)
    elif torch.is_tensor(box_half):
        h = box_half.to(device=src.device, dtype=src.dtype).reshape(3)
    else:
        hh = float(box_half)
        h = torch.tensor([hh, hh, hh], device=src.device, dtype=src.dtype)
    inv_d = 1.0 / (ray_dir + 1e-9)
    t1 = (-h[None, :] - src) * inv_d
    t2 = (h[None, :] - src) * inv_d
    t_min = torch.minimum(t1, t2).amax(dim=-1)
    t_max = torch.maximum(t1, t2).amin(dim=-1)
    t_near = t_min.clamp(0.0, 1.0)
    t_far = t_max.clamp(0.0, 1.0)
    valid = t_far > t_near
    t_near = torch.where(valid, t_near, torch.zeros_like(t_near))
    t_far = torch.where(valid, t_far, torch.zeros_like(t_far))
    return t_near, t_far


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
    volume_size_mm: float | tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sampled points [B,S,3] in world coordinates and delta [B]."""
    src, det_center, det_x, det_y = _source_and_detector(angles, sod=sod, sdd=sdd)
    uu = (u_idx.float() - (det_w - 1) * 0.5) * float(det_spacing_w)
    vv = (v_idx.float() - (det_h - 1) * 0.5) * float(det_spacing_h)
    det = det_center + uu[:, None] * det_x + vv[:, None] * det_y
    ray = det - src
    if isinstance(volume_size_mm, (tuple, list)):
        box_half = (
            float(volume_size_mm[0]) * 0.5,
            float(volume_size_mm[1]) * 0.5,
            float(volume_size_mm[2]) * 0.5,
        )
    else:
        box_half = float(volume_size_mm) * 0.5
    t_near, t_far = ray_aabb_intersect(src, ray, box_half=box_half)
    tau_norm = torch.linspace(0.0, 1.0, int(n_samples), device=angles.device, dtype=angles.dtype)[None, :]
    tau = t_near[:, None] + (t_far - t_near)[:, None] * tau_norm
    pts = src[:, None, :] + tau[:, :, None] * ray[:, None, :]
    seg_len = torch.linalg.norm(ray, dim=-1).clamp_min(1e-8) * (t_far - t_near).clamp_min(0.0)
    delta = seg_len.clamp_min(1e-8) / max(int(n_samples) - 1, 1)
    return pts, delta


def world_to_unit(points_world: torch.Tensor, volume_size_mm: float | tuple[float, float, float]) -> torch.Tensor:
    """Map world mm coordinates from box [-size/2, size/2] to [0,1]."""
    if isinstance(volume_size_mm, (tuple, list)):
        s = torch.tensor(
            [float(volume_size_mm[0]), float(volume_size_mm[1]), float(volume_size_mm[2])],
            device=points_world.device,
            dtype=points_world.dtype,
        )
        return points_world / s[None, None, :] + 0.5
    return points_world / float(volume_size_mm) + 0.5
