"""Volume rendering helpers for continuous-time attenuation fields."""

from __future__ import annotations

import numpy as np
import torch
from torch.autograd import Function

try:
    import tigre
except Exception:  # pragma: no cover
    tigre = None


def render_ray_batch(
    model,
    points_unit: torch.Tensor,
    condition: torch.Tensor,
    delta: torch.Tensor,
    projection_mode: str = "line_integral",
) -> torch.Tensor:
    """Beer-Lambert rendering for a ray batch. Returns intensity [B,1]."""
    if points_unit.ndim != 3 or points_unit.shape[-1] != 3:
        raise ValueError(f"points_unit must be [B,S,3], got {tuple(points_unit.shape)}")
    b, s, _ = points_unit.shape
    if condition.ndim == 1:
        condition = condition[:, None]
    c = condition[:, None, :].expand(b, s, condition.shape[-1]).reshape(b * s, condition.shape[-1])
    xyz = points_unit.reshape(b * s, 3)
    mu = model(xyz, c).reshape(b, s)
    integral = (mu.sum(dim=1) * delta).clamp_min(0.0)
    mode = str(projection_mode).lower()
    if mode == "line_integral":
        return integral[:, None]
    if mode == "beer_lambert":
        return torch.exp(-integral)[:, None]
    raise ValueError(f"Unsupported projection_mode={projection_mode}")


@torch.no_grad()
def query_volume(model, t_value: float, resolution: int = 128, batch_size: int = 262144) -> torch.Tensor:
    """Query the full 3D volume at fixed scalar time t, shape [R,R,R]."""
    return query_volume_condition(model, condition=[float(t_value)], resolution=resolution, batch_size=batch_size)


@torch.no_grad()
def query_volume_condition(model, condition, resolution: int = 128, batch_size: int = 262144) -> torch.Tensor:
    """Query the full 3D volume for an arbitrary condition vector, shape [R,R,R]."""
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
    cond = torch.as_tensor(condition, device=device, dtype=coords.dtype).reshape(1, -1)
    for i in range(0, coords.shape[0], batch_size):
        c = coords[i : i + batch_size]
        cc = cond.expand(c.shape[0], cond.shape[-1])
        out.append(model(c, cc).squeeze(-1))
    return torch.cat(out, dim=0).reshape(resolution, resolution, resolution)


class _TigreProjectorFn(Function):
    @staticmethod
    def forward(ctx, volume_xyz: torch.Tensor, angles_rad: torch.Tensor, geo: dict) -> torch.Tensor:
        if tigre is None:
            raise ImportError("tigre is required for TIGRE-compatible projector")
        vol = volume_xyz.detach().float().cpu().numpy().astype(np.float32)
        ang = angles_rad.detach().float().cpu().numpy().astype(np.float32)
        g = tigre.geometry()
        g.nVoxel = np.asarray(geo["nVoxel"], dtype=np.int32)
        g.sVoxel = np.asarray(geo["sVoxel"], dtype=np.float32)
        g.dVoxel = g.sVoxel / g.nVoxel
        g.nDetector = np.asarray(geo["nDetector"], dtype=np.int32)
        g.dDetector = np.asarray(geo["dDetector"], dtype=np.float32)
        g.sDetector = g.nDetector * g.dDetector
        g.DSO = float(geo["DSO"])
        g.DSD = float(geo["DSD"])
        g.offOrigin = np.asarray(geo.get("offOrigin", [0.0, 0.0, 0.0]), dtype=np.float32)
        g.offDetector = np.asarray(geo.get("offDetector", [0.0, 0.0]), dtype=np.float32)
        g.mode = "cone"
        proj = tigre.Ax(vol, g, ang).astype(np.float32)
        ctx.geo = geo
        ctx.save_for_backward(angles_rad)
        return torch.from_numpy(proj).to(device=volume_xyz.device, dtype=volume_xyz.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if tigre is None:
            raise ImportError("tigre is required for TIGRE-compatible projector")
        (angles_rad,) = ctx.saved_tensors
        geo = ctx.geo
        g = tigre.geometry()
        g.nVoxel = np.asarray(geo["nVoxel"], dtype=np.int32)
        g.sVoxel = np.asarray(geo["sVoxel"], dtype=np.float32)
        g.dVoxel = g.sVoxel / g.nVoxel
        g.nDetector = np.asarray(geo["nDetector"], dtype=np.int32)
        g.dDetector = np.asarray(geo["dDetector"], dtype=np.float32)
        g.sDetector = g.nDetector * g.dDetector
        g.DSO = float(geo["DSO"])
        g.DSD = float(geo["DSD"])
        g.offOrigin = np.asarray(geo.get("offOrigin", [0.0, 0.0, 0.0]), dtype=np.float32)
        g.offDetector = np.asarray(geo.get("offDetector", [0.0, 0.0]), dtype=np.float32)
        g.mode = "cone"
        go = grad_output.detach().float().cpu().numpy().astype(np.float32)
        ang = angles_rad.detach().float().cpu().numpy().astype(np.float32)
        grad_vol = tigre.Atb(go, g, ang).astype(np.float32)
        grad_t = torch.from_numpy(grad_vol).to(device=grad_output.device, dtype=grad_output.dtype)
        return grad_t, None, None


def project_volume_tigre_autograd(volume_xyz: torch.Tensor, angles_rad: torch.Tensor, geo: dict) -> torch.Tensor:
    """TIGRE-compatible cone-beam projector with autograd via adjoint backprojection."""
    return _TigreProjectorFn.apply(volume_xyz, angles_rad, geo)
