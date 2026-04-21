"""V1: impulse/gradient/block probes for volume-side coordinate consistency."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tigre
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.dynamic import render_ray_batch
from anatgs.dynamic.ray import build_ray_batch, world_to_unit
from anatgs.geom import to_radians


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    dr = float(np.max(b) - np.min(b) + 1e-8)
    return float(10.0 * math.log10((dr * dr) / mse))


class VoxelOracleModel(nn.Module):
    def __init__(self, vol_zyx: np.ndarray):
        super().__init__()
        self.register_buffer("volume", torch.from_numpy(vol_zyx.astype(np.float32))[None, None])

    def forward(self, xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        x = xyz[:, 0] * 2.0 - 1.0
        y = xyz[:, 1] * 2.0 - 1.0
        z = xyz[:, 2] * 2.0 - 1.0
        grid = torch.stack([x, y, z], dim=-1).view(1, 1, 1, -1, 3)
        vals = F.grid_sample(self.volume, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
        return vals.view(-1, 1)


def _make_geo(bundle: np.lib.npyio.NpzFile) -> tigre.geometry:
    geo = tigre.geometry()
    geo.nVoxel = np.array([355, 280, 115], dtype=np.int32)
    geo.sVoxel = np.asarray(bundle["s_voxel"], dtype=np.float32).reshape(3) if "s_voxel" in bundle else np.array([355.0, 280.0, 345.0], dtype=np.float32)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.asarray(bundle["n_detector"], dtype=np.int32).reshape(2) if "n_detector" in bundle else np.array([256, 256], dtype=np.int32)
    geo.dDetector = np.asarray(bundle["d_detector"], dtype=np.float32).reshape(2) if "d_detector" in bundle else np.array([1.5, 1.5], dtype=np.float32)
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.DSO = float(np.asarray(bundle["sod"]).reshape(-1)[0]) if "sod" in bundle else 750.0
    geo.DSD = float(np.asarray(bundle["sdd"]).reshape(-1)[0]) if "sdd" in bundle else 1200.0
    geo.offOrigin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    geo.offDetector = np.array([0.0, 0.0], dtype=np.float32)
    geo.mode = "cone"
    return geo


def _source_detector_basis(angle: float, sod: float, sdd: float):
    ca, sa = math.cos(angle), math.sin(angle)
    src = np.array([sod * ca, sod * sa, 0.0], dtype=np.float32)
    u = np.array([ca, sa, 0.0], dtype=np.float32)  # detector plane normal
    det_center = src - u * sdd
    det_x = np.array([-sa, ca, 0.0], dtype=np.float32)
    det_y = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return src, det_center, det_x, det_y, u


def _intersect_ray_aabb(src: np.ndarray, dst: np.ndarray, box_min: np.ndarray, box_max: np.ndarray):
    d = dst - src
    inv = 1.0 / (d + 1e-9)
    t1 = (box_min - src) * inv
    t2 = (box_max - src) * inv
    tmin = np.max(np.minimum(t1, t2))
    tmax = np.min(np.maximum(t1, t2))
    t0 = max(tmin, 0.0)
    t1o = min(tmax, 1.0)
    if t1o <= t0:
        return None
    return t0, t1o


def _gamma_impulse(det_h: int, det_w: int, dh: float, dw: float, angle: float, sod: float, sdd: float, impulse_world: np.ndarray) -> np.ndarray:
    src, det_center, det_x, det_y, n = _source_detector_basis(angle, sod, sdd)
    ray = impulse_world - src
    denom = float(np.dot(n, ray))
    if abs(denom) < 1e-8:
        return np.zeros((det_h, det_w), dtype=np.float32)
    lam = float(np.dot(n, det_center - src) / denom)
    hit = src + lam * ray
    du = float(np.dot(hit - det_center, det_x))
    dv = float(np.dot(hit - det_center, det_y))
    u = int(round(du / dw + (det_w - 1) * 0.5))
    v = int(round(dv / dh + (det_h - 1) * 0.5))
    out = np.zeros((det_h, det_w), dtype=np.float32)
    if 0 <= u < det_w and 0 <= v < det_h:
        out[v, u] = 1.0
    return out


def _gamma_linear_z(det_h: int, det_w: int, dh: float, dw: float, angle: float, sod: float, sdd: float, s_voxel: np.ndarray) -> np.ndarray:
    src, det_center, det_x, det_y, _ = _source_detector_basis(angle, sod, sdd)
    Lx, Ly, Lz = float(s_voxel[0]), float(s_voxel[1]), float(s_voxel[2])
    box_min = np.array([-Lx / 2.0, -Ly / 2.0, -Lz / 2.0], dtype=np.float32)
    box_max = -box_min
    out = np.zeros((det_h, det_w), dtype=np.float32)
    for v in range(det_h):
        dv = (v - (det_h - 1) * 0.5) * dh
        for u in range(det_w):
            du = (u - (det_w - 1) * 0.5) * dw
            dst = det_center + du * det_x + dv * det_y
            seg = _intersect_ray_aabb(src, dst, box_min, box_max)
            if seg is None:
                continue
            t0, t1 = seg
            p0 = src + t0 * (dst - src)
            p1 = src + t1 * (dst - src)
            L = float(np.linalg.norm(p1 - p0))
            zmid = 0.5 * float(p0[2] + p1[2])
            out[v, u] = ((zmid / Lz) + 0.5) * L
    return out.astype(np.float32)


def _gamma_block(det_h: int, det_w: int, dh: float, dw: float, angle: float, sod: float, sdd: float, s_voxel: np.ndarray) -> np.ndarray:
    src, det_center, det_x, det_y, _ = _source_detector_basis(angle, sod, sdd)
    Lx, Ly, Lz = float(s_voxel[0]), float(s_voxel[1]), float(s_voxel[2])
    box_min = np.array([-Lx / 2.0, -Ly / 2.0, -Lz / 2.0], dtype=np.float32)
    box_max = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # octant block
    out = np.zeros((det_h, det_w), dtype=np.float32)
    for v in range(det_h):
        dv = (v - (det_h - 1) * 0.5) * dh
        for u in range(det_w):
            du = (u - (det_w - 1) * 0.5) * dw
            dst = det_center + du * det_x + dv * det_y
            seg = _intersect_ray_aabb(src, dst, box_min, box_max)
            if seg is None:
                continue
            t0, t1 = seg
            p0 = src + t0 * (dst - src)
            p1 = src + t1 * (dst - src)
            out[v, u] = float(np.linalg.norm(p1 - p0))
    return out.astype(np.float32)


def _render_beta(model: nn.Module, angle: float, geo, n_samples: int, volume_size, device: torch.device, chunk: int = 32768) -> np.ndarray:
    det_h, det_w = int(geo.nDetector[0]), int(geo.nDetector[1])
    uu, vv = np.meshgrid(np.arange(det_w, dtype=np.int64), np.arange(det_h, dtype=np.int64), indexing="xy")
    u = torch.from_numpy(uu.reshape(-1)).to(device=device)
    v = torch.from_numpy(vv.reshape(-1)).to(device=device)
    ang = torch.full((u.shape[0],), float(angle), device=device, dtype=torch.float32)
    pts_w, delta = build_ray_batch(
        angles=ang,
        u_idx=u,
        v_idx=v,
        det_h=det_h,
        det_w=det_w,
        det_spacing_h=float(geo.dDetector[0]),
        det_spacing_w=float(geo.dDetector[1]),
        sod=float(geo.DSO),
        sdd=float(geo.DSD),
        n_samples=int(n_samples),
        volume_size_mm=volume_size,
    )
    pts = world_to_unit(pts_w, volume_size_mm=volume_size).clamp(0.0, 1.0)
    cond = torch.zeros((u.shape[0], 1), device=device, dtype=torch.float32)
    out = torch.empty((u.shape[0], 1), device=device, dtype=torch.float32)
    for i in range(0, u.shape[0], int(chunk)):
        sl = slice(i, min(i + int(chunk), u.shape[0]))
        out[sl] = render_ray_batch(model, pts[sl], cond[sl], delta[sl], projection_mode="line_integral")
    return out[:, 0].view(det_h, det_w).detach().cpu().numpy().astype(np.float32)


def _save_three(panel_dir: Path, name: str, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray):
    for k, arr in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(arr, cmap="gray")
        plt.axis("off")
        plt.title(f"{name} {k}")
        plt.tight_layout()
        fig.savefig(panel_dir / f"{name}_{k}.png", dpi=180)
        plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(alpha, cmap="gray")
    axs[0].set_title("alpha TIGRE")
    axs[1].imshow(beta, cmap="gray")
    axs[1].set_title("beta renderer")
    axs[2].imshow(gamma, cmap="gray")
    axs[2].set_title("gamma analytic")
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(panel_dir / f"{name}_triplet.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--out_dir", default="results/step1_8_volume/V1_axis_probe")
    ap.add_argument("--n_samples", type=int, default=128)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = np.load(args.bundle)
    geo = _make_geo(bundle)
    angles = np.deg2rad(np.array([0.0, 90.0, 180.0], dtype=np.float32))

    # volume dims in ZYX
    nz, ny, nx = int(geo.nVoxel[2]), int(geo.nVoxel[1]), int(geo.nVoxel[0])
    sx, sy, sz = float(geo.sVoxel[0]), float(geo.sVoxel[1]), float(geo.sVoxel[2])
    volume_size = (sx, sy, sz)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"tests": []}

    def voxel_to_world(ix: int, iy: int, iz: int) -> np.ndarray:
        x = -sx / 2.0 + (ix + 0.5) * (sx / nx)
        y = -sy / 2.0 + (iy + 0.5) * (sy / ny)
        z = -sz / 2.0 + (iz + 0.5) * (sz / nz)
        return np.array([x, y, z], dtype=np.float32)

    tests = []
    # V1-a center impulse
    va = np.zeros((nz, ny, nx), dtype=np.float32)
    ixa, iya, iza = nx // 2, ny // 2, nz // 2
    va[iza, iya, ixa] = 1.0
    tests.append(("V1-a_center_impulse", va, ("impulse", voxel_to_world(ixa, iya, iza))))
    # V1-b z-offset impulse
    vb = np.zeros((nz, ny, nx), dtype=np.float32)
    ixb, iyb, izb = nx // 2, ny // 2, int(round(nz * 0.78))
    vb[izb, iyb, ixb] = 1.0
    tests.append(("V1-b_z_offset_impulse", vb, ("impulse", voxel_to_world(ixb, iyb, izb))))
    # V1-c linear z gradient
    zlin = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None, None]
    vc = np.broadcast_to(zlin, (nz, ny, nx)).copy()
    tests.append(("V1-c_linear_z", vc, ("linear_z", None)))
    # V1-d octant block
    vd = np.zeros((nz, ny, nx), dtype=np.float32)
    vd[: nz // 2, : ny // 2, : nx // 2] = 1.0
    tests.append(("V1-d_octant_block", vd, ("octant", None)))

    for name, vol_zyx, ginfo in tests:
        model = VoxelOracleModel(vol_zyx).to(device).eval()
        vol_xyz = np.transpose(vol_zyx, (2, 1, 0)).copy().astype(np.float32)
        test_rows = []
        for a in angles:
            alpha = tigre.Ax(vol_xyz, geo, np.array([a], dtype=np.float32))[0].astype(np.float32)
            if "projection_v_flipped" in bundle and int(np.asarray(bundle["projection_v_flipped"]).reshape(-1)[0]) != 0:
                alpha = alpha[::-1, :]
            beta = _render_beta(model, float(a), geo, n_samples=int(args.n_samples), volume_size=volume_size, device=device)
            if ginfo[0] == "impulse":
                gamma = _gamma_impulse(
                    det_h=int(geo.nDetector[0]),
                    det_w=int(geo.nDetector[1]),
                    dh=float(geo.dDetector[0]),
                    dw=float(geo.dDetector[1]),
                    angle=float(a),
                    sod=float(geo.DSO),
                    sdd=float(geo.DSD),
                    impulse_world=ginfo[1],
                )
            elif ginfo[0] == "linear_z":
                gamma = _gamma_linear_z(
                    det_h=int(geo.nDetector[0]),
                    det_w=int(geo.nDetector[1]),
                    dh=float(geo.dDetector[0]),
                    dw=float(geo.dDetector[1]),
                    angle=float(a),
                    sod=float(geo.DSO),
                    sdd=float(geo.DSD),
                    s_voxel=np.asarray(geo.sVoxel, dtype=np.float32),
                )
            else:
                gamma = _gamma_block(
                    det_h=int(geo.nDetector[0]),
                    det_w=int(geo.nDetector[1]),
                    dh=float(geo.dDetector[0]),
                    dw=float(geo.dDetector[1]),
                    angle=float(a),
                    sod=float(geo.DSO),
                    sdd=float(geo.DSD),
                    s_voxel=np.asarray(geo.sVoxel, dtype=np.float32),
                )
            tag = f"{name}_a{int(round(np.rad2deg(a))):03d}"
            _save_three(out_dir, tag, alpha, beta, gamma)
            test_rows.append(
                {
                    "angle_deg": float(np.rad2deg(a)),
                    "psnr_alpha_beta": float(_psnr(beta, alpha)),
                    "psnr_alpha_gamma": float(_psnr(gamma, alpha)),
                    "psnr_beta_gamma": float(_psnr(beta, gamma)),
                }
            )
        report["tests"].append({"name": name, "rows": test_rows})

    (out_dir / "V1_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved probe images and metrics to {out_dir}")


if __name__ == "__main__":
    main()
