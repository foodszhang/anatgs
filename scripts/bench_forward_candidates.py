"""Phase-A benchmark for candidate forward projectors against TIGRE reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tigre
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu
from anatgs.dynamic import project_volume_tigre_autograd, render_ray_batch
from anatgs.dynamic.ray import build_ray_batch, world_to_unit
from anatgs.geom import to_radians


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    dr = float(np.max(b) - np.min(b) + 1e-8)
    return float(10.0 * math.log10((dr * dr) / mse))


class _VoxelModel(nn.Module):
    def __init__(self, vol_zyx: torch.Tensor):
        super().__init__()
        self.volume = nn.Parameter(vol_zyx[None, None].float())

    def forward(self, xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        x = xyz[:, 0] * 2.0 - 1.0
        y = xyz[:, 1] * 2.0 - 1.0
        z = xyz[:, 2] * 2.0 - 1.0
        grid = torch.stack([x, y, z], dim=-1).view(1, 1, 1, -1, 3)
        vals = F.grid_sample(self.volume, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
        return vals.view(-1, 1)


def _make_geo(bundle: np.lib.npyio.NpzFile):
    geo = tigre.geometry()
    n_voxel = np.array([355, 280, 115], dtype=np.int32)
    s_voxel = np.asarray(bundle["s_voxel"], dtype=np.float32).reshape(3)
    n_detector = np.asarray(bundle["n_detector"], dtype=np.int32).reshape(2)
    d_detector = np.asarray(bundle["d_detector"], dtype=np.float32).reshape(2)
    geo.nVoxel = n_voxel
    geo.sVoxel = s_voxel
    geo.dVoxel = s_voxel / n_voxel
    geo.nDetector = n_detector
    geo.dDetector = d_detector
    geo.sDetector = n_detector * d_detector
    geo.DSO = float(np.asarray(bundle["sod"]).reshape(-1)[0])
    geo.DSD = float(np.asarray(bundle["sdd"]).reshape(-1)[0])
    geo.offOrigin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    geo.offDetector = np.array([0.0, 0.0], dtype=np.float32)
    geo.mode = "cone"
    geo_dict = {
        "nVoxel": n_voxel.tolist(),
        "sVoxel": s_voxel.tolist(),
        "nDetector": n_detector.tolist(),
        "dDetector": d_detector.tolist(),
        "DSO": float(geo.DSO),
        "DSD": float(geo.DSD),
    }
    return geo, geo_dict


def _render_raymarch(vol_zyx: torch.Tensor, angles: np.ndarray, geo, n_samples: int, device: torch.device):
    model = _VoxelModel(vol_zyx.to(device)).to(device)
    h, w = int(geo.nDetector[0]), int(geo.nDetector[1])
    uu, vv = np.meshgrid(np.arange(w, dtype=np.int64), np.arange(h, dtype=np.int64), indexing="xy")
    u = torch.from_numpy(uu.reshape(-1)).to(device=device)
    v = torch.from_numpy(vv.reshape(-1)).to(device=device)
    cond = torch.zeros((u.shape[0], 1), device=device)
    out_views = []
    for a in angles:
        ang = torch.full((u.shape[0],), float(a), device=device)
        pts_w, delta = build_ray_batch(
            angles=ang,
            u_idx=u,
            v_idx=v,
            det_h=h,
            det_w=w,
            det_spacing_h=float(geo.dDetector[0]),
            det_spacing_w=float(geo.dDetector[1]),
            sod=float(geo.DSO),
            sdd=float(geo.DSD),
            n_samples=n_samples,
            volume_size_mm=(float(geo.sVoxel[0]), float(geo.sVoxel[1]), float(geo.sVoxel[2])),
        )
        pts = world_to_unit(pts_w, volume_size_mm=(float(geo.sVoxel[0]), float(geo.sVoxel[1]), float(geo.sVoxel[2]))).clamp(0.0, 1.0)
        pred = render_ray_batch(model, pts, cond, delta, projection_mode="line_integral")[:, 0].view(h, w)
        out_views.append(pred)
    proj = torch.stack(out_views, dim=0)
    loss = proj.mean()
    loss.backward()
    gnorm = float(model.volume.grad.norm().detach().cpu().item())
    return proj.detach().cpu().numpy().astype(np.float32), gnorm


def _render_tigre_autograd(vol_zyx: torch.Tensor, angles: np.ndarray, geo_dict: dict, device: torch.device):
    vol_xyz = torch.from_numpy(np.transpose(vol_zyx.cpu().numpy(), (2, 1, 0)).copy()).to(device=device).requires_grad_(True)
    ang = torch.from_numpy(angles.astype(np.float32)).to(device=device)
    proj = project_volume_tigre_autograd(vol_xyz, ang, geo_dict)
    loss = proj.mean()
    loss.backward()
    gnorm = float(vol_xyz.grad.norm().detach().cpu().item())
    return proj.detach().cpu().numpy().astype(np.float32), gnorm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--timepoint", type=int, default=90)
    ap.add_argument("--n_views", type=int, default=16)
    ap.add_argument("--n_samples", type=int, default=128)
    ap.add_argument("--out_dir", default="results/step1_9_renderer")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "A2_triplets").mkdir(parents=True, exist_ok=True)

    b = np.load(args.bundle)
    angles = to_radians(np.asarray(b["angles"], dtype=np.float32), angle_unit="auto")[: int(args.n_views)]
    geo, geo_dict = _make_geo(b)

    p = Path(args.raw_dir) / "ground_truth" / "volumes" / f"volume_{int(args.timepoint)}.nii.gz"
    xyz = np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32), dtype=np.float32)
    vol_zyx = torch.from_numpy(hu_to_mu(np.transpose(xyz, (2, 1, 0)).copy().astype(np.float32)))
    vol_xyz = np.transpose(vol_zyx.numpy(), (2, 1, 0)).copy().astype(np.float32)
    alpha = tigre.Ax(vol_xyz, geo, angles).astype(np.float32)
    if "projection_v_flipped" in b and int(np.asarray(b["projection_v_flipped"]).reshape(-1)[0]) != 0:
        alpha = alpha[:, ::-1, :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = []

    # Candidate 1: current ray-march renderer
    proj_rm, grad_rm = _render_raymarch(vol_zyx, angles, geo, n_samples=int(args.n_samples), device=device)
    candidates.append(("renderer_raymarch", proj_rm, grad_rm))

    # Candidate 2: TIGRE autograd wrapper
    proj_ta, grad_ta = _render_tigre_autograd(vol_zyx, angles, geo_dict, device=device)
    if "projection_v_flipped" in b and int(np.asarray(b["projection_v_flipped"]).reshape(-1)[0]) != 0:
        proj_ta = proj_ta[:, ::-1, :]
    candidates.append(("tigre_autograd", proj_ta, grad_ta))

    # Optional probes for external libraries (availability only in this bench).
    availability = {"LEAP": False, "tomosipo": False, "diffdrr": False}
    try:
        import leapctype  # noqa: F401

        availability["LEAP"] = True
    except Exception:
        pass
    try:
        import tomosipo  # noqa: F401

        availability["tomosipo"] = True
    except Exception:
        pass
    try:
        import diffdrr  # noqa: F401

        availability["diffdrr"] = True
    except Exception:
        pass

    summary = {"availability": availability, "candidates": []}
    for name, proj, gnorm in candidates:
        per_view = []
        for i in range(proj.shape[0]):
            a = alpha[i]
            p2 = proj[i]
            per_view.append(
                {
                    "view": int(i),
                    "angle_deg": float(np.rad2deg(angles[i])),
                    "mse": float(np.mean((p2 - a) ** 2)),
                    "psnr": float(_psnr(p2, a)),
                    "mean_ratio": float(np.mean(p2) / (np.mean(a) + 1e-12)),
                    "std_ratio": float(np.std(p2) / (np.std(a) + 1e-12)),
                }
            )
        mpsnr = float(np.mean([x["psnr"] for x in per_view]))
        mmr = float(np.mean([x["mean_ratio"] for x in per_view]))
        msr = float(np.mean([x["std_ratio"] for x in per_view]))
        summary["candidates"].append(
            {
                "name": name,
                "mean_psnr_vs_tigre": mpsnr,
                "mean_ratio_vs_tigre": mmr,
                "std_ratio_vs_tigre": msr,
                "grad_norm": float(gnorm),
                "per_view": per_view,
            }
        )
        # triplets for 0/90/180 closest in selected views
        angles_deg = np.rad2deg(angles)
        for td in [0.0, 90.0, 180.0]:
            k = int(np.argmin(np.abs(((angles_deg - td + 180.0) % 360.0) - 180.0)))
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(alpha[k], cmap="gray")
            axs[0].set_title("TIGRE")
            axs[1].imshow(proj[k], cmap="gray")
            axs[1].set_title(name)
            axs[2].imshow(proj[k] - alpha[k], cmap="bwr")
            axs[2].set_title("diff")
            for ax in axs:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / "A2_triplets" / f"{name}_a{int(td):03d}.png", dpi=180)
            plt.close(fig)

    (out_dir / "A2_alignment_bench.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
