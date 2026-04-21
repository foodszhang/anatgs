"""D1: head-to-head forward projection comparison (TIGRE vs model renderer)."""

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


class VoxelOracleModel(nn.Module):
    """Renderer-compatible model that samples from a fixed voxel volume."""

    def __init__(self, vol_zyx: np.ndarray):
        super().__init__()
        t = torch.from_numpy(np.asarray(vol_zyx, dtype=np.float32))
        self.register_buffer("volume", t[None, None])  # [1,1,D,H,W]

    def forward(self, xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        x = xyz[:, 0] * 2.0 - 1.0
        y = xyz[:, 1] * 2.0 - 1.0
        z = xyz[:, 2] * 2.0 - 1.0
        grid = torch.stack([x, y, z], dim=-1).view(1, 1, 1, -1, 3)
        vals = F.grid_sample(
            self.volume,
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )
        return vals.view(-1, 1)


def _make_geo(bundle: np.lib.npyio.NpzFile) -> tigre.geometry:
    geo = tigre.geometry()
    geo.nVoxel = np.array([355, 280, 115], dtype=np.int32)
    geo.sVoxel = np.array([355.0, 280.0, 345.0], dtype=np.float32)
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


def _render_view(model: nn.Module, angle: float, det_h: int, det_w: int, dh: float, dw: float, sod: float, sdd: float, n_samples: int, volume_size_mm: float | tuple[float, float, float], device: torch.device, chunk: int = 32768) -> np.ndarray:
    uu, vv = np.meshgrid(np.arange(det_w, dtype=np.int64), np.arange(det_h, dtype=np.int64), indexing="xy")
    u = torch.from_numpy(uu.reshape(-1)).to(device=device)
    v = torch.from_numpy(vv.reshape(-1)).to(device=device)
    a = torch.full((u.shape[0],), float(angle), device=device, dtype=torch.float32)
    pts_w, delta = build_ray_batch(
        angles=a,
        u_idx=u,
        v_idx=v,
        det_h=det_h,
        det_w=det_w,
        det_spacing_h=dh,
        det_spacing_w=dw,
        sod=sod,
        sdd=sdd,
        n_samples=n_samples,
        volume_size_mm=volume_size_mm,
    )
    pts = world_to_unit(pts_w, volume_size_mm=volume_size_mm).clamp(0.0, 1.0)
    cond = torch.zeros((u.shape[0], 1), device=device, dtype=torch.float32)
    out = torch.empty((u.shape[0], 1), device=device, dtype=torch.float32)
    for i in range(0, u.shape[0], int(chunk)):
        sl = slice(i, min(i + int(chunk), u.shape[0]))
        out[sl] = render_ray_batch(model, pts[sl], cond[sl], delta[sl], projection_mode="line_integral")
    return out[:, 0].view(det_h, det_w).detach().cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--timepoint", type=int, default=90)
    ap.add_argument("--n_views", type=int, default=16)
    ap.add_argument("--n_samples", type=int, default=128)
    ap.add_argument("--volume_size_mm", type=float, default=-1.0, help="If <0, use bundle s_voxel anisotropic size.")
    ap.add_argument("--projector", choices=["raymarch", "tigre_autograd"], default="raymarch")
    ap.add_argument("--out_json", default="results/step1_7_model/D1_forward_compare.json")
    ap.add_argument("--out_triplets", default="results/step1_7_model/D1_triplets")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_triplets = Path(args.out_triplets)
    out_triplets.mkdir(parents=True, exist_ok=True)

    b = np.load(args.bundle)
    projs_bundle = (b["projections"] if "projections" in b else b["projs"]).astype(np.float32)
    angles_all = to_radians(np.asarray(b["angles"], dtype=np.float32), angle_unit="auto")
    t_idx = b["t_idx_at_view"].astype(np.int32)
    sel = np.arange(min(int(args.n_views), angles_all.shape[0]), dtype=np.int32)
    angles = angles_all[sel]
    geo = _make_geo(b)
    if float(args.volume_size_mm) > 0:
        volume_size = float(args.volume_size_mm)
    elif "s_voxel" in b:
        sv = np.asarray(b["s_voxel"], dtype=np.float32).reshape(3)
        volume_size = (float(sv[0]), float(sv[1]), float(sv[2]))
    else:
        volume_size = 384.0

    p = Path(args.raw_dir) / "ground_truth" / "volumes" / f"volume_{int(args.timepoint)}.nii.gz"
    xyz = np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32), dtype=np.float32)
    gt_zyx = hu_to_mu(np.transpose(xyz, (2, 1, 0)).copy().astype(np.float32))
    gt_xyz = np.transpose(gt_zyx, (2, 1, 0)).copy().astype(np.float32)

    # Alpha: TIGRE forward
    alpha = tigre.Ax(gt_xyz, geo, angles).astype(np.float32)
    if "projection_v_flipped" in b and int(np.asarray(b["projection_v_flipped"]).reshape(-1)[0]) != 0:
        alpha = alpha[:, ::-1, :]

    # Beta: renderer forward with fixed voxel oracle (v=0, no warping path).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoxelOracleModel(gt_zyx).to(device).eval()
    if args.projector == "tigre_autograd":
        geo_dict = {
            "nVoxel": [int(geo.nVoxel[0]), int(geo.nVoxel[1]), int(geo.nVoxel[2])],
            "sVoxel": [float(geo.sVoxel[0]), float(geo.sVoxel[1]), float(geo.sVoxel[2])],
            "nDetector": [int(geo.nDetector[0]), int(geo.nDetector[1])],
            "dDetector": [float(geo.dDetector[0]), float(geo.dDetector[1])],
            "DSO": float(geo.DSO),
            "DSD": float(geo.DSD),
        }
        vol_xyz_t = torch.from_numpy(gt_xyz).to(device=device)
        beta = project_volume_tigre_autograd(vol_xyz_t, torch.from_numpy(angles).to(device=device), geo_dict).detach().cpu().numpy().astype(np.float32)
    else:
        beta = []
        for a in angles:
            pb = _render_view(
                model=model,
                angle=float(a),
                det_h=int(geo.nDetector[0]),
                det_w=int(geo.nDetector[1]),
                dh=float(geo.dDetector[0]),
                dw=float(geo.dDetector[1]),
                sod=float(geo.DSO),
                sdd=float(geo.DSD),
                n_samples=int(args.n_samples),
                volume_size_mm=volume_size,
                device=device,
            )
            beta.append(pb)
        beta = np.stack(beta, axis=0).astype(np.float32)

    rows = []
    for i in range(alpha.shape[0]):
        a = alpha[i]
        bb = beta[i]
        mse = float(np.mean((bb - a) ** 2))
        rows.append(
            {
                "view_index": int(sel[i]),
                "time_index_at_view": int(t_idx[sel[i]]),
                "angle_deg": float(np.rad2deg(angles[i])),
                "mse_beta_vs_alpha": mse,
                "psnr_beta_vs_alpha": float(_psnr(bb, a)),
                "mean_ratio_beta_over_alpha": float(np.mean(bb) / (np.mean(a) + 1e-12)),
                "std_ratio_beta_over_alpha": float(np.std(bb) / (np.std(a) + 1e-12)),
            }
        )

    # Triplets closest to 0/90/180 degrees.
    target_deg = [0.0, 90.0, 180.0]
    angles_deg_all = np.rad2deg(angles)
    for td in target_deg:
        k = int(np.argmin(np.abs(((angles_deg_all - td + 180.0) % 360.0) - 180.0)))
        a = alpha[k]
        bb = beta[k]
        diff = bb - a
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(a, cmap="gray")
        axs[0].set_title(f"alpha TIGRE ({angles_deg_all[k]:.1f}deg)")
        axs[1].imshow(bb, cmap="gray")
        axs[1].set_title("beta renderer")
        im = axs[2].imshow(diff, cmap="bwr")
        axs[2].set_title("beta-alpha")
        for ax in axs:
            ax.axis("off")
        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_triplets / f"triplet_target_{int(td):03d}.png", dpi=180)
        plt.close(fig)

    out = {
        "setup": {
            "timepoint_gt": int(args.timepoint),
            "selected_view_indices": [int(x) for x in sel.tolist()],
            "selected_angle_deg": [float(x) for x in angles_deg_all.tolist()],
            "n_samples_renderer": int(args.n_samples),
            "volume_size_mm_renderer": volume_size,
            "projector": args.projector,
        },
        "per_view": rows,
        "summary": {
            "mean_psnr_beta_vs_alpha": float(np.mean([r["psnr_beta_vs_alpha"] for r in rows])),
            "mean_mse_beta_vs_alpha": float(np.mean([r["mse_beta_vs_alpha"] for r in rows])),
            "global_mean_ratio_beta_over_alpha": float(np.mean([r["mean_ratio_beta_over_alpha"] for r in rows])),
            "global_std_ratio_beta_over_alpha": float(np.mean([r["std_ratio_beta_over_alpha"] for r in rows])),
        },
    }
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
