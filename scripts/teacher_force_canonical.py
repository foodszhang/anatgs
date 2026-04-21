"""D2: teacher-forcing canonical INR fit and projection sanity."""

from __future__ import annotations

import argparse
import json
import math
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu
from anatgs.dynamic import ContinuousTimeField, project_volume_tigre_autograd, render_ray_batch
from anatgs.dynamic.ray import build_ray_batch, world_to_unit
from anatgs.geom import to_radians


def _psnr(a: np.ndarray, b: np.ndarray, data_range_mode: str = "maxmin") -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    if data_range_mode == "target_max":
        dr = float(np.max(b) + 1e-8)
    elif data_range_mode == "unit":
        dr = 1.0
    else:
        dr = float(np.max(b) - np.min(b) + 1e-8)
    return float(10.0 * math.log10((dr * dr) / mse))


def _query_model_on_grid(model: ContinuousTimeField, shape_zyx: tuple[int, int, int], device: torch.device, chunk: int = 262144) -> np.ndarray:
    d, h, w = [int(x) for x in shape_zyx]
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    xyz = torch.stack(
        [
            (xx.reshape(-1).float() + 0.5) / float(w),
            (yy.reshape(-1).float() + 0.5) / float(h),
            (zz.reshape(-1).float() + 0.5) / float(d),
        ],
        dim=-1,
    )
    cond = torch.zeros((1, 1), device=device, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, xyz.shape[0], int(chunk)):
            p = xyz[i : i + int(chunk)]
            c = cond.expand(p.shape[0], 1)
            out.append(model(p, c).squeeze(-1))
    return torch.cat(out, dim=0).reshape(d, h, w).cpu().numpy().astype(np.float32)


def _resize(vol: np.ndarray, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    if vol.shape == shape_zyx:
        return vol.astype(np.float32)
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape_zyx, mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)


def _render_views(model: ContinuousTimeField, angles: np.ndarray, det_h: int, det_w: int, dh: float, dw: float, sod: float, sdd: float, n_samples: int, volume_size_mm: float | tuple[float, float, float], device: torch.device, chunk: int = 32768) -> np.ndarray:
    uu, vv = np.meshgrid(np.arange(det_w, dtype=np.int64), np.arange(det_h, dtype=np.int64), indexing="xy")
    u = torch.from_numpy(uu.reshape(-1)).to(device=device)
    v = torch.from_numpy(vv.reshape(-1)).to(device=device)
    cond = torch.zeros((u.shape[0], 1), device=device, dtype=torch.float32)
    preds = []
    for a in angles:
        ang = torch.full((u.shape[0],), float(a), device=device, dtype=torch.float32)
        pts_w, delta = build_ray_batch(
            angles=ang,
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
        out = torch.empty((u.shape[0], 1), device=device, dtype=torch.float32)
        for i in range(0, u.shape[0], int(chunk)):
            sl = slice(i, min(i + int(chunk), u.shape[0]))
            out[sl] = render_ray_batch(model, pts[sl], cond[sl], delta[sl], projection_mode="line_integral")
        preds.append(out[:, 0].view(det_h, det_w).detach().cpu().numpy().astype(np.float32))
    return np.stack(preds, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--config", default="configs/4d_continuous.yaml")
    ap.add_argument("--timepoint", type=int, default=90)
    ap.add_argument("--fit_iters", type=int, default=4000)
    ap.add_argument("--fit_batch", type=int, default=131072)
    ap.add_argument("--fit_lr", type=float, default=1e-3)
    ap.add_argument("--n_views", type=int, default=16)
    ap.add_argument("--n_samples", type=int, default=128)
    ap.add_argument("--volume_size_mm", type=float, default=-1.0, help="If <0, use bundle s_voxel anisotropic size.")
    ap.add_argument("--projector", choices=["raymarch", "tigre_autograd"], default="raymarch")
    ap.add_argument("--psnr_data_range", choices=["maxmin", "target_max", "unit"], default="target_max")
    ap.add_argument("--target_norm", choices=["none", "max"], default="none")
    ap.add_argument("--grid_lr", type=float, default=-1.0, help="If >0, use split-LR for hash-grid params.")
    ap.add_argument("--out_json", default="results/step1_7_model/D2_teacher_force.json")
    ap.add_argument("--out_slices", default="results/step1_7_model/D2_volume_slices.png")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_slices = Path(args.out_slices)
    out_slices.parent.mkdir(parents=True, exist_ok=True)

    bundle = np.load(args.bundle)
    projs = (bundle["projections"] if "projections" in bundle else bundle["projs"]).astype(np.float32)
    angles = to_radians(np.asarray(bundle["angles"], dtype=np.float32), angle_unit="auto")
    t_idx = bundle["t_idx_at_view"].astype(np.int32)
    if float(args.volume_size_mm) > 0:
        volume_size = float(args.volume_size_mm)
    elif "s_voxel" in bundle:
        sv = np.asarray(bundle["s_voxel"], dtype=np.float32).reshape(3)
        volume_size = (float(sv[0]), float(sv[1]), float(sv[2]))
    else:
        volume_size = 384.0

    p = Path(args.raw_dir) / "ground_truth" / "volumes" / f"volume_{int(args.timepoint)}.nii.gz"
    xyz = np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32), dtype=np.float32)
    gt = hu_to_mu(np.transpose(xyz, (2, 1, 0)).copy().astype(np.float32))
    mu_scale = float(np.max(gt) + 1e-8)
    gt_fit = (gt / mu_scale).astype(np.float32) if args.target_norm == "max" else gt.astype(np.float32)
    d, h, w = [int(x) for x in gt.shape]

    # Build static canonical model.
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    mcfg = dict(cfg.get("model", {}))
    mcfg["use_signal"] = False
    mcfg["use_motion_field"] = False
    mcfg["use_svf"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuousTimeField(mcfg).to(device)
    if float(args.grid_lr) > 0:
        grid_params = list(model.spatial_encoder.parameters())
        grid_ids = {id(p) for p in grid_params}
        other_params = [p for p in model.parameters() if id(p) not in grid_ids]
        opt = torch.optim.Adam(
            [
                {"params": grid_params, "lr": float(args.grid_lr)},
                {"params": other_params, "lr": float(args.fit_lr)},
            ]
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(args.fit_lr))
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_opt_params = int(sum(p.numel() for g in opt.param_groups for p in g["params"]))

    gt_t = torch.from_numpy(gt_fit).to(device=device)
    fit_log = []
    for it in range(1, int(args.fit_iters) + 1):
        idx = torch.randint(0, int(gt_t.numel()), (int(args.fit_batch),), device=device)
        z = idx // int(h * w)
        y = (idx // int(w)) % int(h)
        x = idx % int(w)
        xyz_s = torch.stack(
            [
                (x.float() + 0.5) / float(w),
                (y.float() + 0.5) / float(h),
                (z.float() + 0.5) / float(d),
            ],
            dim=-1,
        )
        cond = torch.zeros((xyz_s.shape[0], 1), device=device, dtype=torch.float32)
        pred = model(xyz_s, cond).squeeze(-1)
        target = gt_t[z, y, x]
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it == 1 or it % 200 == 0:
            fit_log.append({"iter": int(it), "loss": float(loss.item())})

    pred_vol_fit = _query_model_on_grid(model, gt.shape, device=device)
    pred_vol = (pred_vol_fit * mu_scale).astype(np.float32) if args.target_norm == "max" else pred_vol_fit
    vol_psnr = _psnr(pred_vol, gt, data_range_mode=str(args.psnr_data_range))
    best_psnr = float(vol_psnr)
    best_tf = {"perm": [0, 1, 2], "flip": [0, 0, 0]}
    for perm in itertools.permutations([0, 1, 2]):
        p = np.transpose(pred_vol, perm)
        for fl in itertools.product([0, 1], [0, 1], [0, 1]):
            q = p
            if fl[0]:
                q = np.flip(q, axis=0)
            if fl[1]:
                q = np.flip(q, axis=1)
            if fl[2]:
                q = np.flip(q, axis=2)
            pv = _psnr(_resize(q, gt.shape), gt, data_range_mode=str(args.psnr_data_range))
            if pv > best_psnr:
                best_psnr = float(pv)
                best_tf = {"perm": [int(x) for x in perm], "flip": [int(x) for x in fl]}

    # Teacher-forced projection check on 16 nearest views to timepoint.
    sel = np.argsort(np.abs(t_idx - int(args.timepoint)))[: int(args.n_views)]
    sel = np.sort(sel).astype(np.int32)
    angles_sel = angles[sel]
    target_proj = projs[sel]
    if args.projector == "tigre_autograd":
        geo_dict = {
            "nVoxel": [int(w), int(h), int(d)],
            "sVoxel": [float(np.asarray(bundle["s_voxel"]).reshape(3)[0]), float(np.asarray(bundle["s_voxel"]).reshape(3)[1]), float(np.asarray(bundle["s_voxel"]).reshape(3)[2])] if "s_voxel" in bundle else [355.0, 280.0, 345.0],
            "nDetector": [int(target_proj.shape[1]), int(target_proj.shape[2])],
            "dDetector": [float(np.asarray(bundle["d_detector"]).reshape(2)[0]), float(np.asarray(bundle["d_detector"]).reshape(2)[1])] if "d_detector" in bundle else [1.5, 1.5],
            "DSO": float(np.asarray(bundle["sod"]).reshape(-1)[0]) if "sod" in bundle else 750.0,
            "DSD": float(np.asarray(bundle["sdd"]).reshape(-1)[0]) if "sdd" in bundle else 1200.0,
        }
        vol_xyz = torch.from_numpy(np.transpose(pred_vol, (2, 1, 0)).copy()).to(device=device)
        pred_proj = project_volume_tigre_autograd(vol_xyz, torch.from_numpy(angles_sel.astype(np.float32)).to(device=device), geo_dict).detach().cpu().numpy().astype(np.float32)
    else:
        pred_proj = _render_views(
            model=model,
            angles=angles_sel,
            det_h=int(target_proj.shape[1]),
            det_w=int(target_proj.shape[2]),
            dh=float(np.asarray(bundle["d_detector"]).reshape(2)[0]) if "d_detector" in bundle else 1.5,
            dw=float(np.asarray(bundle["d_detector"]).reshape(2)[1]) if "d_detector" in bundle else 1.5,
            sod=float(np.asarray(bundle["sod"]).reshape(-1)[0]) if "sod" in bundle else 750.0,
            sdd=float(np.asarray(bundle["sdd"]).reshape(-1)[0]) if "sdd" in bundle else 1200.0,
            n_samples=int(args.n_samples),
            volume_size_mm=volume_size,
            device=device,
        )
    per_view_psnr = [_psnr(pred_proj[i], target_proj[i], data_range_mode=str(args.psnr_data_range)) for i in range(pred_proj.shape[0])]
    proj_psnr = float(np.mean(per_view_psnr))

    # Slice comparison figure.
    zmid = gt.shape[0] // 2
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(gt[zmid], cmap="gray")
    axs[0].set_title("GT t=90")
    axs[1].imshow(pred_vol[zmid], cmap="gray")
    axs[1].set_title("INR fit")
    diff = pred_vol[zmid] - gt[zmid]
    im = axs[2].imshow(diff, cmap="bwr")
    axs[2].set_title("diff")
    for ax in axs:
        ax.axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_slices, dpi=180)
    plt.close(fig)

    out = {
        "volume_fit": {
            "timepoint": int(args.timepoint),
            "fit_iters": int(args.fit_iters),
            "fit_batch": int(args.fit_batch),
            "fit_lr": float(args.fit_lr),
            "target_norm": str(args.target_norm),
            "mu_scale": mu_scale,
            "grid_lr": float(args.grid_lr),
            "n_params": n_params,
            "n_opt_params": n_opt_params,
            "psnr_data_range": str(args.psnr_data_range),
            "psnr_inr_vs_gt": float(vol_psnr),
            "best_psnr_under_perm_flip": float(best_psnr),
            "best_perm_flip": best_tf,
            "fit_log": fit_log,
        },
        "teacher_projection": {
            "projector": args.projector,
            "n_views": int(args.n_views),
            "view_indices": [int(x) for x in sel.tolist()],
            "mean_psnr_pred_vs_bundle": float(proj_psnr),
            "per_view_psnr": [float(x) for x in per_view_psnr],
        },
    }
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"psnr_inr_vs_gt": vol_psnr, "teacher_proj_psnr": proj_psnr}, indent=2))


if __name__ == "__main__":
    main()
