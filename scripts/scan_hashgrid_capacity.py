"""A: hash-grid capacity scan on pure voxel L2 fit (GT t=90)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu
from anatgs.dynamic import ContinuousTimeField


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


def _query_model_grid(model: ContinuousTimeField, shape_zyx: tuple[int, int, int], device: torch.device, chunk: int = 262144) -> np.ndarray:
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


def _per_level_scale(base_res: int, finest_res: int, levels: int) -> float:
    if levels <= 1:
        return 1.0
    return float(np.exp(np.log(float(finest_res) / float(base_res)) / float(levels - 1)))


def _sample_pred_target_stats(
    model: ContinuousTimeField,
    gt_t: torch.Tensor,
    shape_zyx: tuple[int, int, int],
    device: torch.device,
    n_samples: int = 500000,
) -> dict:
    d, h, w = [int(x) for x in shape_zyx]
    idx = torch.randint(0, int(gt_t.numel()), (int(n_samples),), device=device)
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
    with torch.no_grad():
        pred = model(xyz_s, cond).squeeze(-1)
    target = gt_t[z, y, x]
    return {
        "pred": {
            "min": float(pred.min().item()),
            "max": float(pred.max().item()),
            "mean": float(pred.mean().item()),
        },
        "target": {
            "min": float(target.min().item()),
            "max": float(target.max().item()),
            "mean": float(target.mean().item()),
        },
    }


def _position_dependency(model: ContinuousTimeField, device: torch.device, n_points: int = 4096) -> dict:
    x1 = torch.rand((int(n_points), 3), device=device)
    x2 = torch.rand((int(n_points), 3), device=device)
    cond = torch.zeros((int(n_points), 1), device=device, dtype=torch.float32)
    with torch.no_grad():
        y1 = model(x1, cond)
        y2 = model(x2, cond)
    return {
        "mean_abs_y1_minus_y2": float((y1 - y2).abs().mean().item()),
        "std_y1": float(y1.std().item()),
        "mean_y1": float(y1.mean().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--timepoint", type=int, default=90)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=131072)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--config_names", default="", help="Comma-separated config names to run (default: all)")
    ap.add_argument("--output_activation", choices=["softplus", "linear"], default="softplus")
    ap.add_argument("--output_scale", type=float, default=1.0)
    ap.add_argument("--output_max", type=float, default=10.0)
    ap.add_argument("--psnr_data_range", choices=["maxmin", "target_max", "unit"], default="target_max")
    ap.add_argument("--target_norm", choices=["none", "max"], default="none", help="Normalize fit target before training.")
    ap.add_argument("--grid_lr", type=float, default=-1.0, help="If >0, use split-LR optimizer for spatial hash-grid params.")
    ap.add_argument("--out_json", default="results/step1_10_canonical/A_hashgrid_scan.json")
    ap.add_argument("--out_curves", default="results/step1_10_canonical/A_curves.png")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_curves = Path(args.out_curves)
    out_curves.parent.mkdir(parents=True, exist_ok=True)

    p = Path(args.raw_dir) / "ground_truth" / "volumes" / f"volume_{int(args.timepoint)}.nii.gz"
    xyz = np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32), dtype=np.float32)
    gt = hu_to_mu(np.transpose(xyz, (2, 1, 0)).copy().astype(np.float32))
    d, h, w = [int(x) for x in gt.shape]

    configs = [
        {"name": "C0_baseline", "log2_hash": 19, "levels": 14, "feat": 2, "base_res": 16, "finest_res": 256},
        {"name": "C1", "log2_hash": 21, "levels": 14, "feat": 2, "base_res": 16, "finest_res": 256},
        {"name": "C2", "log2_hash": 21, "levels": 16, "feat": 2, "base_res": 16, "finest_res": 512},
        {"name": "C3", "log2_hash": 21, "levels": 16, "feat": 4, "base_res": 16, "finest_res": 512},
        {"name": "C4_aggressive", "log2_hash": 22, "levels": 16, "feat": 4, "base_res": 16, "finest_res": 768},
    ]
    if args.config_names.strip():
        wanted = {x.strip() for x in args.config_names.split(",") if x.strip()}
        configs = [c for c in configs if c["name"] in wanted]
        if not configs:
            raise ValueError(f"No config matched --config_names={args.config_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu_scale = float(np.max(gt) + 1e-8)
    if args.target_norm == "max":
        gt_fit = (gt / mu_scale).astype(np.float32)
    else:
        gt_fit = gt.astype(np.float32)
    gt_t = torch.from_numpy(gt_fit).to(device=device)

    results = []
    plt.figure(figsize=(9, 5))
    for cfg in configs:
        mcfg = {
            "use_tcnn": True,
            "time_enc": "pe",
            "time_n_freqs": 8,
            "hidden_dim": 128,
            "use_signal": False,
            "signal_dim": 1,
            "use_motion_field": False,
            "use_svf": False,
            "hash_levels": int(cfg["levels"]),
            "hash_feat_per_level": int(cfg["feat"]),
            "hash_log2_size": int(cfg["log2_hash"]),
            "hash_base_resolution": int(cfg["base_res"]),
            "hash_per_level_scale": _per_level_scale(int(cfg["base_res"]), int(cfg["finest_res"]), int(cfg["levels"])),
            "output_activation": str(args.output_activation),
            "output_scale": float(args.output_scale),
            "output_max": float(args.output_max),
        }
        model = ContinuousTimeField(mcfg).to(device)
        if float(args.grid_lr) > 0:
            grid_params = list(model.spatial_encoder.parameters())
            grid_ids = {id(p) for p in grid_params}
            other_params = [p for p in model.parameters() if id(p) not in grid_ids]
            opt = torch.optim.Adam(
                [
                    {"params": grid_params, "lr": float(args.grid_lr)},
                    {"params": other_params, "lr": float(args.lr)},
                ]
            )
        else:
            opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
        n_params = int(sum(p.numel() for p in model.parameters()))
        n_opt_params = int(sum(p.numel() for g in opt.param_groups for p in g["params"]))

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)
        curve = []
        init_stats = _sample_pred_target_stats(model, gt_t, gt.shape, device=device)
        init_pos_dep = _position_dependency(model, device=device)
        for it in range(1, int(args.iters) + 1):
            idx = torch.randint(0, int(gt_t.numel()), (int(args.batch),), device=device)
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
            if it == 1 or it % 50 == 0:
                curve.append({"iter": int(it), "loss": float(loss.item())})

        pred_vol_fit = _query_model_grid(model, gt.shape, device=device)
        pred_vol = (pred_vol_fit * mu_scale).astype(np.float32) if args.target_norm == "max" else pred_vol_fit
        final_stats = _sample_pred_target_stats(model, gt_t, gt.shape, device=device)
        final_pos_dep = _position_dependency(model, device=device)
        psnr = float(_psnr(pred_vol, gt, data_range_mode=str(args.psnr_data_range)))
        psnr_maxmin = float(_psnr(pred_vol, gt, data_range_mode="maxmin"))
        psnr_target_max = float(_psnr(pred_vol, gt, data_range_mode="target_max"))
        psnr_unit = float(_psnr(pred_vol, gt, data_range_mode="unit"))
        max_mem_mb = (
            float(torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0))
            if torch.cuda.is_available()
            else float("nan")
        )
        results.append(
            {
                "config": cfg,
                "per_level_scale": float(mcfg["hash_per_level_scale"]),
                "n_params": n_params,
                "n_opt_params": n_opt_params,
                "peak_gpu_mem_mb": max_mem_mb,
                "target_norm": str(args.target_norm),
                "mu_scale": mu_scale,
                "grid_lr": float(args.grid_lr),
                "psnr": psnr,
                "psnr_maxmin": psnr_maxmin,
                "psnr_target_max": psnr_target_max,
                "psnr_unit": psnr_unit,
                "init_stats": init_stats,
                "final_stats": final_stats,
                "init_position_dependency": init_pos_dep,
                "final_position_dependency": final_pos_dep,
                "curve": curve,
            }
        )
        plt.plot([x["iter"] for x in curve], [x["loss"] for x in curve], label=f"{cfg['name']} ({psnr:.2f}dB)")

    plt.yscale("log")
    plt.xlabel("iter")
    plt.ylabel("voxel L2 loss")
    plt.title("Hash-grid capacity scan curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_curves, dpi=180)
    plt.close()

    best = max(results, key=lambda x: x["psnr"]) if results else None
    out = {"timepoint": int(args.timepoint), "iters": int(args.iters), "batch": int(args.batch), "lr": float(args.lr), "results": results, "best": best}
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"best": {"name": best["config"]["name"], "psnr": best["psnr"]} if best else None}, indent=2))


if __name__ == "__main__":
    main()
