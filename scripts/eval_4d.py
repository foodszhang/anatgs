"""Evaluate per-phase reconstruction quality for 4D experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.dynamic import ContinuousTimeField

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:  # pragma: no cover
    ssim = None


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _ssim3d(a: np.ndarray, b: np.ndarray) -> float:
    if ssim is None:
        return float("nan")
    return float(ssim(a, b, data_range=1.0))


def _centroid(mask: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    return idx.mean(axis=0).astype(np.float32)


def _resize_to_shape(vol: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if vol.shape == shape:
        return vol
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _jacobian_determinant_ratio(mapped: np.ndarray) -> float:
    """
    mapped: [D,H,W,3], unit-grid coordinates after deformation.
    Returns fraction of negative Jacobian determinants.
    """
    du_dz = np.gradient(mapped[..., 0], axis=0)
    du_dy = np.gradient(mapped[..., 0], axis=1)
    du_dx = np.gradient(mapped[..., 0], axis=2)
    dv_dz = np.gradient(mapped[..., 1], axis=0)
    dv_dy = np.gradient(mapped[..., 1], axis=1)
    dv_dx = np.gradient(mapped[..., 1], axis=2)
    dw_dz = np.gradient(mapped[..., 2], axis=0)
    dw_dy = np.gradient(mapped[..., 2], axis=1)
    dw_dx = np.gradient(mapped[..., 2], axis=2)
    j11, j12, j13 = du_dz, du_dy, du_dx
    j21, j22, j23 = dv_dz, dv_dy, dv_dx
    j31, j32, j33 = dw_dz, dw_dy, dw_dx
    det = (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )
    return float(np.mean(det < 0.0))


def _make_unit_grid(res: int, device: torch.device) -> torch.Tensor:
    g = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, res, device=device),
            torch.linspace(0.0, 1.0, res, device=device),
            torch.linspace(0.0, 1.0, res, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    return g.reshape(-1, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Dir containing pred_phase_00.npy ...")
    ap.add_argument("--gt_dir", required=True, help="Dir containing phase_00.npy ...")
    ap.add_argument("--gtv_mask", default="", help="Optional mask .npy for centroid trajectory")
    ap.add_argument("--model_ckpt", default="", help="Optional checkpoint for motion metrics")
    ap.add_argument("--motion_grid", type=int, default=40, help="Grid size for Jacobian/inverse metrics")
    ap.add_argument("--signal_dim", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    pred_paths = sorted(pred_dir.glob("pred_phase_*.npy"))
    gt_paths = sorted(gt_dir.glob("phase_*.npy"))
    if not pred_paths:
        raise FileNotFoundError(f"No pred_phase_*.npy under {pred_dir}")
    if not gt_paths:
        raise FileNotFoundError(f"No phase_*.npy under {gt_dir}")
    n = min(len(pred_paths), len(gt_paths))

    rows = []
    mask = np.load(args.gtv_mask).astype(np.uint8) if args.gtv_mask else None
    c_gt = _centroid(mask) if mask is not None else None
    motion_stats: dict[int, dict[str, float]] = {}

    if args.model_ckpt:
        ckpt = torch.load(args.model_ckpt, map_location="cpu")
        cfg = dict((ckpt.get("cfg", {}) or {}).get("model", {}))
        if "use_signal" not in cfg:
            cfg["use_signal"] = False
        if "signal_dim" not in cfg:
            cfg["signal_dim"] = int(args.signal_dim)
        model = ContinuousTimeField(cfg)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        res = int(args.motion_grid)
        xyz = _make_unit_grid(res, device=device)
        with torch.no_grad():
            for i in range(n):
                t = float(i) / 10.0
                if model.use_signal:
                    phase = 2.0 * np.pi * t
                    cond_np = np.zeros((1, int(cfg.get("signal_dim", args.signal_dim))), dtype=np.float32)
                    cond_np[0, 0] = np.sin(phase)
                    if cond_np.shape[1] > 1:
                        cond_np[0, 1] = np.cos(phase)
                    cond = torch.from_numpy(cond_np).to(device=device)
                else:
                    cond = torch.tensor([[t]], device=device, dtype=torch.float32)
                cond_full = cond.expand(xyz.shape[0], cond.shape[-1])
                mapped_inv = model.map_points(xyz, cond_full, inverse=True)
                mapped_fwd = model.map_points(xyz, cond_full, inverse=False)
                back = model.map_points(mapped_fwd, cond_full, inverse=True)
                inv_err = torch.linalg.norm(back - xyz, dim=-1).mean().item()
                mapped_grid = mapped_inv.reshape(res, res, res, 3).cpu().numpy()
                motion_stats[i] = {
                    "jacobian_negative_ratio": _jacobian_determinant_ratio(mapped_grid),
                    "invertibility_error": float(inv_err),
                }

    for i in range(n):
        pred = np.load(pred_paths[i]).astype(np.float32)
        gt = np.load(gt_paths[i]).astype(np.float32)
        if pred.shape != gt.shape:
            pred = _resize_to_shape(pred, gt.shape)
        row = {"phase": i, "psnr": _psnr(pred, gt), "ssim": _ssim3d(pred, gt)}
        if mask is not None:
            # Simple trajectory proxy: threshold on predicted volume inside GT mask extent.
            pred_bin = (pred > float(pred[mask > 0].mean())).astype(np.uint8) * mask
            c_pred = _centroid(pred_bin)
            row["tumor_tre_vox"] = float(np.linalg.norm(c_pred - c_gt))
        if i in motion_stats:
            row.update(motion_stats[i])
        rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-phase metrics to {out_path}")
    if args.model_ckpt:
        with (out_path.parent / f"{out_path.stem}_motion_summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "mean_jacobian_negative_ratio": float(np.mean([r.get("jacobian_negative_ratio", np.nan) for r in rows])),
                    "mean_invertibility_error": float(np.mean([r.get("invertibility_error", np.nan) for r in rows])),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
