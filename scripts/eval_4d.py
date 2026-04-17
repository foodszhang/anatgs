"""Evaluate per-phase reconstruction quality for 4D experiments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Dir containing pred_phase_00.npy ...")
    ap.add_argument("--gt_dir", required=True, help="Dir containing phase_00.npy ...")
    ap.add_argument("--gtv_mask", default="", help="Optional mask .npy for centroid trajectory")
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
            row["gtv_centroid_error_vox"] = float(np.linalg.norm(c_pred - c_gt))
        rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-phase metrics to {out_path}")


if __name__ == "__main__":
    main()
