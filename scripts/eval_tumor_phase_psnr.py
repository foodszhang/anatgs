"""Compute phase-wise tumor-region PSNR from per-phase masks."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _resize_to_shape(vol: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if vol.shape == shape:
        return vol
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    mask_dir = Path(args.mask_dir)
    rows = []
    for i in range(10):
        pred_p = pred_dir / f"pred_phase_{i:02d}.npy"
        gt_p = gt_dir / f"phase_{i:02d}.npy"
        m_p = mask_dir / f"gtv_phase_{i:02d}.npy"
        if not (pred_p.exists() and gt_p.exists() and m_p.exists()):
            continue
        pred = np.load(pred_p).astype(np.float32)
        gt = np.load(gt_p).astype(np.float32)
        m = np.load(m_p).astype(np.uint8) > 0
        if pred.shape != gt.shape:
            pred = _resize_to_shape(pred, gt.shape)
        n = int(m.sum())
        if n < 10:
            continue
        rows.append(
            {
                "phase": i,
                "voxels": n,
                "tumor_psnr": _psnr(pred[m], gt[m]),
                "gt_mean": float(gt[m].mean()),
                "pred_mean": float(pred[m].mean()),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "voxels", "tumor_psnr", "gt_mean", "pred_mean"])
        w.writeheader()
        w.writerows(rows)
    if rows:
        mean_psnr = float(np.mean([r["tumor_psnr"] for r in rows]))
        print(f"Saved {out} (n={len(rows)} phases, tumor_psnr_mean={mean_psnr:.4f})")
    else:
        print(f"Saved {out} (no valid phase masks found)")


if __name__ == "__main__":
    main()
