#!/usr/bin/env python3
"""Organ-level PSNR evaluation for R²-Gaussian reconstructed volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORGAN_NAMES = [
    "background",
    "soft_tissue",
    "bone",
    "liver",
    "kidney",
    "spleen",
    "pancreas",
    "heart_vessels",
    "lung",
    "gi_tract",
]


def compute_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    mse = float(np.mean((pred - gt) ** 2))
    if mse < 1e-10:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to vol_pred.npy")
    parser.add_argument("--gt", required=True, help="Path to vol_gt.npy")
    parser.add_argument("--seg", required=True, help="Path to seg_256.npy in [z,y,x]")
    parser.add_argument("--csv_output", default="", help="Optional output CSV path")
    parser.add_argument("--min_voxels", type=int, default=100)
    args = parser.parse_args()

    vol_pred = np.load(args.pred).astype(np.float32)
    vol_gt = np.load(args.gt).astype(np.float32)
    seg = np.load(args.seg).astype(np.int16)

    seg_xyz = seg.transpose(2, 1, 0)  # [z,y,x] -> [x,y,z]
    if not (vol_pred.shape == vol_gt.shape == seg_xyz.shape):
        raise ValueError(
            f"Shape mismatch: pred={vol_pred.shape}, gt={vol_gt.shape}, seg_xyz={seg_xyz.shape}"
        )

    rows: list[dict[str, float | int | str]] = []
    global_psnr = compute_psnr(vol_pred, vol_gt)
    print(f"Global PSNR: {global_psnr:.4f} dB")
    print("-" * 72)

    for k, name in enumerate(ORGAN_NAMES):
        mask = seg_xyz == k
        n = int(mask.sum())
        if n < args.min_voxels:
            print(f"{k:>2} {name:15s}: skipped (voxels={n})")
            continue
        organ_psnr = compute_psnr(vol_pred, vol_gt, mask)
        gt_mean = float(vol_gt[mask].mean())
        pred_mean = float(vol_pred[mask].mean())
        gt_std = float(vol_gt[mask].std())
        pred_std = float(vol_pred[mask].std())
        rows.append(
            {
                "label": k,
                "organ": name,
                "voxels": n,
                "psnr": organ_psnr,
                "gt_mean": gt_mean,
                "pred_mean": pred_mean,
                "gt_std": gt_std,
                "pred_std": pred_std,
            }
        )
        print(
            f"{k:>2} {name:15s}: PSNR={organ_psnr:6.2f} dB | "
            f"gt_mean={gt_mean:.4f} pred_mean={pred_mean:.4f} | voxels={n}"
        )

    if args.csv_output:
        out = Path(args.csv_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Saved CSV: {out}")


if __name__ == "__main__":
    main()
