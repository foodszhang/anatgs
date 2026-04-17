"""Evaluate std ratio, FWHM ratio and contrast recovery for dynamic volumes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _resize_to_shape(vol: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if vol.shape == shape:
        return vol
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _fwhm_hist_width(values: np.ndarray, bins: int = 128) -> float:
    if values.size < 10:
        return float("nan")
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    peak = int(hist.max())
    if peak <= 0:
        return float("nan")
    half = peak * 0.5
    idx = np.where(hist >= half)[0]
    if idx.size == 0:
        return float("nan")
    left = float(edges[int(idx[0])])
    right = float(edges[int(idx[-1]) + 1])
    return max(0.0, right - left)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--mask_dir", required=True, help="Directory containing gtv_phase_XX.npy")
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
        if pred.shape != gt.shape:
            pred = _resize_to_shape(pred, gt.shape)
        mask = np.load(m_p).astype(np.uint8) > 0
        if int(mask.sum()) < 10:
            continue

        pred_std = float(pred.std())
        gt_std = float(gt.std())
        std_ratio = pred_std / max(gt_std, 1e-8)

        pred_t = pred[mask]
        gt_t = gt[mask]
        pred_bg = pred[~mask]
        gt_bg = gt[~mask]
        pred_contrast = float(pred_t.mean() - pred_bg.mean())
        gt_contrast = float(gt_t.mean() - gt_bg.mean())
        contrast_recovery = pred_contrast / max(abs(gt_contrast), 1e-8)

        pred_f = _fwhm_hist_width(pred_t)
        gt_f = _fwhm_hist_width(gt_t)
        fwhm_ratio = pred_f / gt_f if np.isfinite(pred_f) and np.isfinite(gt_f) and gt_f > 0 else float("nan")

        rows.append(
            {
                "phase": i,
                "std_ratio": std_ratio,
                "fwhm_ratio": fwhm_ratio,
                "contrast_recovery": contrast_recovery,
                "pred_std": pred_std,
                "gt_std": gt_std,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["phase", "std_ratio", "fwhm_ratio", "contrast_recovery", "pred_std", "gt_std"],
        )
        w.writeheader()
        w.writerows(rows)
    if rows:
        arr = np.array([[r["std_ratio"], r["fwhm_ratio"], r["contrast_recovery"]] for r in rows], dtype=np.float64)
        print(
            f"Saved {out} | mean std_ratio={np.nanmean(arr[:,0]):.4f}, "
            f"fwhm_ratio={np.nanmean(arr[:,1]):.4f}, contrast_recovery={np.nanmean(arr[:,2]):.4f}"
        )
    else:
        print(f"Saved {out} (no valid phases)")


if __name__ == "__main__":
    main()

