"""Honest XCAT evaluation against raw GT phase means."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:  # pragma: no cover
    ssim = None


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    dr = float(np.max(b) - np.min(b) + 1e-8)
    return float(10.0 * math.log10((dr * dr) / mse))


def _ssim3d(a: np.ndarray, b: np.ndarray) -> float:
    if ssim is None:
        return float("nan")
    return float(ssim(a, b, data_range=float(np.max(b) - np.min(b) + 1e-8)))


def _resize(vol: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if vol.shape == shape:
        return vol.astype(np.float32)
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)


def _centroid(mask: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    return idx.mean(axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(pred_dir.glob("pred_phase_*.npy"))
    if len(pred_paths) != 10:
        raise FileNotFoundError(f"Expected 10 pred_phase_*.npy under {pred_dir}, got {len(pred_paths)}")

    vols = sorted((raw_dir / "ground_truth" / "volumes").glob("*.nii.gz"))
    masks = sorted((raw_dir / "ground_truth" / "tumor_masks").glob("*.nii.gz"))
    if len(vols) < 10:
        raise FileNotFoundError("Insufficient GT volumes")

    gt_phase = []
    gt_mask_phase = []
    for p in range(10):
        idx = [i for i in range(len(vols)) if (i % 10) == p]
        vv = []
        mm = []
        for i in idx:
            arr = np.asarray(nib.load(str(vols[i])).get_fdata(dtype=np.float32), dtype=np.float32)
            zyx = hu_to_mu(np.transpose(arr, (2, 1, 0)).copy().astype(np.float32))
            vv.append(zyx)
            if i < len(masks):
                m = np.asarray(nib.load(str(masks[i])).get_fdata(dtype=np.float32), dtype=np.float32)
                mm.append((np.transpose(m, (2, 1, 0)).copy() > 0.5).astype(np.uint8))
        gt_phase.append(np.mean(np.stack(vv, axis=0), axis=0).astype(np.float32))
        if mm:
            gt_mask_phase.append((np.mean(np.stack(mm, axis=0), axis=0) > 0.5).astype(np.uint8))
        else:
            gt_mask_phase.append(None)

    rows = []
    for p in range(10):
        pred = np.load(pred_paths[p]).astype(np.float32)
        gt = gt_phase[p]
        pred = _resize(pred, gt.shape)
        row = {"phase": p, "psnr": _psnr(pred, gt), "ssim": _ssim3d(pred, gt)}
        gmask = gt_mask_phase[p]
        if gmask is not None and np.sum(gmask) > 0:
            w = pred * gmask.astype(np.float32)
            if float(np.sum(w)) > 1e-8:
                coords = np.argwhere(gmask > 0).astype(np.float32)
                vals = w[gmask > 0].astype(np.float32)
                c_pred = np.sum(coords * vals[:, None], axis=0) / (float(np.sum(vals)) + 1e-8)
                c_gt = _centroid(gmask)
                row["tumor_tre_vox"] = float(np.linalg.norm(c_pred - c_gt))
        rows.append(row)

    out = {
        "per_phase": rows,
        "mean_psnr": float(np.nanmean([r["psnr"] for r in rows])),
        "mean_ssim": float(np.nanmean([r["ssim"] for r in rows])),
        "mean_tumor_tre_vox": float(np.nanmean([r.get("tumor_tre_vox", np.nan) for r in rows])),
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
