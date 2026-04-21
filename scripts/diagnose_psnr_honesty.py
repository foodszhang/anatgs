"""Q1 diagnostic: check PSNR under multiple axis conventions."""

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
import tigre

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu
from anatgs.dynamic import ContinuousTimeField, query_volume_condition
from anatgs.geom import to_radians


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    dr = float(np.max(b) - np.min(b) + 1e-8)
    return float(10.0 * math.log10((dr * dr) / mse))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64) - float(np.mean(a))
    bb = b.astype(np.float64) - float(np.mean(b))
    denom = math.sqrt(float(np.sum(aa**2) * np.sum(bb**2)) + 1e-12)
    return float(np.sum(aa * bb) / denom)


def _resize(vol: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if vol.shape == shape:
        return vol.astype(np.float32)
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)


def _load_gt_zyx(raw_dir: Path, t_idx: int, ras: bool = False) -> np.ndarray:
    p = raw_dir / "ground_truth" / "volumes" / f"volume_{int(t_idx)}.nii.gz"
    img = nib.load(str(p))
    if ras:
        img = nib.as_closest_canonical(img)
    arr_xyz = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    arr_zyx = np.transpose(arr_xyz, (2, 1, 0)).copy().astype(np.float32)
    return hu_to_mu(arr_zyx).astype(np.float32)


def _make_geo(bundle: np.lib.npyio.NpzFile) -> tigre.geometry:
    geo = tigre.geometry()
    geo.nVoxel = np.array([355, 280, 115], dtype=np.int32)
    geo.sVoxel = np.array([355.0, 280.0, 345.0], dtype=np.float32)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    if "n_detector" in bundle:
        geo.nDetector = np.asarray(bundle["n_detector"], dtype=np.int32).reshape(2)
    else:
        geo.nDetector = np.array([256, 256], dtype=np.int32)
    if "d_detector" in bundle:
        geo.dDetector = np.asarray(bundle["d_detector"], dtype=np.float32).reshape(2)
    else:
        geo.dDetector = np.array([1.5, 1.5], dtype=np.float32)
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.DSO = float(np.asarray(bundle["sod"]).reshape(-1)[0]) if "sod" in bundle else 750.0
    geo.DSD = float(np.asarray(bundle["sdd"]).reshape(-1)[0]) if "sdd" in bundle else 1200.0
    geo.offOrigin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    geo.offDetector = np.array([0.0, 0.0], dtype=np.float32)
    geo.mode = "cone"
    return geo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/step1_xcat/S1-A/model_iter_2000.pt")
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--out_json", default="results/step1_5_diag/Q1_psnr_honesty.json")
    ap.add_argument("--out_slices", default="results/step1_5_diag/Q1_slices")
    ap.add_argument("--times", default="0,45,90,135,181")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_slices = Path(args.out_slices)
    out_slices.mkdir(parents=True, exist_ok=True)

    bundle = np.load(args.bundle)
    t_idx_at_view = bundle["t_idx_at_view"].astype(np.int32)
    cond_all = bundle["signal_features"].astype(np.float32)
    projs = bundle["projections"].astype(np.float32)
    angles = to_radians(bundle["angles"].astype(np.float32), angle_unit="auto")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    mcfg = dict((ckpt.get("cfg", {}) or {}).get("model", {})
                )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuousTimeField(mcfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device).eval()

    times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    raw_dir = Path(args.raw_dir)
    geo = _make_geo(bundle)

    metrics = {"a_current": [], "b_axisfix_like": [], "c_gt_ras": []}
    per_time = {}
    for t in times:
        v = np.where(t_idx_at_view == t)[0]
        if v.size == 0:
            continue
        cond = cond_all[int(v[0])]
        pred = query_volume_condition(model, condition=cond, resolution=96).detach().cpu().numpy().astype(np.float32)
        gt_a = _load_gt_zyx(raw_dir, t, ras=False)
        gt_c = _load_gt_zyx(raw_dir, t, ras=True)
        pred_a = _resize(pred, gt_a.shape)
        pred_b = _resize(np.transpose(pred, (2, 1, 0)).copy(), gt_a.shape)
        pred_c = pred_a.copy()

        # FDK volume for slice triplets (local neighborhood for visibility only).
        idx_local = np.where((t_idx_at_view >= t - 2) & (t_idx_at_view <= t + 2))[0]
        fdk = tigre.algorithms.fdk(projs[idx_local], geo, angles[idx_local])
        fdk_zyx = np.transpose(fdk, (2, 1, 0)).copy().astype(np.float32)

        row = {
            "a_current": {
                "psnr": _psnr(pred_a, gt_a),
                "mae": _mae(pred_a, gt_a),
                "ncc": _ncc(pred_a, gt_a),
            },
            "b_axisfix_like": {
                "psnr": _psnr(pred_b, gt_a),
                "mae": _mae(pred_b, gt_a),
                "ncc": _ncc(pred_b, gt_a),
            },
            "c_gt_ras": {
                "psnr": _psnr(pred_c, gt_c),
                "mae": _mae(pred_c, gt_c),
                "ncc": _ncc(pred_c, gt_c),
            },
        }
        per_time[str(t)] = row
        for k in metrics:
            metrics[k].append(row[k])

        if t in {0, 90, 181}:
            z = gt_a.shape[0] // 2
            p2 = pred_a[z]
            g2 = gt_a[z]
            f2 = _resize(fdk_zyx, gt_a.shape)[z]
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            axs[0].imshow(p2, cmap="gray")
            axs[0].set_title("Model")
            axs[1].imshow(g2, cmap="gray")
            axs[1].set_title("GT")
            axs[2].imshow(f2, cmap="gray")
            axs[2].set_title("FDK")
            for ax in axs:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_slices / f"time_{t:03d}_triplet.png", dpi=180)
            plt.close(fig)

    summary = {
        "per_time": per_time,
        "mean": {
            k: {
                "psnr": float(np.mean([x["psnr"] for x in v])),
                "mae": float(np.mean([x["mae"] for x in v])),
                "ncc": float(np.mean([x["ncc"] for x in v])),
            }
            for k, v in metrics.items()
        },
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
