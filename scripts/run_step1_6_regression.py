"""Run Step-1.6 geometry regressions R1/R2/R3 and dump JSON + slices."""

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
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu
from anatgs.dynamic import ContinuousTimeField, query_volume
from anatgs.geom import to_radians


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    dr = float(np.max(b) - np.min(b) + 1e-8)
    return float(10.0 * math.log10((dr * dr) / mse))


def _resize(vol: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if vol.shape == shape:
        return vol.astype(np.float32)
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)


def _load_gt_zyx(raw_dir: Path, t: int) -> np.ndarray:
    p = raw_dir / "ground_truth" / "volumes" / f"volume_{int(t)}.nii.gz"
    xyz = np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32), dtype=np.float32)
    return hu_to_mu(np.transpose(xyz, (2, 1, 0)).copy().astype(np.float32))


def _make_geo(bundle: np.lib.npyio.NpzFile) -> tigre.geometry:
    geo = tigre.geometry()
    # XCAT volume shape (XYZ) = (355, 280, 115), spacing (1,1,3) mm.
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
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--static_ckpt", required=True, help="Checkpoint from 200-iter static warmup.")
    ap.add_argument("--out_json", default="results/step1_6_geom/A3_regression.json")
    ap.add_argument("--out_slices", default="results/step1_6_geom/A3_slices")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_slices = Path(args.out_slices)
    out_slices.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    b = np.load(args.bundle)
    projs = (b["projections"] if "projections" in b else b["projs"]).astype(np.float32)
    angles = to_radians(np.asarray(b["angles"], dtype=np.float32), angle_unit="auto")
    t_idx = b["t_idx_at_view"].astype(np.int32)
    phase = (t_idx % 10).astype(np.int32)
    geo = _make_geo(b)

    # R1: full-view FDK vs time-mean GT.
    fdk_full = tigre.algorithms.fdk(projs, geo, angles)
    fdk_full_zyx = np.transpose(fdk_full, (2, 1, 0)).copy().astype(np.float32)
    gts = [_load_gt_zyx(raw_dir, t) for t in range(182)]
    gt_mean = np.mean(np.stack(gts, axis=0), axis=0).astype(np.float32)
    r1_psnr = _psnr(fdk_full_zyx, gt_mean)

    # R2: phase-bin FDK vs phase-mean GT.
    phase_psnr = []
    for p in range(10):
        idx = np.where(phase == p)[0]
        rec = tigre.algorithms.fdk(projs[idx], geo, angles[idx])
        rec_zyx = np.transpose(rec, (2, 1, 0)).copy().astype(np.float32)
        gt_phase = np.mean(np.stack([g for i, g in enumerate(gts) if (i % 10) == p], axis=0), axis=0).astype(np.float32)
        phase_psnr.append(_psnr(rec_zyx, gt_phase))
    r2_psnr = float(np.mean(phase_psnr))

    # R3: 200-iter static model volume vs time-mean GT.
    ckpt = torch.load(args.static_ckpt, map_location="cpu")
    mcfg = dict((ckpt.get("cfg", {}) or {}).get("model", {}))
    model = ContinuousTimeField(mcfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    pred = query_volume(model, t_value=0.0, resolution=96).detach().cpu().numpy().astype(np.float32)
    pred = _resize(pred, gt_mean.shape)
    r3_psnr = _psnr(pred, gt_mean)

    # Slices for quick visual verification.
    z = gt_mean.shape[0] // 2
    for name, vol in {
        "gt_timeavg": gt_mean,
        "fdk_full": _resize(fdk_full_zyx, gt_mean.shape),
        "static_model": pred,
    }.items():
        plt.figure(figsize=(4, 4))
        plt.imshow(vol[z], cmap="gray")
        plt.axis("off")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(out_slices / f"{name}.png", dpi=180)
        plt.close()

    out = {
        "R1_full_fdk_psnr": float(r1_psnr),
        "R2_phasebin10_fdk_psnr_mean": float(r2_psnr),
        "R2_phasebin10_fdk_psnr_per_phase": [float(x) for x in phase_psnr],
        "R3_static_warmup_psnr": float(r3_psnr),
        "thresholds": {"R1": 20.0, "R2": 18.0, "R3": 20.0},
        "pass": {"R1": bool(r1_psnr >= 20.0), "R2": bool(r2_psnr >= 18.0), "R3": bool(r3_psnr >= 20.0)},
    }
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
