"""Q1 diagnostic: FDK sanity on full and phase-binned reconstructions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import tigre

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu
from anatgs.geom import to_radians


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    dr = float(np.max(b) - np.min(b) + 1e-8)
    return float(10.0 * math.log10((dr * dr) / mse))


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


def _load_gt_zyx(raw_dir: Path, t: int) -> np.ndarray:
    xyz = np.asarray(
        nib.load(str(raw_dir / "ground_truth" / "volumes" / f"volume_{int(t)}.nii.gz")).get_fdata(dtype=np.float32),
        dtype=np.float32,
    )
    return hu_to_mu(np.transpose(xyz, (2, 1, 0)).copy().astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--out_json", default="results/step1_5_diag/Q1_fdk_sanity.json")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    b = np.load(args.bundle)
    projs = b["projections"].astype(np.float32)
    angles = to_radians(b["angles"].astype(np.float32), angle_unit="auto")
    t_idx = b["t_idx_at_view"].astype(np.int32)
    phase = (t_idx % 10).astype(np.int32)

    geo = _make_geo(b)

    # Full-view reconstruction vs temporal-average GT.
    fdk_full = tigre.algorithms.fdk(projs, geo, angles)
    fdk_full_zyx = np.transpose(fdk_full, (2, 1, 0)).copy().astype(np.float32)
    gts = [_load_gt_zyx(raw_dir, t) for t in range(182)]
    gt_mean = np.mean(np.stack(gts, axis=0), axis=0).astype(np.float32)
    psnr_full = _psnr(fdk_full_zyx, gt_mean)

    # Phase-bin 10 reconstruction vs phase-averaged GT.
    phase_psnr = []
    for p in range(10):
        idx = np.where(phase == p)[0]
        rec = tigre.algorithms.fdk(projs[idx], geo, angles[idx])
        rec_zyx = np.transpose(rec, (2, 1, 0)).copy().astype(np.float32)
        gt_phase = np.mean(np.stack([g for i, g in enumerate(gts) if (i % 10) == p], axis=0), axis=0).astype(np.float32)
        phase_psnr.append(_psnr(rec_zyx, gt_phase))

    out = {
        "fdk_full_vs_gt_timeavg_psnr": float(psnr_full),
        "fdk_phasebin10_psnr_per_phase": [float(x) for x in phase_psnr],
        "fdk_phasebin10_psnr_mean": float(np.mean(phase_psnr)),
    }
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
