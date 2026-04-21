"""Evaluate D3 checkpoint curve: volume PSNR and projection PSNR."""

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
from anatgs.dynamic import ContinuousTimeField, project_volume_tigre_autograd, query_volume
from anatgs.geom import to_radians


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    dr = float(np.max(b) - np.min(b) + 1e-8)
    return float(10.0 * math.log10((dr * dr) / mse))


def _resize(vol: np.ndarray, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    if vol.shape == shape_zyx:
        return vol.astype(np.float32)
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape_zyx, mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)


def _load_gt_mean(raw_dir: Path) -> np.ndarray:
    vols = sorted((raw_dir / "ground_truth" / "volumes").glob("*.nii.gz"))
    arr = []
    for p in vols:
        xyz = np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32), dtype=np.float32)
        arr.append(hu_to_mu(np.transpose(xyz, (2, 1, 0)).copy().astype(np.float32)))
    return np.mean(np.stack(arr, axis=0), axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--volume_res", type=int, default=96)
    ap.add_argument("--proj_views", type=int, default=64, help="Number of evenly sampled views for projection PSNR.")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_mean = _load_gt_mean(Path(args.raw_dir))
    bundle = np.load(args.bundle)
    projs = (bundle["projections"] if "projections" in bundle else bundle["projs"]).astype(np.float32)
    angles = to_radians(np.asarray(bundle["angles"], dtype=np.float32), angle_unit="auto")
    n_views = int(projs.shape[0])
    k = min(int(args.proj_views), n_views)
    sel = np.linspace(0, n_views - 1, num=k, dtype=np.int32)
    target_proj = projs[sel]
    angles_sel = angles[sel]
    sv = np.asarray(bundle["s_voxel"], dtype=np.float32).reshape(3) if "s_voxel" in bundle else np.asarray([355.0, 280.0, 345.0], dtype=np.float32)
    dd = np.asarray(bundle["d_detector"], dtype=np.float32).reshape(2) if "d_detector" in bundle else np.asarray([1.5, 1.5], dtype=np.float32)
    geo_dict = {
        "nVoxel": [int(gt_mean.shape[2]), int(gt_mean.shape[1]), int(gt_mean.shape[0])],
        "sVoxel": [float(sv[0]), float(sv[1]), float(sv[2])],
        "nDetector": [int(target_proj.shape[1]), int(target_proj.shape[2])],
        "dDetector": [float(dd[0]), float(dd[1])],
        "DSO": float(np.asarray(bundle["sod"]).reshape(-1)[0]) if "sod" in bundle else 750.0,
        "DSD": float(np.asarray(bundle["sdd"]).reshape(-1)[0]) if "sdd" in bundle else 1200.0,
    }

    rows = []
    ckpts = sorted(run_dir.glob("model_iter_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    for ck in ckpts:
        it = int(ck.stem.split("_")[-1])
        state = torch.load(ck, map_location="cpu")
        mcfg = dict((state.get("cfg", {}) or {}).get("model", {}))
        model = ContinuousTimeField(mcfg).to(device).eval()
        model.load_state_dict(state["model"], strict=False)

        pred = query_volume(model, t_value=0.0, resolution=int(args.volume_res)).detach().cpu().numpy().astype(np.float32)
        pred = _resize(pred, gt_mean.shape)
        vol_psnr = _psnr(pred, gt_mean)

        vol_xyz = torch.from_numpy(np.transpose(pred, (2, 1, 0)).copy()).to(device=device)
        pred_proj = project_volume_tigre_autograd(
            vol_xyz,
            torch.from_numpy(angles_sel.astype(np.float32)).to(device=device),
            geo_dict,
        ).detach().cpu().numpy().astype(np.float32)
        proj_psnr = float(np.mean([_psnr(pred_proj[i], target_proj[i]) for i in range(pred_proj.shape[0])]))
        rows.append({"iter": it, "volume_psnr": float(vol_psnr), "projection_psnr": proj_psnr})

    out = {"run_dir": str(run_dir), "proj_views": int(k), "rows": rows, "best_volume": max(rows, key=lambda x: x["volume_psnr"]) if rows else None}
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"best_volume": out["best_volume"], "last": rows[-1] if rows else None}, indent=2))


if __name__ == "__main__":
    main()
