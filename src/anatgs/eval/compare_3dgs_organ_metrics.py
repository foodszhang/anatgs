"""Compare 3DGS methods on global + organ-level metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

sys.path.append("/home/foods/pro/anatcoder/src")
from anatcoder.eval.organ_metrics import evaluate_per_organ

CLASS_NAMES = {
    1: "soft_tissue",
    2: "bone",
    3: "liver",
    4: "kidney",
    5: "spleen",
    6: "pancreas",
    7: "heart_vessels",
    8: "lung",
    9: "gi_tract",
}


def _load_latest_pred(case_out_dir: Path) -> np.ndarray:
    pc_root = case_out_dir / "point_cloud"
    iters = sorted(pc_root.glob("iteration_*"))
    if not iters:
        raise FileNotFoundError(f"No iteration outputs under {pc_root}")
    pred = np.load(iters[-1] / "vol_pred.npy")
    return np.asarray(pred, dtype=np.float32)


def _global_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--method-root", required=True, help="Root dir containing <method>/<case>/")
    ap.add_argument("--gt-root", required=True, help="AnatCoder preprocessed root containing case*/volume.npy,seg.npy")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    method_root = Path(args.method_root)
    gt_root = Path(args.gt_root)

    rows: list[dict[str, object]] = []
    for case in args.cases:
        gt_case = gt_root / case
        gt = np.load(gt_case / "volume.npy").astype(np.float32)
        seg = np.load(gt_case / "seg.npy").astype(np.int16)
        for method in args.methods:
            pred = _load_latest_pred(method_root / method / case)
            organ = evaluate_per_organ(pred, gt, seg, class_names=CLASS_NAMES, min_voxels=100)
            row: dict[str, object] = {
                "case": case,
                "method": method,
                "global_psnr": _global_psnr(pred, gt),
            }
            for name, vals in organ.items():
                row[f"{name}_psnr"] = float(vals["psnr"])
            rows.append(row)

    all_keys = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
