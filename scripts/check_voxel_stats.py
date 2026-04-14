#!/usr/bin/env python
"""Inspect voxel statistics across saved training iterations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import sys

sys.path.append("./")
from r2_gaussian.gaussian import GaussianModel, query


def _stats(name: str, vol: np.ndarray, gt: np.ndarray) -> dict[str, float | str]:
    vol = vol.astype(np.float32)
    gt = gt.astype(np.float32)
    mse = float(np.mean((vol - gt) ** 2))
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
    corr = float(np.corrcoef(vol.reshape(-1), gt.reshape(-1))[0, 1])
    return {
        "name": name,
        "min": float(vol.min()),
        "max": float(vol.max()),
        "mean": float(vol.mean()),
        "std": float(vol.std()),
        "p1": float(np.percentile(vol, 1)),
        "p50": float(np.percentile(vol, 50)),
        "p99": float(np.percentile(vol, 99)),
        "mae": float(np.mean(np.abs(vol - gt))),
        "psnr": psnr,
        "corr": corr,
    }


def _load_iter_volume(model_path: Path, iteration: int) -> np.ndarray | None:
    p = model_path / "point_cloud" / f"iteration_{iteration}" / "vol_pred.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def _infer_latest_iter(model_path: Path) -> int:
    pcd_dir = model_path / "point_cloud"
    iters = []
    for d in pcd_dir.glob("iteration_*"):
        try:
            iters.append(int(d.name.split("_")[-1]))
        except ValueError:
            continue
    if not iters:
        raise RuntimeError(f"No iteration_* folders found in {pcd_dir}")
    return max(iters)


def _iter0_from_init(source_dir: Path, init_path: Path) -> np.ndarray:
    with (source_dir / "meta_data.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    scanner = meta["scanner"]
    pts = np.load(init_path).astype(np.float32)
    xyz = pts[:, :3]
    density = pts[:, 3:4]
    g = GaussianModel(scale_bound=None)
    g.create_from_pcd(xyz, density, spatial_lr_scale=1.0)
    pipe = SimpleNamespace(compute_cov3D_python=False, debug=False)
    with torch.no_grad():
        out = query(
            g,
            scanner["offOrigin"],
            scanner["nVoxel"],
            scanner["sVoxel"],
            pipe,
        )["vol"]
    return out.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True, help="R2 scene folder")
    ap.add_argument("--model_path", required=True, help="Training output model path")
    ap.add_argument("--iters", nargs="*", type=int, default=[100, 500, 1000], help="Saved iteration snapshots")
    ap.add_argument("--init_path", default="", help="Optional init_*.npy for iter0 stats")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    model_path = Path(args.model_path)
    with (source_dir / "meta_data.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    gt = np.load(source_dir / meta["vol"]).astype(np.float32)

    rows = [_stats("gt", gt, gt)]

    init_path = Path(args.init_path) if args.init_path else (source_dir / f"init_{source_dir.name}.npy")
    if init_path.exists():
        try:
            v0 = _iter0_from_init(source_dir, init_path)
            rows.append(_stats("iter0", v0, gt))
        except Exception:
            pass

    for it in args.iters:
        vol = _load_iter_volume(model_path, int(it))
        if vol is not None:
            rows.append(_stats(f"iter{it}", vol, gt))

    final_it = _infer_latest_iter(model_path)
    vol_final = _load_iter_volume(model_path, final_it)
    if vol_final is not None and f"iter{final_it}" not in [r["name"] for r in rows]:
        rows.append(_stats(f"iter{final_it}", vol_final, gt))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Voxel Stats\n\n")
        f.write("| name | PSNR(dB) | MAE | Corr | min | max | mean | std | p1 | p50 | p99 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['name']} | {r['psnr']:.4f} | {r['mae']:.6f} | {r['corr']:.6f} | "
                f"{r['min']:.6f} | {r['max']:.6f} | {r['mean']:.6f} | {r['std']:.6f} | "
                f"{r['p1']:.6f} | {r['p50']:.6f} | {r['p99']:.6f} |\n"
            )


if __name__ == "__main__":
    main()

