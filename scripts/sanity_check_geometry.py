#!/usr/bin/env python
"""Geometry sanity check between stored projections and TIGRE forward projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return float("inf")
    data_max = float(np.max(gt))
    data_min = float(np.min(gt))
    peak = max(data_max - data_min, data_max, 1.0)
    return float(10.0 * np.log10((peak * peak) / mse))


def _build_geo(scanner: dict):
    import tigre  # type: ignore

    geo = tigre.geometry()
    geo.mode = scanner["mode"]
    geo.DSD = float(scanner["DSD"])
    geo.DSO = float(scanner["DSO"])
    geo.nVoxel = np.array(scanner["nVoxel"], dtype=np.int32)
    geo.dVoxel = np.array(scanner["dVoxel"], dtype=np.float32)
    geo.sVoxel = np.array(scanner["sVoxel"], dtype=np.float32)
    geo.nDetector = np.array(scanner["nDetector"], dtype=np.int32)
    geo.dDetector = np.array(scanner["dDetector"], dtype=np.float32)
    geo.sDetector = np.array(scanner["sDetector"], dtype=np.float32)
    geo.offOrigin = np.array(scanner.get("offOrigin", [0.0, 0.0, 0.0]), dtype=np.float32)
    geo.offDetector = np.array(scanner.get("offDetector", [0.0, 0.0]), dtype=np.float32)
    geo.accuracy = float(scanner.get("accuracy", 0.5))
    return geo


def _variant_transforms():
    return {
        "identity": lambda x: x,
        "flip_ud": lambda x: np.flip(x, axis=1),
        "flip_lr": lambda x: np.flip(x, axis=2),
        "flip_ud_lr": lambda x: np.flip(np.flip(x, axis=1), axis=2),
        "transpose": lambda x: np.transpose(x, (0, 2, 1)),
        "transpose_flip_ud": lambda x: np.flip(np.transpose(x, (0, 2, 1)), axis=1),
        "transpose_flip_lr": lambda x: np.flip(np.transpose(x, (0, 2, 1)), axis=2),
        "transpose_flip_ud_lr": lambda x: np.flip(np.flip(np.transpose(x, (0, 2, 1)), axis=1), axis=2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True, help="R2 scene folder with meta_data.json")
    ap.add_argument("--output", required=True, help="JSON output path")
    ap.add_argument("--vis_dir", default="results/geometry_vis", help="Visualization output dir")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    with (source_dir / "meta_data.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    vol = np.load(source_dir / meta["vol"]).astype(np.float32)
    proj_entries = meta["proj_train"]
    angles = np.array([float(x["angle"]) for x in proj_entries], dtype=np.float32)
    proj_gt = np.stack([np.load(source_dir / x["file_path"]).astype(np.float32) for x in proj_entries], axis=0)

    geo = _build_geo(meta["scanner"])
    import tigre  # type: ignore

    proj_tigre = tigre.Ax(vol, geo, angles)
    if proj_tigre.shape != proj_gt.shape:
        raise RuntimeError(f"Projection shape mismatch: TIGRE={proj_tigre.shape} vs GT={proj_gt.shape}")

    best_name = None
    best_psnr = float("-inf")
    best_proj = None
    variant_scores = {}
    for name, fn in _variant_transforms().items():
        cur = fn(proj_tigre)
        if cur.shape != proj_gt.shape:
            continue
        psnr = _psnr(cur, proj_gt)
        mae = float(np.mean(np.abs(cur - proj_gt)))
        variant_scores[name] = {"psnr": psnr, "mae": mae}
        if psnr > best_psnr:
            best_psnr = psnr
            best_name = name
            best_proj = cur

    assert best_proj is not None and best_name is not None
    per_view = []
    for i in range(proj_gt.shape[0]):
        gt_i = proj_gt[i]
        pr_i = best_proj[i]
        per_view.append(
            {
                "view": int(i),
                "mae": float(np.mean(np.abs(pr_i - gt_i))),
                "psnr": _psnr(pr_i, gt_i),
                "gt_mean": float(gt_i.mean()),
                "gt_std": float(gt_i.std()),
                "pred_mean": float(pr_i.mean()),
                "pred_std": float(pr_i.std()),
            }
        )

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        show_idx = np.linspace(0, proj_gt.shape[0] - 1, 3, dtype=int).tolist()
        for i in show_idx:
            gt_i = proj_gt[i]
            pr_i = best_proj[i]
            diff = np.abs(pr_i - gt_i)
            vmax = max(float(gt_i.max()), float(pr_i.max()))
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(gt_i, cmap="gray", vmin=0.0, vmax=vmax)
            axes[0].set_title(f"GT view {i}")
            axes[1].imshow(pr_i, cmap="gray", vmin=0.0, vmax=vmax)
            axes[1].set_title(f"TIGRE ({best_name})")
            axes[2].imshow(diff, cmap="magma")
            axes[2].set_title("abs diff")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(vis_dir / f"case001_view{i:03d}.png", dpi=140)
            plt.close(fig)
    except Exception:
        pass

    summary = {
        "source_dir": str(source_dir),
        "num_views": int(proj_gt.shape[0]),
        "projection_shape": list(proj_gt.shape[1:]),
        "best_variant": best_name,
        "variant_scores": variant_scores,
        "global_mae": float(np.mean(np.abs(best_proj - proj_gt))),
        "global_psnr": _psnr(best_proj, proj_gt),
        "per_view": per_view,
    }

    output_json = Path(args.output)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_path = output_json.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Geometry Sanity (case001)\n\n")
        f.write(f"- source: `{source_dir}`\n")
        f.write(f"- best variant: `{best_name}`\n")
        f.write(f"- global MAE: `{summary['global_mae']:.6f}`\n")
        f.write(f"- global PSNR: `{summary['global_psnr']:.4f} dB`\n\n")
        f.write("## Variant scores\n\n")
        f.write("| variant | PSNR(dB) | MAE |\n|---|---:|---:|\n")
        for name, stat in variant_scores.items():
            f.write(f"| {name} | {stat['psnr']:.4f} | {stat['mae']:.6f} |\n")
        f.write("\n## Per-view stats\n\n")
        f.write("| view | PSNR(dB) | MAE | gt_mean | pred_mean |\n|---:|---:|---:|---:|---:|\n")
        for row in per_view:
            f.write(
                f"| {row['view']} | {row['psnr']:.4f} | {row['mae']:.6f} | {row['gt_mean']:.6f} | {row['pred_mean']:.6f} |\n"
            )


if __name__ == "__main__":
    main()
