"""Summarize D3 static-only runs (loss curves, PSNR, histograms)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu
from anatgs.dynamic import ContinuousTimeField, query_volume


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


def _load_gt_mean(raw_dir: Path) -> np.ndarray:
    vols = sorted((raw_dir / "ground_truth" / "volumes").glob("*.nii.gz"))
    arrs = []
    for p in vols:
        xyz = np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32), dtype=np.float32)
        arrs.append(hu_to_mu(np.transpose(xyz, (2, 1, 0)).copy().astype(np.float32)))
    return np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/step1_7_model/D3_static_only_runs")
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--out_md", default="results/step1_7_model/D3_summary.md")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    gt_mean = _load_gt_mean(Path(args.raw_dir))

    rows = []
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    for rd in run_dirs:
        ckpts = sorted(rd.glob("model_iter_*.pt"))
        if not ckpts:
            continue
        ckpt_path = ckpts[-1]
        ckpt = torch.load(ckpt_path, map_location="cpu")
        mcfg = dict((ckpt.get("cfg", {}) or {}).get("model", {}))
        model = ContinuousTimeField(mcfg)
        model.load_state_dict(ckpt["model"], strict=False)
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
        pred = query_volume(model, t_value=0.0, resolution=96).detach().cpu().numpy().astype(np.float32)
        pred = _resize(pred, gt_mean.shape)
        p = _psnr(pred, gt_mean)

        # Hist plot.
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.hist(gt_mean.reshape(-1), bins=120, alpha=0.6, label="GT mean", density=True)
        ax.hist(pred.reshape(-1), bins=120, alpha=0.6, label=rd.name, density=True)
        ax.set_title(f"Histogram {rd.name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(rd / "histogram_vs_gt.png", dpi=180)
        plt.close(fig)

        # Loss curve.
        logp = rd / "train_log.csv"
        if logp.exists():
            df = pd.read_csv(logp)
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(df["iter"], df["loss_proj"], label="loss_proj")
            ax.set_yscale("log")
            ax.set_title(f"Loss curve {rd.name}")
            ax.set_xlabel("iter")
            ax.legend()
            fig.tight_layout()
            fig.savefig(rd / "loss_curve.png", dpi=180)
            plt.close(fig)

        rows.append(
            {
                "run": rd.name,
                "psnr_vs_gt_time_mean": float(p),
                "final_loss_proj": float(pd.read_csv(rd / "train_log.csv")["loss_proj"].iloc[-1]) if (rd / "train_log.csv").exists() else float("nan"),
            }
        )

    rows = sorted(rows, key=lambda x: x["psnr_vs_gt_time_mean"], reverse=True)
    best = rows[0] if rows else None
    md = ["# D3 Static-only Summary\n\n"]
    if rows:
        md.append("| run | PSNR vs GT time-mean | final loss_proj |\n")
        md.append("|---|---:|---:|\n")
        for r in rows:
            md.append(f"| {r['run']} | {r['psnr_vs_gt_time_mean']:.4f} | {r['final_loss_proj']:.6f} |\n")
        md.append("\n")
        md.append(f"Best run: **{best['run']}** with PSNR **{best['psnr_vs_gt_time_mean']:.4f} dB**.\n")
    else:
        md.append("No valid runs found.\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"best": best, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
