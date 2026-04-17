"""Summarize smoke 4D evaluation CSVs into one table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--output", default="results/4d_smoke_summary.csv")
    args = ap.parse_args()

    base = Path(args.results_dir)
    items = [
        ("static", "patient_smoke_120v_static_eval.csv"),
        ("continuous", "patient_smoke_120v_continuous_eval.csv"),
        ("binned_phase00", "patient_smoke_120v_binned_phase00_eval.csv"),
        ("binned_phase05", "patient_smoke_120v_binned_phase05_eval.csv"),
    ]
    rows = []
    for method, fname in items:
        p = base / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        row = {
            "method": method,
            "psnr_mean": float(df["psnr"].mean()),
            "psnr_min": float(df["psnr"].min()),
            "ssim_mean": float(df["ssim"].mean()) if "ssim" in df.columns else np.nan,
        }
        if "gtv_centroid_error_vox" in df.columns:
            row["gtv_centroid_error_mean"] = float(df["gtv_centroid_error_vox"].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()

