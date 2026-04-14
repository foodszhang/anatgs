#!/usr/bin/env python
"""Compute per-organ intensity stats and parameter calibration hints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.append("./src")
from anatgs.anatomy.organ_params import DEFAULT_ORGAN_PARAMS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    args = ap.parse_args()

    vol = np.load(args.volume).astype(np.float32)
    seg = np.load(args.seg).astype(np.int16)
    rows = []
    for k in range(10):
        mask = seg == k
        count = int(mask.sum())
        if count == 0:
            continue
        vals = vol[mask]
        p = DEFAULT_ORGAN_PARAMS.get(k, {})
        rows.append(
            {
                "organ_id": k,
                "name": p.get("name", f"label_{k}"),
                "voxel_count": count,
                "fraction": count / seg.size,
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "p1": float(np.percentile(vals, 1)),
                "p50": float(np.percentile(vals, 50)),
                "p99": float(np.percentile(vals, 99)),
                "init_opacity": float(p.get("init_opacity", np.nan)),
                "opacity_minus_mean": float(p.get("init_opacity", np.nan) - vals.mean()),
            }
        )

    df = pd.DataFrame(rows).sort_values("organ_id")
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Organ Params Calibration\n\n")
        f.write("| organ_id | name | mean | std | p1 | p50 | p99 | init_opacity | opacity-mean |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in df.iterrows():
            f.write(
                f"| {int(r.organ_id)} | {r['name']} | {r['mean']:.4f} | {r['std']:.4f} | "
                f"{r['p1']:.4f} | {r['p50']:.4f} | {r['p99']:.4f} | {r['init_opacity']:.4f} | {r['opacity_minus_mean']:.4f} |\n"
            )


if __name__ == "__main__":
    main()

