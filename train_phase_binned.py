"""Phase-binned baseline: train one model per phase index."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--config", default="configs/4d_continuous.yaml")
    ap.add_argument("--geo_config", default="configs/4d_cbct_geo.yaml")
    ap.add_argument("--phases", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--volume_res", type=int, default=128)
    args = ap.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    bundle = np.load(args.data)
    available = set(int(x) for x in np.unique(bundle["phase_indices"]).tolist()) if "phase_indices" in bundle else set()
    for phase in args.phases:
        if available and int(phase) not in available:
            print(f"[skip] phase={int(phase):02d} not present in bundle phase_indices={sorted(available)}")
            continue
        cmd = [
            sys.executable,
            "train_4d.py",
            "--data",
            args.data,
            "--output",
            str(out_root / f"phase_{int(phase):02d}"),
            "--iterations",
            str(args.iterations),
            "--config",
            args.config,
            "--geo_config",
            args.geo_config,
            "--phase_filter",
            str(int(phase)),
            "--time_mode",
            "continuous",
            "--volume_res",
            str(args.volume_res),
        ]
        code = subprocess.call(cmd)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
