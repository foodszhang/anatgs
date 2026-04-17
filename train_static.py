"""Static baseline: force all projection timestamps to t=0."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--config", default="configs/4d_continuous.yaml")
    ap.add_argument("--geo_config", default="configs/4d_cbct_geo.yaml")
    ap.add_argument("--volume_res", type=int, default=128)
    args = ap.parse_args()

    cmd = [
        sys.executable,
        "train_4d.py",
        "--data",
        args.data,
        "--output",
        args.output,
        "--iterations",
        str(args.iterations),
        "--config",
        args.config,
        "--geo_config",
        args.geo_config,
        "--time_mode",
        "fixed0",
        "--volume_res",
        str(args.volume_res),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
