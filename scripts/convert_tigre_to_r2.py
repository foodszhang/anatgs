#!/usr/bin/env python
"""Convert AnatCoder TIGRE data to R²-Gaussian scene format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from anatgs.data.convert_tigre import convert_case_to_r2_format


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed_case_dir", required=True)
    ap.add_argument("--projections_dir", required=True, help=".../caseXXX/<n>views")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--DSD", type=float, default=1536.0)
    ap.add_argument("--DSO", type=float, default=1000.0)
    ap.add_argument("--d_voxel", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    ap.add_argument("--d_detector", nargs=2, type=float, default=[1.5, 1.5])
    args = ap.parse_args()

    out = convert_case_to_r2_format(
        preprocessed_case_dir=args.preprocessed_case_dir,
        projection_view_dir=args.projections_dir,
        output_dir=args.output_dir,
        dsd=args.DSD,
        dso=args.DSO,
        d_voxel_mm=tuple(args.d_voxel),
        d_detector_mm=tuple(args.d_detector),
    )
    print(out)


if __name__ == "__main__":
    main()
