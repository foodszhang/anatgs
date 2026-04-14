#!/usr/bin/env python
"""Generate anatomy-guided init point cloud from segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from anatgs.anatomy.init import save_anatomy_init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True, help="Path to seg.npy")
    ap.add_argument("--output", required=True, help="Output init .npy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output)
    meta_path = out.with_suffix(".npz")
    init = save_anatomy_init(args.seg, out, out_tags_path=out.with_name(out.stem + "_organ_tags.npy"), out_meta_path=meta_path, seed=args.seed)
    print(f"points={init['means'].shape[0]}")
    tags, counts = np.unique(init["organ_tags"], return_counts=True)
    for t, c in zip(tags.tolist(), counts.tolist(), strict=False):
        print(f"organ {t}: {c}")


if __name__ == "__main__":
    main()

