#!/usr/bin/env python
"""Wrapper entry for 3DGS organ-level comparison."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from anatgs.eval.compare_3dgs_organ_metrics import main


if __name__ == "__main__":
    main()

