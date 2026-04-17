"""Build per-phase tumor masks from multi-phase RTSTRUCT files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pydicom

from rtstruct_to_mask import _extract_contours, _load_ct_geometry, _rasterize_contours, _resize_mask_like_prep


def _extract_percent(text: str) -> float | None:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", str(text))
    return float(m.group(1)) if m else None


def _extract_percent_from_roi_name(name: str) -> float | None:
    # e.g., Tumor_c00 / Tumor_c90 / Tumor_c10
    m = re.search(r"_c(\d{2})$", str(name))
    if not m:
        return None
    return float(int(m.group(1)))


def _ct_phase_map(ct_study_dir: Path) -> dict[float, Path]:
    out: dict[float, Path] = {}
    for d in sorted(ct_study_dir.glob("CT_*")):
        if not d.is_dir():
            continue
        f = next(d.glob("*.dcm"), None)
        if f is None:
            continue
        ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        p = _extract_percent(getattr(ds, "SeriesDescription", ""))
        if p is not None:
            out[p] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtstruct_study_dir", required=True)
    ap.add_argument("--ct_study_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--roi_keyword", default="tumor")
    args = ap.parse_args()

    rt_dir = Path(args.rtstruct_study_dir)
    ct_dir = Path(args.ct_study_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct_map = _ct_phase_map(ct_dir)
    if not ct_map:
        raise RuntimeError("No CT phase directories with percentage description found.")
    print("CT phase map:", {k: v.name for k, v in sorted(ct_map.items())})

    masks = []
    # iterate each RTSTRUCT file (typically one per phase in this dataset)
    rt_files = sorted(rt_dir.glob("RTSTRUCT_*/*.dcm"))
    if not rt_files:
        raise RuntimeError(f"No RTSTRUCT dcm files found under {rt_dir}")

    for rf in rt_files:
        contours = _extract_contours(str(rf), roi_keywords=None)
        picks = [n for n in contours.keys() if args.roi_keyword.lower() in n.lower()]
        if not picks:
            continue
        roi_name = picks[0]
        phase_percent = _extract_percent_from_roi_name(roi_name)
        if phase_percent is None:
            continue
        # pick nearest CT phase by percentage
        ct_percent = min(ct_map.keys(), key=lambda x: abs(float(x) - float(phase_percent)))
        ct_phase_dir = ct_map[ct_percent]
        origin, spacing, shape, z_positions = _load_ct_geometry(str(ct_phase_dir))
        mask = _rasterize_contours(contours[roi_name], origin, spacing, shape, z_positions)
        mask = _resize_mask_like_prep(mask, int(args.target_size))
        phase_idx = int(round(phase_percent / 10.0)) % 10
        np.save(out_dir / f"gtv_phase_{phase_idx:02d}.npy", mask.astype(np.uint8))
        masks.append((phase_idx, roi_name, int(mask.sum()), rf.name, ct_phase_dir.name))

    masks.sort(key=lambda x: x[0])
    for x in masks:
        print(f"phase={x[0]:02d} roi={x[1]} voxels={x[2]} rt={x[3]} ct={x[4]}")
    print(f"Saved {len(masks)} phase masks to {out_dir}")


if __name__ == "__main__":
    main()

