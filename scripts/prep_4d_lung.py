"""Preprocess 4D-Lung phases to normalized cubic numpy volumes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None

try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None


def _load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    if path.is_dir():
        if sitk is None:
            raise ImportError("SimpleITK is required for DICOM directory input.")
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(path))
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        if sitk is None:
            raise ImportError("SimpleITK is required for NIfTI input.")
        img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [z,y,x]
    spacing = tuple(float(x) for x in img.GetSpacing())[::-1]  # to z,y,x
    return arr, spacing


def _resize_to_cube(vol_zyx: np.ndarray, target_size: int) -> np.ndarray:
    z, y, x = vol_zyx.shape
    m = max(z, y, x)
    out = np.full((m, m, m), float(np.min(vol_zyx)), dtype=np.float32)
    z0 = (m - z) // 2
    y0 = (m - y) // 2
    x0 = (m - x) // 2
    out[z0 : z0 + z, y0 : y0 + y, x0 : x0 + x] = vol_zyx
    t = torch.from_numpy(out)[None, None]
    t = F.interpolate(t, size=(target_size, target_size, target_size), mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _normalize_hu(vol: np.ndarray, hu_min: float = -1000.0, hu_max: float = 2000.0) -> np.ndarray:
    v = np.clip(vol, hu_min, hu_max)
    return ((v - hu_min) / (hu_max - hu_min)).astype(np.float32)


def _extract_phase_percent(path: Path) -> float | None:
    if pydicom is None or not path.is_dir():
        return None
    try:
        first = next(path.glob("*.dcm"))
    except StopIteration:
        return None
    try:
        ds = pydicom.dcmread(str(first), stop_before_pixels=True, force=True)
    except Exception:
        return None
    desc = str(getattr(ds, "SeriesDescription", ""))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", desc)
    if m is None:
        return None
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw patient dir containing 10 phase folders/files")
    ap.add_argument("--output", required=True, help="Output processed patient dir")
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--phase_glob", default="phase*")
    ap.add_argument("--gtv_mask", default="", help="Optional .npy mask path (already aligned)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_paths = sorted(in_dir.glob(args.phase_glob))
    if len(phase_paths) == 0:
        raise FileNotFoundError(f"No phase files/directories found by {args.phase_glob} under {in_dir}")
    # For IDC DICOM studies, preserve respiratory phase ordering by parsing
    # phase percentage in SeriesDescription, e.g. "... Gated, 30.0%".
    phase_info = []
    for p in phase_paths:
        phase_info.append((p, _extract_phase_percent(p)))
    if any(pp is not None for _, pp in phase_info):
        phase_info.sort(key=lambda x: (x[1] is None, 1e9 if x[1] is None else x[1], str(x[0])))
        phase_paths = [p for p, _ in phase_info]

    records = []
    for i, p in enumerate(phase_paths):
        vol, spacing = _load_volume(p)
        vol = _normalize_hu(_resize_to_cube(vol, target_size=int(args.target_size)))
        np.save(out_dir / f"phase_{i:02d}.npy", vol.astype(np.float32))
        records.append(
            {
                "phase": i,
                "source": str(p),
                "shape": list(vol.shape),
                "spacing_zyx": list(spacing),
                "phase_percent": _extract_phase_percent(p),
            }
        )

    if args.gtv_mask:
        mask = np.load(args.gtv_mask).astype(np.uint8)
        if mask.shape != (int(args.target_size), int(args.target_size), int(args.target_size)):
            raise ValueError(f"gtv_mask shape mismatch: {mask.shape}")
        np.save(out_dir / "gtv_mask.npy", mask)

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({"phases": records, "target_size": int(args.target_size)}, f, indent=2)

    print(f"Saved {len(records)} phases to {out_dir}")


if __name__ == "__main__":
    main()
