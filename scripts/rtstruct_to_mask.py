"""Rasterize RTSTRUCT contours into voxel mask without rt-utils."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from skimage.draw import polygon


def _load_ct_geometry(ct_dir: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], np.ndarray]:
    """Read CT geometry from DICOM slices in one CT series."""
    dcm_files = sorted(glob.glob(os.path.join(ct_dir, "*.dcm")))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in ct_dir={ct_dir}")
    slices = [pydicom.dcmread(f, stop_before_pixels=True, force=True) for f in dcm_files]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    origin = np.array(slices[0].ImagePositionPatient, dtype=np.float64)  # x,y,z
    row_spacing = float(slices[0].PixelSpacing[0])
    col_spacing = float(slices[0].PixelSpacing[1])
    if len(slices) > 1:
        z_spacing = float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2])
    else:
        z_spacing = float(getattr(slices[0], "SliceThickness", 1.0))
    spacing = np.array([row_spacing, col_spacing, z_spacing], dtype=np.float64)
    shape = (len(slices), int(slices[0].Rows), int(slices[0].Columns))  # z,y,x
    z_positions = np.array([float(s.ImagePositionPatient[2]) for s in slices], dtype=np.float64)
    return origin, spacing, shape, z_positions


def _extract_contours(rtstruct_path: str, roi_keywords: list[str] | None = None) -> dict[str, list[np.ndarray]]:
    """Extract contour 3D points grouped by ROI name."""
    rs = pydicom.dcmread(rtstruct_path, force=True)
    roi_map: dict[int, str] = {}
    for item in getattr(rs, "StructureSetROISequence", []):
        roi_map[int(item.ROINumber)] = str(item.ROIName)

    out: dict[str, list[np.ndarray]] = {}
    for roi_contour in getattr(rs, "ROIContourSequence", []):
        roi_num = int(roi_contour.ReferencedROINumber)
        roi_name = roi_map.get(roi_num, f"ROI_{roi_num}")
        if roi_keywords:
            lowered = roi_name.lower()
            if not any(k.lower() in lowered for k in roi_keywords):
                continue

        contours: list[np.ndarray] = []
        for contour in getattr(roi_contour, "ContourSequence", []):
            pts = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
            if pts.shape[0] >= 3:
                contours.append(pts)
        out[roi_name] = contours
    return out


def _rasterize_contours(
    contours: list[np.ndarray],
    origin_xyz: np.ndarray,
    spacing_row_col_z: np.ndarray,
    shape_zyx: tuple[int, int, int],
    z_positions: np.ndarray,
) -> np.ndarray:
    """Rasterize world-space contour polygons to binary zyx mask."""
    mask = np.zeros(shape_zyx, dtype=np.uint8)
    row_spacing, col_spacing, _ = spacing_row_col_z
    for pts in contours:
        z_val = float(np.mean(pts[:, 2]))
        z_idx = int(np.argmin(np.abs(z_positions - z_val)))
        row = (pts[:, 1] - origin_xyz[1]) / row_spacing
        col = (pts[:, 0] - origin_xyz[0]) / col_spacing
        rr, cc = polygon(row, col, shape=(shape_zyx[1], shape_zyx[2]))
        mask[z_idx, rr, cc] = 1
    return mask


def _resize_mask_like_prep(mask_zyx: np.ndarray, target_size: int) -> np.ndarray:
    """Apply same spatial transform style as prep_4d_lung: pad to cube then resize."""
    z, y, x = mask_zyx.shape
    m = max(z, y, x)
    out = np.zeros((m, m, m), dtype=np.uint8)
    z0 = (m - z) // 2
    y0 = (m - y) // 2
    x0 = (m - x) // 2
    out[z0 : z0 + z, y0 : y0 + y, x0 : x0 + x] = mask_zyx
    t = torch.from_numpy(out.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(target_size, target_size, target_size), mode="nearest")
    return (t.squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtstruct", required=True)
    ap.add_argument("--ct_dir", required=True, help="One CT DICOM series directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--roi_name", default="", help="Exact ROI name to use (optional)")
    ap.add_argument("--roi_keywords", default="GTV,tumor,PTV", help="Comma-separated ROI keyword filter")
    args = ap.parse_args()

    origin, spacing, shape, z_positions = _load_ct_geometry(args.ct_dir)
    print(f"CT shape={shape} spacing(row,col,z)={spacing.tolist()}")

    kws = [k.strip() for k in str(args.roi_keywords).split(",") if k.strip()]
    all_rois = _extract_contours(args.rtstruct, roi_keywords=None)
    print(f"RTSTRUCT ROIs ({len(all_rois)}): {list(all_rois.keys())}")

    if args.roi_name:
        picked_name = args.roi_name
        if picked_name not in all_rois:
            raise ValueError(f"roi_name '{picked_name}' not found in RTSTRUCT")
        picked_contours = all_rois[picked_name]
    else:
        filtered = _extract_contours(args.rtstruct, roi_keywords=kws)
        if not filtered:
            raise RuntimeError("No ROI matched keywords. Use --roi_name from printed ROI list.")
        picked_name = list(filtered.keys())[0]
        picked_contours = filtered[picked_name]

    print(f"Using ROI: {picked_name} with {len(picked_contours)} contours")
    mask = _rasterize_contours(picked_contours, origin, spacing, shape, z_positions)
    if args.target_size > 0:
        mask = _resize_mask_like_prep(mask, int(args.target_size))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mask.astype(np.uint8))
    print(f"Saved {out} shape={mask.shape} voxels={int(mask.sum())}")


if __name__ == "__main__":
    main()

