"""Generate XCAT projection bundle with real RPM-aligned surrogate values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

import sys

sys.path.append("./src")

from anatgs.dynamic.signal import decompose_surrogate
from anatgs.data.xcat import hu_to_mu

try:
    import tigre
except Exception:  # pragma: no cover
    tigre = None


def _build_geo(cfg: dict):
    if tigre is None:
        raise ImportError("tigre is required to generate projections.")
    geo = tigre.geometry()
    geo.nVoxel = np.array(cfg.get("nVoxel", [355, 280, 115]), dtype=np.int32)
    geo.sVoxel = np.array(cfg.get("sVoxel", [355.0, 280.0, 345.0]), dtype=np.float32)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array(cfg.get("nDetector", [256, 256]), dtype=np.int32)
    geo.dDetector = np.array(cfg.get("dDetector", [1.5, 1.5]), dtype=np.float32)
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.DSO = float(cfg.get("DSO", 750.0))
    geo.DSD = float(cfg.get("DSD", 1200.0))
    geo.offOrigin = np.array(cfg.get("offOrigin", [0.0, 0.0, 0.0]), dtype=np.float32)
    geo.offDetector = np.array(cfg.get("offDetector", [0.0, 0.0]), dtype=np.float32)
    geo.mode = "cone"
    return geo


def _extract_idx(path: Path) -> int:
    stem = path.stem
    if stem.endswith(".nii"):
        stem = Path(stem).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse index from {path.name}")
    return int(digits)


def _load_volume_zyx(path: Path) -> np.ndarray:
    arr_xyz = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    arr_zyx = np.transpose(arr_xyz, (2, 1, 0)).copy().astype(np.float32)
    return hu_to_mu(arr_zyx)


def _normalize_rpm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (2.0 * (x - lo) / (hi - lo) - 1.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--geo_config", default="configs/4d_cbct_geo.yaml")
    ap.add_argument("--views_per_timepoint", type=int, default=5)
    ap.add_argument("--output", default="data/xcat_miccai24/projections/full_910v")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    vol_dir = raw_dir / "ground_truth" / "volumes"
    rpm_path = raw_dir / "rpm_signal.txt"
    if not vol_dir.exists():
        raise FileNotFoundError(f"Missing volume directory: {vol_dir}")
    if not rpm_path.exists():
        raise FileNotFoundError(f"Missing rpm signal file: {rpm_path}")

    vol_paths = sorted(vol_dir.glob("*.nii.gz"), key=_extract_idx)
    if len(vol_paths) == 0:
        raise FileNotFoundError(f"No volumes found under {vol_dir}")
    n_time = len(vol_paths)
    rpm = np.loadtxt(str(rpm_path), dtype=np.float32).reshape(-1)
    if rpm.shape[0] != n_time:
        raise ValueError(f"rpm length {rpm.shape[0]} != n_time {n_time}")
    rpm_norm = _normalize_rpm(rpm)

    with open(args.geo_config, "r", encoding="utf-8") as f:
        geo = _build_geo(yaml.safe_load(f) or {})
    # Keep SAD/SID from the chosen geometry config, but align voxel grid to XCAT.
    first_vol = _load_volume_zyx(vol_paths[0])
    # first_vol is [Z,Y,X], TIGRE expects [X,Y,Z]
    nx, ny, nz = int(first_vol.shape[2]), int(first_vol.shape[1]), int(first_vol.shape[0])
    geo.nVoxel = np.array([nx, ny, nz], dtype=np.int32)
    spacing_xyz = np.array([1.0, 1.0, 3.0], dtype=np.float32)
    geo.sVoxel = geo.nVoxel.astype(np.float32) * spacing_xyz
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    # Ensure detector FoV can cover XCAT transverse extent at isocenter.
    # Keep SAD/SID and detector resolution; only adapt detector physical size.
    required_det_mm = float(max(geo.sVoxel[0], geo.sVoxel[1]) * (geo.DSD / geo.DSO) * 1.05)
    cur_det_mm = float(min(geo.sDetector[0], geo.sDetector[1]))
    if cur_det_mm < required_det_mm:
        det_mm = required_det_mm
        geo.dDetector = np.array(
            [det_mm / float(geo.nDetector[0]), det_mm / float(geo.nDetector[1])],
            dtype=np.float32,
        )
        geo.sDetector = geo.nDetector.astype(np.float32) * geo.dDetector

    n_vpt = int(args.views_per_timepoint)
    base_angles = np.linspace(0.0, 2.0 * np.pi, n_vpt, endpoint=False, dtype=np.float32)
    projs: list[np.ndarray] = []
    angles_rad: list[float] = []
    rpm_at_view: list[float] = []
    t_idx_at_view: list[int] = []
    timestamps: list[float] = []

    for t_idx, p in enumerate(vol_paths):
        vol_zyx = _load_volume_zyx(p)
        vol_xyz = np.transpose(vol_zyx, (2, 1, 0)).copy()
        # Rotate angle pattern over timepoint index to avoid repeated angular aliasing.
        angles_t = (base_angles + 2.0 * np.pi * (t_idx / max(n_time, 1))) % (2.0 * np.pi)
        proj = tigre.Ax(vol_xyz, geo, angles_t.astype(np.float32))
        for j in range(n_vpt):
            projs.append(np.asarray(proj[j], dtype=np.float32))
            angles_rad.append(float(angles_t[j]))
            t_sub = t_idx + (j / max(n_vpt, 1))
            t_idx_at_view.append(int(t_idx))
            timestamps.append(float(t_sub / max(n_time - 1, 1)))
            rpm_at_view.append(float(np.interp(t_sub, np.arange(n_time, dtype=np.float32), rpm_norm)))

    projs_np = np.stack(projs, axis=0).astype(np.float32)
    angles_np = np.asarray(angles_rad, dtype=np.float32)
    rpm_at_view_np = np.asarray(rpm_at_view, dtype=np.float32)
    t_idx_at_view_np = np.asarray(t_idx_at_view, dtype=np.int32)
    timestamps_np = np.asarray(timestamps, dtype=np.float32)
    phase_indices = (t_idx_at_view_np % 10).astype(np.int16)
    signal_features, sig_scalar = decompose_surrogate(rpm_at_view_np, timestamps_np)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "bundle.npz",
        # Requested output fields
        projs=projs_np,
        angles=angles_np,
        rpm_at_view=rpm_at_view_np,
        t_idx_at_view=t_idx_at_view_np,
        angle_unit=np.asarray("radian"),
        projection_axes=np.asarray("[N,V,U]"),
        projection_v_flipped=np.asarray(0, dtype=np.uint8),
        volume_axes=np.asarray("[Z,Y,X]"),
        n_detector=np.asarray(geo.nDetector, dtype=np.int32),
        d_detector=np.asarray(geo.dDetector, dtype=np.float32),
        s_voxel=np.asarray(geo.sVoxel, dtype=np.float32),
        sod=np.asarray(float(geo.DSO), dtype=np.float32),
        sdd=np.asarray(float(geo.DSD), dtype=np.float32),
        # Backward-compatible fields for current train_4d pipeline
        projections=projs_np,
        timestamps=timestamps_np,
        phase_indices=phase_indices,
        surrogate_signal=sig_scalar.astype(np.float32),
        surrogate_time=timestamps_np,
        signal_features=signal_features.astype(np.float32),
    )
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "raw_dir": str(raw_dir),
                "n_timepoints": int(n_time),
                "views_per_timepoint": int(n_vpt),
                "n_views_total": int(projs_np.shape[0]),
                "bundle": str(out_dir / "bundle.npz"),
                "angle_unit": "radian",
                "projection_axes": "[N,V,U]",
                "projection_v_flipped": False,
                "volume_axes": "[Z,Y,X]",
                "n_detector": [int(geo.nDetector[0]), int(geo.nDetector[1])],
                "d_detector": [float(geo.dDetector[0]), float(geo.dDetector[1])],
                "s_voxel": [float(geo.sVoxel[0]), float(geo.sVoxel[1]), float(geo.sVoxel[2])],
                "sod": float(geo.DSO),
                "sdd": float(geo.DSD),
            },
            f,
            indent=2,
        )
    print(f"Saved XCAT bundle to {out_dir / 'bundle.npz'} with shape {projs_np.shape}")


if __name__ == "__main__":
    main()
