"""Utilities to normalize geometry conventions across XCAT/TIGRE/model pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def detect_angle_unit(angles: np.ndarray) -> str:
    """Best-effort angle unit detection from value range."""
    a = np.asarray(angles, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return "radian"
    m = float(np.nanmax(np.abs(a)))
    if m > (2.0 * np.pi + 0.5):
        return "degree"
    return "radian"


def to_radians(angles: np.ndarray, angle_unit: str = "auto") -> np.ndarray:
    """Convert angles to radians (idempotent for radian inputs)."""
    a = np.asarray(angles, dtype=np.float32)
    unit = angle_unit.strip().lower()
    if unit == "auto":
        unit = detect_angle_unit(a)
    if unit in {"deg", "degree", "degrees"}:
        return np.deg2rad(a).astype(np.float32)
    if unit in {"rad", "radian", "radians"}:
        return a.astype(np.float32)
    raise ValueError(f"Unsupported angle_unit={angle_unit}")


def reverse_angle_direction(angles_rad: np.ndarray) -> np.ndarray:
    """Reverse angular direction while keeping values in [0, 2pi)."""
    a = np.asarray(angles_rad, dtype=np.float32)
    twopi = np.float32(2.0 * np.pi)
    return np.mod(-a, twopi).astype(np.float32)


def xyz_to_zyx(volume_xyz: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(volume_xyz), (2, 1, 0)).copy()


def zyx_to_xyz(volume_zyx: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(volume_zyx), (2, 1, 0)).copy()


def apply_axis_permute_flip(volume: np.ndarray, permute: tuple[int, int, int], flips: tuple[bool, bool, bool]) -> np.ndarray:
    out = np.transpose(np.asarray(volume), permute).copy()
    for ax, do_flip in enumerate(flips):
        if do_flip:
            out = np.flip(out, axis=ax)
    return out.copy()


def _read_npz_scalar(arr: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in arr:
        return default
    v = arr[key]
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return str(v.item())
        if v.size == 1:
            return str(v.reshape(-1)[0])
    return str(v)


def describe_bundle_convention(bundle_path: str | Path) -> dict:
    """Inspect key convention markers from an npz projection bundle."""
    arr = np.load(str(bundle_path))
    projs = arr["projections"] if "projections" in arr else arr["projs"]
    angles = np.asarray(arr["angles"], dtype=np.float32)
    unit_raw = _read_npz_scalar(arr, "angle_unit", default="auto")
    unit_detected = detect_angle_unit(angles)
    unit_used = unit_detected if unit_raw == "auto" else unit_raw
    conv = {
        "projection_shape": tuple(int(x) for x in projs.shape),
        "projection_axes": _read_npz_scalar(arr, "projection_axes", default="[N,V,U]"),
        "projection_v_flipped": _read_npz_scalar(arr, "projection_v_flipped", default="unknown"),
        "volume_axes": _read_npz_scalar(arr, "volume_axes", default="unknown"),
        "angle_unit_raw": unit_raw,
        "angle_unit_detected": unit_detected,
        "angle_unit_used": unit_used,
        "angles_min": float(np.min(angles)),
        "angles_max": float(np.max(angles)),
    }
    return conv
