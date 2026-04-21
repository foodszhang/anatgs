"""Generate time-stamped 4D cone-beam projections from phase volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import sys

sys.path.append("./src")

from anatgs.dynamic.signal import decompose_surrogate

try:
    import tigre
except Exception:  # pragma: no cover
    tigre = None


def _load_phases(patient_dir: Path) -> list[np.ndarray]:
    phases = sorted(patient_dir.glob("phase_*.npy"))
    if not phases:
        raise FileNotFoundError(f"No phase_*.npy under {patient_dir}")
    return [np.load(p).astype(np.float32) for p in phases]


def _build_geo(cfg: dict):
    if tigre is None:
        raise ImportError("tigre is required to generate projections.")
    geo = tigre.geometry()
    geo.nVoxel = np.array(cfg.get("nVoxel", [256, 256, 256]), dtype=np.int32)
    geo.sVoxel = np.array(cfg.get("sVoxel", [384.0, 384.0, 384.0]), dtype=np.float32)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient_dir", required=True, help="Processed patient dir with phase_*.npy")
    ap.add_argument("--n_projections", type=int, required=True)
    ap.add_argument("--n_breath_cycles", type=float, default=1.0)
    ap.add_argument("--irregular", action="store_true")
    ap.add_argument("--output", required=True, help="Output projection dir")
    ap.add_argument("--geo_config", default="configs/4d_cbct_geo.yaml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    patient_dir = Path(args.patient_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_vols = _load_phases(patient_dir)
    phase_vols_xyz = [np.transpose(v, (2, 1, 0)).copy() for v in phase_vols]
    with open(args.geo_config, "r", encoding="utf-8") as f:
        geo = _build_geo(yaml.safe_load(f) or {})

    angles = np.linspace(0.0, 2.0 * np.pi, int(args.n_projections), endpoint=False, dtype=np.float32)
    timestamps = np.linspace(0.0, 1.0, int(args.n_projections), endpoint=False, dtype=np.float32)
    projections: list[np.ndarray] = []
    phase_indices: list[int] = []
    surrogate_signal: list[float] = []
    n_phases = len(phase_vols)

    for angle, t in zip(angles, timestamps):
        if args.irregular:
            breath_phase = (t * float(args.n_breath_cycles) + 0.1 * rng.normal()) % 1.0
            amp = 1.0 + 0.15 * rng.normal()
        else:
            breath_phase = (t * float(args.n_breath_cycles)) % 1.0
            amp = 1.0
        p = int(breath_phase * n_phases) % n_phases
        proj = tigre.Ax(phase_vols_xyz[p], geo, np.array([angle], dtype=np.float32))[:, ::-1, :]
        projections.append(np.asarray(proj[0], dtype=np.float32))
        phase_indices.append(p)
        surrogate_signal.append(float(amp * np.sin(2.0 * np.pi * breath_phase)))

    projections_np = np.stack(projections, axis=0)
    phase_indices_np = np.asarray(phase_indices, dtype=np.int16)
    surrogate_signal_np = np.asarray(surrogate_signal, dtype=np.float32)
    signal_features_np, surrogate_norm_np = decompose_surrogate(surrogate_signal_np, timestamps)
    np.savez_compressed(
        out_dir / "bundle.npz",
        projections=projections_np,
        angles=angles,
        timestamps=timestamps,
        phase_indices=phase_indices_np,
        surrogate_signal=surrogate_norm_np.astype(np.float32),
        surrogate_time=timestamps,
        signal_features=signal_features_np.astype(np.float32),
    )
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "patient_dir": str(patient_dir),
                "n_projections": int(args.n_projections),
                "n_breath_cycles": float(args.n_breath_cycles),
                "irregular": bool(args.irregular),
                "n_phases": n_phases,
                "bundle": str(out_dir / "bundle.npz"),
            },
            f,
            indent=2,
        )
    print(f"Saved bundle to {out_dir / 'bundle.npz'} with shape {projections_np.shape}")


if __name__ == "__main__":
    main()
