"""Generate a lightweight synthetic 4D patient with periodic motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _make_grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    z, y, x = np.meshgrid(xs, xs, xs, indexing="ij")
    return z, y, x


def _blob(z: np.ndarray, y: np.ndarray, x: np.ndarray, cz: float, cy: float, cx: float, sigma: float) -> np.ndarray:
    d2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n_phases", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    z, y, x = _make_grid(int(args.size))

    # Static anatomy: background + bone-like ring + soft tissue body
    body = (x**2 + y**2 + (z * 1.2) ** 2 < 0.75).astype(np.float32) * 0.18
    bone = ((x**2 + y**2 + (z * 1.2) ** 2 < 0.55) & (x**2 + y**2 + (z * 1.2) ** 2 > 0.48)).astype(np.float32) * 0.55
    lung = (((x + 0.22) ** 2 + (y * 1.15) ** 2 + (z * 1.25) ** 2 < 0.18) | ((x - 0.22) ** 2 + (y * 1.15) ** 2 + (z * 1.25) ** 2 < 0.18)).astype(np.float32) * -0.12

    meta = {"n_phases": int(args.n_phases), "size": int(args.size), "phases": []}
    for p in range(int(args.n_phases)):
        t = p / float(args.n_phases)
        # Respiratory superior-inferior motion + slight AP shift
        dz = 0.15 * np.sin(2.0 * np.pi * t)
        dy = 0.05 * np.sin(2.0 * np.pi * t + np.pi / 4.0)
        tumor = _blob(z, y, x, cz=0.1 + dz, cy=-0.05 + dy, cx=0.0, sigma=0.08) * 0.42
        vol = body + bone + lung + tumor
        vol = np.clip(vol, 0.0, 1.0).astype(np.float32)
        np.save(out_dir / f"phase_{p:02d}.npy", vol)
        meta["phases"].append({"phase": p, "t": t, "dz": float(dz), "dy": float(dy)})

    # canonical tumor mask at phase 0 for optional trajectory proxy
    tumor0 = _blob(z, y, x, cz=0.1, cy=-0.05, cx=0.0, sigma=0.08)
    np.save(out_dir / "gtv_mask.npy", (tumor0 > 0.2).astype(np.uint8))

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Synthetic 4D patient saved to {out_dir}")


if __name__ == "__main__":
    main()

