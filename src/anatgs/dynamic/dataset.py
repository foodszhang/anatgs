"""Dataset utilities for time-stamped 4D projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .ray import build_ray_batch, world_to_unit


@dataclass
class GeometryConfig:
    det_h: int
    det_w: int
    det_spacing_h: float
    det_spacing_w: float
    sod: float
    sdd: float
    volume_size_mm: float
    n_samples: int


class ProjectionDataset:
    """Projection sampler for continuous-time field optimization."""

    def __init__(
        self,
        bundle_path: str | Path,
        geo: GeometryConfig,
        phase_filter: int | None = None,
        time_mode: str = "continuous",
    ):
        arr = np.load(str(bundle_path))
        self.projections = np.asarray(arr["projections"], dtype=np.float32)
        self.angles = np.asarray(arr["angles"], dtype=np.float32)
        self.timestamps = np.asarray(arr["timestamps"], dtype=np.float32)
        self.phase_indices = np.asarray(arr["phase_indices"], dtype=np.int16) if "phase_indices" in arr else None
        if phase_filter is not None:
            if self.phase_indices is None:
                raise ValueError("phase_filter requested but phase_indices missing in bundle")
            keep = self.phase_indices == int(phase_filter)
            self.projections = self.projections[keep]
            self.angles = self.angles[keep]
            self.timestamps = self.timestamps[keep]
            self.phase_indices = self.phase_indices[keep]
        if self.projections.ndim != 3:
            raise ValueError(f"Expected projections [N,H,W], got {self.projections.shape}")
        self.geo = geo
        self.time_mode = str(time_mode)

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        n = self.projections.shape[0]
        pidx = np.random.randint(0, n, size=int(batch_size))
        u = np.random.randint(0, self.geo.det_w, size=int(batch_size))
        v = np.random.randint(0, self.geo.det_h, size=int(batch_size))
        target = self.projections[pidx, v, u]
        angles = self.angles[pidx]
        t = self.timestamps[pidx]
        if self.time_mode == "fixed0":
            t = np.zeros_like(t, dtype=np.float32)

        angles_t = torch.from_numpy(angles).to(device=device)
        u_t = torch.from_numpy(u).to(device=device)
        v_t = torch.from_numpy(v).to(device=device)
        points_world, delta = build_ray_batch(
            angles=angles_t,
            u_idx=u_t,
            v_idx=v_t,
            det_h=self.geo.det_h,
            det_w=self.geo.det_w,
            det_spacing_h=self.geo.det_spacing_h,
            det_spacing_w=self.geo.det_spacing_w,
            sod=self.geo.sod,
            sdd=self.geo.sdd,
            n_samples=self.geo.n_samples,
            volume_size_mm=self.geo.volume_size_mm,
        )
        points_unit = world_to_unit(points_world, volume_size_mm=self.geo.volume_size_mm).clamp(0.0, 1.0)
        return {
            "points_unit": points_unit,
            "delta": delta,
            "timestamps": torch.from_numpy(t).to(device=device),
            "target": torch.from_numpy(target).to(device=device)[:, None],
        }
