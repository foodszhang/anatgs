"""Dataset utilities for time-stamped 4D projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from anatgs.geom import to_radians

from .ray import build_ray_batch, world_to_unit
from .signal import decompose_surrogate, interpolate_trace, phase_only_from_timestamps


@dataclass
class GeometryConfig:
    det_h: int
    det_w: int
    det_spacing_h: float
    det_spacing_w: float
    sod: float
    sdd: float
    volume_size_mm: float
    volume_size_xyz: tuple[float, float, float] | None
    n_samples: int


class ProjectionDataset:
    """Projection sampler for continuous-time field optimization."""

    def __init__(
        self,
        bundle_path: str | Path,
        geo: GeometryConfig,
        phase_filter: int | None = None,
        time_mode: str = "continuous",
        use_signal: bool = False,
        signal_dim: int = 5,
    ):
        arr = np.load(str(bundle_path))
        if "projections" in arr:
            self.projections = np.asarray(arr["projections"], dtype=np.float32)
        elif "projs" in arr:
            self.projections = np.asarray(arr["projs"], dtype=np.float32)
        else:
            raise ValueError("bundle must contain 'projections' or 'projs'")
        if "projection_v_flipped" in arr:
            pv = int(np.asarray(arr["projection_v_flipped"]).reshape(-1)[0])
            if pv != 0:
                self.projections = self.projections[:, ::-1, :].copy()
        self.angles = to_radians(np.asarray(arr["angles"], dtype=np.float32), angle_unit="auto")
        if "timestamps" in arr:
            self.timestamps = np.asarray(arr["timestamps"], dtype=np.float32)
        elif "t_idx_at_view" in arr:
            t_idx = np.asarray(arr["t_idx_at_view"], dtype=np.float32)
            denom = max(float(np.max(t_idx)), 1.0)
            self.timestamps = (t_idx / denom).astype(np.float32)
        else:
            n = self.projections.shape[0]
            self.timestamps = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        if "phase_indices" in arr:
            self.phase_indices = np.asarray(arr["phase_indices"], dtype=np.int16)
        elif "t_idx_at_view" in arr:
            self.phase_indices = (np.asarray(arr["t_idx_at_view"], dtype=np.int32) % 10).astype(np.int16)
        else:
            self.phase_indices = None
        keep = None
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
        proj_h = int(self.projections.shape[1])
        proj_w = int(self.projections.shape[2])
        det_spacing_h = float(geo.det_spacing_h)
        det_spacing_w = float(geo.det_spacing_w)
        sod = float(geo.sod)
        sdd = float(geo.sdd)
        if "d_detector" in arr:
            dd = np.asarray(arr["d_detector"], dtype=np.float32).reshape(-1)
            if dd.size >= 2:
                det_spacing_h = float(dd[0])
                det_spacing_w = float(dd[1])
        if "sod" in arr:
            sod = float(np.asarray(arr["sod"]).reshape(-1)[0])
        if "sdd" in arr:
            sdd = float(np.asarray(arr["sdd"]).reshape(-1)[0])
        if int(geo.det_h) != proj_h or int(geo.det_w) != proj_w:
            geo = GeometryConfig(
                det_h=proj_h,
                det_w=proj_w,
                det_spacing_h=det_spacing_h,
                det_spacing_w=det_spacing_w,
                sod=sod,
                sdd=sdd,
                volume_size_mm=float(geo.volume_size_mm),
                volume_size_xyz=tuple(float(x) for x in np.asarray(arr["s_voxel"]).reshape(3))
                if "s_voxel" in arr
                else geo.volume_size_xyz,
                n_samples=int(geo.n_samples),
            )
        else:
            geo = GeometryConfig(
                det_h=int(geo.det_h),
                det_w=int(geo.det_w),
                det_spacing_h=det_spacing_h,
                det_spacing_w=det_spacing_w,
                sod=sod,
                sdd=sdd,
                volume_size_mm=float(geo.volume_size_mm),
                volume_size_xyz=tuple(float(x) for x in np.asarray(arr["s_voxel"]).reshape(3))
                if "s_voxel" in arr
                else geo.volume_size_xyz,
                n_samples=int(geo.n_samples),
            )
        self.geo = geo
        self.time_mode = str(time_mode)
        self.use_signal = bool(use_signal)
        self.signal_dim = int(signal_dim)
        self.signal_features: np.ndarray | None = None
        self.signal_scalar: np.ndarray | None = None
        if self.use_signal:
            if "signal_features" in arr:
                sf = np.asarray(arr["signal_features"], dtype=np.float32)
                if keep is not None and sf.shape[0] != self.timestamps.shape[0] and sf.shape[0] == keep.shape[0]:
                    sf = sf[keep]
                if sf.shape[0] != self.timestamps.shape[0]:
                    raise ValueError("signal_features length mismatch with timestamps")
                if sf.shape[1] < self.signal_dim:
                    raise ValueError(f"signal_features dim {sf.shape[1]} < signal_dim {self.signal_dim}")
                self.signal_features = sf[:, : self.signal_dim]
            else:
                if "surrogate_signal" in arr:
                    raw_trace = np.asarray(arr["surrogate_signal"], dtype=np.float32).reshape(-1)
                    if keep is not None and raw_trace.shape[0] == keep.shape[0]:
                        raw_trace = raw_trace[keep]
                    if "surrogate_time" in arr:
                        raw_time = np.asarray(arr["surrogate_time"], dtype=np.float32).reshape(-1)
                        if keep is not None and raw_time.shape[0] == keep.shape[0]:
                            raw_time = raw_time[keep]
                        trace_on_views = interpolate_trace(raw_time, raw_trace, self.timestamps)
                    elif raw_trace.shape[0] == self.timestamps.shape[0]:
                        trace_on_views = raw_trace
                    else:
                        raise ValueError("surrogate_signal length mismatch and surrogate_time missing")
                    sf_all, sig_scalar = decompose_surrogate(trace_on_views, self.timestamps)
                    self.signal_scalar = sig_scalar
                else:
                    if "rpm_at_view" in arr:
                        rpm = np.asarray(arr["rpm_at_view"], dtype=np.float32).reshape(-1)
                        if rpm.shape[0] != self.timestamps.shape[0]:
                            raise ValueError("rpm_at_view length mismatch with timestamps")
                        sf_all, sig_scalar = decompose_surrogate(rpm, self.timestamps)
                        self.signal_scalar = sig_scalar
                    else:
                        sf_all = phase_only_from_timestamps(self.timestamps, n_cycles=1.0)
                        sig_scalar = (sf_all[:, 0] + 1.0) * 0.5
                        self.signal_scalar = sig_scalar.astype(np.float32)
                if sf_all.shape[1] < self.signal_dim:
                    pad = np.zeros((sf_all.shape[0], self.signal_dim - sf_all.shape[1]), dtype=np.float32)
                    sf_all = np.concatenate([sf_all, pad], axis=1)
                self.signal_features = sf_all[:, : self.signal_dim].astype(np.float32)
            if self.signal_scalar is None:
                if "surrogate_signal" in arr and np.asarray(arr["surrogate_signal"]).shape[0] == self.timestamps.shape[0]:
                    self.signal_scalar = np.asarray(arr["surrogate_signal"], dtype=np.float32)
                else:
                    self.signal_scalar = ((self.signal_features[:, 0] + 1.0) * 0.5).astype(np.float32)

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
        cond_np = t[:, None].astype(np.float32)
        signal_scalar = None
        if self.use_signal:
            if self.signal_features is None:
                raise ValueError("use_signal=True but no signal features are available")
            cond_np = self.signal_features[pidx].astype(np.float32)
            signal_scalar = self.signal_scalar[pidx].astype(np.float32) if self.signal_scalar is not None else None

        angles_t = torch.from_numpy(angles).to(device=device)
        u_t = torch.from_numpy(u).to(device=device)
        v_t = torch.from_numpy(v).to(device=device)
        volume_size = self.geo.volume_size_xyz if self.geo.volume_size_xyz is not None else self.geo.volume_size_mm
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
            volume_size_mm=volume_size,
        )
        points_unit = world_to_unit(points_world, volume_size_mm=volume_size).clamp(0.0, 1.0)
        out = {
            "points_unit": points_unit,
            "delta": delta,
            "timestamps": torch.from_numpy(t).to(device=device),
            "condition": torch.from_numpy(cond_np).to(device=device),
            "target": torch.from_numpy(target).to(device=device)[:, None],
        }
        if signal_scalar is not None:
            out["signal_scalar"] = torch.from_numpy(signal_scalar).to(device=device)
        return out
