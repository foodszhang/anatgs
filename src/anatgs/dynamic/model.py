"""Continuous spatiotemporal attenuation field with optional motion conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_enc import TemporalEncoder
from .svf import integrate_stationary_velocity

try:
    import tinycudann as tcnn
except Exception:  # pragma: no cover - fallback path
    tcnn = None


class _TorchSpatialEncoder(nn.Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.out_dim = int(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.out_dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.mlp(xyz)


class ContinuousTimeField(nn.Module):
    """Continuous attenuation field with optional signal-parameterized motion."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = dict(cfg or {})
        hidden = int(cfg.get("hidden_dim", 128))
        n_freqs = int(cfg.get("time_n_freqs", 8))
        time_method = str(cfg.get("time_enc", "pe"))
        hash_levels = int(cfg.get("hash_levels", 16))
        hash_feat_per_level = int(cfg.get("hash_feat_per_level", 2))
        hash_log2 = int(cfg.get("hash_log2_size", 19))
        hash_base_res = int(cfg.get("hash_base_resolution", 16))
        hash_per_level_scale = float(cfg.get("hash_per_level_scale", 1.4472))

        self.use_signal = bool(cfg.get("use_signal", False))
        self.signal_dim = int(cfg.get("signal_dim", 5))
        self.use_motion_field = bool(cfg.get("use_motion_field", self.use_signal or bool(cfg.get("use_svf", False))))
        self.use_svf = bool(cfg.get("use_svf", False))
        self.svf_steps = int(cfg.get("svf_steps", 7))
        self.max_velocity = float(cfg.get("max_velocity", 0.10))
        self.output_activation = str(cfg.get("output_activation", "softplus")).lower()
        self.output_scale = float(cfg.get("output_scale", 1.0))
        self.output_max = cfg.get("output_max", 10.0)
        self.output_max = None if self.output_max is None else float(self.output_max)
        self.time_encoder = TemporalEncoder(method=time_method, n_freqs=n_freqs, embed_dim=16)
        self.use_tcnn = bool(cfg.get("use_tcnn", True)) and tcnn is not None

        if self.use_tcnn:
            self.spatial_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": hash_levels,
                    "n_features_per_level": hash_feat_per_level,
                    "log2_hashmap_size": hash_log2,
                    "base_resolution": hash_base_res,
                    "per_level_scale": hash_per_level_scale,
                },
            )
            spatial_dim = hash_levels * hash_feat_per_level
        else:
            self.spatial_encoder = _TorchSpatialEncoder(out_dim=32)
            spatial_dim = 32

        self.spatial_dim = int(spatial_dim)
        if self.use_motion_field:
            if self.use_signal:
                self.cond_encoder = nn.Sequential(
                    nn.Linear(self.signal_dim, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 64),
                )
                cond_dim = 64
            else:
                self.cond_encoder = self.time_encoder
                cond_dim = self.time_encoder.out_dim
            self.velocity_head = nn.Sequential(
                nn.Linear(self.spatial_dim + cond_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 3),
            )
            self.canonical_head = nn.Sequential(
                nn.Linear(self.spatial_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
            self.head = None
        else:
            self.head = nn.Sequential(
                nn.Linear(self.spatial_dim + self.time_encoder.out_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
            self.cond_encoder = None
            self.velocity_head = None
            self.canonical_head = None

    def _prepare_condition(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 1:
            cond = cond[:, None]
        if self.use_signal:
            if cond.ndim != 2 or cond.shape[-1] != self.signal_dim:
                raise ValueError(f"signal condition must be [N,{self.signal_dim}], got {tuple(cond.shape)}")
        else:
            if cond.ndim != 2 or cond.shape[-1] != 1:
                raise ValueError(f"time condition must be [N,1], got {tuple(cond.shape)}")
        return cond

    def _encode_condition(self, cond: torch.Tensor) -> torch.Tensor:
        cond = self._prepare_condition(cond)
        if self.use_signal:
            return self.cond_encoder(cond)
        return self.time_encoder(cond)

    def _spatial(self, xyz: torch.Tensor) -> torch.Tensor:
        xyz = xyz.clamp(0.0, 1.0)
        spatial = self.spatial_encoder(xyz)
        if spatial.dtype != torch.float32:
            spatial = spatial.float()
        return spatial

    def velocity(self, xyz: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if not self.use_motion_field:
            return torch.zeros_like(xyz)
        if cond.shape[0] != xyz.shape[0]:
            raise ValueError(f"Batch mismatch: xyz={xyz.shape[0]} cond={cond.shape[0]}")
        cond_feat = self._encode_condition(cond)
        feat = torch.cat([self._spatial(xyz), cond_feat], dim=-1)
        v = self.velocity_head(feat)
        return torch.tanh(v) * self.max_velocity

    def map_points(self, xyz: torch.Tensor, cond: torch.Tensor, inverse: bool = True) -> torch.Tensor:
        if not self.use_motion_field:
            return xyz.clamp(0.0, 1.0)
        direction = -1.0 if inverse else 1.0
        if self.use_svf:
            mapped = integrate_stationary_velocity(
                velocity_fn=lambda pts: self.velocity(pts, cond),
                xyz=xyz,
                n_steps=self.svf_steps,
                direction=direction,
            )
        else:
            mapped = xyz + direction * self.velocity(xyz, cond)
        return mapped.clamp(0.0, 1.0)

    def forward(self, xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [N,3], got {tuple(xyz.shape)}")
        cond = self._prepare_condition(t)
        if cond.shape[0] != xyz.shape[0]:
            raise ValueError(f"Batch mismatch: xyz={xyz.shape[0]} cond={cond.shape[0]}")
        if self.use_motion_field:
            xyz_canonical = self.map_points(xyz.clamp(0.0, 1.0), cond, inverse=True)
            raw = self.canonical_head(self._spatial(xyz_canonical))
        else:
            tau = self.time_encoder(cond)
            feat = torch.cat([self._spatial(xyz), tau], dim=-1)
            raw = self.head(feat)
        if self.output_activation == "softplus":
            out = F.softplus(raw)
        elif self.output_activation == "linear":
            out = raw
        else:
            raise ValueError(f"Unsupported output_activation: {self.output_activation}")
        out = out * self.output_scale
        out = out.clamp_min(0.0)
        if self.output_max is not None:
            out = out.clamp_max(self.output_max)
        return out
