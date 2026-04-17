"""Continuous spatiotemporal attenuation field."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_enc import TemporalEncoder

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
    """μ(x, y, z, t) with hash-grid-like spatial encoding + temporal encoding."""

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

        self.head = nn.Sequential(
            nn.Linear(spatial_dim + self.time_encoder.out_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [N,3], got {tuple(xyz.shape)}")
        if t.ndim == 1:
            t = t[:, None]
        if t.shape[0] != xyz.shape[0]:
            raise ValueError(f"Batch mismatch: xyz={xyz.shape[0]} t={t.shape[0]}")
        xyz = xyz.clamp(0.0, 1.0)
        spatial = self.spatial_encoder(xyz)
        if spatial.dtype != torch.float32:
            spatial = spatial.float()
        tau = self.time_encoder(t)
        feat = torch.cat([spatial, tau], dim=-1)
        raw = self.head(feat)
        # Keep attenuation numerically stable for long path integrals.
        return F.softplus(raw).clamp_max(10.0)
