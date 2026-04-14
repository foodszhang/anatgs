"""Per-organ initialization defaults for AnatGS."""

from __future__ import annotations

DEFAULT_ORGAN_PARAMS: dict[int, dict[str, float | str]] = {
    0: {"name": "background", "density": 0.005, "init_scale": 4.0, "init_opacity": 0.01},
    1: {"name": "soft_tissue", "density": 0.5, "init_scale": 1.5, "init_opacity": 0.15},
    2: {"name": "bone", "density": 2.0, "init_scale": 0.8, "init_opacity": 0.70},
    3: {"name": "liver", "density": 0.8, "init_scale": 1.2, "init_opacity": 0.18},
    4: {"name": "kidney", "density": 0.8, "init_scale": 1.0, "init_opacity": 0.14},
    5: {"name": "spleen", "density": 0.8, "init_scale": 1.2, "init_opacity": 0.12},
    6: {"name": "pancreas", "density": 1.0, "init_scale": 0.8, "init_opacity": 0.10},
    7: {"name": "heart_vessels", "density": 1.2, "init_scale": 0.6, "init_opacity": 0.13},
    8: {"name": "lung", "density": 0.1, "init_scale": 3.0, "init_opacity": 0.04},
    9: {"name": "gi_tract", "density": 0.6, "init_scale": 1.0, "init_opacity": 0.12},
}

