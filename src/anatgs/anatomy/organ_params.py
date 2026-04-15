"""Per-organ initialization defaults for AnatGS."""

from __future__ import annotations

DEFAULT_ORGAN_PARAMS: dict[int, dict[str, float | str]] = {
    0: {"name": "background", "density": 0.005, "init_scale": 4.0, "init_opacity": 0.01},
    # Init opacity calibrated for R2 normalized density domain ([0,1]) with conservative scale.
    1: {"name": "soft_tissue", "density": 0.5, "init_scale": 1.5, "init_opacity": 0.11},
    2: {"name": "bone", "density": 2.0, "init_scale": 0.8, "init_opacity": 0.14},
    3: {"name": "liver", "density": 0.8, "init_scale": 1.2, "init_opacity": 0.14},
    4: {"name": "kidney", "density": 0.8, "init_scale": 1.0, "init_opacity": 0.13},
    5: {"name": "spleen", "density": 0.8, "init_scale": 1.2, "init_opacity": 0.13},
    6: {"name": "pancreas", "density": 1.0, "init_scale": 0.8, "init_opacity": 0.13},
    7: {"name": "heart_vessels", "density": 1.2, "init_scale": 0.6, "init_opacity": 0.13},
    8: {"name": "lung", "density": 0.1, "init_scale": 3.0, "init_opacity": 0.03},
    9: {"name": "gi_tract", "density": 0.6, "init_scale": 1.0, "init_opacity": 0.12},
}
