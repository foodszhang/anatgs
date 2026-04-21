"""SVF utilities for approximately diffeomorphic motion integration."""

from __future__ import annotations

import torch


def integrate_stationary_velocity(
    velocity_fn,
    xyz: torch.Tensor,
    n_steps: int = 7,
    direction: float = 1.0,
) -> torch.Tensor:
    """Integrate a stationary velocity field with stable fixed-step updates."""
    n = max(int(n_steps), 1)
    dt = float(direction) / float(2**n)
    disp = dt * velocity_fn(xyz)
    current = xyz + disp
    for _ in range(n):
        disp = disp + dt * velocity_fn(current)
        current = xyz + disp
    return current

