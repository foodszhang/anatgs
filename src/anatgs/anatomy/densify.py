"""Anatomy-aware pruning helpers for R²-Gaussian."""

from __future__ import annotations

import torch


def organ_aware_prune_mask(
    density_values: torch.Tensor,
    min_density: float,
    organ_tags: torch.Tensor | None,
    protected_organs: set[int] | None = None,
    background_organ: int = 0,
    background_density_scale: float = 2.0,
) -> torch.Tensor:
    """Return prune mask with organ protection and stricter background pruning."""
    prune_mask = (density_values < float(min_density)).squeeze()
    if organ_tags is None:
        return prune_mask

    tags = organ_tags.to(dtype=torch.long)
    if protected_organs:
        protected = torch.zeros_like(prune_mask, dtype=torch.bool)
        for oid in protected_organs:
            protected |= tags == int(oid)
        prune_mask &= ~protected

    background = tags == int(background_organ)
    if background.any():
        bg_mask = (density_values.squeeze() < float(min_density) * float(background_density_scale)) & background
        prune_mask |= bg_mask
    return prune_mask

