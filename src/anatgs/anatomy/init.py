"""Anatomy-guided Gaussian initialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .organ_params import DEFAULT_ORGAN_PARAMS


def _boundary_mask(seg: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(seg, dtype=bool)
    boundary[1:, :, :] |= seg[1:, :, :] != seg[:-1, :, :]
    boundary[:-1, :, :] |= seg[:-1, :, :] != seg[1:, :, :]
    boundary[:, 1:, :] |= seg[:, 1:, :] != seg[:, :-1, :]
    boundary[:, :-1, :] |= seg[:, :-1, :] != seg[:, 1:, :]
    boundary[:, :, 1:] |= seg[:, :, 1:] != seg[:, :, :-1]
    boundary[:, :, :-1] |= seg[:, :, :-1] != seg[:, :, 1:]
    return boundary


def anatomy_guided_init(
    seg_volume: np.ndarray,
    organ_params: dict[int, dict[str, float | str]] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Sample anatomy-conditioned Gaussian point cloud and side metadata."""
    seg = np.asarray(seg_volume, dtype=np.int16)
    if seg.ndim != 3:
        raise ValueError(f"seg_volume must be [D,H,W], got {seg.shape}")
    params = organ_params or DEFAULT_ORGAN_PARAMS
    rng = np.random.default_rng(int(seed))
    boundary = _boundary_mask(seg)
    shape = np.array(seg.shape, dtype=np.float32)

    means: list[np.ndarray] = []
    densities: list[np.ndarray] = []
    organ_tags: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    boundary_tags: list[np.ndarray] = []

    for organ_id in sorted(params.keys()):
        cfg = params[int(organ_id)]
        vox = np.argwhere(seg == int(organ_id))
        if vox.size == 0:
            continue
        dens = float(cfg["density"])
        n_vox = int(vox.shape[0])
        if dens >= 1.0:
            n_samples = int(np.ceil(n_vox * dens))
        else:
            n_samples = max(1, int(np.ceil(n_vox * dens)))
        idx = rng.integers(0, n_vox, size=n_samples)
        pts = vox[idx].astype(np.float32)
        pts += rng.uniform(-0.5, 0.5, size=pts.shape).astype(np.float32)

        is_boundary = boundary[vox[idx, 0], vox[idx, 1], vox[idx, 2]]
        boundary_vox = np.argwhere((seg == int(organ_id)) & boundary)
        if boundary_vox.size > 0:
            n_extra = max(1, int(0.2 * n_samples))
            bidx = rng.integers(0, boundary_vox.shape[0], size=n_extra)
            bpts = boundary_vox[bidx].astype(np.float32)
            bpts += rng.uniform(-0.5, 0.5, size=bpts.shape).astype(np.float32)
            pts = np.concatenate([pts, bpts], axis=0)
            is_boundary = np.concatenate([is_boundary, np.ones((n_extra,), dtype=bool)], axis=0)

        # Match R2 init coordinate convention: voxel index space -> world in [-1, 1].
        pts_world = (pts / shape[None, :]) * 2.0 - 1.0
        pts_world = np.clip(pts_world, -1.0, 1.0 - 2.0 / float(np.max(shape)))
        means.append(pts_world.astype(np.float32))
        densities.append(np.full((pts.shape[0], 1), float(cfg["init_opacity"]), dtype=np.float32))
        organ_tags.append(np.full((pts.shape[0],), int(organ_id), dtype=np.int16))
        scales.append(np.full((pts.shape[0], 3), float(cfg["init_scale"]), dtype=np.float32))
        boundary_tags.append(is_boundary.astype(np.uint8))

    if not means:
        raise ValueError("No Gaussian points were sampled from segmentation volume.")

    means_np = np.concatenate(means, axis=0)
    density_np = np.concatenate(densities, axis=0)
    tags_np = np.concatenate(organ_tags, axis=0)
    scales_np = np.concatenate(scales, axis=0)
    boundary_np = np.concatenate(boundary_tags, axis=0)
    return {
        "means": means_np,
        "densities": density_np,
        "organ_tags": tags_np,
        "scales": scales_np,
        "boundary_tags": boundary_np,
    }


def save_anatomy_init(
    seg_path: str | Path,
    out_init_path: str | Path,
    out_tags_path: str | Path | None = None,
    out_meta_path: str | Path | None = None,
    organ_params: dict[int, dict[str, float | str]] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create and save `init_*.npy` and sidecar organ-tag metadata."""
    seg = np.load(str(seg_path))
    init = anatomy_guided_init(seg, organ_params=organ_params, seed=seed)
    out_init = Path(out_init_path)
    out_init.parent.mkdir(parents=True, exist_ok=True)
    points = np.concatenate([init["means"], init["densities"]], axis=1).astype(np.float32)
    np.save(out_init, points)

    tags_path = Path(out_tags_path) if out_tags_path else out_init.with_name(out_init.stem + "_organ_tags.npy")
    np.save(tags_path, init["organ_tags"].astype(np.int16))

    if out_meta_path is not None:
        meta = {
            "scales": init["scales"].astype(np.float32),
            "boundary_tags": init["boundary_tags"].astype(np.uint8),
        }
        np.savez_compressed(str(out_meta_path), **meta)
    return init
