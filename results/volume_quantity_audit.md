# Volume quantity audit — strict baseline

## Code-path audit

1. `src/anatgs/data/convert_tigre.py` writes `volume.npy` directly to `vol_gt.npy` (float32), no extra scale transform in conversion.
2. `r2_gaussian/dataset/dataset_readers.py` loads `vol_gt.npy` directly into `scene.vol_gt`; projections are scaled by `scene_scale` together with geometry scaling.
3. `r2_gaussian/gaussian/render_query.py` builds `vol_pred` using Gaussian voxelization with `opacities=density` (`pc.get_density`), i.e. predicted voxelized density/attenuation field from Gaussian parameters.
4. `r2_gaussian/utils/image_utils.py::metric_vol` computes PSNR directly on raw voxel values with default `pixel_max=1.0`.

## Numerical-domain check

- `vol_gt.npy` stats: min=0.0, max=1.0, mean=0.2622, std=0.1715.
- `vol_pred` across init/iter100/iter1000/iter5000/iter30000 stays in approximately `[0, 1.3]`.
- So current eval is not comparing obviously different units/orders of magnitude (no gross unit mismatch found).

## Important caveat (2D metric interpretation)

- `metric_proj` normalizes each projection slice by its own max before PSNR/SSIM.
- Therefore high reported 2D PSNR can understate intensity-scale errors and is not a strong guarantee of 3D fidelity by itself.

## Verdict

H2 (“completely different physical quantity”) is **not supported as a hard mismatch** in current pipeline.

The main issue appears to be representational/objective under-constraint (2D fit can improve while 3D structure quality plateaus), not a simple units bug.
