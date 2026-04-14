# 2D→3D gap diagnosis — case001/50v strict vanilla

## Core trend

- Val projection PSNR: **36.58 dB (iter100) → 52.12 dB (iter30000)**.
- Global voxel PSNR: **13.27 dB (iter100 best) → 12.94 dB (iter30000)**.
- 2D/3D PSNR gap (`proj_psnr_val - global_psnr`): **23.31 dB → 39.18 dB**.

## Model-evolution context

- Gaussian count: **50,000 → 111,942** (then stabilizes).
- Voxel mean/std drifts mildly (not catastrophic collapse): mean ~0.251 → 0.262, std ~0.151 → 0.167.
- Densify/prune activity mostly in early-mid phase; late phase has little topology change but 3D PSNR does not recover.

## Pattern decision

Most consistent with **Pattern A (+ B)**:

1. **A**: projection keeps improving while 3D quality does not.
2. **B**: only train views are available (`proj_test` is empty), so 2D quality can be strongly optimized without validating volumetric consistency/generalization.

Less support for **Pattern C** as primary cause:
- Turning off densify+prune changes final 3D PSNR only marginally (~12.95 vs ~12.94).

## Data source

- `results/proj_voxel_gap_case001_50v.csv`
