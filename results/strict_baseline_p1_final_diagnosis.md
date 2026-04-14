# Strict baseline P1 final diagnosis — case001/50v

## Top-3 most likely root causes

1. **Projection objective is under-constrained for volumetric recovery (dominant).**  
   Evidence: projection PSNR rises from 36.58→52.12 dB while 3D PSNR declines from 13.27→12.94 dB; low corr to GT stays ~0.10–0.11.

2. **Recovered volume collapses to a low-contrast affine surrogate rather than structure-faithful field.**  
   Evidence: linear fit remains `pred≈(0.09~0.11)*gt + 0.23`; MAE grows slightly and 3D PSNR plateaus despite better 2D.

3. **Initialization/adaptive control are secondary, not primary.**  
   Evidence: random init (very poor at iter0) and FDK init end in the same ~12.9–13.3 dB basin; disabling densify/prune changes final 3D PSNR only marginally.

## H1–H4 status

- **H1 init bottleneck**: partially true for early stage, **not primary** for final plateau.
- **H2 quantity mismatch**: no hard unit mismatch found; GT and pred remain comparable range (GT in [0,1], pred around [0,1.3]).
- **H3 objective mismatch / ill-posedness**: **strongly supported**.
- **H4 late Gaussian evolution destroys volume**: weak-to-moderate support; not primary from no-densify/prune ablation.

## What to fix next (priority)

1. **Objective side (highest priority)**: add stronger 3D-consistency constraints (or physically grounded calibration constraints) rather than relying on projection-only fit.
2. **Evaluation side**: add non-train-view projection validation split (current `proj_test` empty) and keep 3D stats/correlation as first-class stop criteria.
3. **Calibration side**: explicitly regularize global affine/contrast drift in `vol_pred` (current slope/intercept drift indicates weak voxel-level anchoring).

## Additional fix done in this round

- Fixed `train.py` argument precedence bug: `--max_iterations` now correctly overrides config-loaded `iterations` after config merge.

## Go/No-Go

**No-Go** for reintroducing AnatGS modules now.  
Strict baseline still does not stand up as a reliable 3D reconstruction baseline on case001/50v.
