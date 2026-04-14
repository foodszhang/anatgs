# Strict baseline ablation — case001/50v

## Result table

| run | init | densify/prune | best voxel PSNR (iter) | final voxel PSNR | final val proj PSNR | final #Gaussians |
|---|---|---|---:|---:|---:|---:|
| strict_default_fdk | FDK | on | 13.27 (100) | 12.94 | 52.12 | 111,942 |
| strict_no_densify_no_prune_fdk | FDK | off | 13.27 (100) | 12.95 | 49.99 | 50,000 |
| strict_default_random_init | random | on | 13.30 (300) | 12.92 | 54.45 | 233,115 |

## Interpretation

- Disabling densify/prune does **not** materially lift final 3D PSNR.
- Random init still converges to the same ~12.9–13.3 dB basin.
- The dominant bottleneck is unlikely to be adaptive control alone or init alone; objective/identifiability mismatch is more plausible.

## Note about run length control

During this phase, a bug in `train.py` caused `--max_iterations` to be overwritten by config-loaded `iterations`.
That bug is now fixed so command-line max-iter control works as expected.

## Data source

- `results/strict_baseline_ablation_case001_50v.csv`
