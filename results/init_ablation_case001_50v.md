# Init ablation — case001/50v (baseline-only)

## Comparison

| run | init | densify/prune | init PSNR | iter100 PSNR | best PSNR (iter) | final PSNR | final val proj PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| strict_default_fdk | FDK | on | 13.58 | 13.27 | 13.27 (100) | 12.94 | 52.12 |
| strict_default_random_init | random | on | -10.45 | 5.24 | 13.30 (300) | 12.92 | 54.45 |

## Readout

- Random init catches up to nearly the same final 3D PSNR as FDK init.
- The gap is mostly in early iterations; final plateau remains nearly identical.
- This further supports that init is **not** the dominant final blocker.

## Data source

- `results/init_ablation_case001_50v.csv`
