# Init diagnosis — case001/50v (strict baseline)

## Key findings

| init | init voxel PSNR | init MAE | init corr | density mean/std |
|---|---:|---:|---:|---:|
| FDK init (`init_case001_50v.npy`) | 13.58 dB | 0.1566 | 0.1172 | 0.0402 / 0.0255 |
| Random init (`init_case001_50v_random.npy`) | -10.45 dB | 3.2290 | -0.0195 | 0.5027 / 0.2880 |

- FDK init is much better than random at iter 0, but both converge to a similar final 3D PSNR plateau (~12.92–12.95 dB in strict runs).
- This indicates init quality affects **early convergence**, but is **not** the dominant reason final 3D quality is stuck.

## Artifacts

- Numeric stats: `results/init_stats_case001_50v.csv`
- Slice visuals: `results/init_vis_case001_50v/fdk_init_zmid.png`, `results/init_vis_case001_50v/random_init_zmid.png`

## Conclusion

H1 (init is the main bottleneck) is **not supported** as the primary root cause for the 50v final plateau.
