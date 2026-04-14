# Volume stats over training — case001/50v

## Snapshot summary

| snapshot | PSNR to GT | corr | MAE | mean/std | p1/p50/p99 | linear fit `pred≈a*gt+b` |
|---|---:|---:|---:|---:|---:|---:|
| init_fdk_voxelized | 13.58 | 0.1172 | 0.1566 | 0.2726 / 0.1415 | 0.0765 / 0.2273 / 0.7285 | a=0.0967, b=0.2472 |
| iter100 | 13.27 | 0.1002 | 0.1590 | 0.2512 / 0.1508 | 0.0619 / 0.1926 / 0.6922 | a=0.0881, b=0.2281 |
| iter1000 | 13.03 | 0.1084 | 0.1627 | 0.2591 / 0.1625 | 0.0702 / 0.1867 / 0.6928 | a=0.1028, b=0.2321 |
| iter5000 | 12.97 | 0.1109 | 0.1639 | 0.2609 / 0.1653 | 0.0731 / 0.1862 / 0.6985 | a=0.1069, b=0.2329 |
| iter30000 | 12.94 | 0.1120 | 0.1647 | 0.2615 / 0.1668 | 0.0648 / 0.1878 / 0.7049 | a=0.1089, b=0.2330 |

## Readout

- Correlation to GT remains low (~0.10–0.11) across training.
- Predicted volume is consistently compressed toward a low-contrast affine surrogate (small slope `a≈0.09–0.11`, positive bias `b≈0.23`).
- This is consistent with “projection fit improves, but recovered 3D structure remains weak”.

## Artifacts

- Numeric table: `results/volume_stats_case001_50v.csv`
- Scatter visualization: `results/volume_scatter_case001_50v.png`
