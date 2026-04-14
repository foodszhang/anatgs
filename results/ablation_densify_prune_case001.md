# Densify/Prune Ablation (case001/10v)

| setting | best global PSNR | best iter | final global PSNR | final-iter100 | best val proj PSNR | final gaussians | densify added | prune removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 13.5710 | 200 | 13.3944 | 0.3049 | 34.08 | 81534 | 31856 | 322 |
| no_densify_no_prune | 13.5709 | 200 | 13.3589 | 0.2692 | 39.12 | 50000 | 0 | 0 |
| no_densify | 13.5707 | 200 | 13.3621 | 0.2725 | 39.13 | 49819 | 0 | 181 |
| no_prune | 13.5709 | 200 | 13.3942 | 0.3046 | 34.07 | 81766 | 31766 | 0 |
