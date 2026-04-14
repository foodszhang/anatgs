# Strict Vanilla Summary — case001/50v

## Key results

| Item | Value |
|---|---:|
| Training iterations | 30000 |
| Best global voxel PSNR (train monitor) | 13.2693 dB @ iter 100 |
| Final global voxel PSNR (train monitor) | 12.9436 dB @ iter 30000 |
| Final validation projection PSNR | 52.1216 dB |
| Official `test.py` 3D PSNR | 12.9436 dB |
| Official `test.py` 3D SSIM | 0.1792 |

## Convergence shape

- Projection-side quality improves strongly (val projection PSNR grows to ~52 dB).
- Voxel-side global PSNR peaks very early and then drifts downward to a low plateau (~12.94 dB).
- This reproduces the projection/voxel decoupling pattern under strict vanilla settings (no AnatGS branches).

## Baseline sanity verdict (50v)

Strict vanilla is **runnable and reproducible**, but reconstruction quality is still **not in a healthy baseline range** for CT benchmarking on this data chain.
