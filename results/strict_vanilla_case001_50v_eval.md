# Strict Vanilla Eval — case001/50v

## Run setup

- **Config**: `configs/gs_r2_strict.yaml` (no anatomy init / no organ-aware prune / no seg input)
- **Data**: `data/anatcoder_cases/case001_50v`
- **Model output**: `output/strict_vanilla_case001_50v`
- **Convergence CSV**: `results/strict_vanilla_case001_50v_convergence.csv`

## Commands

```bash
conda run -n ntorch python train.py \
  -s data/anatcoder_cases/case001_50v \
  -m output/strict_vanilla_case001_50v \
  --config configs/gs_r2_strict.yaml \
  --eval_interval 100 \
  --debug_metrics_csv results/strict_vanilla_case001_50v_convergence.csv

conda run -n ntorch python test.py \
  -m output/strict_vanilla_case001_50v \
  -s data/anatcoder_cases/case001_50v \
  --iteration -1 \
  --skip_render_test
```

> `--skip_render_test` is required because this scene has no `proj_test` split.

## Official `test.py` metrics (iter 30000)

- **2D render_train PSNR**: `52.0517 dB`
- **2D render_train SSIM**: `0.9984`
- **3D reconstruction PSNR**: `12.9436 dB`
- **3D reconstruction SSIM**: `0.1792`

## Train monitor vs test.py consistency

- Train monitor at iter 30000: `global_PSNR = 12.9436 dB`
- `test.py` eval3d at iter 30000: `psnr_3d = 12.9436 dB`

The 3D PSNR values are numerically aligned (same query/eval path).
