# Final Diagnosis (case001/10v, AnatGS P0 blocking)

## Top-3 root causes

1. **TIGRE→R² conversion detector spacing 写错（1.0 mm 而非 1.5 mm）**
   - **证据**：修复前几何 sanity（同一 GT volume 正向投影）`global MAE=12.2955`, `PSNR=-25.19 dB`；修复后 `MAE=0`, `PSNR=inf`，逐视角完全一致。
   - **影响**：baseline 学的是错误几何，直接导致低 PSNR 和异常收敛形态。

2. **Anatomy init 坐标域错误（体素索引坐标直接喂给高斯）**
   - **证据**：修复前 `init_case001_10v_anat.npy` 坐标范围约 `[0,127]`；vanilla init 范围约 `[-1,0.984]`。修复前 AnatGS 训练中出现大规模异常点裁剪（高斯数从 `213782` 级别快速塌缩）。
   - **修复**：在 `src/anatgs/anatomy/init.py` 中将采样点映射到 `[-1,1]` 并 clamp。

3. **projection 优化与 voxel 质量提升不一致（10-view 下可观测）**
   - **证据**：修复后 projection PSNR 大幅上升（vanilla 最高 `34.08 dB`, anatgs 最高 `47.70 dB`），但 global voxel PSNR 仅在 `13~14 dB`，并未随 projection 指标同比例提升。
   - **消融结论**：`disable_densify/prune` 后 global voxel PSNR 几乎不变（best 都约 `13.57 dB`），说明 densify/prune 不是当前上限主因。

## 已修改文件

- `src/anatgs/data/convert_tigre.py`
- `scripts/convert_tigre_to_r2.py`
- `scripts/sanity_check_geometry.py`
- `src/anatgs/anatomy/init.py`
- `r2_gaussian/gaussian/gaussian_model.py`
- `train.py`
- `scripts/check_voxel_stats.py`
- `scripts/compute_organ_stats.py`

## 修复前 vs 修复后（case001/10v）

| method | 修复前 best global PSNR | 修复前 final | 修复后 best global PSNR | 修复后 final |
|---|---:|---:|---:|---:|
| vanilla_3dgs | 10.92 dB @100 | 10.01 dB | 13.57 dB @200 | 13.39 dB |
| anatgs_oracle | 12.71 dB @100 | 10.35 dB | 14.21 dB @100 | 13.29 dB |

## 未完全解决项

- 10-view 条件下，projection 指标高但 voxel PSNR 上限仍低（~13-14 dB）。
- `organ_params` 的 `init_opacity` 相对 case001 器官强度均值普遍偏低（见 `results/organ_params_calibration.md`），但这不是当前最大瓶颈。

## Go / No-Go

**No-Go（暂不进入 5-case 主实验）**。  
几何与 anatomy-init 坐标问题已修复并显著提升 baseline，但 vanilla 仍未达到可扩展批量实验的稳态质量阈值（10-view 仍低于 15 dB）。应先继续最小闭环优化（监督一致性/正则/初始化策略）后再扩展 5-case。

