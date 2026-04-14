# Phase 1 Diagnosis (case001/10v)

## 判定
- **主要属于 A**：projection-side 指标持续变好，但 voxel-side PSNR 不同步提升。
- **不是 D 主因**：densify/prune 变量变化存在，但并未解释主要性能上限（见后续消融）。

## 证据
- Vanilla（修复后）：
  - val projection PSNR: `19.09 dB @100 -> 34.08 dB @500 -> 32.28 dB @1000`
  - global voxel PSNR: `13.09 dB @100 -> 13.46 dB @500 -> 13.39 dB @1000`
- AnatGS（修复后）：
  - val projection PSNR: `29.01 dB @100 -> 43.42 dB @500 -> 47.70 dB @1000`
  - global voxel PSNR: `14.21 dB @100 -> 13.43 dB @500 -> 13.29 dB @1000`

结论：当前 10-view 下训练目标主要被 projection 约束驱动，voxel 重建质量提升有限，说明还需要继续做更底层的建模/监督一致性改进，而不是单纯拉长训练步数。

