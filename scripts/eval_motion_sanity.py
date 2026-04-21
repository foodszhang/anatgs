"""Step-1.6 B2/B3 motion sanity metrics on XCAT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu
from anatgs.dynamic import ContinuousTimeField, amsterdam_shroud, query_volume_condition


def _resize(vol: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if vol.shape == shape:
        return vol.astype(np.float32)
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)


def _load_gt_zyx(raw_dir: Path, t: int) -> np.ndarray:
    p = raw_dir / "ground_truth" / "volumes" / f"volume_{int(t)}.nii.gz"
    arr = np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32), dtype=np.float32)
    return hu_to_mu(np.transpose(arr, (2, 1, 0)).copy().astype(np.float32))


def _vtv_and_range(vols: np.ndarray) -> tuple[float, float]:
    # vols [T,Z,Y,X]
    var_map = np.var(vols, axis=0)
    rng_map = np.max(vols, axis=0) - np.min(vols, axis=0)
    return float(np.mean(var_map)), float(np.median(rng_map))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--out_json", default="results/step1_6_geom/B_motion_metrics.json")
    ap.add_argument("--res", type=int, default=64)
    args = ap.parse_args()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    bundle = np.load(args.bundle)
    t_idx = bundle["t_idx_at_view"].astype(np.int32)
    cond_all = bundle["signal_features"].astype(np.float32)
    rpm_view = bundle["rpm_at_view"].astype(np.float32)

    # one condition + rpm per timepoint
    cond_t = []
    rpm_t = []
    for t in range(182):
        idx = np.where(t_idx == t)[0]
        cond_t.append(cond_all[idx[0]])
        rpm_t.append(float(np.mean(rpm_view[idx])))
    cond_t = np.asarray(cond_t, dtype=np.float32)
    rpm_t = np.asarray(rpm_t, dtype=np.float32)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    mcfg = dict((ckpt.get("cfg", {}) or {}).get("model", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuousTimeField(mcfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device).eval()

    # 10 evenly sampled timepoints for VTV/range.
    t_sel = np.linspace(0, 181, 10).round().astype(np.int32)
    vols_model = []
    vols_gt = []
    with torch.no_grad():
        for t in t_sel:
            vm = query_volume_condition(model, condition=cond_t[int(t)], resolution=int(args.res)).detach().cpu().numpy()
            vg = _resize(_load_gt_zyx(raw_dir, int(t)), vm.shape)
            vols_model.append(vm.astype(np.float32))
            vols_gt.append(vg.astype(np.float32))
    vols_model_np = np.stack(vols_model, axis=0)
    vols_gt_np = np.stack(vols_gt, axis=0)
    vtv_model, range_med_model = _vtv_and_range(vols_model_np)
    vtv_gt, range_med_gt = _vtv_and_range(vols_gt_np)
    vtv_ratio = float(vtv_model / (vtv_gt + 1e-12))

    # B-c: shroud std ratio from full 182 timepoints at low resolution.
    with torch.no_grad():
        proj_recon = []
        for t in range(182):
            vol = query_volume_condition(model, condition=cond_t[t], resolution=32).detach()
            proj_recon.append(vol.sum(dim=-1))
        proj_recon = torch.stack(proj_recon, dim=0)
        s_recon = amsterdam_shroud(proj_recon).cpu().numpy().astype(np.float32)
    std_recon = float(np.std(s_recon))
    std_rpm = float(np.std(rpm_t))
    std_ratio = float(std_recon / (std_rpm + 1e-12))

    out = {
        "vtv_model": float(vtv_model),
        "vtv_gt": float(vtv_gt),
        "vtv_ratio_model_over_gt": vtv_ratio,
        "range_median_model": float(range_med_model),
        "range_median_gt": float(range_med_gt),
        "shroud_std_recon": std_recon,
        "shroud_std_rpm": std_rpm,
        "shroud_std_ratio_recon_over_rpm": std_ratio,
        "gates": {
            "B_b_vtv_ratio_ge_0_3": bool(vtv_ratio >= 0.3),
            "B_c_shroud_std_ratio_ge_0_1": bool(std_ratio >= 0.1),
        },
        "selected_timepoints": [int(x) for x in t_sel.tolist()],
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
