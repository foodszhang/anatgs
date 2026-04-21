"""Q2 diagnostic: shroud signal degradation analysis on S1-C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu
from anatgs.dynamic import ContinuousTimeField, amsterdam_shroud, query_volume_condition


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64) - float(np.mean(a))
    bb = b.astype(np.float64) - float(np.mean(b))
    denom = np.sqrt(float(np.sum(aa**2) * np.sum(bb**2)) + 1e-12)
    return float(np.sum(aa * bb) / denom)


def _load_gt_zyx(raw_dir: Path, t: int) -> np.ndarray:
    xyz = np.asarray(
        nib.load(str(raw_dir / "ground_truth" / "volumes" / f"volume_{int(t)}.nii.gz")).get_fdata(dtype=np.float32),
        dtype=np.float32,
    )
    return hu_to_mu(np.transpose(xyz, (2, 1, 0)).copy().astype(np.float32))


def _shift_z(vol: torch.Tensor, shift_vox: float) -> torch.Tensor:
    # vol [D,H,W]
    d, h, w = vol.shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, d, device=vol.device, dtype=vol.dtype),
        torch.linspace(-1.0, 1.0, h, device=vol.device, dtype=vol.dtype),
        torch.linspace(-1.0, 1.0, w, device=vol.device, dtype=vol.dtype),
        indexing="ij",
    )
    dz = 2.0 * float(shift_vox) / max(d - 1, 1)
    grid = torch.stack([xx, yy, zz - dz], dim=-1)[None]
    out = F.grid_sample(vol[None, None], grid, mode="bilinear", align_corners=True, padding_mode="border")
    return out[0, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/step1_xcat/S1-C/model_iter_2000.pt")
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--out_dir", default="results/step1_5_diag")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    bundle = np.load(args.bundle)
    t_idx = bundle["t_idx_at_view"].astype(np.int32)
    cond_all = bundle["signal_features"].astype(np.float32)
    rpm_view = bundle["rpm_at_view"].astype(np.float32)

    # one condition per timepoint
    cond_t = []
    rpm_t = []
    for t in range(182):
        idx = np.where(t_idx == t)[0]
        cond_t.append(cond_all[idx[0]])
        rpm_t.append(float(np.mean(rpm_view[idx])))
    cond_t = np.asarray(cond_t, dtype=np.float32)
    rpm_t = np.asarray(rpm_t, dtype=np.float32)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    mcfg = dict((ckpt.get("cfg", {}) or {}).get("model", {})
                )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuousTimeField(mcfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device).eval()

    # Reconstructed shroud signal
    with torch.no_grad():
        proj_recon = []
        for t in range(182):
            vol = query_volume_condition(model, condition=cond_t[t], resolution=64).detach()
            proj_recon.append(vol.sum(dim=-1))
        proj_recon = torch.stack(proj_recon, dim=0)
        s_recon = amsterdam_shroud(proj_recon).cpu().numpy().astype(np.float32)

    # GT shroud signal
    proj_gt = []
    for t in range(182):
        gt = _load_gt_zyx(raw_dir, t)
        proj_gt.append(torch.from_numpy(gt).sum(dim=-1))
    proj_gt = torch.stack(proj_gt, dim=0)
    s_gt = amsterdam_shroud(proj_gt).cpu().numpy().astype(np.float32)

    # correlations / std
    corr = {
        "corr_recon_rpm": _corr(s_recon, rpm_t),
        "corr_recon_gtshroud": _corr(s_recon, s_gt),
        "corr_gtshroud_rpm": _corr(s_gt, rpm_t),
    }
    corr_abs = {f"abs_{k}": float(abs(v)) for k, v in corr.items()}
    stds = {
        "std_recon": float(np.std(s_recon)),
        "std_gt_shroud": float(np.std(s_gt)),
        "std_rpm": float(np.std(rpm_t)),
    }

    np.savez_compressed(
        out_dir / "Q2_signals.npz",
        s_recon=s_recon,
        s_gt_shroud=s_gt,
        s_rpm=rpm_t,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(s_recon, label="s_recon")
    plt.plot(s_gt, label="s_gt_shroud")
    plt.plot(rpm_t, label="s_rpm")
    plt.legend()
    plt.xlabel("timepoint")
    plt.ylabel("signal")
    plt.tight_layout()
    plt.savefig(out_dir / "Q2_signal_traces.png", dpi=180)
    plt.close()

    out_corr = {**corr, **corr_abs, **stds}
    (out_dir / "Q2_corr_matrix.json").write_text(json.dumps(out_corr, indent=2), encoding="utf-8")

    # synthetic shift test
    base = torch.from_numpy(_load_gt_zyx(raw_dir, 90))
    n = 64
    phase = np.sin(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)).astype(np.float32)
    shifted_proj = []
    for i in range(n):
        shift_mm = 2.5 * (phase[i] + 1.0)  # 0..5mm
        shift_vox = float(shift_mm / 3.0)  # z spacing = 3mm
        sv = _shift_z(base, shift_vox)
        shifted_proj.append(sv.sum(dim=-1))
    shifted_proj = torch.stack(shifted_proj, dim=0)
    s_shift = amsterdam_shroud(shifted_proj).cpu().numpy().astype(np.float32)
    corr_shift = _corr(s_shift, phase)
    out_shift = {"corr": float(corr_shift)}
    (out_dir / "Q2_synthetic_shift_test.json").write_text(json.dumps(out_shift, indent=2), encoding="utf-8")

    print(json.dumps({"corr": out_corr, "synthetic_shift": out_shift}, indent=2))


if __name__ == "__main__":
    main()
