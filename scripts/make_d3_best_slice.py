"""Create D3 best-vs-GT slice figure for a static checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.append("./src")

from anatgs.data.xcat import hu_to_mu
from anatgs.dynamic import ContinuousTimeField, query_volume


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    vols = sorted((Path(args.raw_dir) / "ground_truth" / "volumes").glob("*.nii.gz"))
    gt = []
    for p in vols:
        xyz = np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32), dtype=np.float32)
        gt.append(hu_to_mu(np.transpose(xyz, (2, 1, 0)).copy().astype(np.float32)))
    gt_mean = np.mean(np.stack(gt, axis=0), axis=0).astype(np.float32)

    ck = torch.load(args.ckpt, map_location="cpu")
    mcfg = dict((ck.get("cfg", {}) or {}).get("model", {}))
    model = ContinuousTimeField(mcfg)
    model.load_state_dict(ck["model"], strict=False)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    pred = query_volume(model, t_value=0.0, resolution=96).detach().cpu().numpy().astype(np.float32)
    pred_t = torch.from_numpy(pred)[None, None]
    pred_rs = (
        F.interpolate(pred_t, size=gt_mean.shape, mode="trilinear", align_corners=False)[0, 0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    z = gt_mean.shape[0] // 2
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(gt_mean[z], cmap="gray")
    axs[0].set_title("GT mean")
    axs[1].imshow(pred_rs[z], cmap="gray")
    axs[1].set_title("D3 best")
    im = axs[2].imshow(pred_rs[z] - gt_mean[z], cmap="bwr")
    axs[2].set_title("diff")
    for ax in axs:
        ax.axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
