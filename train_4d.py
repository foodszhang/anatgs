"""Train a phase-free continuous-time attenuation field on 4D projection bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.append("./src")

from anatgs.dynamic import (
    ContinuousTimeField,
    ProjectionDataset,
    projection_mse_loss,
    query_volume,
    render_ray_batch,
    temporal_smoothness_loss,
)
from anatgs.dynamic.dataset import GeometryConfig


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _geo_from_cfg(geo_cfg: dict, n_samples: int) -> GeometryConfig:
    n_detector = geo_cfg.get("nDetector", None)
    d_detector = geo_cfg.get("dDetector", None)
    s_voxel = geo_cfg.get("sVoxel", None)
    if n_detector is not None and len(n_detector) == 2:
        det_h_default = int(n_detector[0])
        det_w_default = int(n_detector[1])
    else:
        det_h_default = 256
        det_w_default = 256
    if d_detector is not None and len(d_detector) == 2:
        det_spacing_h_default = float(d_detector[0])
        det_spacing_w_default = float(d_detector[1])
    else:
        det_spacing_h_default = 1.5
        det_spacing_w_default = 1.5
    if s_voxel is not None and len(s_voxel) == 3:
        volume_size_default = float(max(s_voxel))
    else:
        volume_size_default = 384.0
    return GeometryConfig(
        det_h=int(geo_cfg.get("det_h", det_h_default)),
        det_w=int(geo_cfg.get("det_w", det_w_default)),
        det_spacing_h=float(geo_cfg.get("det_spacing_h", det_spacing_h_default)),
        det_spacing_w=float(geo_cfg.get("det_spacing_w", det_spacing_w_default)),
        sod=float(geo_cfg.get("sod", geo_cfg.get("DSO", 750.0))),
        sdd=float(geo_cfg.get("sdd", geo_cfg.get("DSD", 1200.0))),
        volume_size_mm=float(geo_cfg.get("volume_size_mm", volume_size_default)),
        n_samples=int(n_samples),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Projection bundle .npz produced by scripts/gen_4d_projections.py")
    ap.add_argument("--config", default="configs/4d_continuous.yaml")
    ap.add_argument("--geo_config", default="configs/4d_cbct_geo.yaml")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--lambda_temporal", type=float, default=None)
    ap.add_argument("--time_mode", choices=["continuous", "fixed0"], default="continuous")
    ap.add_argument("--phase_filter", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--volume_res", type=int, default=128)
    ap.add_argument("--projection_mode", choices=["line_integral", "beer_lambert"], default=None)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--projection_scale", type=float, default=None)
    ap.add_argument("--lambda_contrast", type=float, default=None)
    ap.add_argument("--contrast_ref_volume", default="", help="Static reference volume .npy for stats matching.")
    ap.add_argument("--contrast_points", type=int, default=None)
    ap.add_argument("--contrast_interval", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    geo_cfg = _load_yaml(args.geo_config)
    train_cfg = dict(cfg.get("train", {}))
    model_cfg = dict(cfg.get("model", {}))

    iterations = int(args.iterations if args.iterations is not None else train_cfg.get("iterations", 50000))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 4096))
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-3))
    n_samples = int(args.n_samples if args.n_samples is not None else train_cfg.get("n_samples", 128))
    lambda_temporal = float(
        args.lambda_temporal if args.lambda_temporal is not None else train_cfg.get("lambda_temporal", 1e-3)
    )
    eval_every = int(args.eval_every if args.eval_every is not None else train_cfg.get("eval_every", 1000))
    save_every = int(args.save_every if args.save_every is not None else train_cfg.get("save_every", 5000))
    projection_mode = str(
        args.projection_mode if args.projection_mode is not None else train_cfg.get("projection_mode", "line_integral")
    )
    grad_clip = float(args.grad_clip if args.grad_clip is not None else train_cfg.get("grad_clip", 1.0))
    projection_scale = float(
        args.projection_scale if args.projection_scale is not None else train_cfg.get("projection_scale", 1.0)
    )
    if projection_scale <= 0:
        raise ValueError("projection_scale must be > 0")
    lambda_contrast = float(
        args.lambda_contrast if args.lambda_contrast is not None else train_cfg.get("lambda_contrast", 0.0)
    )
    contrast_points = int(args.contrast_points if args.contrast_points is not None else train_cfg.get("contrast_points", 8192))
    contrast_interval = int(
        args.contrast_interval if args.contrast_interval is not None else train_cfg.get("contrast_interval", 10)
    )
    contrast_ref_volume = str(args.contrast_ref_volume or train_cfg.get("contrast_ref_volume", "")).strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geo = _geo_from_cfg(geo_cfg, n_samples=n_samples)
    ds = ProjectionDataset(args.data, geo, phase_filter=args.phase_filter, time_mode=args.time_mode)
    model = ContinuousTimeField(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ref_vol = None
    ref_shape = None
    if lambda_contrast > 0.0:
        if not contrast_ref_volume:
            raise ValueError("--lambda_contrast > 0 requires --contrast_ref_volume")
        rv = np.load(contrast_ref_volume).astype(np.float32)
        if rv.ndim != 3:
            raise ValueError(f"contrast_ref_volume must be 3D, got {rv.shape}")
        ref_vol = torch.from_numpy(rv).to(device=device)
        ref_shape = torch.tensor(ref_vol.shape, device=device, dtype=torch.float32)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("iter,loss_proj,loss_temp,loss_contrast,loss_total\n")

    for it in range(1, iterations + 1):
        batch = ds.sample_batch(batch_size=batch_size, device=device)
        pred = render_ray_batch(
            model,
            batch["points_unit"],
            batch["timestamps"],
            batch["delta"],
            projection_mode=projection_mode,
        )
        loss_proj = projection_mse_loss(pred / projection_scale, batch["target"] / projection_scale)
        xyz_rand = torch.rand((batch_size, 3), device=device)
        t_rand = torch.rand((batch_size, 1), device=device)
        loss_temp = temporal_smoothness_loss(model, xyz=xyz_rand, t=t_rand, dt=0.01)
        loss_contrast = torch.zeros((), device=device)
        if (
            lambda_contrast > 0.0
            and ref_vol is not None
            and contrast_points > 0
            and contrast_interval > 0
            and (it % contrast_interval == 0)
        ):
            n = int(contrast_points)
            idx = torch.randint(0, int(ref_vol.numel()), (n,), device=device)
            z = idx // int(ref_vol.shape[1] * ref_vol.shape[2])
            y = (idx // int(ref_vol.shape[2])) % int(ref_vol.shape[1])
            x = idx % int(ref_vol.shape[2])
            xyz = torch.stack(
                [
                    (x.float() + 0.5) / ref_shape[2],
                    (y.float() + 0.5) / ref_shape[1],
                    (z.float() + 0.5) / ref_shape[0],
                ],
                dim=-1,
            )
            t_ref = torch.full((n, 1), float(batch["timestamps"].mean().item()), device=device)
            pred_vals = model(xyz, t_ref).squeeze(-1)
            ref_vals = ref_vol[z, y, x]
            loss_contrast = (pred_vals.mean() - ref_vals.mean()) ** 2 + (
                pred_vals.std(unbiased=False) - ref_vals.std(unbiased=False)
            ) ** 2
        loss = loss_proj + lambda_temporal * loss_temp + lambda_contrast * loss_contrast

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{it},{loss_proj.item():.8f},{loss_temp.item():.8f},{loss_contrast.item():.8f},{loss.item():.8f}\n"
            )

        if it % eval_every == 0 or it == 1:
            print(
                f"[iter {it}] loss_proj={loss_proj.item():.6f} "
                f"loss_temp={loss_temp.item():.6f} loss_contrast={loss_contrast.item():.6f} total={loss.item():.6f}"
            )

        if it % save_every == 0 or it == iterations:
            ckpt = out_dir / f"model_iter_{it}.pt"
            torch.save({"iter": it, "model": model.state_dict(), "cfg": cfg}, ckpt)

    # Export phase snapshots for eval_4d.py.
    # If data was generated with multiple breathing cycles, map phase p to the
    # matching timestamp in the first cycle to keep phase/time alignment.
    n_cycles = 1.0
    data_meta = Path(args.data).with_name("meta.json")
    if data_meta.exists():
        with data_meta.open("r", encoding="utf-8") as f:
            n_cycles = float((json.load(f) or {}).get("n_breath_cycles", 1.0))
    n_cycles = max(n_cycles, 1e-8)
    export_t_values: list[float] = []
    for p in range(10):
        t = (p / 10.0) / n_cycles
        export_t_values.append(float(t))
        vol = query_volume(model, t_value=t, resolution=int(args.volume_res)).detach().cpu().numpy().astype(np.float32)
        np.save(out_dir / f"pred_phase_{p:02d}.npy", vol)
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data": args.data,
                "iterations": iterations,
                "batch_size": batch_size,
                "lr": lr,
                "n_samples": n_samples,
                "lambda_temporal": lambda_temporal,
                "projection_mode": projection_mode,
                "grad_clip": grad_clip,
                "projection_scale": projection_scale,
                "lambda_contrast": lambda_contrast,
                "contrast_points": contrast_points,
                "contrast_interval": contrast_interval,
                "contrast_ref_volume": contrast_ref_volume,
                "time_mode": args.time_mode,
                "phase_filter": args.phase_filter,
                "export_n_breath_cycles": n_cycles,
                "export_t_values": export_t_values,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
