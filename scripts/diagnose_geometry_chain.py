"""Inspect geometry/coordinate conventions across GT, projection, model-forward and model-volume chains."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

import sys

sys.path.append("./src")

from anatgs.geom import describe_bundle_convention


def _first_volume(raw_dir: Path) -> Path:
    vols = sorted((raw_dir / "ground_truth" / "volumes").glob("*.nii.gz"))
    if not vols:
        raise FileNotFoundError(f"No NIfTI volumes under {raw_dir / 'ground_truth' / 'volumes'}")
    return vols[0]


def _row(name: str, start: str, end: str, axis: str, zdir: str, angle: str, spacing_origin: str) -> str:
    return (
        f"| {name} | {start} | {end} | {axis} | {zdir} | {angle} | {spacing_origin} |\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/xcat_miccai24/raw")
    ap.add_argument("--bundle", default="data/xcat_miccai24/projections/full_910v/bundle.npz")
    ap.add_argument("--out_dir", default="results/step1_6_geom")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = _first_volume(raw_dir)
    img = nib.load(str(p))
    ras = nib.as_closest_canonical(img)
    ax_raw = "".join(nib.aff2axcodes(img.affine))
    ax_ras = "".join(nib.aff2axcodes(ras.affine))
    spacing_raw = tuple(float(x) for x in img.header.get_zooms()[:3])
    spacing_ras = tuple(float(x) for x in ras.header.get_zooms()[:3])

    bconv = describe_bundle_convention(args.bundle)
    b = np.load(args.bundle)
    shape = tuple(int(x) for x in (b["projections"] if "projections" in b else b["projs"]).shape)

    # Expected model conventions from current code.
    l3_angle = "radian (ray.py uses cos/sin directly), +angle CCW around +Z, 0 at +X"
    l3_axis = "projections [N,V,U]; detector u-axis tangential, v-axis +Z in world"
    l4_axis = "volume query returns [Z,Y,X] over unit cube [0,1]^3"

    md = []
    md.append("# A1 Geometry Chain Inspection\n\n")
    md.append("| 链路 | 起点 | 终点 | axis顺序 | Z方向 | 角度约定 | spacing/origin |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    md.append(
        _row(
            "L1 GT链",
            "XCAT NIfTI raw",
            "GT tensor (pipeline)",
            f"raw nib axis={ax_raw}; canonical={ax_ras}; pipeline uses [Z,Y,X]",
            "pipeline assumes index-z increasing toward +Z world",
            "N/A",
            f"raw spacing={spacing_raw} mm; canonical spacing={spacing_ras} mm; origin from NIfTI affine",
        )
    )
    md.append(
        _row(
            "L2 投影链",
            "NIfTI -> gen_xcat_projections.py",
            "bundle.npz",
            f"bundle projections shape={shape}, axes={bconv['projection_axes']}",
            f"projection_v_flipped={bconv['projection_v_flipped']}",
            f"bundle angle_unit(raw={bconv['angle_unit_raw']}, detected={bconv['angle_unit_detected']}, used={bconv['angle_unit_used']})",
            "TIGRE geo: DSO/DSD from config, XCAT voxel spacing 1x1x3 mm",
        )
    )
    md.append(
        _row(
            "L3 模型forward",
            "bundle -> ProjectionDataset",
            "render_ray_batch pred projection",
            l3_axis,
            "ray.py det_y=[0,0,1] (+Z)",
            l3_angle,
            "world box centered at origin, side=volume_size_mm",
        )
    )
    md.append(
        _row(
            "L4 模型volume",
            "canonical INR + M1/M2",
            "mu(x,t)",
            l4_axis,
            "map_points works in unit coords; inverse warp for canonical query",
            "conditioned by time/signal (not projection angle)",
            "unit-grid domain, no explicit physical origin in volume query",
        )
    )

    mismatches = []
    if bconv["angle_unit_detected"] != "radian":
        mismatches.append(
            "- **角度单位不一致**：bundle检测为 degree，而 L3 ray forward 直接按 radian 使用；会导致严重几何错误。"
        )
    if str(bconv["projection_axes"]) != "[N,V,U]":
        mismatches.append("- **投影轴约定不一致**：bundle projection_axes 不是 [N,V,U]。")
    if bconv["projection_v_flipped"] == "unknown":
        mismatches.append("- **detector v 方向未显式标注**：bundle缺失 projection_v_flipped 元信息。")
    if not mismatches:
        mismatches.append("- 当前检查未发现显式单位/元信息不一致；若性能异常需继续核查 rpm-phase 映射与 bbox 尺度。")

    (out_dir / "A1_chain_inspection.md").write_text("".join(md), encoding="utf-8")
    (out_dir / "A2_mismatch_list.md").write_text(
        "# A2 Mismatch List\n\n" + "\n".join(mismatches) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {(out_dir / 'A1_chain_inspection.md')}")
    print(f"Saved {(out_dir / 'A2_mismatch_list.md')}")


if __name__ == "__main__":
    main()
