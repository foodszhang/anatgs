#!/usr/bin/env python3
"""Prepare TotalSegmentator CT/seg for R²-Gaussian synthetic pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk

# 10-class merge map aligned with AnatGS/AnatCoder usage.
CLASS_NAMES: dict[int, str] = {
    0: "background",
    1: "soft_tissue",
    2: "bone",
    3: "liver",
    4: "kidney",
    5: "spleen",
    6: "pancreas",
    7: "heart_vessels",
    8: "lung",
    9: "gi_tract",
}

MERGE_BY_NAME: dict[int, set[str]] = {
    2: {
        "clavicula_left",
        "clavicula_right",
        "femur_left",
        "femur_right",
        "hip_left",
        "hip_right",
        "sacrum",
        "scapula_left",
        "scapula_right",
        *{f"vertebrae_C{i}" for i in [6, 7]},
        *{f"vertebrae_T{i}" for i in range(1, 13)},
        *{f"vertebrae_L{i}" for i in range(1, 6)},
        *{f"rib_left_{i}" for i in range(1, 13)},
        *{f"rib_right_{i}" for i in range(1, 13)},
    },
    3: {"liver"},
    4: {"kidney_left", "kidney_right"},
    5: {"spleen"},
    6: {"pancreas"},
    7: {
        "aorta",
        "inferior_vena_cava",
        "pulmonary_artery",
        "portal_vein_and_splenic_vein",
        "heart",
        "heart_atrium_left",
        "heart_atrium_right",
        "heart_myocardium",
        "heart_ventricle_left",
        "heart_ventricle_right",
    },
    8: {
        "lung_lower_lobe_left",
        "lung_lower_lobe_right",
        "lung_middle_lobe_right",
        "lung_upper_lobe_left",
        "lung_upper_lobe_right",
    },
    9: {"stomach", "duodenum", "small_bowel", "colon"},
}

SOFT_TISSUE_FALLBACK: set[str] = {
    "adrenal_gland_left",
    "adrenal_gland_right",
    "autochthon_left",
    "autochthon_right",
    "esophagus",
    "gallbladder",
    "gluteus_maximus_left",
    "gluteus_maximus_right",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "gluteus_minimus_left",
    "gluteus_minimus_right",
    "humerus_left",
    "humerus_right",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    "iliopsoas_left",
    "iliopsoas_right",
    "trachea",
    "urinary_bladder",
}


def expand_to_cube(array: np.ndarray) -> np.ndarray:
    max_dim = max(array.shape)
    padding = [(max_dim - s) // 2 for s in array.shape]
    padding = [(pad, max_dim - s - pad) for pad, s in zip(padding, array.shape)]
    return np.pad(array, padding, mode="constant", constant_values=0)


def resample_isotropic_1mm(
    volume_zyx: np.ndarray, spacing_zyx: np.ndarray, order: int
) -> np.ndarray:
    resize_factor = spacing_zyx / np.array([1.0, 1.0, 1.0], dtype=np.float32)
    new_real_shape = np.array(volume_zyx.shape, dtype=np.float32) * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / np.array(volume_zyx.shape, dtype=np.float32)
    if np.allclose(real_resize_factor, 1.0, atol=1e-3):
        return volume_zyx
    return ndimage.zoom(volume_zyx, real_resize_factor, mode="nearest", order=order)


def resize_to_target_cube(volume: np.ndarray, target_size: int, order: int) -> np.ndarray:
    factors = np.array([target_size, target_size, target_size], dtype=np.float32) / np.array(
        volume.shape, dtype=np.float32
    )
    if np.allclose(factors, 1.0, atol=1e-6):
        return volume
    return ndimage.zoom(volume, factors, mode="nearest", order=order)


def preprocess_spatial(volume_zyx: np.ndarray, spacing_zyx: np.ndarray, target_size: int, order: int) -> np.ndarray:
    vol = resample_isotropic_1mm(volume_zyx, spacing_zyx, order=order)
    vol = expand_to_cube(vol)
    vol = resize_to_target_cube(vol, target_size, order=order)
    return vol


def prepare_ct(input_nii: str, target_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = sitk.ReadImage(input_nii)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # z,y,x in HU
    spacing_zyx = np.array(image.GetSpacing(), dtype=np.float32)[::-1]  # SITK is x,y,z

    volume = np.clip(volume, -1000.0, 2000.0)
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax <= vmin:
        raise ValueError(f"Invalid HU range after clipping: min={vmin}, max={vmax}")
    volume = (volume - vmin) / (vmax - vmin)

    volume = preprocess_spatial(volume, spacing_zyx, target_size, order=1)
    volume = np.clip(volume, 0.0, 1.0).astype(np.float32)
    return volume, spacing_zyx


def _class_for_totalseg_name(name: str) -> int:
    for class_id, members in MERGE_BY_NAME.items():
        if name in members:
            return class_id
    if name in SOFT_TISSUE_FALLBACK:
        return 1
    return 0


def merge_seg_from_directory(seg_dir: Path, ref_image: sitk.Image) -> np.ndarray:
    seg_files = sorted(list(seg_dir.glob("*.nii.gz")) + list(seg_dir.glob("*.nii")))
    if not seg_files:
        raise FileNotFoundError(f"No segmentation files found in {seg_dir}")
    merged = np.zeros(sitk.GetArrayFromImage(ref_image).shape, dtype=np.int16)
    for seg_file in seg_files:
        organ = seg_file.name.replace(".nii.gz", "").replace(".nii", "")
        class_id = _class_for_totalseg_name(organ)
        if class_id == 0:
            continue
        src = sitk.ReadImage(str(seg_file))
        resampled = sitk.Resample(
            src,
            ref_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            src.GetPixelID(),
        )
        mask = sitk.GetArrayFromImage(resampled) > 0
        merged[mask] = int(class_id)
    return merged


def prepare_seg(
    seg_input: str | None, seg_dir: str | None, ct_input: str, spacing_zyx: np.ndarray, target_size: int
) -> np.ndarray | None:
    seg_arr: np.ndarray | None = None
    ct_img = sitk.ReadImage(ct_input)
    if seg_input:
        seg_img = sitk.ReadImage(seg_input)
        seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.int16)
    elif seg_dir:
        seg_arr = merge_seg_from_directory(Path(seg_dir), ct_img)
    else:
        auto_seg = Path(ct_input).with_name("segmentation.nii.gz")
        if auto_seg.exists():
            seg_img = sitk.ReadImage(str(auto_seg))
            seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.int16)
    if seg_arr is None:
        return None
    if seg_arr.shape != sitk.GetArrayFromImage(ct_img).shape:
        # Align to CT grid first when seg has different size/origin/spacing.
        if seg_input:
            seg_src = sitk.ReadImage(seg_input)
        else:
            seg_src = sitk.ReadImage(str(Path(ct_input).with_name("segmentation.nii.gz")))
        seg_resampled = sitk.Resample(
            seg_src,
            ct_img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            seg_src.GetPixelID(),
        )
        seg_arr = sitk.GetArrayFromImage(seg_resampled).astype(np.int16)
    seg_arr = preprocess_spatial(seg_arr.astype(np.float32), spacing_zyx, target_size, order=0)
    seg_arr = np.rint(seg_arr).astype(np.int16)
    seg_arr[seg_arr < 0] = 0
    seg_arr[seg_arr > 9] = 0
    return seg_arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to case CT .nii/.nii.gz")
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--seg_input", default=None, help="Path to merged segmentation .nii/.nii.gz")
    parser.add_argument("--seg_dir", default=None, help="Path to TotalSegmentator segmentations dir")
    parser.add_argument("--seg_output", default=None, help="Output .npy path for merged 10-class seg")
    parser.add_argument("--target_size", type=int, default=256, help="Output cube size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vol, spacing_zyx = prepare_ct(args.input, args.target_size)
    np.save(output_path, vol)
    print(
        "Saved",
        str(output_path),
        f"shape={tuple(vol.shape)}",
        f"dtype={vol.dtype}",
        f"range=[{vol.min():.6f},{vol.max():.6f}]",
        f"mean={vol.mean():.6f}",
        f"std={vol.std():.6f}",
    )

    if args.seg_output is not None:
        seg = prepare_seg(
            seg_input=args.seg_input,
            seg_dir=args.seg_dir,
            ct_input=args.input,
            spacing_zyx=spacing_zyx,
            target_size=args.target_size,
        )
        if seg is None:
            print("No segmentation source found; skip seg output.")
        else:
            seg_output_path = Path(args.seg_output)
            seg_output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(seg_output_path, seg)
            uniq = np.unique(seg)
            print(
                "Saved",
                str(seg_output_path),
                f"shape={tuple(seg.shape)}",
                f"dtype={seg.dtype}",
                f"labels={uniq.tolist()}",
            )


if __name__ == "__main__":
    main()
