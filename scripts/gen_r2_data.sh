#!/usr/bin/env bash
set -euo pipefail

# One-shot generation of R²-Gaussian format data from TotalSegmentator CTs.
#
# Usage:
#   bash scripts/gen_r2_data.sh
#   bash scripts/gen_r2_data.sh /path/to/raw /path/to/out case001

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${1:-/home/foods/pro/anatcoder/data/raw}"
OUT_ROOT="${2:-${ROOT_DIR}/data/r2_format}"
CASE_FILTER="${3:-}"
TARGET_SIZE="${TARGET_SIZE:-256}"
N_TEST="${N_TEST:-100}"
N_TRAIN_LIST="${N_TRAIN_LIST:-50 10}"

PREP_SCRIPT="${ROOT_DIR}/scripts/prep_totalseg_for_r2.py"
GEN_SCRIPT="${ROOT_DIR}/data_generator/synthetic_dataset/generate_data.py"
INIT_SCRIPT="${ROOT_DIR}/initialize_pcd.py"
SCANNER_CFG="${ROOT_DIR}/data_generator/synthetic_dataset/scanner/cone_beam.yml"
VOL_PROC_DIR="${ROOT_DIR}/data_generator/volume_processed"

if [[ ! -f "${GEN_SCRIPT}" || ! -f "${INIT_SCRIPT}" || ! -f "${SCANNER_CFG}" ]]; then
  echo "Missing official pipeline files under ${ROOT_DIR}." >&2
  exit 1
fi

mkdir -p "${VOL_PROC_DIR}" "${OUT_ROOT}"

mapfile -t CASE_DIRS < <(find "${RAW_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "case*" | sort)
if [[ -n "${CASE_FILTER}" ]]; then
  CASE_DIRS=("${RAW_ROOT}/${CASE_FILTER}")
fi

if [[ ${#CASE_DIRS[@]} -eq 0 ]]; then
  echo "No case directories found in ${RAW_ROOT}" >&2
  exit 1
fi

for case_dir in "${CASE_DIRS[@]}"; do
  case_name="$(basename "${case_dir}")"
  ct_path="${case_dir}/ct.nii.gz"
  if [[ ! -f "${ct_path}" ]]; then
    echo "[skip] ${case_name}: missing ${ct_path}"
    continue
  fi

  vol_out="${VOL_PROC_DIR}/${case_name}.npy"
  echo "[prep] ${case_name} -> ${vol_out}"
  python "${PREP_SCRIPT}" --input "${ct_path}" --output "${vol_out}" --target_size "${TARGET_SIZE}"

  for n_train in ${N_TRAIN_LIST}; do
    out_dir="${OUT_ROOT}/cone_ntrain_${n_train}_angle_360"
    echo "[gen] ${case_name} n_train=${n_train} -> ${out_dir}"
    python "${GEN_SCRIPT}" \
      --vol "${vol_out}" \
      --scanner "${SCANNER_CFG}" \
      --output "${out_dir}" \
      --n_train "${n_train}" \
      --n_test "${N_TEST}"

    case_scene="${out_dir}/${case_name}_cone"
    init_path="${case_scene}/init_${case_name}_cone.npy"
    if [[ -f "${init_path}" ]]; then
      echo "[init] ${case_name} n_train=${n_train}: existing init found, skip"
    else
      echo "[init] ${case_name} n_train=${n_train}: generating init"
      python "${INIT_SCRIPT}" --data "${case_scene}" --evaluate
    fi
  done
done

echo "Done. Output root: ${OUT_ROOT}"
