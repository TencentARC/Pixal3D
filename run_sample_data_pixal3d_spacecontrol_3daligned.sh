#!/usr/bin/env bash
set -euo pipefail

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-pixal3d}"
fi

cd "$(dirname "$0")"

GPU="${GPU:-2}"
SEED="${SEED:-42}"
TAUS="${TAUS:-3 4 6}"
VIDEO_BG_COLOR="${VIDEO_BG_COLOR:-white}"
CONTROL="${CONTROL:-/root/dev/TRELLIS.2/assets/sample_data/DH-001_SZ270/DH-001_SZ270_SURF_bbox_normalized.ply}"
THREADS="${THREADS:-16}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-22000}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" python run_sample_data_pixal3d_spacecontrol_3daligned.py \
  --gpu "$GPU" \
  --min_gpu_free_mb "$MIN_GPU_FREE_MB" \
  --seed "$SEED" \
  --video_bg_color "$VIDEO_BG_COLOR" \
  --control_mesh "$CONTROL" \
  --taus $TAUS \
  "$@"
