#!/usr/bin/env bash
set -euo pipefail

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-pixal3d}"
fi

cd "$(dirname "$0")"

GPU="${GPU:-4}"
SEED="${SEED:-42}"
VIDEO_BG_COLOR="${VIDEO_BG_COLOR:-white}"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" python run_sample_data_pixal3d.py \
  --seed "$SEED" \
  --video_bg_color "$VIDEO_BG_COLOR" \
  "$@"
