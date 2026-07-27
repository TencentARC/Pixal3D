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
TAU="${TAU:-6}"
VIDEO_BG_COLOR="${VIDEO_BG_COLOR:-white}"
CONTROL="${CONTROL:-/root/dev/TRELLIS.2/assets/sample_data/DH-001_SZ270/DH-001_SZ270_SURF_bbox_normalized.ply}"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" python run_sample_data_pixal3d.py \
  --seed "$SEED" \
  --video_bg_color "$VIDEO_BG_COLOR" \
  --spatial_control_mesh_path "$CONTROL" \
  --space_control_tau "$TAU" \
  --auto_align_control \
  --output_tag "pixal3d-spacecontrol-tau${TAU}" \
  "$@"
