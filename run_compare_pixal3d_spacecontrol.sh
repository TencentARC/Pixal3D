#!/usr/bin/env bash
set -euo pipefail

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-pixal3d}"
fi

cd "$(dirname "$0")"

POINTS="${POINTS:-100000}"
VIEWS="${VIEWS:-12}"
RESOLUTION="${RESOLUTION:-512}"
TAUS="${TAUS:-6 8 9}"
OUT_DIR="${OUT_DIR:-outputs/sample_data/spacecontrol_comparison_tau6_tau8_tau9}"

read -r -a TAU_ARGS <<< "$TAUS"

PYTHONUNBUFFERED=1 python compare_pixal3d_spacecontrol.py \
  --points "$POINTS" \
  --views "$VIEWS" \
  --resolution "$RESOLUTION" \
  --out_dir "$OUT_DIR" \
  --taus "${TAU_ARGS[@]}" \
  "$@"
