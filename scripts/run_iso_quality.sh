#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT_DIR"

INPUT="${1:-output/inputs/0_img_2048.png}"
OUTPUT="${2:-output/pixal3d_mps_1536_cuda_parity.glb}"
LOG="${OUTPUT%.glb}.log"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv; run bash setup_macos.sh first." >&2
  exit 2
fi

if [[ ! -f "$INPUT" && "$INPUT" == "output/inputs/0_img_2048.png" ]]; then
  mkdir -p "$(dirname -- "$INPUT")"
  sips --resampleHeightWidth 2048 2048 \
    assets/images/0_img.png \
    --out "$INPUT" >/dev/null
fi
if [[ ! -f "$INPUT" ]]; then
  echo "Input image not found: $INPUT" >&2
  exit 2
fi

mkdir -p "$(dirname -- "$OUTPUT")" "$(dirname -- "$LOG")"
export PYTHONUNBUFFERED=1
export PIXAL3D_NAF_CHUNK_ROWS="${PIXAL3D_NAF_CHUNK_ROWS:-1}"
export SPARSE_ATTN_BACKEND=sdpa

echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Log:    $LOG"

caffeinate -dimsu .venv/bin/python inference.py \
  --image "$INPUT" \
  --output "$OUTPUT" \
  --resolution 1536 \
  --low_vram \
  --export-profile cuda-parity \
  2>&1 | tee "$LOG"
