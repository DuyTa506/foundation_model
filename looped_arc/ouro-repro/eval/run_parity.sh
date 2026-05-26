#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro"
CKPT="${1:-${ROOT}/artifacts/checkpoint_final.pt}"
OUT="${ROOT}/artifacts/eval/parity_report.json"
mkdir -p "$(dirname "${OUT}")"

python3 "${ROOT}/eval/compare_hf.py" \
  --hf-model "${HF_MODEL:-ByteDance/Ouro-1.4B}" \
  --local-ckpt "${CKPT}" \
  --prompt "${PROMPT:-Solve 2x + 3 = 11.}" \
  --out "${OUT}"

echo "Wrote parity report: ${OUT}"

