#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro"
OUT_DIR="${ROOT}/artifacts/sanity"
mkdir -p "${OUT_DIR}"

python3 "${ROOT}/train/sanity_overfit.py" \
  --steps "${STEPS:-200}" \
  --out "${OUT_DIR}/sanity_report.json"

echo "Sanity report written to ${OUT_DIR}/sanity_report.json"

