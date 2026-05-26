#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro"
MODEL="${1:-ByteDance/Ouro-1.4B-Thinking}"
OUT_DIR="${ROOT}/artifacts/eval"
mkdir -p "${OUT_DIR}"

TASKS="gsm8k,arc_challenge,hellaswag,winogrande,mmlu"

lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL},trust_remote_code=True" \
  --tasks "${TASKS}" \
  --batch_size auto \
  --output_path "${OUT_DIR}/lm_eval_${MODEL//\//_}.json"

