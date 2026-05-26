#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro"
CFG="${ROOT}/train/sft_llamafactory/sft_ouro_1_4b.yaml"

# Full run against local base checkpoint converted to HF-compatible directory.
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${ROOT}/artifacts/base_hf_export}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/sft_full}"

llamafactory-cli train "${CFG}" \
  model_name_or_path="${MODEL_NAME_OR_PATH}" \
  output_dir="${OUTPUT_DIR}" \
  num_train_epochs="${NUM_TRAIN_EPOCHS:-2.0}"

