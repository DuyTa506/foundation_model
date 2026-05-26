#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro"
CFG="${ROOT}/train/sft_llamafactory/sft_ouro_1_4b.yaml"

# Smoke run on public Ouro base
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-ByteDance/Ouro-1.4B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/sft_smoke}"

llamafactory-cli train "${CFG}" \
  model_name_or_path="${MODEL_NAME_OR_PATH}" \
  output_dir="${OUTPUT_DIR}" \
  num_train_epochs="${NUM_TRAIN_EPOCHS:-0.05}" \
  max_samples="${MAX_SAMPLES:-2048}"

