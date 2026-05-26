#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro"
STAGE="${1:-stage1_stable}"
CONFIG="${ROOT}/train/stages/${STAGE}.yaml"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-ouro-repro-subset}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${STAGE}}"

torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
  "${ROOT}/train/pretrain.py" \
  --config "${CONFIG}"

