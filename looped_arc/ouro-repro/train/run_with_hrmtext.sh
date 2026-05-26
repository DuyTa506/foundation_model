#!/usr/bin/env bash
set -euo pipefail

# Bridge command to run an Ouro-like looped UT baseline inside HRM-Text.
# This uses configs added under hierachical_arc/HRM-Text/config/arch/{net,size}.

ROOT="/home/duy/Downloads/duy_dev/foundation_model"
HRM_TEXT="${ROOT}/hierachical_arc/HRM-Text"

cd "${HRM_TEXT}"

torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" pretrain.py \
  arch/net@arch=ouro_looplm \
  arch/size@arch=ouro_1.4b \
  arch.attn_type=causal \
  global_batch_size="${GLOBAL_BATCH_SIZE:-131072}" \
  lr="${LR:-2.2e-4}" \
  data.path="${DATA_PATH:-/dev/shm/sampled}"

