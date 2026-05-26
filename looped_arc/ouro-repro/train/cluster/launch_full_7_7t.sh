#!/usr/bin/env bash
set -euo pipefail

# Example multi-node launcher skeleton for full 7.7T run.
# Fill node topology + rendezvous values from your scheduler environment.

ROOT="/home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-8}"
NODE_RANK="${NODE_RANK:?set NODE_RANK}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

run_stage () {
  local stage="$1"
  local cfg="${ROOT}/train/stages/${stage}.yaml"
  torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${ROOT}/train/pretrain.py" \
    --config "${cfg}"
}

run_stage stage1_stable
run_stage stage2_ct_anneal
run_stage stage3_longct
run_stage stage4_midtrain

