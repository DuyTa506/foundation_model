#!/usr/bin/env bash
# Launch HF Trainer pretraining via accelerate (FSDP or DeepSpeed ZeRO-2).
# Replaces launch_pretrain_megatron_ds.sh.
#
# Usage:
#   bash scripts/launch_pretrain_hf.sh --config configs/training_8xH200_hf_pretrain.yaml
#   bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_32k.yaml
#   bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_128k.yaml
#   bash scripts/launch_pretrain_hf.sh --smoke_test   # 5-step sanity check
#
# Requires accelerate config at scripts/accelerate_fsdp.yaml (auto-generated below
# if missing) or set ACCELERATE_CONFIG_FILE env var.

set -euo pipefail

CONFIG=""
SMOKE_TEST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --smoke_test) SMOKE_TEST="--smoke_test"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${CONFIG}" && -z "${SMOKE_TEST}" ]]; then
  echo "Usage: bash scripts/launch_pretrain_hf.sh --config <yaml> [--smoke_test]"
  exit 1
fi

CONFIG="${CONFIG:-configs/training_8xH200_hf_pretrain.yaml}"

# ── Generate accelerate FSDP config if missing ───────────────────────────────
ACCELERATE_CFG="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_fsdp.yaml}"

if [[ ! -f "${ACCELERATE_CFG}" ]]; then
  echo "[launch] generating accelerate FSDP config at ${ACCELERATE_CFG}"
  mkdir -p "$(dirname "${ACCELERATE_CFG}")"
  cat > "${ACCELERATE_CFG}" << 'ACCEL_EOF'
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1           # FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
ACCEL_EOF
fi

# ── Set environment ──────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
# Flash attention if available (significant speedup for attention at longer contexts)
export FLASH_ATTENTION=1

echo "[launch] config=${CONFIG}"
echo "[launch] accelerate_config=${ACCELERATE_CFG}"

# ── Launch ────────────────────────────────────────────────────────────────────
accelerate launch \
  --config_file "${ACCELERATE_CFG}" \
  scripts/pretrain_hf.py \
  --config "${CONFIG}" \
  ${SMOKE_TEST}

echo "[ok] launch complete"
