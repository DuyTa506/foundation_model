#!/usr/bin/env bash
# Launch HF Trainer pretraining via accelerate (FSDP or DeepSpeed ZeRO-2).
#
# Usage:
#   bash scripts/launch_pretrain_hf.sh --config configs/training_8xH200_hf_pretrain.yaml
#   bash scripts/launch_pretrain_hf.sh --config ... --gpu_ids 0,1,2,3        # use GPUs 0-3
#   bash scripts/launch_pretrain_hf.sh --config ... --gpu_ids 4,5,6,7        # use GPUs 4-7
#   bash scripts/launch_pretrain_hf.sh --config ... --gpu_ids 0,2,5,7        # any 4 GPUs
#   bash scripts/launch_pretrain_hf.sh --smoke_test
#
# --gpu_ids 4,5,6,7  : comma-separated GPU IDs to use (sets CUDA_VISIBLE_DEVICES).
#                      GPU count is inferred automatically from the list.
# --gpus N           : use N GPUs starting from GPU 0 (shorthand when IDs are 0..N-1).

set -euo pipefail

CONFIG=""
SMOKE_TEST=""
GPUS=""          # number of GPUs (inferred from --gpu_ids if given)
GPU_IDS=""       # explicit CUDA_VISIBLE_DEVICES string, e.g. "4,5,6,7"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2";   shift 2 ;;
    --gpu_ids)    GPU_IDS="$2";  shift 2 ;;
    --gpus)       GPUS="$2";     shift 2 ;;
    --smoke_test) SMOKE_TEST="--smoke_test"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${CONFIG}" && -z "${SMOKE_TEST}" ]]; then
  echo "Usage: bash scripts/launch_pretrain_hf.sh --config <yaml> [--gpus N] [--smoke_test]"
  exit 1
fi

CONFIG="${CONFIG:-configs/training_8xH200_hf_pretrain.yaml}"

# ── Resolve GPU IDs and count ─────────────────────────────────────────────────
# Priority: --gpu_ids > --gpus > gpus_per_node in yaml > default 8
if [[ -n "${GPU_IDS}" ]]; then
  # --gpu_ids 4,5,6,7  →  CUDA_VISIBLE_DEVICES=4,5,6,7, GPUS=4
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l | tr -d ' ')
elif [[ -n "${GPUS}" ]]; then
  # --gpus 4  →  use GPU 0..3 (CUDA_VISIBLE_DEVICES not set, accelerate picks first N)
  : # GPUS already set
else
  GPUS=$(python3 -c "
import yaml
with open('${CONFIG}') as f: c = yaml.safe_load(f)
print(c.get('hardware', {}).get('gpus_per_node', 8))
" 2>/dev/null || echo "8")
fi

echo "[launch] GPUs: ${GPUS}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'(all)'}"
export WORLD_SIZE="${GPUS}"

# ── Generate accelerate FSDP config (always regenerate to pick up GPUS) ──────
ACCELERATE_CFG="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_fsdp.yaml}"
mkdir -p "$(dirname "${ACCELERATE_CFG}")"

# Read mixed_precision from config (bf16 for Ampere+, fp16 for Turing)
MIXED_PRECISION=$(python3 -c "
import yaml
with open('${CONFIG}') as f: c = yaml.safe_load(f)
t = c.get('training', {})
if t.get('bf16'): print('bf16')
elif t.get('fp16'): print('fp16')
else: print('bf16')
" 2>/dev/null || echo "bf16")

cat > "${ACCELERATE_CFG}" << ACCEL_EOF
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
mixed_precision: ${MIXED_PRECISION}
num_machines: 1
num_processes: ${GPUS}
rdzv_backend: static
same_network: true
ACCEL_EOF

echo "[launch] accelerate config → ${ACCELERATE_CFG}  (num_processes=${GPUS}, mixed_precision=${MIXED_PRECISION})"

# ── Set environment ───────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export FLASH_ATTENTION=1

echo "[launch] config=${CONFIG}"

# ── Launch ────────────────────────────────────────────────────────────────────
accelerate launch \
  --config_file "${ACCELERATE_CFG}" \
  --num_processes "${GPUS}" \
  scripts/pretrain_hf.py \
  --config "${CONFIG}" \
  ${SMOKE_TEST}

echo "[ok] launch complete"
