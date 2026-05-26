#!/usr/bin/env bash
set -euo pipefail

CONFIG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "Usage: bash scripts/launch_pretrain_megatron_ds.sh --config <yaml>"
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}"
  exit 1
fi

python - <<'PY' "${CONFIG}"
import json
import os
import subprocess
import sys

import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

run_name = cfg["run"]["name"]
out_dir = cfg["run"]["output_dir"]
hardware = cfg["hardware"]
parallelism = cfg["parallelism"]
optim = cfg["optimization"]
launcher = cfg["launcher"]["torchrun"]
env_cfg = cfg["launcher"].get("env", {})
data = cfg["data"]
model = cfg["model"]

os.makedirs(out_dir, exist_ok=True)

for k, v in env_cfg.items():
    os.environ[str(k)] = str(v)

cmd = [
    "torchrun",
    "--nproc_per_node", str(launcher["nproc_per_node"]),
    "--master_port", str(launcher["master_port"]),
    "pretrain_gpt.py",
    "--tensor-model-parallel-size", str(parallelism["tensor_parallel_size"]),
    "--pipeline-model-parallel-size", str(parallelism["pipeline_parallel_size"]),
    "--micro-batch-size", str(optim["micro_batch_size"]),
    "--global-batch-size", str(
        optim["micro_batch_size"]
        * parallelism["data_parallel_size"]
        * parallelism["gradient_accumulation_steps"]
    ),
    "--seq-length", str(model["seq_length"]),
    "--train-iters", str(optim["train_steps"]),
    "--lr", str(optim["learning_rate"]),
    "--min-lr", str(optim["min_lr"]),
    "--lr-warmup-iters", str(optim["warmup_steps"]),
    "--weight-decay", str(optim["weight_decay"]),
    "--clip-grad", str(optim["grad_clip"]),
    "--bf16",
    "--save-interval", str(cfg["checkpointing"]["save_interval"]),
    "--save", out_dir,
    "--data-path", data["indexed_dataset_prefix"],
    "--tokenizer-type", "HFTokenizer",
    "--tokenizer-name-or-path", model["tokenizer"],
    "--deepspeed",
    "--deepspeed_config", json.dumps({
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": int(cfg["deepspeed"]["zero_stage"])},
        "gradient_clipping": float(cfg["deepspeed"]["gradient_clipping"]),
    }),
]

print("[launch] " + " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"[ok] finished run={run_name}")
PY
