# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Project Overview

An **~0.88B parameter** Vietnamese-first small language model (SLM) trained entirely from scratch — no pretrained checkpoint, all weights randomly initialized. Target domains: mathematics, science, and language. Architecture: Llama-like with GQA, custom EN+VI tokenizer (64K vocab), and a MiniCPM-inspired training recipe (WSD scheduler, UltraClean filtering, hybrid-thinking SFT, GRPO/RLVR).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `torch==2.7.0`, `transformers`, `accelerate==1.7.0`, `trl>=0.15`, `datatrove[processing]`, `lighteval`, `vllm==0.9.1`.

## Smoke test → Scale workflow

**Always validate locally before running on the cluster.** Smoke configs are completely separate from production configs — never run `--smoke_test` against the real `training_8xH200_hf_pretrain.yaml`, it will use wrong batch/seq dimensions.

### Step 1 — Grab a small corpus slice
```bash
# Download only wikipedia_vi (fast, ~500MB) for local testing
HF_TOKEN=hf_xxx python scripts/download_datasets.py \
  --source_ids wikipedia_vi --cache_dir /tmp/hf_cache_test

# Tokenize → .npy shards (3M tokens takes ~30s)
python scripts/smoke_tokenize.py \
  --tokenizer_path outputs/tokenizer \
  --cache_dir /tmp/hf_cache_test \
  --source_id wikipedia_vi \
  --seq_len 512 --max_tokens 3000000 \
  --output_dir outputs/smoke_tokenized
```

### Step 2 — Init tiny model
```bash
# ~20M params, fits in 4GB VRAM
python scripts/init_model_from_scratch.py \
  --config configs/model_tiny_smoke.yaml \
  --tokenizer_path outputs/tokenizer \
  --output_dir outputs/model_smoke_init
```

### Step 3 — Run 20-step pretrain
```bash
WORLD_SIZE=1 python scripts/pretrain_hf.py \
  --config configs/training_smoke_test.yaml

# Or 5-step quick sanity check
WORLD_SIZE=1 python scripts/pretrain_hf.py \
  --config configs/training_smoke_test.yaml --smoke_test
```

**Pass criteria:** initial loss ≈ `ln(64000) = 11.07` (random init), loss decreasing by step 20.

### Step 3b — Test generation (optional sanity check)
```bash
# Single prompt (raw mode — no chat template)
python scripts/generate.py --model outputs/smoke_train \
  --prompt "Tính 1 + 1" --max_new_tokens 50

# Interactive REPL
python scripts/generate.py --model outputs/smoke_train

# Chat mode (for SFT/RLVR checkpoints)
python scripts/generate.py --model outputs/sft \
  --chat --prompt "Giải phương trình x^2 - 4 = 0" --think
```

After only 20 steps the tiny model outputs gibberish — that's expected. The goal is to confirm the pipeline loads and generates without errors.

### Step 4 — Scale to 4×H200
Only after smoke test passes, switch to production configs:
```bash
# Init full 0.88B model
python scripts/init_model_from_scratch.py --config configs/model_llama_1b_en_vi.yaml

# Base pretrain with wandb logging
bash scripts/launch_pretrain_hf.sh --config configs/training_8xH200_hf_pretrain.yaml
```

### Smoke test for GRPO
```bash
accelerate launch scripts/launch_rl_grpo.py --config configs/training_rl_grpo.yaml --smoke_test
```

## Commands

### Curation pipeline (stages run sequentially)
```bash
# Pre-download HF sources (run once before curation; avoids mid-pipeline failures)
HF_TOKEN=hf_xxx python scripts/download_datasets.py --cache_dir /data/hf_cache
python scripts/download_datasets.py --dry_run  # preview only

# Run curation steps 00–07 in order
python scripts/curate/00_materialize.py --config configs/curation_pipeline.yaml --output_dir outputs/curated/raw
python scripts/curate/01_quality_filter.py --config configs/curation_pipeline.yaml
python scripts/curate/02_language_id.py
python scripts/curate/03_ultraclean_filter.py
python scripts/curate/04_dedup_minhash.py
python scripts/curate/05_decontaminate.py
python scripts/curate/06_pii_redact.py
python scripts/curate/07_tokenize_pack.py --tokenizer_path outputs/tokenizer
```

### Tokenizer training (Stage 0)
```bash
# Primary: read directly from HF cache (no materialization needed)
python scripts/train_tokenizer.py \
  --config configs/tokenizer_en_vi.yaml \
  --curation_config configs/curation_pipeline.yaml \
  --cache_dir /data/hf_cache

# Limit to specific sources or token budget for faster iteration
python scripts/train_tokenizer.py \
  --config configs/tokenizer_en_vi.yaml \
  --curation_config configs/curation_pipeline.yaml \
  --cache_dir /data/hf_cache \
  --source_ids wikipedia_vi c4_vi \
  --max_corpus_tokens 500000000
```

### Model initialization (required before pretrain)
```bash
python scripts/init_model_from_scratch.py --config configs/model_llama_1b_en_vi.yaml
# Output: outputs/model_init/  (config.json + model.safetensors + tokenizer/)
```

### Pretraining
```bash
# Base pretrain (4096 context, WSD scheduler)
bash scripts/launch_pretrain_hf.sh --config configs/training_8xH200_hf_pretrain.yaml

# Context extension — must run in order (skipping stages is unstable)
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_16k.yaml   # ABF 4k→16k
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_32k.yaml   # ABF 16k→32k
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_64k.yaml   # YaRN 32k→64k
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_128k.yaml  # YaRN 64k→128k

# Mid-training (optional math/VI strengthening)
bash scripts/launch_pretrain_hf.sh --config configs/training_midtrain.yaml
```

### SFT and RLVR
```bash
# Synthesize Vietnamese reasoning data first
python scripts/data/synth_vi_reasoning.py --mode translate --source_dataset AI-MO/NuminaMath-CoT --max_samples 50000
python scripts/data/synth_vi_reasoning.py --mode distill --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_samples 100000

# SFT (hybrid-thinking)
accelerate launch scripts/launch_finetune_trl_sft.py \
  --training_config configs/training_finetune_trl_sft.yaml \
  --dataset_config configs/datasets_en_vi_math_finetune.yaml

# GRPO/RLVR
accelerate launch scripts/launch_rl_grpo.py --config configs/training_rl_grpo.yaml
```

### Evaluation
```bash
python scripts/run_eval_lighteval.py --model_path outputs/rl --stage final
python scripts/run_eval_lighteval.py --model_path outputs/pretrain_128k --stage longctx_128k --max_context 131072
python scripts/run_eval_lighteval.py --model_path outputs/pretrain --tasks math,vi_math --dry_run
```

## Architecture

### Pipeline stages (ordered)
| Stage | Script | Config | Output |
|---|---|---|---|
| 0 | `train_tokenizer.py` | `tokenizer_en_vi.yaml` | `outputs/tokenizer/` |
| 1 (00–07) | `curate/*.py` | `curation_pipeline.yaml` | `outputs/curated/` |
| 2 | `init_model_from_scratch.py` | `model_llama_1b_en_vi.yaml` | `outputs/model_init/` |
| 2 | `pretrain_hf.py` via `launch_pretrain_hf.sh` | `training_8xH200_hf_pretrain.yaml` | `outputs/pretrain/` |
| 2b | same script | `training_longctx_*.yaml` | `outputs/pretrain_{16,32,64,128}k/` |
| 3 | same script | `training_midtrain.yaml` | `outputs/midtrain/` |
| 4 | `launch_finetune_trl_sft.py` | `training_finetune_trl_sft.yaml` | `outputs/sft/` |
| 5 | `launch_rl_grpo.py` | `training_rl_grpo.yaml` | `outputs/rl/` |
| 6 | `run_eval_lighteval.py` | — | `outputs/eval/` |

### Key design decisions

**No pretrained checkpoint.** `init_model_from_scratch.py` creates random weights using `LlamaForCausalLM(config)` — never `from_pretrained`. All scripts pass `local_files_only=True` when loading models.

**WSD (Warmup-Stable-Decay) LR scheduler** is implemented manually in `pretrain_hf.py` as a `TrainerCallback`, not via HF's built-in schedulers. The decay phase switches to a VI-dominant data mix (~70–80% VI).

**Vocab size is padded to a multiple of 256** in `init_model_from_scratch.py` for GPU tensor-core efficiency. The tokenizer's actual vocab must match the model's `vocab_size`.

**Depth-scaled residual init**: after HF's default `_init_weights`, `o_proj` and `down_proj` are additionally multiplied by `1/sqrt(2 * num_hidden_layers)` to stabilize residual stream variance.

**Decoupled `head_dim=128`**: query/key/value projections produce `num_attention_heads * head_dim = 2048` dimensions, not `hidden_size=1536`. This is intentional (MiniCPM5-1B spec).

**Packed token shards**: curation step 07 produces `.npy` or `.ds` (datatrove) files of pre-packed token sequences. `PackedTokenDataset` in `pretrain_hf.py` reads these; no dynamic packing happens at training time.

**Context extension** stages must run in order (4k → 16k → 32k with ABF; 32k → 64k → 128k with YaRN). Skipping stages causes instability.

### GRPO rewards (`scripts/rewards/math_verify.py`)
Three reward components combined in `launch_rl_grpo.py`:
- **Correctness** (weight 1.0): sympy symbolic equivalence after extracting `\boxed{}` or last number
- **Format** (weight 0.1): exactly one `<think>...</think>` block with non-empty content followed by an answer
- **Language consistency** (weight 0.15): for VI prompts, penalizes EN/ZH reasoning in the `<think>` trace (uses GlotLID)

Two-stage length schedule: after `switch_step` (default 2500), overlong completions receive a penalty proportional to excess tokens.

### Chat template (ChatML, Vietnamese default)
```
<|im_start|>system
Bạn là một trợ lý AI thông minh, thành thạo tiếng Việt và tiếng Anh.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
[<think>{reasoning}</think>]   ← only when enable_thinking=True
{answer}
<|im_end|>
```

### SFT dataset format (JSONL)
```json
{"prompt": "...", "response": "...", "reasoning": "<optional CoT>", "mode": "think", "language": "vi"}
```
`mode` is `"think"` or `"no_think"`. The `reasoning` field is only used in think mode to inject the `<think>` block if not already present in `response`.

### GRPO dataset format (JSONL, prompt-only)
```json
{"prompt": "Giải phương trình ...", "answer": "42", "language": "vi"}
```

## Configs overview

| Config | Purpose |
|---|---|
| `model_tiny_smoke.yaml` | **Smoke test only** — ~20M param toy model, 4GB VRAM |
| `training_smoke_test.yaml` | **Smoke test only** — 20 steps, single GPU, fp16, no wandb |
| `curation_pipeline.yaml` | All curation settings: language ID, quality filters, dedup, decontamination, PII, data source weights, tokenization |
| `model_llama_1b_en_vi.yaml` | Production model architecture (do not use `from_pretrained`) |
| `tokenizer_en_vi.yaml` | Tokenizer training (byte-level BPE, VI:EN ~60:40) |
| `training_8xH200_hf_pretrain.yaml` | **Production** base pretrain on 4–8×H200, FSDP, WSD, wandb |
| `training_longctx_{16,32,64,128}k.yaml` | Context extension stages |
| `training_midtrain.yaml` | Optional math/VI mid-training |
| `training_finetune_trl_sft.yaml` | SFT via TRL `SFTTrainer` |
| `training_rl_grpo.yaml` | GRPO/RLVR via TRL `GRPOTrainer` + vLLM rollouts |
| `datasets_en_vi_math_{pretrain,posttrain,finetune}.yaml` | Data manifests per stage |

## Logging

```yaml
# configs/training_8xH200_hf_pretrain.yaml
logging:
  report_to: wandb              # or: tensorboard  or: wandb,tensorboard
  wandb_project: slm_math_vi
  wandb_run_name: llama_1b_en_vi_pretrain
```

`--smoke_test` always forces `report_to: none` regardless of config. `WANDB_PROJECT` / `WANDB_NAME` env vars are set automatically from config fields before Trainer init. Tensorboard logs land in `{output_dir}/tensorboard/`.

## Known GPU constraints

- **Turing (GTX 1650 Ti, RTX 20xx)**: fp16 only, no bfloat16. Use `fp16: true` in training config.
- **Ampere+ (A100, H100, H200)**: bfloat16 preferred. Use `bf16: true`.
- `pretrain_hf.py` loads model in float32 and lets HF Trainer handle AMP. **Do not** set `torch_dtype=float16` when calling `from_pretrained` — it breaks the grad scaler.
- Older torch (2.5.x / cu121 for CUDA 12.2) requires `transformers==4.47.x`. torch 2.7.0 (cu126+) works with latest transformers.

## Gated HuggingFace datasets

Require `HF_TOKEN`: `uonlp/CulturaX`, `openbmb/Ultra-FineWeb`, `openbmb/UltraData-Math`.
