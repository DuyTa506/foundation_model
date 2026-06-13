# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Project Overview

An **~1.0B parameter** (1.004B) Vietnamese-first small language model (SLM) trained entirely from scratch — no pretrained checkpoint, all weights randomly initialized. Target domains: mathematics, science, and language. Architecture: Llama-like, **32 layers**, GQA, **tied input/output embeddings**, decoupled `head_dim=128`, custom EN+VI tokenizer (64K vocab), and a MiniCPM-inspired training recipe (WSD scheduler, UltraClean filtering, hybrid-thinking SFT, GRPO/RLVR).

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
# Init full 1.0B model (32 layers, tied embeddings)
python scripts/init_model_from_scratch.py --config configs/model_llama_1b_en_vi.yaml

# Base pretrain — 4 GPUs, any IDs
bash scripts/launch_pretrain_hf.sh --config configs/training_8xH200_hf_pretrain.yaml \
  --gpu_ids 4,5,6,7

# Adjust GPU count or IDs at any time without touching yaml:
bash scripts/launch_pretrain_hf.sh --config configs/training_8xH200_hf_pretrain.yaml \
  --gpu_ids 0,1,2,3          # GPUs 0-3
bash scripts/launch_pretrain_hf.sh --config configs/training_8xH200_hf_pretrain.yaml \
  --gpus 6                   # first 6 GPUs (0-5)
```

**GPU config fields (in every training yaml):**
```yaml
hardware:
  gpus_per_node: 4           # default GPU count; overridden by --gpus / --gpu_ids
```

**Batch size is pre-calibrated for 4 GPUs.** `gradient_accumulation_steps` in each
config is set so that `global_batch_tokens = micro_batch × grad_accum × gpus × seq_len`
stays constant. If you switch to 8 GPUs, halve the `gradient_accumulation_steps`:

| Config | seq | grad_accum (4 GPU) | grad_accum (8 GPU) | tok/step |
|---|---|---|---|---|
| base pretrain | 4096 | 16 | 8 | 2M |
| longctx 16k | 16384 | 64 | 32 | 8M |
| longctx 32k | 32768 | 128 | 64 | 16M |
| longctx 64k | 65536 | 192 | 96 | 50M |
| longctx 128k | 131072 | 256 | 128 | 134M |
| midtrain | 8192 | 32 | 16 | 4M |

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
# Pass --cache_dir so 00_materialize reuses already-downloaded data instead of re-downloading:
#   1. streamed parquet at <cache_dir>/streamed/  (new fraction-aware download)  → read directly
#   2. else existing HF arrow cache under <cache_dir> (old full download)        → reused, no re-download
#   3. else                                                                      → download from HF
python scripts/curate/00_materialize.py --config configs/curation_pipeline.yaml --cache_dir /data/hf_cache --output_dir outputs/curated/raw
python scripts/curate/01_quality_filter.py --config configs/curation_pipeline.yaml  # language-routed (VI relaxed / EN full)
python scripts/curate/02_language_id.py
python scripts/curate/03_ultraclean_filter.py
python scripts/curate/04_dedup_minhash.py
python scripts/curate/05_decontaminate.py
python scripts/curate/06_pii_redact.py
# Stage 6.5: enforce target mixture + re-stamp source (cap each source to weight×target);
# tokenize the MIXED output, not raw pii_clean.
python scripts/curate/build_mixed_corpus.py --config configs/curation_pipeline.yaml \
  --input_dir outputs/curated/pii_clean --output_dir outputs/curated/mixed --target_tokens 50e9
python scripts/curate/07_tokenize_pack.py --input_dir outputs/curated/mixed --tokenizer_path outputs/tokenizer
```

**Quality filter is language-routed** (`_curate_utils.build_quality_router`): the
English-tuned Gopher/C4/FineWeb rules — esp. Gopher's `min_stop_words=2` against ENGLISH
stop words — reject ~90% of Vietnamese (a pure-VI doc has 0 English stop words). VI gets a
relaxed chain (`min_stop_words=0`, `min_avg_word_length=2`, drop FineWeb, neuter C4's EN
sentence/punct rules); EN keeps the full chain. Verify with `measure_filter_survival.py`
(measured: vi 9%→92%, EN unchanged). **The per-source `weight:` is NOT applied during
filtering** — it's enforced only by `build_mixed_corpus.py` (stage 6.5), which also
re-stamps `metadata.source` from the intact `dataset` field (raw `source` is blank).
**Measure real token count via `.ds` bytes/2** — the HF `epoch` counter is meaningless for
a length-less IterableDataset (it's just step/max_steps).

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
bash scripts/launch_pretrain_hf.sh --config configs/training_8xH200_hf_pretrain.yaml \
  --gpu_ids 4,5,6,7

# Pack long-context shards from base pretrain cache — no new downloads needed.
# Multi-doc packing: concatenate docs with EOS to fill target seq_len.
# Run once per seq_len BEFORE the corresponding context-extension stage.
python scripts/pack_longctx_shards.py --seq_len 16384  --dry_run  # preview
python scripts/pack_longctx_shards.py --tokenizer_path outputs/tokenizer \
  --seq_len 16384  --cache_dir /data/hf_cache --output_dir outputs/curated/tokenized_16k
python scripts/pack_longctx_shards.py --tokenizer_path outputs/tokenizer \
  --seq_len 32768  --cache_dir /data/hf_cache --output_dir outputs/curated/tokenized_32k
python scripts/pack_longctx_shards.py --tokenizer_path outputs/tokenizer \
  --seq_len 65536  --cache_dir /data/hf_cache --output_dir outputs/curated/tokenized_64k
python scripts/pack_longctx_shards.py --tokenizer_path outputs/tokenizer \
  --seq_len 131072 --cache_dir /data/hf_cache --output_dir outputs/curated/tokenized_128k

# Context extension — must run in order (skipping stages is unstable)
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_16k.yaml  --gpu_ids 4,5,6,7
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_32k.yaml  --gpu_ids 4,5,6,7
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_64k.yaml  --gpu_ids 4,5,6,7
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_128k.yaml --gpu_ids 4,5,6,7

# Mid-training (optional math/VI strengthening) — pack 8192-len shards first
python scripts/pack_longctx_shards.py --tokenizer_path outputs/tokenizer \
  --seq_len 8192 --cache_dir /data/hf_cache --output_dir outputs/curated/tokenized_midtrain
bash scripts/launch_pretrain_hf.sh --config configs/training_midtrain.yaml --gpu_ids 4,5,6,7
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

**WSD (Warmup-Stable-Decay) LR scheduler** is a real `LambdaLR` owned by a `WSDTrainer(Trainer)` subclass via `create_scheduler` (NOT a callback). A prior version set the LR in an `on_step_begin` callback while `TrainingArguments` installed a constant scheduler that overwrote it every step — so the optimizer fought the schedule and wandb logged a flat peak LR. Owning the scheduler makes warmup/decay both effective and correctly logged. Config: warmup 1k → stable 41k → exponential decay 8k (~16%), peak `4e-4` → min `4e-5`. During the decay phase the data stream optionally switches to a VI-dominant high-quality mix (see **decay-phase anneal** below).

**Tied embeddings + 32 layers**: `tie_word_embeddings: true` shares the input/output embedding table (~98M saved at 64K vocab); that budget buys depth (24 → 32 layers) for a deeper-narrow ~1.004B model. Changing layer count auto-adjusts the depth-scaled residual init (reads `num_hidden_layers`).

**Vocab size is padded to a multiple of 256** in `init_model_from_scratch.py` for GPU tensor-core efficiency. The tokenizer's actual vocab must match the model's `vocab_size`.

**Depth-scaled residual init**: after HF's default `_init_weights`, `o_proj` and `down_proj` are additionally multiplied by `1/sqrt(2 * num_hidden_layers)` to stabilize residual stream variance.

**Decoupled `head_dim=128`**: query/key/value projections produce `num_attention_heads * head_dim = 2048` dimensions, not `hidden_size=1536`. This is intentional (MiniCPM5-1B spec).

**Packed token shards**: curation step 07 produces `.npy` or `.ds` (datatrove) files of pre-packed token sequences. `PackedTokenDataset` in `pretrain_hf.py` reads these; no dynamic packing happens at training time.

**`PackedTokenDataset` emits `labels == input_ids`** — `LlamaForCausalLM` shifts internally (loss compares `logits[:-1]` to `labels[1:]`). A prior version pre-shifted labels (`chunk[1:L+1]`), causing a double-shift that trained next-next-token prediction; do not "fix" this back.

**Train-time data ordering** (single ~104B-token pass = curriculum, all default-on via `data:` knobs): `shard_interleave` reads all shards concurrently and draws each sequence from a random live shard (dissolves the source-clustered sawtooth that shard-order shuffle alone can't, keeps mix ~proportional to token counts); plus a reservoir `shuffle_buffer_size` and per-epoch `shuffle_shards`.

**Decay-phase anneal** (default-on in the production config): `data.decay_shards_dir` streams a high-quality VI+math subset during the LR-decay phase. Build it with `scripts/data/build_decay_shards.py` (filters `outputs/curated/pii_clean` by `metadata.source` to `curation_pipeline.yaml:decay_phase_mix.sources`, then re-tokenizes via stage 07). `PhaseSwitchDataset` + `DecayPhaseCallback` flip the stream at `decay_start = warmup + stable`. A missing/empty dir warns and falls back to the broad mix (no crash).

**Eval loss** (default-on): `data.val_shards_dir` (held-out, NOT in `tokenized_shards_dir`) logs `eval_loss` every `eval_steps`; deterministic + capped by `eval_max_sequences`. Build a small **source-stratified** set with `scripts/data/build_val_shard.py` (samples `pii_clean` per source ∝ corpus weight so val mirrors the train mix — preferred over holding out a whole 1B `.ds` shard, which can be source-skewed; note the sample overlaps train, so it's a *monitoring* val). A missing/empty dir warns and reports train loss only (no crash).

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
| `training_8xH200_hf_pretrain.yaml` | **Production** base pretrain on 4–8×H200, FSDP `SHARD_GRAD_OP` (ZeRO-2, not FULL_SHARD), WSD, decay anneal, eval loss, wandb |
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
