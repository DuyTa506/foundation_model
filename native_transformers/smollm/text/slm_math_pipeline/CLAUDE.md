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

## Commands

### Smoke test / sanity check
```bash
# Single-GPU smoke test for pretrain (5 steps)
python scripts/pretrain_hf.py --config configs/training_8xH200_hf_pretrain.yaml --smoke_test

# Smoke test for GRPO
accelerate launch scripts/launch_rl_grpo.py --config configs/training_rl_grpo.yaml --smoke_test
```

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
python scripts/train_tokenizer.py \
  --config configs/tokenizer_en_vi.yaml \
  --corpus_dirs outputs/curated/raw
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
| `curation_pipeline.yaml` | All curation settings: language ID, quality filters, dedup, decontamination, PII, data source weights, tokenization |
| `model_llama_1b_en_vi.yaml` | Model architecture (do not use `from_pretrained`) |
| `tokenizer_en_vi.yaml` | Tokenizer training (byte-level BPE, VI:EN ~60:40) |
| `training_8xH200_hf_pretrain.yaml` | Base pretrain on 8×H200, FSDP, WSD scheduler |
| `training_longctx_{16,32,64,128}k.yaml` | Context extension stages |
| `training_midtrain.yaml` | Optional math/VI mid-training |
| `training_finetune_trl_sft.yaml` | SFT via TRL `SFTTrainer` |
| `training_rl_grpo.yaml` | GRPO/RLVR via TRL `GRPOTrainer` + vLLM rollouts |
| `datasets_en_vi_math_{pretrain,posttrain,finetune}.yaml` | Data manifests per stage |

## Gated HuggingFace datasets

Require `HF_TOKEN`: `uonlp/CulturaX`, `openbmb/Ultra-FineWeb`, `openbmb/UltraData-Math`.
