# EN+VI Math/Science SLM — From Scratch (MiniCPM-inspired)

A **~0.88B parameter** small language model with **Vietnamese as the primary language**,
also supporting English. Target domains: mathematics, science, and language.
Training recipe inspired by [MiniCPM](https://github.com/OpenBMB/MiniCPM)
(WSD scheduler, UltraClean filtering, hybrid-thinking SFT, GRPO/RLVR).

---

## Architecture

```
hidden_size:         1536
intermediate_size:   4608
num_hidden_layers:   24
num_attention_heads: 16   (GQA: num_key_value_heads=2)
head_dim:            128  (decoupled; q/k/v -> 2048 dim)
rope_theta:          5_000_000
max_position_embeddings: 131072  (train base at 4096, extend to 128K)
vocab_size:          64000  (custom EN+VI tokenizer, trained from scratch)
dtype:               bfloat16
init:                normal(0, 0.02) + depth-scaled residual projections
```

> **No pretrained checkpoint is used.** All weights are randomly initialized
> from scratch via `LlamaForCausalLM(config)`.

---

## Pipeline Overview

```
Stage 0   train_tokenizer          → from-scratch tokenizer (VI:EN ≈ 60:40)
Stage 1   curate/ 00→07            → filtered + tokenized data shards
Stage 2   init_model_from_scratch  → random-init checkpoint
Stage 2   pretrain_hf (WSD 4k)    → base pretrain ~100B–1T tokens
Stage 2b  pretrain_hf (16k)       → context extension ABF:  4k → 16k
Stage 2b  pretrain_hf (32k)       → context extension ABF: 16k → 32k
Stage 2b  pretrain_hf (64k)       → context extension YaRN: 32k → 64k
Stage 2b  pretrain_hf (128k)      → context extension YaRN: 64k → 128k
Stage 3   pretrain_hf midtrain    → math/science/VI strengthening (optional)
Stage 4   launch_finetune_trl_sft  → hybrid-thinking SFT (think + no_think)
Stage 5   launch_rl_grpo           → GRPO/RLVR (verifiable math rewards)
Stage 6   run_eval_lighteval       → EN+VI math/science/long-ctx eval
```

---

## Setup

```bash
cd native_transformers/smollm/text/slm_math_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart

### Stage 0 — Train tokenizer
```bash
python scripts/train_tokenizer.py \
  --config configs/tokenizer_en_vi.yaml \
  --corpus_dirs outputs/curated/raw   # after running 00_materialize
```

### Stage 1 — Build dataset
```bash
# 1a. Write source manifest
python scripts/build_dataset_index.py \
  --config configs/curation_pipeline.yaml \
  --output_dir outputs/curated_manifest

# 1b. Materialize real text from HuggingFace
python scripts/curate/00_materialize.py \
  --config configs/curation_pipeline.yaml \
  --output_dir outputs/curated/raw

# 1c. Heuristic quality filtering (Gopher + C4 + FineWeb)
python scripts/curate/01_quality_filter.py \
  --config configs/curation_pipeline.yaml

# 1d. Language identification (GlotLID: en/vi)
python scripts/curate/02_language_id.py

# 1e. UltraClean fastText quality classifier (MiniCPM recipe)
python scripts/curate/03_ultraclean_filter.py

# 1f. MinHash-LSH near-dedup
python scripts/curate/04_dedup_minhash.py

# 1g. Decontamination (remove eval set overlaps)
python scripts/curate/05_decontaminate.py

# 1h. PII redaction
python scripts/curate/06_pii_redact.py

# 1i. Tokenize + pack into shards
python scripts/curate/07_tokenize_pack.py \
  --tokenizer_path outputs/tokenizer
```

### Stage 2 — Init model + Pretrain
```bash
# Create random-init checkpoint
python scripts/init_model_from_scratch.py \
  --config configs/model_llama_1b_en_vi.yaml

# Base pretrain (context 4096, WSD scheduler)
bash scripts/launch_pretrain_hf.sh \
  --config configs/training_8xH200_hf_pretrain.yaml

# Context extension: 4k -> 16k (ABF)
bash scripts/launch_pretrain_hf.sh \
  --config configs/training_longctx_16k.yaml

# Context extension: 16k -> 32k (ABF)
bash scripts/launch_pretrain_hf.sh \
  --config configs/training_longctx_32k.yaml

# Context extension: 32k -> 64k (YaRN)
bash scripts/launch_pretrain_hf.sh \
  --config configs/training_longctx_64k.yaml

# Context extension: 64k -> 128k (YaRN)
bash scripts/launch_pretrain_hf.sh \
  --config configs/training_longctx_128k.yaml
```

### Stage 3 — Mid-training (optional)
```bash
bash scripts/launch_pretrain_hf.sh \
  --config configs/training_midtrain.yaml
```

### Stage 4 — SFT (hybrid-thinking)
```bash
# Synthesize Vietnamese reasoning data first
python scripts/data/synth_vi_reasoning.py \
  --mode translate \
  --source_dataset AI-MO/NuminaMath-CoT \
  --max_samples 50000

python scripts/data/synth_vi_reasoning.py \
  --mode distill \
  --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --max_samples 100000

# Run SFT
accelerate launch scripts/launch_finetune_trl_sft.py \
  --training_config configs/training_finetune_trl_sft.yaml \
  --dataset_config configs/datasets_en_vi_math_finetune.yaml
```

### Stage 5 — GRPO / RLVR
```bash
accelerate launch scripts/launch_rl_grpo.py \
  --config configs/training_rl_grpo.yaml
```

### Stage 6 — Eval
```bash
# Math + science + Vietnamese eval
python scripts/run_eval_lighteval.py \
  --model_path outputs/rl \
  --stage final

# Long-context eval
python scripts/run_eval_lighteval.py \
  --model_path outputs/pretrain_128k \
  --stage longctx_128k \
  --max_context 131072
```

---

## Data Mix (VI-first, ~60% VI / 40% EN)

### Pretrain
| Source | Language | Share | Notes |
|---|---|---|---|
| Ultra-FineWeb-VI (FineWeb2-HQ vie + CulturaX vi) | VI | 38% | UltraClean-filtered; HQ VI backbone |
| Wikipedia VI + VTSNLP curated | VI | 12% | |
| VI math/science (synth + verified-translated) | VI | 10% | |
| finemath 3/4plus + open-web-math | EN | 22% | STEM backbone for cross-lingual transfer |
| Ultra-FineWeb EN + fineweb-edu | EN | 16% | |
| peS2o + StackExchange | EN | 8% | science backbone |

VI is upsampled ~4–6× due to the limited unique VI token budget (~30–80B after filtering).

### SFT
Two modes per sample:
- **`mode: think`** → keeps `<think>…</think>` reasoning trace + answer
- **`mode: no_think`** → answer only (no reasoning trace)

---

## Chat Template (ChatML, Vietnamese default)

```
<|im_start|>system
Bạn là một trợ lý AI thông minh, thành thạo tiếng Việt và tiếng Anh.
Hãy trả lời bằng ngôn ngữ của người dùng.
Với các câu hỏi toán học hoặc khoa học, hãy trình bày từng bước rõ ràng.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
[<think>
{reasoning} ← only when enable_thinking=true
</think>]
{answer}
<|im_end|>
```

Thinking is toggled via `enable_thinking=True/False` in `apply_chat_template`.

---

## Notes

- **No internet checkpoints.** All scripts use `local_files_only=True`.
- **WSD scheduler**: Warmup → Stable → Exponential Decay. The decay phase uses a VI-dominant data mix (~70–80% VI).
- **Long-context**: staged extension 4k → 16k → 32k (ABF) → 64k → 128k (YaRN). Each step is a 2× factor; skipping stages is unstable.
- **GRPO rewards**: correctness (sympy equivalence) + format (single `<think>` block) + language consistency (penalizes VI prompts generating EN/ZH reasoning).
