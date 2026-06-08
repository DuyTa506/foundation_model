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
Stage -1  download_datasets         → pre-cache HF datasets to disk
Stage 0   train_tokenizer           → from-scratch tokenizer (VI:EN ≈ 60:40)
Stage 1   curate/ 00→07             → filtered + tokenized data shards
Stage 2   init_model_from_scratch   → random-init checkpoint
Stage 2   pretrain_hf (WSD 4k)     → base pretrain ~100B tokens (50k steps)
Stage 2b  pretrain_hf (16k)        → context extension ABF:  4k → 16k
Stage 2b  pretrain_hf (32k)        → context extension ABF: 16k → 32k
Stage 2b  pretrain_hf (64k)        → context extension YaRN: 32k → 64k
Stage 2b  pretrain_hf (128k)       → context extension YaRN: 64k → 128k
Stage 3   pretrain_hf midtrain     → math/science/VI strengthening (optional)
Stage 4   launch_finetune_trl_sft  → hybrid-thinking SFT (think + no_think)
Stage 5   launch_rl_grpo            → GRPO/RLVR (verifiable math rewards)
Stage 6   run_eval_lighteval        → EN+VI math/science/long-ctx eval
```

---

## Setup

```bash
cd native_transformers/smollm/text/slm_math_pipeline

# Fast (recommended): uv
pip install uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Standard:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart

### Stage -1 — Download datasets

Pre-download all HuggingFace sources before running curation or tokenizer training.
This avoids streaming failures on slow or interrupted connections.

```bash
# --- Pretrain datasets (~212 GB total with split slicing) ---

# Dry-run: see what will be downloaded + estimated sizes
python scripts/download_datasets.py --dry_run

# Download all pretrain sources
HF_TOKEN=hf_xxx python scripts/download_datasets.py \
  --cache_dir /data/hf_cache

# Download only specific sources
python scripts/download_datasets.py \
  --source_ids fineweb2_hq_vi c4_vi finemath_4plus \
  --cache_dir /data/hf_cache

# --- SFT + DPO + GRPO/RL datasets ---

# Dry-run: see SFT/RL sources
python scripts/download_sft_rl_datasets.py --dry_run

# Download all stages
HF_TOKEN=hf_xxx python scripts/download_sft_rl_datasets.py \
  --cache_dir /data/hf_cache

# Download specific stage only
python scripts/download_sft_rl_datasets.py --stages sft \
  --cache_dir /data/hf_cache
```

Gated datasets requiring HF token: `uonlp/CulturaX`, `openbmb/UltraData-Math`.

### Stage 0 — Train tokenizer

The tokenizer reads **directly from the HF datasets cache** — no need to run `00_materialize.py` first.
Corpus composition is controlled by `training_corpus` in `configs/tokenizer_en_vi.yaml`:
`vi_ratio=0.60`, `en_ratio=0.40`, `max_corpus_tokens=30B`.
Each source is budget-capped by weight and round-robin interleaved for diversity.

```bash
# Primary mode: read directly from HF cache (recommended)
python scripts/train_tokenizer.py \
  --config configs/tokenizer_en_vi.yaml \
  --curation_config configs/curation_pipeline.yaml \
  --cache_dir /data/hf_cache

# With HF token (needed if datasets require auth)
HF_TOKEN=hf_xxx python scripts/train_tokenizer.py \
  --config configs/tokenizer_en_vi.yaml \
  --curation_config configs/curation_pipeline.yaml \
  --cache_dir /data/hf_cache

# Legacy mode: read from pre-materialized .jsonl/.txt files
python scripts/train_tokenizer.py \
  --config configs/tokenizer_en_vi.yaml \
  --corpus_dirs outputs/curated/raw
```

Outputs to `output_dir` from config (default: `outputs/tokenizer/`):
- `tokenizer.json` — HF PreTrainedTokenizerFast-compatible
- `tokenizer_config.json` — includes chat template
- `tokenizer_card.json` — fertility report (tokens/word per language)

### Stage 1 — Build dataset

```bash
# 1a. Materialize real text from HuggingFace to disk
python scripts/curate/00_materialize.py \
  --config configs/curation_pipeline.yaml \
  --output_dir outputs/curated/raw

# 1b. Heuristic quality filtering (Gopher + C4 + FineWeb)
python scripts/curate/01_quality_filter.py \
  --config configs/curation_pipeline.yaml

# 1c. Language identification (GlotLID: en/vi)
python scripts/curate/02_language_id.py

# 1d. UltraClean fastText quality classifier (MiniCPM recipe)
python scripts/curate/03_ultraclean_filter.py

# 1e. MinHash-LSH near-dedup
python scripts/curate/04_dedup_minhash.py

# 1f. Decontamination (remove eval set overlaps)
python scripts/curate/05_decontaminate.py

# 1g. PII redaction
python scripts/curate/06_pii_redact.py

# 1h. Tokenize + pack into shards
python scripts/curate/07_tokenize_pack.py \
  --tokenizer_path outputs/tokenizer
```

### Stage 2 — Init model + Pretrain

```bash
# Create random-init checkpoint
python scripts/init_model_from_scratch.py \
  --config configs/model_llama_1b_en_vi.yaml

# Base pretrain: context 4096, WSD scheduler, ~100B tokens (50k steps × 2M tok/step)
bash scripts/launch_pretrain_hf.sh \
  --config configs/training_8xH200_hf_pretrain.yaml

# Context extension: 4k -> 16k (ABF)
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_16k.yaml

# Context extension: 16k -> 32k (ABF)
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_32k.yaml

# Context extension: 32k -> 64k (YaRN)
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_64k.yaml

# Context extension: 64k -> 128k (YaRN)
bash scripts/launch_pretrain_hf.sh --config configs/training_longctx_128k.yaml
```

### Stage 3 — Mid-training (optional)

```bash
bash scripts/launch_pretrain_hf.sh --config configs/training_midtrain.yaml
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

## Data Mix (VI-first, ~74% VI / ~26% EN)

Target: **100B tokens** total (~50k steps × 2M tokens/step on 8×H200).
EN is restricted to math+science only — no general English web text.

### Pretrain sources

| Source | HF Dataset | Lang | Weight | ~Tokens |
|---|---|---|---|---|
| FineWeb2-HQ (vie_Latn) | epfml/FineWeb2-HQ | VI | 0.28 | 28B |
| C4-VI filtered | Symato/c4_vi-filtered_200GB | VI | 0.24 | 24B |
| CulturaX VI | uonlp/CulturaX | VI | 0.12 | 12B |
| VTSNLP curated VI | VTSNLP/vietnamese_curated_dataset | VI | 0.07 | 7B |
| VI math synth (future) | — (synth output) | VI | 0.08 | 8B |
| MADLAD-400 VI | Symato/madlad-400_vi | VI | 0.05 | 5B |
| HPLT VI | Symato/hplt-vi | VI | 0.03 | 3B |
| Wikipedia VI | wikimedia/wikipedia | VI | 0.01 | 1B |
| finemath-4plus | HuggingFaceTB/finemath | EN | 0.12 | 12B |
| open-web-math | open-web-math/open-web-math | EN | 0.06 | 6B |
| UltraData-Math | openbmb/UltraData-Math | EN | 0.04 | 4B |
| FineWeb-Edu | HuggingFaceFW/fineweb-edu | EN | 0.04 | 4B |
| peS2o | allenai/peS2o | EN | 0.02 | 2B |

All large datasets use split slicing (e.g. `train[:35%]`) to limit download size.
Total download: ~212 GB.

### SFT format

Two modes per sample:
- **`mode: think`** — keeps `<think>…</think>` reasoning trace + answer
- **`mode: no_think`** — answer only (no reasoning trace)

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
{reasoning}   ← only when enable_thinking=True
</think>]
{answer}
<|im_end|>
```

Thinking toggled via `enable_thinking=True/False` in `apply_chat_template`.

---

## Notes

- **No internet checkpoints.** All scripts use `local_files_only=True` during training.
- **WSD scheduler**: Warmup (1k steps) → Stable → Exponential Decay (5k steps). Decay phase uses VI-dominant mix (~75% VI) defined in `curation_pipeline.yaml:decay_phase_mix`.
- **Tokenizer**: byte-level BPE, NFC normalization (not NFKC — NFKC strips Vietnamese diacritics), `individual_digits=True` for math.
- **Long-context**: staged extension 4k → 16k → 32k (ABF) → 64k → 128k (YaRN). Each step is a 2× factor; skipping stages is unstable.
- **GRPO rewards**: correctness (sympy equivalence) + format (single `<think>` block) + language consistency (penalizes VI prompts generating EN/ZH reasoning).
- **Wikipedia VI** is only 1% weight (~0.3B unique tokens) to avoid excessive upsampling (~15× at higher weights).
