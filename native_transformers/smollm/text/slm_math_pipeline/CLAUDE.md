# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **~1.0B param Vietnamese-first SLM** trained **from scratch** (no pretrained checkpoint — all weights random init). Domains: math, science, language. Llama-like: 32 layers, GQA, tied embeddings, decoupled `head_dim=128`, custom 64K EN+VI tokenizer. MiniCPM-style recipe: WSD scheduler, UltraClean filtering, hybrid-thinking SFT, GRPO/RLVR.

Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Key deps pinned in `requirements.txt` (`torch==2.7.0`, `accelerate==1.7.0`, `trl>=0.15`, `vllm==0.9.1`).

## Pipeline (run in order)

Stages are scripts driven by configs; outputs of one feed the next via `--input_dir`/`--output_dir`.

| Stage | Script | Config | Output |
|---|---|---|---|
| 0 tokenizer | `train_tokenizer.py` | `tokenizer_en_vi.yaml` | `outputs/tokenizer/` |
| 1 curation | `curate/00–07*.py` + `build_mixed_corpus.py` | `curation_pipeline.yaml` | `outputs/curated/` |
| 2 init | `init_model_from_scratch.py` | `model_llama_1b_en_vi.yaml` | `outputs/model_init/` |
| 2 pretrain | `pretrain_hf.py` (via `launch_pretrain_hf.sh`) | `training_8xH200_hf_pretrain.yaml` | `outputs/pretrain/` |
| 2b longctx | same | `training_longctx_{16,32,64,128}k.yaml` | `outputs/pretrain_*k/` |
| 3 midtrain | same | `training_midtrain.yaml` | `outputs/midtrain/` |
| 4 SFT | `launch_finetune_trl_sft.py` | `training_finetune_trl_sft.yaml` | `outputs/sft/` |
| 5 RLVR | `launch_rl_grpo.py` | `training_rl_grpo.yaml` | `outputs/rl/` |
| 6 eval | `run_eval_lighteval.py` | — | `outputs/eval/` |

**Curation stage order is deliberately NOT numeric**: language-ID (02) runs *before* quality (01) so the filter routes on real per-doc labels; dedup (04) runs *before* the model classifier (03) so it never scores duplicates. Then 05 decontaminate → 06 PII → `build_mixed_corpus.py` (stage 6.5) → 07 tokenize. The README/curation commands have the exact wiring.

## Smoke test before cluster (always)

Smoke configs are **separate files** — never run `--smoke_test` against production yamls (wrong batch/seq dims). Sequence: download `wikipedia_vi` slice → `smoke_tokenize.py` → `init_model_from_scratch.py --config model_tiny_smoke.yaml` → `pretrain_hf.py --config training_smoke_test.yaml`. Pass criteria: initial loss ≈ `ln(64000)=11.07`, decreasing by step 20. (GRPO smoke: `accelerate launch scripts/launch_rl_grpo.py --config configs/training_rl_grpo.yaml --smoke_test`.)

Run full pretrain with `bash scripts/launch_pretrain_hf.sh --config <cfg> --gpu_ids 4,5,6,7` (also `--gpus N`). Batch sizes are pre-calibrated for 4 GPUs; on 8 GPUs halve `gradient_accumulation_steps`.

## Design decisions that will bite you if changed

- **No `from_pretrained` for weights.** `init_model_from_scratch.py` builds random `LlamaForCausalLM(config)`. All model loads pass `local_files_only=True`.
- **WSD scheduler is a real `LambdaLR` owned by `WSDTrainer.create_scheduler`** — NOT a callback. A prior callback version fought the constant scheduler HF installed and logged a flat LR. Don't move it back to a callback.
- **`PackedTokenDataset` emits `labels == input_ids`** (no pre-shift). `LlamaForCausalLM` shifts internally. A prior pre-shift caused next-next-token training — do NOT "fix" it.
- **Quality filter is language-routed** (`_curate_utils.build_quality_router`): EN-web (full Gopher+C4+FineWeb), VI (relaxed — EN-tuned rules reject ~90% of VI), math/science (min-words + GopherRepetition only; C4/FineWeb delete LaTeX). Routing keys on dataset→language/role, not `metadata.language`. Verify with `measure_filter_survival.py`.
- **Per-source `weight:` is NOT applied during filtering** — only by `build_mixed_corpus.py` (6.5), which also re-stamps `metadata.source` from the `dataset` field (raw `source` is blank). **Tokenize the mixed output, not raw `pii_clean`.**
- **Tied embeddings + 32 layers**: `tie_word_embeddings: true` frees budget for depth. Depth-scaled residual init multiplies `o_proj`/`down_proj` by `1/sqrt(2*num_hidden_layers)` and auto-adjusts with layer count.
- **Vocab padded to multiple of 256**; tokenizer vocab must match model `vocab_size`. **Decoupled `head_dim=128`**: QKV project to `num_heads*head_dim=2048`, not `hidden_size` (intentional, MiniCPM spec).
- **Real token count = `.ds` bytes / 2.** The HF `epoch` counter is meaningless for the length-less IterableDataset.
- **Context extension must run 4k→16k→32k (ABF) →64k→128k (YaRN) in order** — skipping is unstable. Pack shards per seq_len with `pack_longctx_shards.py` first.

## Train-time data flow (all default-on knobs under `data:`)

`shard_interleave` draws each sequence from a random live shard (dissolves source-clustered sawtooth) + reservoir `shuffle_buffer_size` + per-epoch `shuffle_shards`. **Decay-phase anneal**: `data.decay_shards_dir` streams a high-quality VI+math mix during LR decay (built by `scripts/data/build_decay_shards.py`); `PhaseSwitchDataset`+`DecayPhaseCallback` flip at `warmup+stable`. **Eval loss**: `data.val_shards_dir` (held-out; build with `scripts/data/build_val_shard.py`, source-stratified). Missing dirs warn + fall back, never crash.

## GRPO rewards (`scripts/rewards/math_verify.py`)

Correctness (1.0, sympy equivalence on `\boxed{}`/last number) + Format (0.1, one `<think>…</think>` then answer) + Language consistency (0.15, penalize non-VI reasoning on VI prompts via GlotLID). Two-stage length penalty after `switch_step` (default 2500).

## Data formats

- **Chat template**: ChatML, VI system default; `<think>…</think>` only when `enable_thinking=True`.
- **SFT JSONL**: `{"prompt","response","reasoning"?,"mode":"think"|"no_think","language"}`. `reasoning` injected as `<think>` only in think mode.
- **GRPO JSONL** (prompt-only): `{"prompt","answer","language"}`.

## GPU notes

- Turing (GTX 16xx, RTX 20xx): fp16 only, no bf16 (`fp16: true`). Ampere+ (A100/H100/H200): `bf16: true`.
- `pretrain_hf.py` loads model in fp32 and lets HF Trainer handle AMP. **Never** pass `torch_dtype=float16` to `from_pretrained` (breaks grad scaler).
- Production pretrain uses FSDP `SHARD_GRAD_OP` (ZeRO-2), not FULL_SHARD.

## Misc

- `--smoke_test` forces `report_to: none`. `WANDB_PROJECT`/`WANDB_NAME` set from config before Trainer init.
- Gated HF datasets need `HF_TOKEN`: `uonlp/CulturaX`, `openbmb/Ultra-FineWeb`, `openbmb/UltraData-Math`.
- `generate.py` for sanity checks: `--prompt` (raw), `--chat --think` (SFT/RLVR), no `--prompt` for REPL.
- The README has full command listings; `research/` holds design notes (`smol_playbook.md`, deep-research report).
