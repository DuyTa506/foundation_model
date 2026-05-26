# Framework Survey for EN+VI SLM Math Pipeline (8xH200)

## Scope
- Target hardware: 1 node, 8xH200 (high memory, high bandwidth).
- Target workflow: pretrain -> finetune -> posttrain.
- Domain focus: English + Vietnamese, math/science.

## Short recommendation
- **Pretrain**: **Megatron-LM + DeepSpeed**.
- **Finetune/Posttrain**: **TRL** (with PEFT/LoRA for quick iteration).
- **Data processing**: keep curation standalone and deterministic (manifest + language filter + dedup + decontamination).

This split gives high throughput for long-token pretraining and practical flexibility for SFT/DPO iterations.

## Comparison by framework

### Megatron-LM
- Best-in-class for large-scale autoregressive pretraining throughput.
- Mature tensor/pipeline/data parallel support and memory optimization knobs.
- Strong fit for 8xH200 when pretraining from a base checkpoint and scaling sequence length or token budget.
- Tradeoff: more setup overhead and stricter data format requirements.

### DeepSpeed
- Complements Megatron-LM for optimizer/memory scaling (ZeRO stages, checkpoint partitioning).
- Useful for stable large-batch training with bf16 on H200.
- Tradeoff: extra config complexity and careful checkpoint/version management.

### Nanotron
- Already used in this repository's SmolLM text pretraining examples.
- Cleaner configs and Hugging Face ecosystem alignment.
- Good option when you want tighter compatibility with existing in-repo scripts.
- Tradeoff: for this project we prioritize Megatron-LM + DeepSpeed baseline per user choice.

### TRL
- Strong for instruction tuning and preference optimization workflows (SFT, DPO).
- Integrates well with transformers + datasets + PEFT.
- Faster iteration for domain adaptation and posttraining compared to pretraining stacks.
- Tradeoff: not intended for high-throughput base pretraining at Megatron scale.

### Axolotl
- Great for practical SFT/LoRA training and experimentation.
- Easier operationally than full Megatron setup.
- Tradeoff: does not replace Megatron-LM + DeepSpeed for pretraining throughput targets.

### Unsloth
- Very effective for efficient finetuning workflows.
- Strong for rapid prototype loops and lower-cost adaptation.
- Tradeoff: not a primary choice for full distributed pretraining on 8xH200.

## Why this stack for this project
1. Pretraining is the stage that benefits most from aggressive distributed optimization.
2. Finetune/posttrain needs agility and dataset churn tolerance, where TRL wins.
3. The repository already has TRL and SmolLM training patterns that can be reused.

## 8xH200 operational notes
- Use bf16 end-to-end by default.
- Start with no pipeline parallel (PP=1) and tune tensor/data parallel first.
- Keep sequence length and global token batch stable while sweeping micro-batch + grad accumulation.
- Save resumable checkpoints frequently enough for long jobs (optimizer + scheduler states where needed).
- Track train throughput (`tokens/sec`), memory headroom, and step-time variance from day one.

## Dataset quality guardrails (high priority)
- Deterministic manifest generation with explicit source identifiers and weights.
- Language filtering restricted to `en` and `vi`.
- Near-duplicate removal before tokenization and again on concatenated output.
- Task decontamination against math/science eval sets before final training shards.
- Store curation metadata (`data_card.json`) with hashes and filtering statistics.

## Should datasets include `<think></think>` reasoning tags?

Short answer: **not by default for the baseline**.

- For a production-oriented SLM, training directly on visible chain-of-thought traces can cause unstable style transfer and over-verbose outputs.
- For this EN+VI math/science baseline, the safer path is:
  - Pretrain: no explicit `<think>` formatting.
  - Finetune: use concise supervised targets (solution + final answer), strip `<think>` tags.
  - Posttrain: if preference data with reasoning traces is used, keep it as internal optimization signal and enforce answer-style outputs during decoding.
- If you want explicit reasoning outputs, run a separate experimental branch with trace-enabled supervision and compare on:
  - answer accuracy,
  - verbosity and latency,
  - safety/style regressions.
