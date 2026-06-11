# Notes: Recommendations for Building a Small Language Model (SLM, <2B params)

Synthesized from: **MiniCPM4 technical report** (OpenBMB, arXiv:2506.07900), **The Smol Training Playbook** (HuggingFace / SmolLM3), the **deep-research comparison report** (Qwen3.5-0.8B, LFM2.5, MiniCPM5-1B, SmolLM3), and the **Qwen3/3.5 architecture article** (Viblo).

---

## 0. Before anything: the "training compass" (Smol Playbook)

- Answer **why** you're training before **what/how**: research insight, production need an existing model can't serve, or filling an open-source gap. If a fine-tune of Qwen/Llama solves your problem, don't pretrain.
- **Validate your evaluation suite first.** Reproduce published numbers of reference models before training anything. Evals drive every decision; a broken eval poisons the whole project.
- **Ablate everything, change one thing at a time.** Every architecture/data change gets tested on a small proxy run (e.g., ~1B model, ~30–100B tokens) before being adopted. No change goes untested — even library upgrades.
- The biggest gains come from **data curation, not architecture tweaks**. Allocate compute/time accordingly.

---

## 1. Model size & token budget (scaling strategy)

- Modern SLMs are **deliberately overtrained far beyond Chinchilla-optimal**, because inference cost dominates for deployed small models: Qwen3 ~36T tokens, SmolLM3 (3B) ~11T, LFM2 ~10–12T. Train as long as your compute budget allows.
- The **counter-example is MiniCPM4: data quality can substitute for data quantity** — it matches Qwen3-8B using only ~8T tokens (22% of Qwen3's data) thanks to aggressive filtering (UltraClean). For a small team, this is the most cost-effective path: invest in filtering, not raw token count.
- Pick model size from your **deployment target** (phone/edge → 0.5–1.2B; laptop/server-light → 1.7–3B). Mind the "critical model size" — below a certain capacity you hit diminishing returns for a target loss.

## 2. Architecture recommendations

- **Safe default**: decoder-only Transformer (Llama-style) with the now-standard stack — **GQA** (e.g., 16Q/2KV at ~1B scale), **RMSNorm (pre-norm) + QK-Norm**, **SwiGLU FFN**, **RoPE**, BF16. This is what SmolLM3 and MiniCPM essentially use; it's well supported by every inference framework.
- **Tie input/output embeddings** at small sizes (Qwen3 does this for 0.6B–4B) — saves a large fraction of params when vocab is big.
- **Long context / on-device efficiency** (only if it's a product requirement, and ablate it):
  - *Trainable sparse attention* — MiniCPM4's **InfLLM v2**: block-wise KV selection with semantic kernels, no extra parameters, degrades to dense attention on short inputs, accelerates **both prefill and decode** (vs MoBA/NSA which compromise one or the other). Achieved ~5% attention density at 128K with ~7× decode speedup on edge GPUs.
  - *Hybrid linear-attention stacks* — Qwen3.5 (Gated DeltaNet + MoE), LFM2 (gated short-conv + GQA layers), MiniCPM5 (≈25% sparse / 75% linear attention layers). Strong for edge latency, but more engineering risk; only adopt with validated kernels.
- **MoE**: gives capacity at fixed active-param cost (Qwen3.5, LFM2-8.3B-A1.5B), but adds memory footprint and infra complexity — usually not worth it under ~2B unless serving cost is the explicit goal.
- **Tokenizer**: byte-level BPE; size the vocab to your languages. Measure **fertility (tokens/word)** on your actual target domains/languages — ~50K vocab is fine for English-only, ~128K–151K if seriously multilingual. Don't blindly copy the latest model's vocab.
- Context extension: pretrain at 4K, extend in a late stage to 32K with **LongRoPE/RoPE scaling**, extrapolate to 128K via **YaRN** (MiniCPM4 and SmolLM3 both follow this pattern).

## 3. Data: where the model is actually won

- Pipeline (common to all four projects): raw crawl → HTML stripping/lang-ID/normalization → **exact dedup → fuzzy dedup (MinHash/LSH) → semantic dedup** → quality classification → **decontamination vs benchmarks** → mixture design.
- **Quality classifiers**: prefer a **fastText classifier over LLM scoring** for web-scale filtering (MiniCPM4: 15T tokens on 80 CPUs in <1,000h vs ~6,000 GPU-hours for an LLM classifier). The key is *seed selection*:
  - MiniCPM4's **efficient verification** trick: instead of training models from scratch to test a data source (~1,200 GPUh), anneal a nearly-trained 1B model on the candidate data for 10B tokens (~110 GPUh) and measure the benchmark delta. Use this loop to pick classifier seed data empirically rather than by human judgment.
- **Synthesize reasoning-intensive data**: web text has low knowledge density. Generate textbook-style and forum-style passages for math/code/science with a <10B open model, feed outputs back as new seeds (iterative self-improvement). MiniCPM4 credits much of its data efficiency to this.
- **Multi-stage curriculum (3 stages is the norm)**:
  1. **Stable/base phase**: bulk web mixture (~75% English web, ~20% multilingual, ~5% code is LFM2's split; tune to your goals).
  2. **Annealing/mid-training**: upsample high-quality sources (textbooks, curated math/code, FineWeb-Edu / UltraFineWeb-style sets) during LR decay; add long-context data here.
  3. Keep the mixture decision empirical — run mixture ablations, don't trust intuition (mixtures behave unintuitively).

## 4. Optimizer & hyperparameters (proven settings)

- **AdamW**, β=(0.9, 0.95), weight decay ≈ 0.1, ε ≈ 1e-8, BF16 mixed precision.
- Peak LR roughly **1e-4 to 6e-4** for sub-2B models (Qwen-1.8B used 6e-4); warmup ~0.1–10% of steps.
- **Prefer WSD (Warmup-Stable-Decay) over cosine**: it lets you extend training, branch annealing experiments, and run the data-verification trick above. MiniCPM4: 7T stable + 1.3T decay. Playbook verdict: choose flexibility/stability over a marginal convergence win.
- Don't hand-tune at full scale: use **µP + small-model hyperparameter search** (ModelTunnel v2). MiniCPM found µP search (~32 GPUh) matched StepLaw-style fitted laws (~1M GPUh) in their setting.
- Use a **better proxy metric than LM loss** for small-scale ablations: MiniCPM's ScalingBench computes loss on GPT-4o-generated reasoning traces for downstream tasks; it maps to downstream accuracy via a sigmoid, giving signal where tiny models score randomly on real benchmarks.
- Optional efficiency boosters (validated in MiniCPM4): **Multi-Token Prediction** auxiliary head (denser supervision; doubles as a speculative-decoding draft later) and **FP8 matmuls** on linear layers only (param grads stay BF16).

## 5. Post-training pipeline

Order: **SFT → preference optimization → (optional) RLVR / distillation**.

- **SFT first, always.** Build a capability-targeted instruction set (the UltraChat v2 / LFM2 pattern): knowledge QA, math + code reasoning (with verifiable answers/unit tests), instruction-following with checkable constraints, long-context QA (8–64K with distractor docs), tool/function calling (prepend a CoT step before the tool call — measurably helps). Filter hard: schema-validate tool calls, dedup (e.g., SemHash), drop samples a small model already solves 4/4 times.
- **Chat template matters**: support dual-mode reasoning (`/think` vs `/no_think`) if you want a hybrid reasoning model (SmolLM3, MiniCPM4.1 both do this). Mask loss to assistant tokens.
- **Preference optimization**: start with **DPO** as baseline; try ORPO/KTO/APO per data type. LR ≈ **10× smaller than SFT**; scan β in **0.01–0.5**; most algorithms overfit after ~1 epoch, so partition data and iterate.
- **RL with verifiable rewards (RLVR)** for math/code: GRPO-style with the standard 2025 fixes — dynamic sampling (drop all-correct/all-wrong prompts), clip-higher, token-level loss, overlong-sample filtering. If rollout GPU utilization is your bottleneck, MiniCPM4's **chunk-wise rollout** (cap tokens per rollout phase, resume unfinished trajectories; stabilized with chunk-level importance sampling, dual-clip, KL with periodic reference refresh, garble filter) cut sampling time ~40–60% at equal quality.
- Distillation: small models benefit from teacher signals (Qwen/Gemma/Llama small variants all use KD; LFM2 uses tempered top-K KD; MiniCPM5 uses **on-policy distillation** after specialist RL models). If you have a good teacher, use it.

## 6. Inference & deployment (decide early, it shapes architecture)

- **Speculative decoding**: EAGLE-2-style single-layer draft; for big vocabs, prune the draft LM head to the top ~25% most frequent tokens (**FR-Spec**, ~75% draft-head FLOP reduction, output distribution unchanged). Train the spec head as a final step.
- **Quantization**: GPTQ-style W4 PTQ, with **prefix-aware calibration (P-GPTQ)** — exclude the first ~4 token positions from the Hessian (their activations are ~10× outliers). For extreme compression, **QAT from a pretrained checkpoint** (BitCPM4 recipe: re-warm LR, spend ≈2× the decay-phase tokens) instead of training low-bit from scratch — ~10× cheaper than BitNet-style.
- Combining spec-decoding + quantization works but retune draft length (verification gets relatively pricier on quantized targets).
- Budget engineering time for the serving stack (CPM.cu / vLLM / SGLang / llama.cpp / ArkInfer-style multi-backend) — the paper's 7× speedups come from kernels + system co-design, not the model alone.

## 7. Evaluation & process discipline

- Held-out vs validation: anything used during ablations is *validation*; keep untouched benchmarks for the final report.
- For small/noisy evals (<~2K problems): report **avg@k**; strip CoT before scoring; pin LLM-judge versions (prefer open-weight judges).
- Watch for **contamination** (e.g., models scoring far better on AIME 2024 than 2025).
- Keep **vibe evals** on your own tasks to catch overfitting to public suites.
- Infra hygiene: pre-flight checklist before launch, automated checkpoint evals, node health monitoring, log everything (seeds, lib versions, configs). Expect throughput mysteries mid-run.

---

## TL;DR recipe for a ~1B SLM on a modest budget

1. Llama-style GQA Transformer, tied embeddings, ~24 layers, vocab sized by fertility on your languages.
2. µP hyperparameter search on ~150M proxies; AdamW, WSD schedule, BF16 (+MTP head).
3. Spend most effort on data: fastText quality filter with empirically-verified seeds, MinHash dedup, decontamination, synthetic reasoning data; 3-stage curriculum with high-quality annealing mix.
4. Train as many tokens as compute allows (quality-filtered 1–8T beats raw 10T+).
5. Extend context 4K → 32K (LongRoPE) late; YaRN to 128K at inference.
6. Post-train: targeted SFT (incl. tool calling + dual reasoning modes) → DPO → RLVR on math/code if budget allows.
7. Ship with FR-Spec speculative decoding + P-GPTQ INT4.
8. Throughout: validated evals, one-change-at-a-time ablations, everything logged.
