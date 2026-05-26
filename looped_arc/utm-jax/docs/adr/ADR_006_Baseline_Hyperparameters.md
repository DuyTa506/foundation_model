# ADR 006: Baseline Hyperparameters

## Status
Accepted

## Context
After implementing the UT block and validating it locally, we needed to pick a configuration for full TPU runs on Sudoku-Extreme (3.83M train / 423K test). We anchored the choice against three recent recursive-reasoning models:

- **URM** — Gao et al. 2025, *Universal Reasoning Model* ([arXiv:2512.14693](https://arxiv.org/abs/2512.14693), [code](https://github.com/UbiquantAI/URM)). ~27M params, 4 layers × 8 inner loops × ACT-16 outer loops. Decoder-only with ConvSwiGLU; uses Truncated Backpropagation Through Loops and the Muon optimizer. Reaches 77.6% on Sudoku without memory tokens — a useful calibration baseline.
- **TRM** — Jolicoeur-Martineau 2025, *Less is More: Recursive Reasoning with Tiny Networks* ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871), [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)). ~5–19M params, ~42 effective layers via nested recursion. Notable for high accuracy with a small parameter footprint and EMA-heavy training.
- **HRM** — Wang et al. 2025, *Hierarchical Reasoning Model* ([arXiv:2506.21734](https://arxiv.org/abs/2506.21734), [code](https://github.com/sapientinc/HRM)). ~27M params, two specialized modules (high-level slow + low-level fast) with ACT up to 16 steps.

Our architecture is a single deep-recursion UT (closer to TRM than HRM) with memory tokens added (ADR 003). The width was chosen on paper by comparing parameter footprints at three candidate sizes — no head-to-head training runs at 128 or 384:

| Hidden size | Heads | Params (parameter-shared UT) | Note                       |
|-------------|-------|------------------------------|----------------------------|
| 128         | 4     | 0.21M                        | Likely underfit            |
| 384         | 12    | 1.80M                        | Lightweight; not pursued   |
| **512**     | **8** | **3.18M**                    | **Selected — TRM-comparable** |

## Decision

`hidden_size = 512` is our v1 baseline. TRM showed that recursion can substitute for parameter count, so 3M is sufficient.

| Hyperparameter      | Value     | Rationale                                                      |
|---------------------|-----------|----------------------------------------------------------------|
| `hidden_size`       | 512       | Matches URM / TRM dimensionality at ~3M total params.          |
| `num_heads`         | 8         | Standard 64-dim head split.                                    |
| `num_memory_tokens` | 16        | Inside the empirical "plateau" found by the paper sweep.       |
| `max_ponder_steps`  | 18        | Slightly above TRM's effective depth; ACT regularizes downward.|
| `ponder_lambda`     | 0.0–0.01  | 0.0 used for the main sweep (clean comparison); 0.01 for compute-savings runs (see ADR 014). |
| `global_batch_size` | 256       | Saturates a v6e-1 HBM at this width.                           |
| `epochs`            | 4         | Sweep budget; exact step count depends on dataset size.        |
| `lr`                | 3e-4      | Standard.                                                      |
| `optimizer`         | AdamW     | Muon is supported but out of scope for the v1 paper.           |
| `clip_grad_norm`    | 1.0       | Bounds gradient magnitude across the unrolled ponder loop.     |
| `use_ema`           | True      | Improves generalization, follows TRM precedent.                |
| `router_init_bias`  | -3.0      | Deep-start default — see ADR 013/014 for the trap this avoids. |

## Consequences

This configuration is the baseline for all sweeps in the paper. Width was held at 512 throughout — earlier attempts at width=768 are documented in archived experimental notes but were not carried forward, partly because they hit the same router init trap that ADR 013 later identified.
