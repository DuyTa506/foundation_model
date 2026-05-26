# ADR 013: ACT Router Initialization Bias

## Status
Accepted

## Context

Phase 1 experiments revealed a critical **router initialization trap** that dominates training dynamics and seed sensitivity. The ACT router (`nnx.Linear(hidden, 1)` + sigmoid) initializes with near-zero weights, producing `p ≈ sigmoid(0) ≈ 0.5` at every ponder step. At `p = 0.5`, cumulative halting probability crosses the `1 - epsilon = 0.95` threshold after just 2 steps, so every token halts immediately.

### Evidence from Phase 1 diagnostics (seed=42, lambda=0)

**The trap.** By step 3k, ALL configurations (T=0, 8, 16, 32) develop the same "shallow halt" pattern regardless of whether they later grokk:

```
p_step0 ≈ 0.35  (pass through)
p_step1 ≈ 0.15  (pass through)
p_step5 ≈ 0.87  (halt!)
halt ≈ 5-7
```

Tokens pass through the first 1-2 steps, then slam the gate shut. All deeper ponder steps receive zero weight and zero gradient. The model is stuck in a shallow local equilibrium.

**The escape.** In configurations that eventually grokk (T=16, T=32), the router gradient suddenly spikes 10-45x, deep-step p collapses from ~0.9 to ~0.05, and halt depth jumps from ~5 to ~14 within 3-5k steps. This is a self-reinforcing cascade: deeper processing → better LM loss → stronger router gradient → even deeper processing.

**The failure.** Stuck configurations (T=0, T=8 at both seeds; T=32 at seed=0) never experience the gradient spike. The router gradient stays at 0.004-0.038 for the entire 60k-step run. The LM loss gradient is never strong enough to escape the trap.

### Key numbers

| Config | Initial p | Trap halt | Escaped? | Router grad at escape |
|--------|----------|-----------|----------|----------------------|
| T=0, S=42 | 0.473 | 5.9 | No | max 0.037 |
| T=8, S=42 | 0.484 | 5.7 | No | max 0.038 |
| T=16, S=42 | 0.482 | 5.1 | Yes @12k | 0.009 → **0.404** |
| T=32, S=42 | 0.480 | 6.3 | Yes @21k | 0.029 → **0.296** |
| T=16, S=0 | ~0.48 | ~5 | Yes @22k | (no diag data) |
| T=32, S=0 | ~0.48 | ~5 | No | (no diag data) |

## Decision

Add a configurable `--router_init_bias` flag (default 0.0, preserving current behavior). Setting it to a negative value (e.g., -3.0) initializes the router's output bias so that `sigmoid(bias) ≈ 0.05`, making every token process all 18 ponder steps by default.

This flips the optimization problem from "escape the shallow trap to discover deep processing" to "start deep and learn where to halt early" — a much smoother optimization landscape.

### Implementation

- `ACTRouter.__init__` accepts `init_bias: float = 0.0` and overrides `self.proj.bias` when non-zero
- `UniversalTransformer.__init__` passes `router_init_bias` through to the router
- `train.py` exposes `--router_init_bias` CLI argument, logged to wandb config

## Proposed Experiment

**Quick validation (1 run, ~2 hours):**

```bash
python3 train.py \
    --data_paths data/sudoku-extreme-full \
    --global_batch_size 256 --epochs 4 \
    --hidden_size 512 --num_heads 8 \
    --num_memory_tokens 16 --max_ponder_steps 18 \
    --ponder_lambda 0.0 --optimizer adamw \
    --seed 0 --use_ema \
    --router_init_bias -3.0 \
    --run_name phase-1b-bias-T16-S0
```

**Why T=16, seed=0:** T=16 already grokks at seed=0 (at step ~22k), so we can compare:
- Without bias (existing): grokking phase transition at step ~22k, preceded by 22k steps stuck at halt=5
- With bias=-3.0: expect immediate deep processing (halt≈18 from step 0), no phase transition needed

**Pass criteria:**
1. Model reaches ≥40% eval EM (matching un-biased T=16 S=0's 50%)
2. No phase transition visible — accuracy improves monotonically from early training
3. halt starts at ~18 and gradually decreases to a learned optimum (rather than starting at 2 and jumping to 15)

**If validation passes, follow-up sweep:**
- T=0, seed=0, bias=-3.0 (does the bias rescue T=0, which never grokks without it?)
- T=32, seed=0, bias=-3.0 (does the bias rescue the T=32/S=0 failure case?)
- T=16, seed=42, bias=-3.0 (does it change the final EM vs the unbiased run?)

This follow-up directly tests whether the initialization trap is the SOLE cause of seed sensitivity, or whether there are deeper landscape properties that matter even when starting in the deep basin.

## Consequences

**Pros:**
- May eliminate seed sensitivity entirely by bypassing the p≈0.5 trap
- Cleaner optimization: the router starts in a functional regime rather than a degenerate one
- Default `0.0` preserves backward compatibility with all existing runs

**Cons:**
- If the model benefits from "learning to ponder deeper" (the current bootstrapping process), forcing deep processing from step 0 may give a qualitatively different solution
- Starting at halt=18 means every training step is maximally expensive (no early halting savings) until the router learns to halt
- Introduces a new hyperparameter; the optimal bias value is unknown (−3.0 is a reasonable guess for "start deep" but may not be optimal)
