# ADR 014: Change ACT Router Default to Deep-Start Initialization

## Context

ADR 013 introduced a configurable `--router_init_bias` flag to bypass the ACT router initialization trap. The trap was identified by Phase 1 diagnostics: with the default Linear init (bias ≈ 0), the router produces `p ≈ sigmoid(0) ≈ 0.5`, causing every token to halt after 2 ponder steps. Escaping this shallow trap requires a rare gradient spike that depends on seed and memory-token count, causing severe seed sensitivity and wasting 10–20k training steps.

### Phase 1 results confirm the trap is the dominant failure mode

Across 3 seeds × 5 memory-token values (15 completed runs):
- **Only 4 out of 15 runs grokked** (T=16/S=0, T=16/S=42, T=32/S=42, T=64/S=0)
- **Seed 123 failed at every T value**, including T=16 which grokked at both other seeds
- **T=0 and T=8 never grokked** at any seed
- All failures show the same diagnostic signature: p trapped at 0.5 → shallow halt (2–7 steps) → router grad stays small → no escape

The default `bias=0.0` init produces a degenerate starting point that the optimizer cannot reliably escape. There is no known benefit to starting shallow: Graves' ACT was designed for models that start deep and learn to halt early, not the reverse.

### Why a negative bias is unconditionally better

With `bias=-3.0` → `sigmoid(-3) ≈ 0.05`:
- Tokens process all 18 ponder steps from step 0 (deep-start)
- The router can learn to halt earlier when deeper processing isn't needed
- The ponder penalty (lambda > 0) works as Graves intended: regularizing depth *downward* from the maximum, rather than fighting against an upward escape
- No phase transition / gradient spike is needed — the optimization landscape is smooth

The old default (`0.0`) has no theoretical justification and empirically causes >70% of runs to fail.

## Decision

Change the default `router_init_bias` from `0.0` to `-3.0` everywhere:
- `ACTRouter.__init__`: `init_bias` default → `-3.0`
- `UniversalTransformer.__init__`: `router_init_bias` default → `-3.0`
- `train.py` argparse: `--router_init_bias` default → `-3.0`

The flag remains configurable for ablation purposes. Passing `--router_init_bias 0.0` recovers the old behavior for comparison runs.

## Impact on Existing and Upcoming Experiments

- **All Phase 1 runs (completed):** unaffected, they used explicit `--router_init_bias` values or the old code without the flag
- **Phase 1b bias ablation (queued via `queue_bias_ablation.sh`):** already specifies `--router_init_bias -3.0` explicitly, unaffected by the default change
- **Future runs:** automatically get the deep-start default unless overridden

## Consequences

**Pros:**
- Eliminates the p ≈ 0.5 initialization trap as the default behavior
- Should dramatically reduce seed sensitivity (pending Phase 1b confirmation)
- Enables nonzero ponder_lambda to function correctly (penalty regularizes depth downward from 18, rather than fighting upward escape)
- Aligns with Graves' ACT design assumption: start at full depth, learn to halt early

**Cons:**
- First ~1k training steps are maximally expensive (all 18 ponder steps) before the router learns to halt
- Not directly comparable to any published UT implementation (most use default init)
- Changes the default behavior of the codebase; old runs cannot be exactly reproduced without `--router_init_bias 0.0`
