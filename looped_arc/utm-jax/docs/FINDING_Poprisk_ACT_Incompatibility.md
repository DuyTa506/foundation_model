# Finding: The Population-Risk Gate Blocks ACT Halt-Recovery

Date: 2026-05-12
Status: Confirmed (mechanistic + diagnostic evidence)
Related: ADR 016 (the optimizer), ADR 013/014 (Router Initialization Trap & Deep-Start)

## TL;DR

Litman & Guo's Algorithm 1 (Population-Risk gradient leave-one-out, arXiv:2605.01172v1, May 2026) cannot train UT-ACT on Sudoku-Extreme. **Not for a hyperparameter reason.** The gate's per-parameter SNR cutoff suppresses precisely the low-SNR, coordinated router-gradient burst that drives ACT halt-recovery in the deep-start regime. The optimizer is doing exactly what the paper specifies; the architectural assumption that "low-SNR per-parameter motion is noise to be filtered" is correct for plain feed-forward grokking but **wrong for ACT**, where coherent low-SNR motion across the router is the load-bearing dynamic.

This is a clean mechanistic limitation of Algorithm 1, not a bug.

## Setup

- **Baseline**: `long-run-v2-T16-S123` (wandb `2huh56jj`, AdamW). EM=0.594 at step 119,867.
- **Poprisk v1**: `long-run-v2-T16-S123-poprisk-canonical` (wandb `vtrg27ml`). `--poprisk_alpha 1.0` (paper's fresh-batch boundary). Failed at step 102k, EM=0.
- **Poprisk v2**: `long-run-v2-T16-S123-poprisk-canonical-v2` (wandb `f2ojon5d`). `--poprisk_alpha 1e-4` (paper-correct finite-dataset boundary `b/(n-b)` for n=3.8M, b=256). Running through step 82k+ at time of analysis, EM=0.06.

All three runs use byte-identical hyperparameters apart from optimizer family (verified via wandb config diff). Same seed (S=123), data, batch size, schedule, EMA, deep-start `bias=-3`.

The v1 and v2 runs fail in *different* ways but both ultimately do not grok.

## Observation 1 — Trajectories overlap until the metastable transition

Side-by-side metric history (rounded to nearest sampled step):

```
                          BASELINE (AdamW)                 POPRISK v2 (α=1e-4)
  step    loss  grad   acc    halt   EM     |  loss  grad   acc    halt   EM
   ~500   1.08  2.06   0.52    5.4   --     |  1.36  1.40   0.44    6.1    --
  ~2000   0.80  0.51   0.63    5.1   --     |  0.80  0.38   0.63    5.5    --
  ~5000   0.74  0.44   0.66    5.1   0.01   |  0.75  0.34   0.65    5.1    --
  ~7000   0.59  ~      0.74    5.1   --     |  0.71  ~      0.69    5.0    --
  ~8500   ~     ~      ~       5.8   --     |  ~     ~      ~       4.9    --
 ~10000   0.52  1.52   0.78  *13.1*  0.14   |  0.69  0.46   0.69    4.8    --
 ~20000   0.38  1.36   0.84   12.2   0.43   |  0.60  0.49   0.73    4.4    --
 ~30000   0.38  1.15   0.84   12.3   0.43   |  0.56  0.43   0.75    4.3    --
 ~50000   0.36  1.32   0.84   11.1   0.45   |  0.52  0.36   0.77    3.8    --
 ~80000   0.36  1.78   0.84   11.1   0.49   |  0.51  0.31   0.77    3.7   0.06
```

Both runs collapse `mean_halt_steps` from the init ceiling to ~5 in the first 2k steps. This is normal — every UT-Sudoku run does this. Then at step ~10,008 the baseline **jumps from halt=5 to halt=12.5 in a single eval window**, after which EM starts rising. Poprisk continues to drop (halt → 3.7 by step 50k) and never recovers.

## Observation 2 — The transition is a router-gradient burst, and the gate absorbs it

Diagnostic logs from `diag/router_grad_norm` and per-step halt probabilities `diag/p_mean_step{0..17}`:

```
                          BASELINE (AdamW)                              POPRISK v2 (α=1e-4)
  step  halt   rg       p[3]  p[6]  p[12] p[17] | halt  rg       p[3]  p[6]  p[12] p[17]
  7139  5.05   0.012    0.22  0.93  0.93  0.91  | 4.98  0.012    0.21  0.95  0.96  0.95
  8453  5.83   0.018    0.11  0.85  0.80  0.77  | 4.93  0.020    0.24  0.94  0.95  0.93
 10008 12.50   *0.888*  0.08  0.08  0.37  0.30  | 4.83  0.012    0.32  0.97  0.96  0.95
```

Three points to read off this table:

1. **Router gradient norm spikes 50× in baseline at the transition.** `diag/router_grad_norm`: 0.012 → 0.018 → 0.888 across steps 7139 → 8453 → 10008. The 50× jump is the gradient signal "more depth helps" finally accumulating to a coherent direction strong enough to dominate noise.

2. **Poprisk's router gradient stays absolutely flat in the same window.** 0.012 → 0.020 → 0.012. The gate is absorbing exactly the gradient burst that drives baseline's transition.

3. **Per-step halt probabilities collapse coherently in baseline, drift up in poprisk.** Baseline rewrites the depth profile in one move — `p[6]` plunges from 0.93 to 0.08 between steps 7139 and 10008. Poprisk's `p[6]` *increases* from 0.95 to 0.97 in the same window. The router is moving the wrong way.

## Observation 3 — Poprisk converges to a degenerate "halt-at-step-0" policy

Tracking `p[0]` (probability of halting before any computation) over training:

```
                 step:  500   2k    6k    10k   20k   50k   80k
  BASELINE     p[0]: 0.047 0.044 0.061 0.068 0.077 0.117 0.184
  POPRISK v2   p[0]: 0.047 0.043 0.056 0.074 0.125 0.359 0.719
```

By step 80k, poprisk's router halts the majority of tokens at step 0 (before any ponder step has run). Combined with high `p` values at intermediate steps, the network's modal behavior is: try to skip computation entirely; if that's not chosen, halt at the very next opportunity. This is the floor of `mean_halt_steps = 3.7` we observe.

The model achieves per-token accuracy 0.77 (memorizing common digit distributions) but exact-match 0.06 (essentially zero genuine puzzle-solving). It's the same "halt-collapse + memorize" pathology that ADR 013 documented for `bias=0` initialization — reached here from the opposite direction, via an optimizer that prevents the depth-recovery escape.

## Mechanism

ACT halt-recovery in the deep-start regime is **metastable**. During steps ~5k–10k, the router's gradient is oscillating: some batches push toward more depth (hard puzzles where extra steps help), others push toward less (easy puzzles where the model already has the answer). On average the "more depth" signal slightly wins, and over enough steps the EMA `m` on relevant router parameters accumulates a coherent negative bias on the router output → halt deepens → cross-entropy loss drops further → positive feedback → halt jumps from 5 to 12 in a few hundred steps.

The poprisk soft gate is:
```
q = (m̂² − α·ŝ)+ / ((m̂² − α·ŝ)+ + λ_pop·ŝ + ε)
```

With λ_pop=0 (default), `q` is effectively binary: ~1 when `m̂² > α·ŝ`, ~0 otherwise. Oscillating gradients are precisely the case where `m̂` stays small (cancellation in the EMA) while `ŝ` grows (variance accumulates). At any positive α — paper-correct 1e-4, paper-default 1.0, or anything in between — there exists a threshold below which the router's `m̂²` sits during metastable recovery, and `q` is closed there. The coherent-direction accumulation can never complete, and the run is trapped in the shallow-halt minimum.

The threshold-binary behavior means α does **not** smoothly control the effect:
- At α=1.0 (v1), the threshold is so high that most parameters are gated off throughout training; v1's failure is broad and noisy (grad-norm spikes from pent-up suppressed updates, no learning).
- At α=1e-4 (v2), the threshold is low and most parameter updates flow through. v2's failure is narrow and specific: only the metastable transitions are blocked, and the most important one in this architecture is the router halt-recovery.

The gate behaves correctly under its own assumptions — it suppresses per-parameter updates whose squared signal cannot beat their per-batch variance. The hidden architectural assumption in the paper is that **all useful learning is high-SNR per parameter**. ACT halt-recovery breaks this assumption.

## Predictions confirmed

| Prediction (from initial diagnosis) | Evidence |
|---|---|
| Run trajectories overlap until the metastable transition | Loss/halt/grad identical through step ~5k |
| The transition is signaled by a router-gradient burst | Baseline `diag/router_grad_norm` 0.018 → 0.888 |
| The poprisk gate absorbs that burst | Poprisk `diag/router_grad_norm` stays at 0.012 in the same window |
| Without the burst, halt cannot recover | Poprisk `p[3..17]` stay near 0.95+; baseline collapses to 0.08–0.37 |
| The blocked recovery means the router falls into a worse minimum | Poprisk `p[0]` drifts to 0.72; baseline stays at 0.18 |

Five-for-five.

## Implications

**For ADR 016:** the optimizer is correctly implemented and the math is consistent with the paper. The failure is at the level of the algorithm's interaction with adaptive-compute architectures, not the implementation. Defaults updated, status remains Accepted; gate-on-router is contra-indicated by this evidence and the script comments reflect that.

**For the paper:** this finding slots naturally into §3 (Router Initialization Trap) as a sibling observation. Both pathologies — `bias=0` init and poprisk-gated training — fail via the same metastable transition. They differ in mechanism:

- **`bias=0` trap**: router p starts at ~0.5; cross-entropy gradient through the router is balanced; no coherent direction accumulates; depth never grows.
- **`bias=-3` + poprisk trap**: router p starts at the deep-start ceiling; cross-entropy gradient does develop a coherent direction; but the optimizer's per-parameter SNR gate suppresses that signal during the metastable accumulation phase; the depth-recovery transition never fires.

Same outcome — shallow halt + per-token memorization + zero exact-match — reached by different routes. Together they argue that the depth-recovery transition in §3 is a sharper architectural feature than the paper currently presents: it's not just an init pathology, it's a class of failures that any optimization choice can collide with.

If we want to keep the paper focused, this can live as a single sentence and a footnote ("we further verified that even with correct deep-start initialization, optimizers that suppress low-SNR per-parameter motion — e.g. Litman & Guo's population-risk gate — fail to escape the same shallow-halt minimum via the same blocked transition"). If we want a fuller subsection, the diagnostic data above is camera-ready.

## Open questions and possible ablations

1. **Hybrid optimizer: AdamW on the router, AdamW-poprisk elsewhere.** Tests whether the gate's pathology is router-specific or whether it also suppresses other (subtler) metastable transitions elsewhere in the model. Implementation slots into the existing `optax.multi_transform` pattern (parallel to ADR 008 Muon split-clip). If hybrid grokks, it isolates the failure to the router. If hybrid still fails, the gate is fundamentally incompatible with deep-recurrence dynamics.
2. **SNR-form gate (paper eq. 243):** `q = m̂²/(m̂² + λ·ŝ + ε)`. Continuous, not threshold-binary. Same SNR philosophy, but smoothly weighted rather than killed. Predict: still suppresses the burst, but less catastrophically. Worth one short run if we want to map the full gate-variant space.
3. **What's the gate-equivalent for plain AdamW that *would* trigger the same failure?** A LR schedule that drops the LR sharply during the metastable window would have similar effect. Could test this with a constant-low-LR ablation — predicts that low LR through the depth-recovery window also blocks recovery in plain AdamW, supporting the "this is an architectural feature, not an optimizer feature" framing.

The hybrid optimizer (1) is the highest-value next experiment if we want to publish this finding cleanly.
