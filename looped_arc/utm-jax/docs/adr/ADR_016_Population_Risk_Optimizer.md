# ADR 016: Population-Risk Gradient LOO — AdamW Gate

Date: 2026-05-11
Status: Accepted (2026-05-11)

## Context

Litman & Guo (Stanford, *A Theory of Generalization in Deep Learning*, arXiv 2605.01172v1, May 2026) derive an SNR-style per-parameter gate on top of Adam from a leave-one-out population-risk argument. The reported headline numbers, taken from the paper's main results:

- **Modular division `a · b⁻¹ mod 97`** at 25% training fraction (a canonical grokking task, Power et al. 2022): 95% held-out accuracy at step 5,950 vs. step 29,450 for AdamW — **4.9× fewer steps to grok**.
- **PINN with noisy initial condition**: ℓ₂ ≤ 0.40 in 2.4× fewer iterations than learning-rate-tuned AdamW.
- **DPO fine-tuning of Qwen2.5-0.5B-Instruct** under 30% swapped UltraFeedback preferences: final reward accuracy 0.566 → 0.641, **3.05× closer to the reference policy** in mean absolute reward drift.

The mechanism, paraphrased: a parameter `k` is "population-safe" — its update reduces population risk to first order — when the squared mean gradient `μ_k²` exceeds its scaled variance `σ_k²/(b−1)`. Parameters where signal does not exceed minibatch-noise are skipped; parameters with clear signal pass through Adam as usual. This is the population-risk analog of Adam's signal-to-noise per coordinate.

Relevance to UTM-Jax:

1. **Sudoku-Extreme has grokking-like dynamics.** Our canonical UT+ACT runs show long memorization plateaus before generalization onset. If the 4.9× modular-arithmetic speedup transfers even partially, it changes the cost of every multiseed.
2. **The change is local to the optimizer.** No interaction with ACT halting, memory tokens, or RoPE indexing. Adds one parameter-sized EMA state `s` plus a scalar gate per step. Compatible with Muon for the matrix params, since this only modifies the AdamW path (router, biases, embeddings, norms).
3. **Cheap to gate behind a flag.** New ablation, not a default change. Existing paper claims (Tables in §3–5) remain reproducible by leaving `--optimizer adamw`.

### The algorithm (Algorithm 1 from the paper, soft form)

```
init  m, v, s ← 0
loop  g_t = ∇L(w_{t−1})                                   # batch-mean gradient
      m_prev ← m
      s ← ρ·s + (1−ρ)·(g_t − m_prev)²                     # streaming variance EMA
      m ← β₁·m + (1−β₁)·g_t
      v ← β₂·v + (1−β₂)·g_t²
      m̂, v̂, ŝ ← bias-correct
      q = gate(m̂, ŝ; α, λ_pop, ε)                         # per-parameter mask in [0, 1]
      w ← w − η·q ⊙ m̂/(√v̂ + ϵ) − η·λ_wd·w
```

The three gate forms (Eqs. 241–243 of the paper):

| Variant | Formula                                                              | Notes |
|---------|----------------------------------------------------------------------|-------|
| `hard`  | `1{m̂² > α·ŝ}`                                                       | Unique binary rule with first-order safety. |
| `soft`  | `(m̂² − α·ŝ)+ / ((m̂² − α·ŝ)+ + λ_pop·ŝ + ε)`                       | Algorithm 1 default. Smooth; preserves first-order safety. |
| `snr`   | `m̂² / (m̂² + λ·ŝ + ε)`                                              | Simplest. No α. SNR-shrinker form used in prior work. |

`α` is the leave-one-out coefficient: 1 in the fresh-batch / online regime (typical at our scale), `b/(n − b)` in the finite-dataset regime. The paper notes `λ_pop` "is typically unnecessary at scale" (Eq. 245).

## Decision

Add a population-risk gate as an opt-in optax transformation, exposed through `train.py`:

1. **New module `optimizers/poprisk.py`** with `scale_by_poprisk_gate(...)` returning an `optax.GradientTransformation`. State: `(m, v, s, count)`. Implements all three gate variants (`hard`, `soft`, `snr`) selected at construction time. Chained as `scale_by_poprisk_gate ∘ scale_by_learning_rate ∘ add_decayed_weights` so it composes with the existing optax stack and Muon's `multi_transform` split for QKV (ADR 008).
2. **New flags in `train.py`** (all opt-in, no default behavior change):
   - `--optimizer adamw_poprisk` (alongside existing `adamw`, `muon`).
   - `--poprisk_gate {hard,soft,snr}`, default `soft`.
   - `--poprisk_rho` (default `0.99`), `--poprisk_alpha` (default `1.0`), `--poprisk_lambda_pop` (default `0.0`), `--poprisk_eps` (default `1e-12`).
3. **Tests in `tests/test_poprisk_optimizer.py`**:
   - `q ∈ [0, 1]` element-wise for all three gates on random inputs.
   - `q ≡ 1` ⇒ updates numerically match plain AdamW to float32 tolerance (deterministic, same seed).
   - `α → ∞` (or `λ_pop → ∞`) ⇒ `q → 0` ⇒ no parameter movement.
   - Bias correction at `t = 1` matches a hand-computed reference.
   - State shapes match params; nothing leaks across pytree leaves.
   - `s` is non-negative and bounded above by `max(g − m_prev)²` after one step.
4. **No default change.** `--optimizer adamw` stays the default. The new optimizer is purely additive — paper reproducibility is preserved.

## Impact on existing and upcoming experiments

- **Paper runs (Tables 1–8, all current figures):** unaffected. They use `--optimizer adamw` or `--optimizer muon` and are not touched.
- **Grokking ablation (proposed for v2 paper):** new experiment family. Compare AdamW vs. AdamW-poprisk on Sudoku-Extreme multiseed; report steps-to-grok at matched accuracy. Would slot under §5 (Making ACT Efficient) or a new "Optimizer" subsection.
- **Muon split-clip path (ADR 008):** unchanged. The gate only modifies the AdamW group; Muon-handled QKV matrices route through the existing Muon transformation.

## Consequences

**Pros:**
- Cheap to add and test (one optax transformation, ~150 LoC). The math is a single-line change to Adam.
- High potential upside if the paper's modular-arithmetic 4.9× transfers even partially: every multiseed gets cheaper.
- Opt-in, so zero risk to current paper numbers.
- Compatible with all existing infrastructure (Muon split-clip, EMA, cosine schedule, resume).

**Cons:**
- One extra parameter-sized state vector → +33% optimizer memory for the AdamW group. Sudoku-Extreme is tiny relative to model size; for our 3.8M canonical model this is a few MB. On TPU v6e-1 this is negligible.
- One extra elementwise op per step (gate + multiply). Compute overhead is in the noise next to attention.
- Adds four hyperparameters. Mitigated by paper-default values (`ρ=0.99, α=1.0, λ_pop=0`) and the note that `λ_pop` "is typically unnecessary at scale".
- Author has only published the May 2026 preprint; no third-party replication yet. Treat as exploratory ablation, not a production change.

## Open questions for review

1. Should `adamw_poprisk` integrate with Muon split-clip in the same `multi_transform` chain, or should it be mutually exclusive with `--optimizer muon` for now? (Default proposal: same chain — gate on AdamW group only.)
2. Are we OK adding four `--poprisk_*` flags to `train.py`, or should they be wrapped in a single `--poprisk_config json_string`?
3. Sudoku-Extreme is not a grokking task in the strict sense (it has continuous EM progress, not the sharp delayed phase transition of modular arithmetic). Is the right place to validate this on a modular-arithmetic side experiment first, before touching Sudoku runs?

## Update 2026-05-12: Hyperparameter regime correction

The first canonical run (`long-run-v2-T16-S123-poprisk-canonical`, wandb id `vtrg27ml`) used `--poprisk_alpha 1.0`, the value originally chosen as the train.py default to match the paper's headline experiments. Live dynamics through step 60k showed:

- `grad_norm` spikes to 165→248→112 in the first 2k steps, oscillates between 5 and 95 thereafter (vs. baseline `grad_norm=0.99` at end of training).
- `accuracy` plateaus around 0.61 from step 40k onward.
- `mean_halt_steps` stuck at 13–14 instead of decaying toward the baseline's terminal ~10. The router cannot escape the deep-start warmup ceiling.
- Run state went to `failed` once before being restarted under a new name.

Configs are byte-identical to the AdamW baseline (`long-run-v2-T16-S123`, id `2huh56jj`) except for the optimizer family and the five `poprisk_*` keys; the divergence is purely from the optimizer.

### Root cause: α=1.0 is the fresh-batch boundary, not the finite-dataset value

Paper §F.4 / eq. 245 specifies two regimes:

- **Fresh-batch (online streaming)**: each batch is an independent fresh draw from population D. `α = 1`.
- **Finite-dataset**: batches drawn from a fixed dataset S of size n. `α = b/(n − b)`.

UTM-Sudoku training is unambiguously finite-dataset: 8 epochs over Sudoku-Extreme (n ≈ 3.8M puzzles, b = 256). The corrected coefficient is:

```
α = b / (n − b) = 256 / (3,840,000 − 256) ≈ 6.7 × 10⁻⁵   →   round to 1e-4
```

This is 4 orders of magnitude smaller than the original default. With α=1.0 in this regime, the soft gate `q = (m̂² − α·ŝ)+ / ((m̂² − α·ŝ)+ + ε)` requires per-parameter SNR > 1 to open. Sudoku gradient noise — amplified by augmentation (digit-permutation, rotation) — keeps most parameters below that threshold most of the time. The result is that few parameters update on any given step, gradient signal accumulates in `m̂` against a closed gate, and intermittent gate openings release pent-up updates as grad-norm spikes. The router's ponder-loss gradient is particularly noise-dominated and gets gated off the most, which is why mean-halt cannot escape the warmup ceiling.

The grad-norm spike at step ≈25k is co-located with the end of `--lambda_warmup_steps 20000`: the full ponder penalty engages, the router suddenly has large gradients, the gate is still blocking them, and the system rings.

### Decision

1. **`train.py` default for `--poprisk_alpha` changed from 1.0 → 0.0**, with a help-text pointer to this section. At α=0 the soft gate reduces to `m̂² / (m̂² + ε) ≈ 1` — essentially AdamW, which is the safe failure mode if a user forgets to set α explicitly.
2. **`scripts/queue_poprisk_canonical.sh` sets `--poprisk_alpha 1e-4`** explicitly (paper-correct finite-dataset value, rounded up from 6.7e-5 for legibility). RUN_NAME bumped to `long-run-v2-T16-S123-poprisk-canonical-v2` so it doesn't collide with the previous wandb id.

### Expected outcome and follow-up

At α=1e-4, `(m̂² − α·ŝ)` is dominated by `m̂²` for any parameter with non-trivial signal, so the soft gate is approximately identity. The v2 run will likely track AdamW closely. **This is the expected theoretical result**: Algorithm 1's leave-one-out denoising mechanism has very little headroom in large-data regimes, because the LOO correction factor itself scales as `b/n`.

If v2 ≈ AdamW (visibly overlapping EM and loss curves), the conclusion is: Algorithm 1 as published does not meaningfully accelerate convergence on UTM-Sudoku-style large-data finite-dataset training. This is a clean negative result.

The next experiment to actually probe whether **SNR-style per-parameter denoising** helps in our regime is the paper's SNR form (eq. 243):

```
q_k^SNR = m̂_k² / (m̂_k² + λ_pop · ŝ_k + ε)
```

which is independent of the LOO regime and provides continuous SNR-weighted shrinkage. Recommended ablation if v2 is flat:

```
--poprisk_gate snr --poprisk_lambda_pop 1.0 --poprisk_alpha 0
```

## Update 2026-05-12 (later): Failure mode confirmed — gate blocks ACT halt-recovery

The v2 run (`f2ojon5d`, α=1e-4) at step 82k confirms a clean failure mode, not a tunable hyperparameter sensitivity. Diagnostic write-up with full data and paper-ready prose: [`FINDING_Poprisk_ACT_Incompatibility.md`](../FINDING_Poprisk_ACT_Incompatibility.md).

Compressed version of the finding:

- v2 trajectory overlaps the baseline AdamW run exactly through step ~5k.
- At step ~10,008, baseline's `diag/router_grad_norm` spikes 50× (0.018 → 0.888) — the metastable transition where the router learns to halt deeper. Halt jumps 5 → 12.5; EM begins to rise.
- v2's `diag/router_grad_norm` in the same window stays at 0.012. The gate absorbs the gradient burst the transition depends on. Halt continues to drop, never recovers.
- By step 80k, v2's router converges to a degenerate "halt-at-step-0" policy (`p[0] = 0.72` vs baseline's 0.18). Per-token accuracy 0.77 (memorizing common digits) but EM=0.06 (no real puzzle solving).

The gate is doing exactly what Algorithm 1 specifies — suppressing per-parameter updates whose squared signal cannot beat per-batch variance. The unstated architectural assumption in the paper is that all useful learning is high-SNR per parameter. ACT halt-recovery violates that assumption: it's a metastable transition driven by coherent low-SNR motion across the router. Any positive α (including the paper-correct finite-dataset 1e-4) closes the gate at the wrong time.

### Implications for this ADR

- The optimizer is correctly implemented; tests still pass; the math matches the paper. The failure is at the algorithm/architecture interaction level.
- Status remains Accepted but gate-on-router is contra-indicated by this evidence. Default `--poprisk_alpha 0.0` already makes the failure unreachable for casual users.
- The next experiment of interest is a **hybrid optimizer** — plain AdamW on router params, AdamW-poprisk on the rest — which isolates whether the gate's pathology is router-specific. Implementation slots into the existing `optax.multi_transform` pattern (parallel to ADR 008's Muon split-clip).

### Update 2026-05-12 (hybrid built)

New flag `--poprisk_skip_router` added to `train.py`. When set together with `--optimizer adamw_poprisk`, the dispatch wraps `clip_by_global_norm` around an `optax.multi_transform` that routes params whose path contains `"router"` through plain `optax.adamw` and everything else through `adamw_poprisk`. Same hyperparameters apply to the poprisk branch as before. Tests in `tests/test_poprisk_optimizer.py` (`test_hybrid_routes_router_to_adamw_and_rest_to_poprisk`, `test_hybrid_label_fn_substring_predicate`) confirm the dispatch matches the standalone chains on the appropriate parameter groups. Launch script: `scripts/queue_poprisk_hybrid.sh`.
