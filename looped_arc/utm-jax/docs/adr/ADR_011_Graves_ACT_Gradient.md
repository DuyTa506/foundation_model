# ADR 011: Correct Gradient Implementation for Graves ACT

## Status
Accepted (2026-04-15)

## Context
While debugging shallow halting behavior across multiple ablations, a code audit revealed that the ponder-penalty gradient in our initial ACT implementation was effectively zero. This invalidates earlier interpretations of "ACT halting depth" in any run that used `ponder_lambda > 0` before this fix.

## The bug

Graves' ACT requires a differentiable penalty proportional to the number of steps. The textbook formulation accumulates per-step weights `p_i` so that `sum_i p_i = 1`, with the final step's weight set to a *remainder* `R = 1 - sum_{i<halt} p_i` to make the total well-defined.

Our initial implementation tracked `n_updates`, an integer step count of the form `where(halted, 0, 1)`. This is non-differentiable. Computing `ponder_loss = mean(n_updates + 1.0)` therefore produced a **constant gradient of zero** with respect to the router parameters.

**Practical impact**: across all earlier experiments (the predecessors of ADR 010 and earlier), the router received no direct penalty gradient. Whatever halting depth those models settled at was driven implicitly by the LM cross-entropy term and the architectural priors, not by `ponder_lambda`. Reported halt-step values from those runs cannot be interpreted as evidence about the penalty's effect.

## Fix

The remainder term is computed only at the halting boundary step:

```
R(t) = 1 - Σ_{i=1}^{N(t)-1} p_i      # at the step where token t halts
```

`ponder_loss` is then the mean of `(n_updates + R)` across the batch, where `R` carries gradient through the cumulative `p_i` chain. The router now responds correctly to `ponder_lambda`.

Implementation: `models/ut.py` (ACT inner loop), changes propagated through `train.py` loss accounting.

## Other patches landed at the same time

These are smaller code-hygiene fixes, not decisions worth their own ADR, but listed here for completeness:

- **Padding-row softmax NaN.** `jnp.where(mask, logits, -jnp.inf)` produces `NaN` for rows that are entirely masked, and the `NaN` propagates backward through gradients even when subsequently zero-gated. Replaced `-jnp.inf` with `-1e9`.
- **Mask semantics.** Previously a single mask conflated "valid input position" and "valid label". Split into `input_mask = (inputs != pad_id)` (used by attention) and `loss_mask = (labels != -100)` (used by cross-entropy).
- **Boolean dtype.** Forced `astype(jnp.bool_)` on attention masks to avoid implicit float32 in some test paths.

## Consequences

- All v1 paper results were generated **after** this fix. Earlier ablation numbers should not be quoted.
- Subsequent ADRs (013, 014) interpret router behavior under the corrected gradient.
