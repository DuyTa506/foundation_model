# ADR 002: Norm-Free Block + SwiGLU MLP

## Status
Accepted

## Context
Standard transformer blocks combine `LayerNorm` (or `RMSNorm`) with `GELU` MLPs. Recent work has explored norm-free alternatives that avoid the per-step reduction over the hidden dimension, and LLaMA-style architectures consistently report SwiGLU outperforming GELU at matched parameter count.

## Decision

- **Norm replacement.** Drop `LayerNorm`/`RMSNorm` from the residual stream. Apply a pointwise non-linearity we refer to as DerfNorm: `Derf(x) = erf(α·x + s)`, with learned scalar `α` and `s`. Initialization mimics variance-preserving setups (`1/sqrt(d)`).
- **MLP.** Replace the standard two-layer MLP with a SwiGLU gate, matching parameter count by targeting an inner dimension of `≈ 0.66 · 4 · d_model`.
- **QK-norm exception.** Independent `RMSNorm` is kept on Q and K projections inside attention to bound logit magnitudes (added later, see ADR 008).

## Consequences

**Pros**
- Removes the cross-feature reduction in the residual path.
- SwiGLU is a well-established quality bump.

**Cons**
- DerfNorm is a non-standard choice. The ablation study in the paper includes a check with RMSNorm in place of DerfNorm to confirm that the main findings (router init trap, memory token necessity) are not artifacts of this normalization choice.
- Norm-free networks can be sensitive to learning rate; warmup is enforced and `tests/test_numerical_stability.py` guards against gradient explosion.
