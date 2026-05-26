# ADR 008: QK-Norm, Decoupled QKV, and Split-Clip Optimizer Topology

## Status
Accepted (architectural changes landed in code). Muon itself is supported but **out of scope for the v1 paper** — all paper results use AdamW.

## Context
While experimenting with the Muon optimizer (momentum + Newton-Schulz orthogonalization on matrix-shaped parameters), three issues surfaced that required architectural changes to attention and the optimizer wiring:

1. Muon's natural gradient magnitude is larger than AdamW's. With global gradient clipping at norm 1.0 applied uniformly, Muon's matrix updates were truncated below their effective range, eroding the optimizer's main advantage. With clipping removed entirely, the unbounded matrix updates pushed attention logits into regions where the softmax saturated.

2. Muon's orthogonalization treats each parameter tensor as a unit. Applying it to a fused `qkv_proj` of shape `(d, 3d)` orthogonalizes Q, K, V *together*, even though the three projections serve semantically distinct roles. Per-projection orthogonalization is the more principled choice.

3. The original `nnx.MultiHeadAttention` had no Q/K-side normalization. Without QK-norm, a few large Muon updates can produce attention logits that swamp the softmax for many subsequent steps before the optimizer recovers.

(Historical note: an earlier version of this ADR attributed an observed "router collapse to 2 halt steps under Muon" to gradient shock from the optimizer. ADR 013 later identified this as the router *initialization* trap — a problem that exists at standard init regardless of optimizer. The architectural choices below are still useful for Muon support, but the original framing was a misdiagnosis.)

## Decision

### 1. Split-clip optimizer topology

Use `optax.multi_transform` to apply gradient clipping selectively. The AdamW parameter group keeps `optax.clip_by_global_norm(1.0)`. The Muon parameter group bypasses global clipping so its natural update geometry is preserved.

### 2. Decoupled QKV projections

In `RoPEMultiHeadAttention`, replace the fused `qkv_proj = nnx.Linear(3 * d)` with three independent `q_proj`, `k_proj`, `v_proj` linear layers. Muon orthogonalizes each separately. Parameter count is unchanged: `(d × d) × 3 == d × 3d`.

### 3. QK-norm

Apply independent `RMSNorm` to the outputs of `q_proj` and `k_proj` before RoPE rotation and attention. This bounds attention logits against optimizer-induced magnitude swings. Kept regardless of optimizer choice — it's a small-cost robustness improvement that benefits AdamW too.

## Consequences

**Pros**
- Muon support is a clean configuration toggle (`--optimizer muon` in `train.py`), not a model rewrite.
- QK-norm is a robust default that costs little and protects against logit blow-ups.
- Decoupled QKV gives finer-grained control for any per-projection optimization (e.g. selective LR scaling).

**Cons**
- The QKV split structurally invalidates checkpoints saved with the older fused projection. Old checkpoints cannot be loaded into the current model without weight remapping.
- Split-clip plumbing adds complexity to the optimizer construction in `train.py` (`label_fn` for the multi-transform PyTree).
