# ADR 005: Rotary Position Embeddings with Independent Indices

## Status
Accepted

## Context
The original implementation used learned absolute positional embeddings. RoPE generalizes better across lengths and is the standard choice in modern decoder architectures (LLaMA, PaLM, Mistral). We migrated to RoPE while taking care of the interaction with memory tokens (ADR 003).

## Decision

- Remove the standalone `PositionalEmbedding` module.
- Implement `RoPEMultiHeadAttention` extending standard scaled dot-product attention. Q and K projections are rotationally transformed using precomputed sinusoidal frequencies before the attention score computation.
- **Type encodings.** A learned vector is added to all memory-token embeddings, and a different learned vector is added to all puzzle-token embeddings. This gives the model a clean signal for "memory vs sequence" without contaminating the rotary phases.
- **Independent rotary indices.** Memory tokens receive positions `0..N-1` (a sequential order among themselves). Puzzle tokens always begin at position `0` regardless of `N`. This means a Sudoku token's rotary phases are identical whether `N=0` or `N=64` — preventing memory-token count from indirectly modulating positional structure.

## Consequences

**Pros**
- Memory tokens have a stable internal order without being absolutely positioned within the sequence.
- Puzzle tokens have memory-token-count-invariant rotary geometry, which is important because we sweep `N` as the main paper experiment.

**Cons**
- We replace `nnx.MultiHeadAttention` with a custom layer (`models/layers.py`). Slightly more code to maintain.

## Note on indexing memory tokens
The numeric indices assigned to memory tokens are intentional — they act as numbered registers, not as "positions in a sentence." This is occasionally flagged as a bug by readers expecting position-free memory; it isn't.
