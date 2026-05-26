# ADR 003: Memory Tokens

## Status
Accepted

## Context
Standard sequence processing ties every token to a fixed input position. We adopt **memory tokens** — additional learnable token slots prepended to the sequence, drawing on Burtsev et al.'s Memory Transformer (arXiv:2006.11527) and the more recent "registers" line of work for vision transformers (Darcet et al., 2023). The hypothesis is that recursive depth processing benefits from a positional-invariant scratchpad: the model can offload intermediate state across ponder steps without competing with positional bindings on the sequence tokens.

## Decision

- A configurable `num_memory_tokens` parameter `N` controls the number of memory slots.
- A learnable parameter `mem_space` of shape `(N, hidden_size)` is initialized at construction.
- Memory embeddings are concatenated to the input embedding sequence before attention. Total sequence length per layer is `(L + N)`.
- Memory tokens **do not** receive sequence positional encodings — they act as location-invariant registers (see ADR 005 for how RoPE is partitioned).
- Memory tokens participate fully in ACT — they have their own halting probabilities computed by the same router.

## Consequences

**Pros**
- Provides explicit per-batch storage that persists across ponder steps without overlapping the puzzle-token positional structure.
- Empirically necessary for our UT+ACT to solve Sudoku-Extreme; T=0 fails across all configurations tested. See the paper for the full curve.

**Cons**
- Attention cost grows from `O(L²)` to `O((L+N)²)`. For `L=81` (Sudoku) and `N≤32` this is mild.
- Halt-step metrics need to be segmented by token type when interpreting traces (sequence vs memory).
