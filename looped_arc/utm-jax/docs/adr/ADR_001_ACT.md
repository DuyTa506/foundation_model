# ADR 001: Adaptive Computation Time via Bounded Unrolled Loop

## Status
Accepted (current). A migration to `jax.lax.scan` is on the table — see *Considered alternatives*.

## Context
The Universal Transformer needs a dynamic-depth pondering mechanism so different tokens can stop processing at different steps. A naive implementation with a Python `while` loop or dynamic early break introduces variable shapes into the XLA compilation, which causes long compile times and breaks `jax.sharding`.

## Decision
We use a Python `for step in range(max_ponder_steps):` loop inside the model's `__call__`, fully unrolled at JAX trace time, with masking-based halting:

- Every token executes the loop for the full `max_ponder_steps` iterations. Halting is handled analytically rather than by exiting the loop.
- Once a token's accumulated halting probability reaches `1 - epsilon`, a `jnp.where` mask freezes its hidden state for all remaining steps.
- The remainder term `R = 1 - sum_{i<halt} p_i` is applied at the halting step only, providing the gradient path required by Graves' ACT formulation.
- Per-step diagnostics (router `p` mean/std, step weights) are accumulated into Python lists during tracing and emitted as a fixed-size dict — possible only because the loop is unrolled.

See `models/ut.py` (the `for step in range(self.max_ponder_steps):` block in the UT forward pass).

## Consequences

**Pros**
- Static compute graph, predictable XLA compile.
- Trivially compatible with `jax.sharding` and `nnx.jit`.
- Per-step Python-side bookkeeping (diagnostic dicts keyed by step index) is straightforward.

**Cons**
- No wall-clock speedup from early halting on TPU — every step is physically executed. This is acceptable here because TPUs do not benefit from dynamic sparsity at the block level anyway, and we care about *learned* halting behavior more than runtime savings.
- Unrolling produces a graph proportional in size to `max_ponder_steps`, which lengthens trace and compile time as the bound grows. Tolerable at our current `max_ponder_steps=18`, but a soft cost.

## Considered alternatives

**`jax.lax.scan` over the ponder steps.** Would keep a single rolled copy of the block in the compiled graph (smaller, faster to trace/compile), at the price of:
- The carry must hold all per-step state (`h`, halting probabilities, remainders, n_updates, halted mask, accumulated states) plus any diagnostics we want to keep.
- Per-step Python-side conditionals like the "last step force-halt" branch (`if step == max_ponder_steps - 1`) need to be expressed as data instead — straightforward, but a refactor.
- Diagnostic accumulators currently kept as Python lists keyed by step would need to become stacked arrays returned from `scan`'s `ys` channel.

We may migrate to `lax.scan` if compile time becomes a real cost (e.g. when sweeping larger `max_ponder_steps`, or stacking the loop inside an outer JIT boundary that retraces often). For now the unrolled form is simpler and the compile-time hit is acceptable.
