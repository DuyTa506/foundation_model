import jax
import jax.numpy as jnp
from flax import nnx

from models.layers import RoPEMultiHeadAttention, SwiGLU, DerfNorm

class MemoryPrepender(nnx.Module):
    def __init__(self, num_memory_tokens: int, features: int, rngs: nnx.Rngs):
        self.num_memory_tokens = num_memory_tokens
        if self.num_memory_tokens > 0:
            self.mem_tokens = nnx.Param(jax.random.normal(rngs(), (num_memory_tokens, features)) * 0.02)
        else:
            self.mem_tokens = None

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.num_memory_tokens <= 0:
            return x
        batch_size = x.shape[0]
        mem = jnp.broadcast_to(self.mem_tokens.value, (batch_size, self.num_memory_tokens, x.shape[-1]))
        return jnp.concatenate([mem, x], axis=1)


class ACTRouter(nnx.Module):
    def __init__(self, features: int, rngs: nnx.Rngs, init_bias: float = -3.0):
        self.proj = nnx.Linear(features, 1, rngs=rngs)
        if init_bias != 0.0:
            self.proj.bias.value = jnp.array([init_bias])

    def __call__(self, x: jax.Array) -> jax.Array:
        # Sigmoid to bound probability [0, 1]
        return jax.nn.sigmoid(self.proj(x))


class UniversalTransformerBlock(nnx.Module):
    def __init__(self, features: int, num_heads: int, rngs: nnx.Rngs, use_rmsnorm: bool = False):
        if use_rmsnorm:
            self.norm1 = nnx.RMSNorm(features, rngs=rngs)
            self.norm2 = nnx.RMSNorm(features, rngs=rngs)
        else:
            self.norm1 = DerfNorm(features, rngs=rngs)
            self.norm2 = DerfNorm(features, rngs=rngs)
        self.mha = RoPEMultiHeadAttention(num_heads=num_heads, in_features=features, rngs=rngs)
        self.ffn = SwiGLU(features, int(features * (8/3)), rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array, rotary_indices: jax.Array, num_memory_tokens: int = 0):
        nx = self.norm1(x)
        attn_out, attn_diag = self.mha(nx, mask=mask, rotary_indices=rotary_indices, num_memory_tokens=num_memory_tokens)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_diag


class UniversalTransformer(nnx.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, max_len: int, num_memory_tokens: int, max_ponder_steps: int, epsilon: float, rngs: nnx.Rngs, disable_act: bool = False, router_init_bias: float = -3.0, use_rmsnorm: bool = False):
        self.hidden_size = hidden_size
        self.max_ponder_steps = max_ponder_steps
        self.epsilon = epsilon
        self.num_memory_tokens = num_memory_tokens
        self.disable_act = disable_act

        self.embed = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
        self.type_embed = nnx.Embed(2, hidden_size, rngs=rngs)
        self.mem_prepender = MemoryPrepender(num_memory_tokens, hidden_size, rngs=rngs)

        self.step_embed = nnx.Embed(max_ponder_steps, hidden_size, rngs=rngs)

        self.block = UniversalTransformerBlock(hidden_size, num_heads, rngs=rngs, use_rmsnorm=use_rmsnorm)
        self.router = ACTRouter(hidden_size, rngs=rngs, init_bias=router_init_bias)

        self.out_proj = nnx.Linear(hidden_size, vocab_size, rngs=rngs)

    def __call__(self, x: jax.Array, pad_mask: jax.Array):
        B, L = x.shape
        # Coerce to bool so & works regardless of caller's dtype.
        pad_mask = pad_mask.astype(jnp.bool_)

        # Embed
        h = self.embed(x)

        # Prepend Memory Tokens
        h = self.mem_prepender(h) # (B, L+N, H)

        # Apply Type Encodings and setup decoupled Rotary Indices
        B, total_len, H = h.shape
        type_indices = jnp.concatenate([
            jnp.zeros((self.num_memory_tokens,), dtype=jnp.int32),
            jnp.ones((L,), dtype=jnp.int32)
        ])
        h = h + self.type_embed(type_indices)[None, :, :]

        rotary_indices = jnp.concatenate([
            jnp.arange(self.num_memory_tokens, dtype=jnp.int32),
            jnp.arange(L, dtype=jnp.int32)
        ])

        # Attention Masks
        mem_mask = jnp.ones((B, self.num_memory_tokens), dtype=jnp.bool_)
        full_mask = jnp.concatenate([mem_mask, pad_mask], axis=1) # (B, L+N)

        q_mask = full_mask[:, None, :, None]
        k_mask = full_mask[:, None, None, :]
        attn_mask = q_mask & k_mask # (B, 1, L+N, L+N)
        
        halting_probabilities = jnp.zeros((B, L + self.num_memory_tokens, 1), dtype=h.dtype)
        remainders = jnp.zeros((B, L + self.num_memory_tokens, 1), dtype=h.dtype)
        n_updates = jnp.zeros((B, L + self.num_memory_tokens, 1), dtype=h.dtype)
        halted = jnp.zeros((B, L + self.num_memory_tokens, 1), dtype=jnp.bool_)
        
        accumulated_states = jnp.zeros_like(h)

        # Diagnostic accumulators
        diag_p_mean = []
        diag_p_std = []
        diag_weight_mean = []
        diag_attn_accum = {}

        for step in range(self.max_ponder_steps):
            # Step embedding
            h_step = h + self.step_embed(jnp.array(step))[None, None, :]

            # Shared Layer Block
            h_next, attn_diag = self.block(h_step, mask=attn_mask, rotary_indices=rotary_indices, num_memory_tokens=self.num_memory_tokens)

            # Halting Logic
            p = self.router(h_next)
            if self.disable_act:
                p = jnp.zeros_like(p)

            # Per-step router diagnostics (raw p, before halt masking)
            diag_p_mean.append(jnp.mean(p))
            diag_p_std.append(jnp.std(p))

            p_masked = jnp.where(halted, 0.0, p)
            new_halting_probabilities = halting_probabilities + p_masked

            natural_halt = (new_halting_probabilities >= (1.0 - self.epsilon)) & (~halted)
            # Last step force-halts every still-active token.
            if step == self.max_ponder_steps - 1:
                just_halted = ~halted
            else:
                just_halted = natural_halt

            # At the halting step the weight is the Graves remainder R = 1 - prev_P;
            # for in-flight tokens it stays p; for already-halted tokens p_masked is 0.
            step_weight = jnp.where(just_halted, 1.0 - halting_probabilities, p_masked)

            # Per-step weight diagnostic
            diag_weight_mean.append(jnp.mean(step_weight))

            accumulated_states = accumulated_states + step_weight * h_next

            # Graves ACT ponder cost: only the halting-step remainder contributes to R.
            remainders = remainders + jnp.where(just_halted, 1.0 - halting_probabilities, 0.0)
            n_updates = n_updates + jnp.where(halted, 0.0, 1.0)

            halted = halted | just_halted
            halting_probabilities = new_halting_probabilities

            # In UT, the representation h is frozen once halted.
            h = jnp.where(halted, h, h_next)

            # Accumulate attention diagnostics for averaging across steps
            for k, v in attn_diag.items():
                if k not in diag_attn_accum:
                    diag_attn_accum[k] = v
                else:
                    diag_attn_accum[k] = diag_attn_accum[k] + v

        # Ponder penalty: N + R per token, averaged.
        ponder_loss = jnp.mean(n_updates + remainders)

        # Assemble diagnostics dict
        diagnostics = {}
        for s in range(self.max_ponder_steps):
            diagnostics[f"diag/p_mean_step{s}"] = diag_p_mean[s]
            diagnostics[f"diag/p_std_step{s}"] = diag_p_std[s]
            diagnostics[f"diag/weight_mean_step{s}"] = diag_weight_mean[s]
        for k, v in diag_attn_accum.items():
            diagnostics[f"diag/{k}"] = v / self.max_ponder_steps

        # Output Projection from the accumulated final states
        logits = self.out_proj(accumulated_states)

        return logits, ponder_loss, jnp.squeeze(n_updates, axis=-1), diagnostics
