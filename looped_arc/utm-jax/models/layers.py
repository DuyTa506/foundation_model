import jax
import jax.numpy as jnp
from flax import nnx

class SwiGLU(nnx.Module):
    def __init__(self, features: int, inner_features: int, rngs: nnx.Rngs):
        self.w_g = nnx.Linear(features, inner_features, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(features, inner_features, use_bias=False, rngs=rngs)
        self.w_o = nnx.Linear(inner_features, features, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = jax.nn.silu(self.w_g(x))
        return self.w_o(gate * self.w_v(x))


class DerfNorm(nnx.Module):
    def __init__(self, features: int, rngs: nnx.Rngs):
        self.alpha = nnx.Param(jax.random.normal(rngs(), (features,)) * 0.02 + 1.0)
        self.s = nnx.Param(jnp.zeros((features,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.scipy.special.erf(self.alpha * x + self.s)


def apply_rope(x: jax.Array, positions: jax.Array) -> jax.Array:
    """Applies Rotary Positional Embeddings to the input array."""
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    
    freqs = jnp.arange(0, half_dim, dtype=jnp.float32)
    inv_freq = 1.0 / (10000.0 ** (freqs / half_dim))
    
    angles = positions[:, None] * inv_freq[None, :]
    
    sin = jnp.sin(angles)[None, :, None, :]
    cos = jnp.cos(angles)[None, :, None, :]
    
    x1, x2 = jnp.split(x, 2, axis=-1)
    
    x_rot_1 = x1 * cos - x2 * sin
    x_rot_2 = x2 * cos + x1 * sin
    
    return jnp.concatenate([x_rot_1, x_rot_2], axis=-1)


class RoPEMultiHeadAttention(nnx.Module):
    def __init__(self, num_heads: int, in_features: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = in_features // num_heads
        
        self.q_proj = nnx.Linear(in_features, in_features, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(in_features, in_features, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(in_features, in_features, use_bias=False, rngs=rngs)
        self.q_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)
        self.k_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)
        self.out_proj = nnx.Linear(in_features, in_features, use_bias=False, rngs=rngs)

    def __call__(self, q_inputs: jax.Array, mask: jax.Array = None, rotary_indices: jax.Array = None, num_memory_tokens: int = 0):
        B, L, _ = q_inputs.shape

        q = self.q_proj(q_inputs).reshape((B, L, self.num_heads, self.head_dim))
        k = self.k_proj(q_inputs).reshape((B, L, self.num_heads, self.head_dim))
        v = self.v_proj(q_inputs).reshape((B, L, self.num_heads, self.head_dim))

        # Apply independent head normalization mapping before rotations
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rotary_indices is not None:
            q = apply_rope(q, rotary_indices)
            k = apply_rope(k, rotary_indices)

        logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)

        if mask is not None:
            # Large finite negative instead of -inf: fully-masked query rows would
            # otherwise produce 0/0 = NaN in softmax and poison gradients via the
            # masked-where pattern downstream.
            logits = jnp.where(mask, logits, -1e9)

        weights = jax.nn.softmax(logits, axis=-1)

        # Attention mass diagnostics: fraction of attention to mem vs seq tokens.
        attn_diag = {}
        if num_memory_tokens > 0:
            N = num_memory_tokens
            # weights: (B, H, Q, K). For each query, weights sum to 1 over keys.
            # Sum over the key-dimension slice to get total mass fraction.
            attn_diag["attn_seq_to_mem"] = weights[:, :, N:, :N].sum(axis=-1).mean()
            attn_diag["attn_seq_to_seq"] = weights[:, :, N:, N:].sum(axis=-1).mean()
            attn_diag["attn_mem_to_mem"] = weights[:, :, :N, :N].sum(axis=-1).mean()
            attn_diag["attn_mem_to_seq"] = weights[:, :, :N, N:].sum(axis=-1).mean()

        output = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
        output = output.reshape((B, L, self.in_features))

        return self.out_proj(output), attn_diag
