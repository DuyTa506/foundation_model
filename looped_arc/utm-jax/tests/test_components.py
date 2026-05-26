import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from models.layers import SwiGLU, DerfNorm
from models.ut import MemoryPrepender, ACTRouter

def test_swiglu_shape_and_initialization():
    rngs = nnx.Rngs(0)
    hidden_dim = 64
    inner_dim = 128
    
    swiglu = SwiGLU(features=hidden_dim, inner_features=inner_dim, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, hidden_dim))
    
    out = swiglu(x)
    
    assert out.shape == x.shape, f"Expected output shape {x.shape}, got {out.shape}"
    

def test_derf_norm_free():
    rngs = nnx.Rngs(0)
    hidden_dim = 64
    
    derf = DerfNorm(features=hidden_dim, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, hidden_dim))
    
    out = derf(x)
    
    # Derf output should be structurally bound between roughly [-1, 1] for large values
    assert out.shape == x.shape
    assert jnp.max(out) <= 1.01
    assert jnp.min(out) >= -1.01

def test_memory_prepender():
    rngs = nnx.Rngs(0)
    hidden_dim = 64
    num_mem_tokens = 5
    
    prepender = MemoryPrepender(num_memory_tokens=num_mem_tokens, features=hidden_dim, rngs=rngs)
    
    # Batch size 3, seq length 10
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 10, hidden_dim))
    
    out = prepender(x)
    
    # Sequence length should increase by num_mem_tokens
    assert out.shape == (3, 15, hidden_dim)

def test_act_router():
    rngs = nnx.Rngs(0)
    hidden_dim = 64
    
    router = ACTRouter(features=hidden_dim, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, hidden_dim))
    
    # Returns the probability of halting
    p_halt = router(x)
    
    assert p_halt.shape == (2, 10, 1)
    
    # Probabilities must be exactly between 0 and 1
    assert jnp.all(p_halt >= 0.0)
    assert jnp.all(p_halt <= 1.0)
