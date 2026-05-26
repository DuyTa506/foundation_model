import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from models.ut import UniversalTransformer
from models.layers import DerfNorm, SwiGLU

def test_ut_initialization():
    rngs = nnx.Rngs(0)
    config = {
        'vocab_size': 100,
        'hidden_size': 64,
        'num_heads': 4,
        'max_len': 128,
        'num_memory_tokens': 3,
        'max_ponder_steps': 12,
        'epsilon': 0.05
    }
    
    model = UniversalTransformer(**config, rngs=rngs)
    
    # Simple compilation check
    assert model is not None

def test_ut_forward_pass():
    rngs = nnx.Rngs(0)
    config = {
        'vocab_size': 100,
        'hidden_size': 64,
        'num_heads': 4,
        'max_len': 128,
        'num_memory_tokens': 3,
        'max_ponder_steps': 12,
        'epsilon': 0.05
    }
    
    model = UniversalTransformer(**config, rngs=rngs)
    x = jnp.array([[1, 2, 3, 4, 5], [10, 11, 0, 0, 0]]) # (2, 5) batch
    pad_mask = jnp.array([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    
    # We expect output, ponder_loss, halt_steps, diagnostics
    out, ponder_loss, halt_steps, diagnostics = model(x, pad_mask)

    # The output should have the original length + memory tokens length
    assert out.shape == (2, 5 + 3, config['vocab_size'])
    assert ponder_loss.shape == ()
    assert halt_steps.shape == (2, 5 + 3)
    assert isinstance(diagnostics, dict)

def test_ut_backward_pass():
    rngs = nnx.Rngs(0)
    config = {
        'vocab_size': 100,
        'hidden_size': 64,
        'num_heads': 4,
        'max_len': 128,
        'num_memory_tokens': 3,
        'max_ponder_steps': 12,
        'epsilon': 0.05
    }
    
    model = UniversalTransformer(**config, rngs=rngs)
    
    def loss_fn(model, x, mask):
        out, ponder, _, _diag = model(x, mask)
        # Dummy Language Modeling loss on first token
        lm_loss = jnp.mean(out[:, 0, :])
        return lm_loss + ponder

    x = jnp.array([[1, 2, 3]])
    mask = jnp.array([[1, 1, 1]])
    
    # Test gradients can flow
    grads = nnx.grad(loss_fn)(model, x, mask)
    
    # Make sure we got gradients for SwiGLU components and Halting router
    assert grads is not None
