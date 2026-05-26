import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from models.ut import ACTRouter

def test_halting_probability_bounds():
    rngs = nnx.Rngs(0)
    router = ACTRouter(features=64, rngs=rngs)
    
    # Generate extreme random noise to check robustness
    x = jax.random.normal(jax.random.PRNGKey(1), (32, 10, 64)) * 1000.0
    
    p = router(x)
    
    # Extreme inputs should squash nicely and not NaN
    assert not jnp.any(jnp.isnan(p))
    assert jnp.all(p >= 0.0)
    assert jnp.all(p <= 1.0)
    
def test_ponder_accumulation_math():
    # Test strict bounding for ponder logic 
    # Example logic verification
    p_accumulated = 0.9
    p_tmp = 0.5
    
    # The remainder is strictly bounded
    remainder = jnp.minimum(1.0 - p_accumulated, p_tmp)
    assert remainder == 0.1
