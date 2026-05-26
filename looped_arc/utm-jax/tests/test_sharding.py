import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import nnx

from models.ut import UniversalTransformer

def test_mesh_sharding_compilation():
    # Only test if we have at least 1 device, simulate layout
    devices = jax.devices()
    if len(devices) < 1:
        pytest.skip("No devices found")
        
    # We mock a small mesh even if single device
    mesh = Mesh(devices, ('batch',))
    sharding = NamedSharding(mesh, PartitionSpec('batch'))
    
    rngs = nnx.Rngs(0)
    config = {
        'vocab_size': 100,
        'hidden_size': 64,
        'num_heads': 4,
        'max_len': 128,
        'num_memory_tokens': 2,
        'max_ponder_steps': 5,
        'epsilon': 0.05
    }
    
    model = UniversalTransformer(**config, rngs=rngs)
    
    @nnx.jit
    def forward_jit(model, x, mask):
        return model(x, mask)

    x = jnp.array([[1, 2], [3, 4]])
    mask = jnp.array([[1, 1], [1, 1]])
    
    x = jax.device_put(x, sharding)
    mask = jax.device_put(mask, sharding)
    
    # Should not crash during compilation/execution
    out, ponder, steps, diagnostics = forward_jit(model, x, mask)
    assert out.shape == (2, 4, config['vocab_size'])
