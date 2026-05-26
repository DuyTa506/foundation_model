import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple

class MuonState(NamedTuple):
    momentum: optax.Updates

def scale_by_muon(momentum: float = 0.95, n_steps: int = 5, use_bfloat16: bool = True) -> optax.GradientTransformation:
    """
    Optax transformation for Muon momentum orthogonalization.
    It expects >=2D parameters and strictly calculates zero-power stabilization.
    Note: Weight decay and learning rate scales should be chained AFTER this.
    """
    def init_fn(params):
        return MuonState(momentum=jax.tree_util.tree_map(jnp.zeros_like, params))
        
    def update_fn(updates, state, params=None):
        def _zeropower_via_newtonschulz5(G):
            # Fallback if somehow a 1D tensor slips through the multi_transform router
            if len(G.shape) < 2:
                return G
            
            orig_dtype = G.dtype
            X = G.astype(jnp.bfloat16) if use_bfloat16 else G
            
            # Need to transpose if height > width symmetrically 
            transposed = False
            if X.shape[0] > X.shape[1]:
                X = X.T
                transposed = True
                
            a, b, c = (3.4445, -4.7750, 2.0315)
            
            # Use max constraint for zero-avoidance (stable Frobenius norm)
            X_norm = jnp.linalg.norm(X)
            X = X / jnp.maximum(X_norm, 1e-7)
            
            for _ in range(n_steps):
                A = X @ X.T
                B = b * A + c * (A @ A)
                X = a * X + B @ X
                
            if transposed:
                X = X.T
                
            # Dynamic matrix-shape learning rate adjustment (derived from Moonlight)
            A, B = G.shape[0], G.shape[1]
            adjusted_ratio = 0.2 * jnp.sqrt(jnp.maximum(A, B))
                
            return (X * adjusted_ratio).astype(orig_dtype)

        # Update running momentum
        new_momentum = jax.tree_util.tree_map(
            lambda m, g: momentum * m + g, state.momentum, updates
        )
        
        # Apply strict orthogonalization process
        orthogonalized_updates = jax.tree_util.tree_map(
            _zeropower_via_newtonschulz5, new_momentum
        )
        
        return orthogonalized_updates, MuonState(momentum=new_momentum)
        
    return optax.GradientTransformation(init_fn, update_fn)
