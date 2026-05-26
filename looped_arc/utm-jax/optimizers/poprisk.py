"""Population-risk gradient leave-one-out gate on AdamW.

Implements Algorithm 1 of:
    Litman & Guo. A Theory of Generalization in Deep Learning.
    arXiv:2605.01172v1, May 2026.

The gate suppresses parameter updates whose squared mean gradient m_hat^2
fails to exceed leave-one-out noise (alpha * s_hat). Three gate variants are
exposed (hard / soft / snr); the soft form is Algorithm 1 in the paper.
See ADR_016 for the design decision and integration notes.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Callable, NamedTuple, Union


class PopriskState(NamedTuple):
    count: jax.Array
    mu: optax.Updates
    nu: optax.Updates
    sigma: optax.Updates


def _gate_hard(m_hat_sq, s_hat, alpha, lambda_pop, eps):
    return (m_hat_sq > alpha * s_hat).astype(m_hat_sq.dtype)


def _gate_soft(m_hat_sq, s_hat, alpha, lambda_pop, eps):
    delta = jnp.maximum(m_hat_sq - alpha * s_hat, 0.0)
    return delta / (delta + lambda_pop * s_hat + eps)


def _gate_snr(m_hat_sq, s_hat, alpha, lambda_pop, eps):
    del alpha
    return m_hat_sq / (m_hat_sq + lambda_pop * s_hat + eps)


_GATES = {"hard": _gate_hard, "soft": _gate_soft, "snr": _gate_snr}


def scale_by_poprisk_gate(
    b1: float = 0.9,
    b2: float = 0.999,
    rho: float = 0.99,
    alpha: float = 1.0,
    lambda_pop: float = 0.0,
    eps_gate: float = 1e-12,
    eps_root: float = 1e-8,
    gate: str = "soft",
) -> optax.GradientTransformation:
    """Adam direction multiplied elementwise by a population-safety gate q in [0, 1].

    Args:
        b1, b2: Adam first/second-moment decay rates.
        rho: gradient-variance EMA decay (paper: rho in eq. 240).
        alpha: leave-one-out coefficient. 1.0 for fresh-batch / online regime
            (default); b/(n-b) for finite-dataset regime. Ignored when gate
            == "snr".
        lambda_pop: population-risk gate sharpness. The paper notes this is
            "typically unnecessary at scale" (eq. 245); 0.0 is a reasonable
            default.
        eps_gate: stabiliser in the gate denominator.
        eps_root: Adam epsilon inside sqrt(v_hat).
        gate: "hard", "soft" (Algorithm 1), or "snr".
    """
    if gate not in _GATES:
        raise ValueError(
            f"poprisk gate must be one of {sorted(_GATES)}, got {gate!r}"
        )
    gate_fn = _GATES[gate]

    def init_fn(params):
        zero = lambda p: jnp.zeros_like(p)
        return PopriskState(
            count=jnp.zeros([], jnp.int32),
            mu=jax.tree_util.tree_map(zero, params),
            nu=jax.tree_util.tree_map(zero, params),
            sigma=jax.tree_util.tree_map(zero, params),
        )

    def update_fn(updates, state, params=None):
        del params
        count = state.count + 1
        t = count.astype(jnp.float32)

        # sigma uses m_prev (state.mu, before the m update) per Algorithm 1 line 4-5.
        new_sigma = jax.tree_util.tree_map(
            lambda s, g, m_prev: rho * s + (1.0 - rho) * jnp.square(g - m_prev),
            state.sigma, updates, state.mu,
        )
        new_mu = jax.tree_util.tree_map(
            lambda m, g: b1 * m + (1.0 - b1) * g, state.mu, updates,
        )
        new_nu = jax.tree_util.tree_map(
            lambda v, g: b2 * v + (1.0 - b2) * jnp.square(g), state.nu, updates,
        )

        bc1 = 1.0 - jnp.power(jnp.float32(b1), t)
        bc2 = 1.0 - jnp.power(jnp.float32(b2), t)
        bcr = 1.0 - jnp.power(jnp.float32(rho), t)

        def _step(mu, nu, sigma):
            m_hat = mu / bc1
            v_hat = nu / bc2
            s_hat = sigma / bcr
            q = gate_fn(jnp.square(m_hat), s_hat, alpha, lambda_pop, eps_gate)
            return q * m_hat / (jnp.sqrt(v_hat) + eps_root)

        new_updates = jax.tree_util.tree_map(_step, new_mu, new_nu, new_sigma)
        return new_updates, PopriskState(count=count, mu=new_mu, nu=new_nu, sigma=new_sigma)

    return optax.GradientTransformation(init_fn, update_fn)


def adamw_poprisk(
    learning_rate: Union[float, Callable],
    b1: float = 0.9,
    b2: float = 0.999,
    rho: float = 0.99,
    alpha: float = 1.0,
    lambda_pop: float = 0.0,
    eps_gate: float = 1e-12,
    eps_root: float = 1e-8,
    weight_decay: float = 0.0,
    gate: str = "soft",
) -> optax.GradientTransformation:
    """AdamW chain with the population-risk gate. Mirrors optax.adamw's signature."""
    lr_transform = (
        optax.scale_by_schedule(learning_rate)
        if callable(learning_rate)
        else optax.scale(learning_rate)
    )
    return optax.chain(
        scale_by_poprisk_gate(b1, b2, rho, alpha, lambda_pop, eps_gate, eps_root, gate),
        optax.add_decayed_weights(weight_decay),
        lr_transform,
        optax.scale(-1.0),
    )
