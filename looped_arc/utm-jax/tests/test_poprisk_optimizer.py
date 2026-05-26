"""Tests for the population-risk gate optimizer (Litman & Guo 2026, ADR_016)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from optimizers.poprisk import (
    PopriskState,
    _GATES,
    adamw_poprisk,
    scale_by_poprisk_gate,
)


def _make_params(seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    return {"a": jax.random.normal(k1, (8,)), "b": jax.random.normal(k2, (4, 5))}


def _make_grads(params, seed=1):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    new_leaves = [jax.random.normal(k, leaf.shape) for k, leaf in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


# ------------------------- state shape / init -------------------------

def test_state_shapes_match_params():
    params = _make_params()
    state = scale_by_poprisk_gate().init(params)
    assert isinstance(state, PopriskState)
    for name, p in params.items():
        assert state.mu[name].shape == p.shape
        assert state.nu[name].shape == p.shape
        assert state.sigma[name].shape == p.shape
    assert state.count.shape == ()


def test_state_starts_at_zero():
    params = _make_params()
    state = scale_by_poprisk_gate().init(params)
    assert int(state.count) == 0
    for leaf in jax.tree_util.tree_leaves(state.mu):
        assert jnp.all(leaf == 0)
    for leaf in jax.tree_util.tree_leaves(state.nu):
        assert jnp.all(leaf == 0)
    for leaf in jax.tree_util.tree_leaves(state.sigma):
        assert jnp.all(leaf == 0)


# ------------------------- gate range -------------------------

@pytest.mark.parametrize("name,fn", list(_GATES.items()))
def test_gate_function_in_unit_interval(name, fn):
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    m_hat = jax.random.normal(k1, (200,))
    s_hat = jnp.abs(jax.random.normal(k2, (200,))) + 1e-3
    q = fn(jnp.square(m_hat), s_hat, alpha=1.0, lambda_pop=0.1, eps=1e-12)
    assert q.shape == m_hat.shape
    assert jnp.all(q >= 0.0), f"gate {name} produced negative q"
    assert jnp.all(q <= 1.0 + 1e-6), f"gate {name} produced q > 1"


# ------------------------- gate limiting behaviour -------------------------

def test_huge_alpha_zeroes_hard_gate():
    """alpha -> infinity with hard gate => q ≡ 0 => zero updates."""
    params = _make_params()
    tx = scale_by_poprisk_gate(gate="hard", alpha=1e12)
    state = tx.init(params)
    grads = _make_grads(params, seed=30)
    updates, _ = tx.update(grads, state)
    for leaf in jax.tree_util.tree_leaves(updates):
        np.testing.assert_allclose(np.asarray(leaf), 0.0, atol=1e-10)


def test_huge_lambda_pop_zeroes_soft_gate():
    params = _make_params()
    tx = scale_by_poprisk_gate(gate="soft", alpha=0.0, lambda_pop=1e12)
    state = tx.init(params)
    grads = _make_grads(params, seed=31)
    updates, _ = tx.update(grads, state)
    for leaf in jax.tree_util.tree_leaves(updates):
        np.testing.assert_allclose(np.asarray(leaf), 0.0, atol=1e-6)


def test_huge_lambda_pop_zeroes_snr_gate():
    params = _make_params()
    tx = scale_by_poprisk_gate(gate="snr", lambda_pop=1e12)
    state = tx.init(params)
    grads = _make_grads(params, seed=32)
    updates, _ = tx.update(grads, state)
    for leaf in jax.tree_util.tree_leaves(updates):
        np.testing.assert_allclose(np.asarray(leaf), 0.0, atol=1e-6)


# ------------------------- numerical equivalence to Adam -------------------------

def test_hard_alpha_zero_matches_scale_by_adam():
    """With alpha=0 and the hard gate, q = 1 wherever m_hat^2 > 0.
    The update direction must then equal optax.scale_by_adam to float32 tolerance.
    """
    params = _make_params()
    tx_pop = scale_by_poprisk_gate(gate="hard", alpha=0.0)
    tx_adam = optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8)

    state_pop = tx_pop.init(params)
    state_adam = tx_adam.init(params)

    for step in range(4):
        grads = _make_grads(params, seed=100 + step)
        u_pop, state_pop = tx_pop.update(grads, state_pop)
        u_adam, state_adam = tx_adam.update(grads, state_adam)
        for name in params:
            np.testing.assert_allclose(
                np.asarray(u_pop[name]),
                np.asarray(u_adam[name]),
                atol=1e-6,
                err_msg=f"step={step}, leaf={name}",
            )


# ------------------------- variance estimator -------------------------

def test_sigma_after_one_step():
    """After one step from zero state, sigma = (1-rho) * g^2 since m_prev = 0."""
    params = _make_params()
    tx = scale_by_poprisk_gate(rho=0.5)
    state = tx.init(params)
    grads = _make_grads(params, seed=40)
    _, state = tx.update(grads, state)
    for name in params:
        expected = 0.5 * jnp.square(grads[name])
        np.testing.assert_allclose(
            np.asarray(state.sigma[name]),
            np.asarray(expected),
            atol=1e-6,
            err_msg=f"leaf {name}",
        )


def test_sigma_nonnegative_over_steps():
    params = _make_params()
    tx = scale_by_poprisk_gate()
    state = tx.init(params)
    for step in range(8):
        grads = _make_grads(params, seed=50 + step)
        _, state = tx.update(grads, state)
        for leaf in jax.tree_util.tree_leaves(state.sigma):
            assert jnp.all(leaf >= 0), f"sigma negative at step {step}"


# ------------------------- bias correction -------------------------

def test_bias_correction_at_t1():
    """At t=1 with eps_root=0 and hard alpha=0, update = sign(g)."""
    params = {"x": jnp.ones((3,))}
    grads = {"x": jnp.array([1.0, 2.0, -3.0])}
    tx = scale_by_poprisk_gate(
        b1=0.9, b2=0.999, gate="hard", alpha=0.0, eps_root=0.0,
    )
    state = tx.init(params)
    upd, _ = tx.update(grads, state)
    expected = jnp.sign(grads["x"])
    # atol loose: 1/(1-b2) = 1000 amplifies float32 rounding by ~1e-4.
    np.testing.assert_allclose(np.asarray(upd["x"]), np.asarray(expected), atol=1e-4)


# ------------------------- end-to-end smoke test -------------------------

@pytest.mark.parametrize("gate", ["hard", "soft", "snr"])
def test_minimizes_simple_quadratic(gate):
    """The full adamw_poprisk chain converges on argmin of 0.5 * ||w - target||^2."""
    target = jnp.array([1.0, -2.0, 0.5])
    tx = adamw_poprisk(learning_rate=0.05, alpha=0.0, lambda_pop=0.0, gate=gate)
    params = jnp.zeros(3)
    state = tx.init(params)

    def loss(w):
        return 0.5 * jnp.sum((w - target) ** 2)

    grad_fn = jax.grad(loss)
    for _ in range(800):
        g = grad_fn(params)
        updates, state = tx.update(g, state, params)
        params = optax.apply_updates(params, updates)

    np.testing.assert_allclose(np.asarray(params), np.asarray(target), atol=2e-2)


# ------------------------- invalid input -------------------------

def test_invalid_gate_raises():
    with pytest.raises(ValueError):
        scale_by_poprisk_gate(gate="bogus")


# ------------------------- hybrid multi-transform (--poprisk_skip_router) -------------------------

def test_hybrid_routes_router_to_adamw_and_rest_to_poprisk():
    """The hybrid optimizer pattern from train.py: router params -> plain AdamW,
    everything else -> adamw_poprisk. Verifies that the multi_transform dispatch
    actually routes correctly and each group gets its own update rule.
    """
    params = {"router": jnp.ones((4,)), "block": jnp.ones((3, 5)), "embed": jnp.ones((6,))}
    grads = _make_grads(params, seed=200)

    lr = 0.01
    poprisk_chain = adamw_poprisk(learning_rate=lr, gate="hard", alpha=0.0, weight_decay=0.0)
    adamw_chain = optax.adamw(learning_rate=lr, weight_decay=0.0)

    def label_fn(path, p):
        path_str = "".join(str(k) for k in path).lower()
        return "adamw" if "router" in path_str else "adamw_poprisk"

    hybrid = optax.multi_transform(
        {"adamw": adamw_chain, "adamw_poprisk": poprisk_chain},
        lambda p: jax.tree_util.tree_map_with_path(label_fn, p),
    )

    state_h = hybrid.init(params)
    state_p = poprisk_chain.init(params)
    state_a = adamw_chain.init(params)

    u_h, _ = hybrid.update(grads, state_h, params)
    u_p, _ = poprisk_chain.update(grads, state_p, params)
    u_a, _ = adamw_chain.update(grads, state_a, params)

    # Router branch of hybrid must match the standalone AdamW update.
    np.testing.assert_allclose(np.asarray(u_h["router"]), np.asarray(u_a["router"]), atol=1e-6,
                                err_msg="hybrid did not route 'router' to plain AdamW")
    # Non-router branches must match the standalone poprisk update.
    for name in ("block", "embed"):
        np.testing.assert_allclose(np.asarray(u_h[name]), np.asarray(u_p[name]), atol=1e-6,
                                    err_msg=f"hybrid did not route {name!r} to adamw_poprisk")


def test_hybrid_label_fn_substring_predicate():
    """Substring check 'router' catches ACTRouter's proj.* params but doesn't
    accidentally match other param names. Sanity check on the label_fn predicate.
    """
    paths = [
        (("router", "proj", "kernel"), "adamw"),
        (("router", "proj", "bias"),   "adamw"),
        (("block", "ffn", "w_up"),     "adamw_poprisk"),
        (("embed", "embedding"),       "adamw_poprisk"),
        (("mem_tokens", "value"),      "adamw_poprisk"),
    ]
    for path, want in paths:
        path_str = "".join(str(k) for k in path).lower()
        got = "adamw" if "router" in path_str else "adamw_poprisk"
        assert got == want, f"path {path} -> {got!r}, expected {want!r}"
