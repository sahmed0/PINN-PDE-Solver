import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from inverse import (
    InversePINN,
    generate_observations,
    generate_inverse_data,
    compute_inverse_loss,
    train_inverse,
)


def test_inverse_pinn_output_and_alpha_is_trainable():
    model = InversePINN(jr.PRNGKey(0), alpha_init=0.05)

    out = model(jnp.array([0.5, 0.3]))
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (1,)
    assert not jnp.isnan(out).any()

    # alpha must be a differentiable array leaf, i.e. survive eqx.is_array filtering.
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    assert any(leaf.shape == () and jnp.allclose(leaf, 0.05) for leaf in leaves), \
        "alpha scalar is not present as a trainable array leaf."


def test_generate_observations_shapes_and_noise():
    key = jr.PRNGKey(1)
    alpha_true = 0.042
    X_obs, u_obs = generate_observations(key, alpha_true, n_obs=40, noise_sigma=0.02)

    assert X_obs.shape == (40, 2)
    assert u_obs.shape == (40, 1)
    # Points lie inside the training domain.
    assert jnp.all(X_obs[:, 0] >= -1.0) and jnp.all(X_obs[:, 0] <= 1.0)
    assert jnp.all(X_obs[:, 1] >= 0.0) and jnp.all(X_obs[:, 1] <= 1.0)

    # Noise must actually be present: observations differ from the clean field.
    from analytical import u_exact
    clean = u_exact(X_obs[:, 0:1], X_obs[:, 1:2], alpha_true)
    assert float(jnp.mean(jnp.abs(u_obs - clean))) > 1e-4


def test_compute_inverse_loss_is_finite_with_valid_gradients():
    model = InversePINN(jr.PRNGKey(2))
    collocation_points, ic_points, bc_points = generate_inverse_data(
        jr.PRNGKey(3), num_collocation=50, num_bc=20, num_ic=20
    )
    obs_points = generate_observations(jr.PRNGKey(4), 0.042, n_obs=30)

    loss_val, grads = eqx.filter_value_and_grad(compute_inverse_loss)(
        model, collocation_points, ic_points, bc_points, obs_points
    )

    assert loss_val.ndim == 0
    assert jnp.isfinite(loss_val)

    found = False
    for leaf in jax.tree_util.tree_leaves(grads):
        if leaf is not None:
            found = True
            assert not jnp.isnan(leaf).any()
    assert found, "No gradients were computed."

    # In particular, alpha must receive a gradient (the residual depends on it).
    assert not jnp.isnan(grads.alpha).any()
    assert float(jnp.abs(grads.alpha)) > 0.0


def test_train_inverse_runs_and_reduces_loss():
    model, history = train_inverse(
        alpha_true=0.042, key=jr.PRNGKey(5), epochs=10, num_collocation=200
    )

    assert isinstance(model, InversePINN)
    assert len(history) >= 2
    assert history[-1]["loss"] < history[0]["loss"]
    assert {"epoch", "loss", "alpha_est", "alpha_error"} <= set(history[0].keys())


def test_train_inverse_converges_near_alpha_true():
    alpha_true = 0.042
    # decay_steps > epochs keeps the LR from annealing to zero before the short
    # run lands (see train_inverse); 500 steps along the full schedule already
    # carries alpha well within tolerance.
    model, _ = train_inverse(
        alpha_true=alpha_true, key=jr.PRNGKey(0), epochs=500,
        num_collocation=1000, decay_steps=2000
    )
    alpha_est = float(model.alpha)
    assert abs(alpha_est - alpha_true) / alpha_true < 0.20, \
        f"alpha estimate {alpha_est} not within 20% of {alpha_true}"
