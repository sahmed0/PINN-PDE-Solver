import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from pinn.burgers import (
    BurgersPINN,
    burgers_reference,
    burgers_residual,
    compute_burgers_loss,
    generate_burgers_data,
    train_burgers,
)


def test_burgers_pinn_output_shape():
    model = BurgersPINN(jr.PRNGKey(0))
    out = model(jnp.array([0.5, 0.3]))
    assert out.shape == (1,)
    assert not jnp.isnan(out).any()


def test_burgers_residual_is_finite():
    model = BurgersPINN(jr.PRNGKey(1))
    r = burgers_residual(model, jnp.array(0.3), jnp.array(0.4))
    assert r.ndim == 0
    assert jnp.isfinite(r)


def test_compute_burgers_loss_finite_with_gradients():
    model = BurgersPINN(jr.PRNGKey(2))
    collocation_points, _ic_points, _bc_points = generate_burgers_data(
        jr.PRNGKey(3), num_collocation=50, num_bc=20, num_ic=20
    )

    loss_val, grads = eqx.filter_value_and_grad(compute_burgers_loss)(
        model, collocation_points
    )

    assert loss_val.ndim == 0
    assert jnp.isfinite(loss_val)

    found = False
    for leaf in jax.tree_util.tree_leaves(grads):
        if leaf is not None:
            found = True
            assert not jnp.isnan(leaf).any()
    assert found, "No gradients were computed."


def test_burgers_ansatz_makes_ic_bc_exact():
    # The hard-constraint ansatz enforces the IC/BCs exactly for ANY model, so the
    # IC and BC MSE must be ~0 even for an untrained BurgersPINN. burgers.py has no
    # loss-components helper, so the two terms are computed inline here.
    model = BurgersPINN(jr.PRNGKey(5))
    _collocation, ic_points, bc_points = generate_burgers_data(
        jr.PRNGKey(6), num_collocation=50, num_bc=40, num_ic=40
    )
    X_ic, u_ic = ic_points
    X_bc, u_bc = bc_points
    loss_ic = float(jnp.mean((jax.vmap(model)(X_ic) - u_ic) ** 2))
    loss_bc = float(jnp.mean((jax.vmap(model)(X_bc) - u_bc) ** 2))
    assert loss_ic < 1e-10
    assert loss_bc < 1e-10


def test_burgers_reference_shape_and_bc():
    x, t, u = burgers_reference()
    assert x.shape == (100,)
    assert t.shape == (100,)
    assert u.shape == (100, 100)

    # Zero Dirichlet BCs at x = -1 and x = 1.
    assert jnp.max(jnp.abs(u[:, 0])) < 1e-6
    assert jnp.max(jnp.abs(u[:, -1])) < 1e-6

    # Initial condition u(x, 0) = -sin(pi x).
    assert jnp.max(jnp.abs(u[0, :] - (-jnp.sin(jnp.pi * x)))) < 1e-3


def test_train_burgers_runs_two_epochs():
    model = train_burgers(jr.PRNGKey(4), epochs=2, num_collocation=200)
    assert isinstance(model, BurgersPINN)
