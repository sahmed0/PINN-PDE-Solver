import jax
import jax.numpy as jnp
import jax.random as jr

from pinn.analytical import (
    evaluate,
    make_grid,
    max_abs_error,
    predict_on_grid,
    relative_l2_error,
    u_exact,
)
from pinn.model import ParametricPINN


def test_u_exact_matches_initial_and_boundary_conditions():
    x = jnp.linspace(-1.0, 1.0, 50)
    # At t = 0 the solution is the initial profile sin(pi x).
    assert jnp.allclose(u_exact(x, 0.0, 0.05), jnp.sin(jnp.pi * x), atol=1e-6)
    # Homogeneous Dirichlet boundaries hold for all t and alpha.
    for t in (0.0, 0.3, 1.0):
        assert jnp.allclose(u_exact(-1.0, t, 0.07), 0.0, atol=1e-6)
        assert jnp.allclose(u_exact(1.0, t, 0.07), 0.0, atol=1e-6)


def test_u_exact_satisfies_heat_equation():
    # The closed form must satisfy u_t = alpha * u_xx pointwise.
    alpha = 0.05

    def u_scalar(x, t):
        return u_exact(x, t, alpha)

    u_t = jax.grad(u_scalar, argnums=1)
    u_x = jax.grad(u_scalar, argnums=0)
    u_xx = jax.grad(u_x, argnums=0)

    for x in (-0.5, 0.1, 0.7):
        for t in (0.0, 0.4, 0.9):
            residual = u_t(x, t) - alpha * u_xx(x, t)
            assert abs(float(residual)) < 1e-5, f"residual {residual} at ({x},{t})"


def test_make_grid_shapes_and_ranges():
    inputs, X, T = make_grid(nx=20, nt=10, alpha=0.03)
    assert inputs.shape == (200, 3)
    assert X.shape == (10, 20) and T.shape == (10, 20)
    # Column 2 of inputs is the constant alpha.
    assert jnp.allclose(inputs[:, 2], 0.03)
    assert jnp.isclose(X.min(), -1.0) and jnp.isclose(X.max(), 1.0)
    assert jnp.isclose(T.min(), 0.0) and jnp.isclose(T.max(), 1.0)


def test_error_metrics_are_zero_against_exact_field():
    # A "model" that returns the exact solution must have ~zero error. This
    # validates the metric plumbing independently of any trained network.
    class ExactModel:
        def __call__(self, inp):
            x, t, alpha = inp[0], inp[1], inp[2]
            return jnp.array([u_exact(x, t, alpha)])

    m = ExactModel()
    assert float(relative_l2_error(m, 0.05, nx=30, nt=30)) < 1e-6
    assert float(max_abs_error(m, 0.05, nx=30, nt=30)) < 1e-6


def test_predict_on_grid_and_evaluate_with_untrained_model():
    model = ParametricPINN(jr.PRNGKey(0))
    u_pred, u_ref = predict_on_grid(model, nx=25, nt=25, alpha=0.05)
    assert u_pred.shape == (25, 25) == u_ref.shape
    assert not jnp.isnan(u_pred).any()

    metrics = evaluate(model, alphas=(0.02, 0.08), nx=25, nt=25)
    assert set(metrics["alphas"]) == {0.02, 0.08}
    assert metrics["mean_rel_l2"] > 0  # untrained network is not exact
    assert jnp.isfinite(metrics["mean_rel_l2"])
