"""
Analytical (closed-form) reference solution for the 1D heat equation and the
error metrics used to validate the PINN against it.

For the heat equation  u_t = alpha * u_xx  on x in [-1, 1], t >= 0, with the
initial condition  u(x, 0) = sin(pi * x)  and homogeneous Dirichlet boundary
conditions  u(-1, t) = u(1, t) = 0, separation of variables gives the exact
solution

    u(x, t) = sin(pi * x) * exp(-alpha * pi**2 * t).

Because sin(+/- pi) = 0 the boundary conditions hold for all t, and at t = 0 it
reduces to the initial profile. This gives us a ground truth everywhere in the
domain (and for every alpha), so we can report a real, physical error instead
of just the training loss.
"""

import jax
import jax.numpy as jnp


def u_exact(x, t, alpha):
    """Exact solution u(x, t) = sin(pi x) exp(-alpha pi^2 t).

    Accepts scalars or broadcastable arrays for x, t and alpha.
    """
    return jnp.sin(jnp.pi * x) * jnp.exp(-alpha * (jnp.pi ** 2) * t)


def make_grid(nx, nt, alpha):
    """Build a regular (x, t) grid for a fixed alpha.

    Returns (inputs, X, T):
        inputs : (nx * nt, 3) array of [x, t, alpha] rows, ready for vmap(model)
        X, T   : (nt, nx) meshgrid arrays matching the flattened ordering
                 (time is the outer/row axis, space the inner/column axis).
    """
    x = jnp.linspace(-1.0, 1.0, nx)
    t = jnp.linspace(0.0, 1.0, nt)
    # indexing="xy" -> X, T both have shape (nt, nx): rows are time, cols space.
    X, T = jnp.meshgrid(x, t, indexing="xy")
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)
    alpha_flat = jnp.full_like(x_flat, alpha)
    inputs = jnp.hstack([x_flat, t_flat, alpha_flat])
    return inputs, X, T


def predict_on_grid(model, nx, nt, alpha):
    """Evaluate the PINN on a grid, returning predicted and exact fields.

    Both returned arrays have shape (nt, nx).
    """
    inputs, X, T = make_grid(nx, nt, alpha)
    u_pred = jax.vmap(model)(inputs).reshape(nt, nx)
    u_ref = u_exact(X, T, alpha)
    return u_pred, u_ref


def relative_l2_error(model, alpha, nx=100, nt=100):
    """Relative L2 error  ||u_pred - u_exact||_2 / ||u_exact||_2  for one alpha."""
    u_pred, u_ref = predict_on_grid(model, nx, nt, alpha)
    num = jnp.linalg.norm(u_pred - u_ref)
    den = jnp.linalg.norm(u_ref)
    return num / den


def max_abs_error(model, alpha, nx=100, nt=100):
    """Maximum absolute (L-infinity) error over the grid for one alpha."""
    u_pred, u_ref = predict_on_grid(model, nx, nt, alpha)
    return jnp.max(jnp.abs(u_pred - u_ref))


def evaluate(model, alphas=(0.01, 0.05, 0.1), nx=100, nt=100):
    """Aggregate error metrics across several alpha values.

    Returns a dict with per-alpha relative L2 / L-infinity errors plus the
    means, suitable for printing a validation report.
    """
    alphas = tuple(float(a) for a in alphas)
    rel_l2 = {a: float(relative_l2_error(model, a, nx, nt)) for a in alphas}
    linf = {a: float(max_abs_error(model, a, nx, nt)) for a in alphas}
    return {
        "alphas": alphas,
        "rel_l2": rel_l2,
        "linf": linf,
        "mean_rel_l2": float(sum(rel_l2.values()) / len(rel_l2)),
        "mean_linf": float(sum(linf.values()) / len(linf)),
    }


def format_report(metrics):
    """Pretty-print the dict returned by `evaluate` as a small table."""
    lines = ["    alpha |  rel L2  |   L-inf", "    ------+----------+---------"]
    for a in metrics["alphas"]:
        lines.append(
            f"    {a:5.3f} | {metrics['rel_l2'][a]:.2e} | {metrics['linf'][a]:.2e}"
        )
    lines.append(
        f"     mean | {metrics['mean_rel_l2']:.2e} | {metrics['mean_linf']:.2e}"
    )
    return "\n".join(lines)
