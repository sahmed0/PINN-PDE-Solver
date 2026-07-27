"""
Burgers' equation as a second PDE, to show the architecture is a general PDE
solver and not a one-off fit to the heat equation.

    u_t + u u_x = nu u_xx,    x in [-1, 1], t in [0, 1]
    u(x, 0) = -sin(pi x)                          (IC)
    u(-1, t) = u(1, t) = 0                         (Dirichlet BCs)
    nu = 0.01 / pi                                 (Raissi et al. 2019)

This problem is nonlinear (the u u_x advection term) and develops a near-shock
around t ~ 0.7 at this small viscosity, where classical solvers need fine grids.
Unlike the heat equation it has no simple closed form under these Dirichlet BCs,
so the reference solution comes from a method-of-lines numerical integration
(see burgers_reference) rather than an analytical formula.
"""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax.experimental.ode import odeint

# Raissi viscosity; the near-shock forms around t ~ 0.7 at this value.
NU = 0.01 / jnp.pi

# nu is fixed here (not a network input), so the inputs are only [x, t]. x is
# already in [-1, 1]; t in [0, 1] is centred/scaled to ~[-1, 1] before the MLP.
INPUT_CENTER = (0.0, 0.5)
INPUT_SCALE = (1.0, 0.5)


class BurgersPINN(eqx.Module):
    """
    Physics-Informed Neural Network for the viscous Burgers' equation.

    Inputs (shape: (2,)):
        - x: Spatial coordinate in [-1, 1]
        - t: Time               in [0, 1]

    Outputs (shape: (1,)):
        - u: Velocity

    The MLP is wider and deeper than the heat model (width 64, depth 4) because
    Burgers' develops a sharp interior gradient the network must resolve.

    The same hard-constraint ansatz idea as ParametricPINN is used, adapted to
    this IC:

        u(x, t) = -sin(pi x) + (1 - x^2) * t * N(x, t)

    - At t = 0:      u = -sin(pi x)                 -> initial condition u(x, 0)
    - At x = +/- 1:  (1 - x^2) = 0 and sin(pi x)=0  -> zero Dirichlet BCs

    so only the interior dynamics are learnt through the PDE residual.
    """
    mlp: eqx.nn.MLP
    input_center: tuple = eqx.field(static=True)
    input_scale: tuple = eqx.field(static=True)

    def __init__(self, key: jr.PRNGKey, width_size: int = 64, depth: int = 4):
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        self.input_center = INPUT_CENTER
        self.input_scale = INPUT_SCALE

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_phys, t_phys = x[0], x[1]

        center = jnp.asarray(self.input_center)
        scale = jnp.asarray(self.input_scale)
        x_norm = (x - center) / scale

        n = self.mlp(x_norm)[0]

        u = -jnp.sin(jnp.pi * x_phys) + (1.0 - x_phys ** 2) * t_phys * n
        return jnp.reshape(u, (1,))


def burgers_residual(model, x, t):
    """
    PDE residual u_t + u u_x - nu u_xx at a single (x, t). Mirrors
    physics.heat_equation_residual but for the 2-input model and with the
    nonlinear advection term u * u_x.
    """
    def u_fn(x_val, t_val):
        inputs = jnp.stack([x_val, t_val])
        return model(inputs)[0]

    u_t_fn = jax.grad(u_fn, argnums=1)
    u_x_fn = jax.grad(u_fn, argnums=0)
    u_xx_fn = jax.grad(u_x_fn, argnums=0)

    u = u_fn(x, t)
    u_t = u_t_fn(x, t)
    u_x = u_x_fn(x, t)
    u_xx = u_xx_fn(x, t)
    return u_t + u * u_x - NU * u_xx


def compute_burgers_loss(model, collocation_points):
    """
    Training loss: the PDE residual MSE.

    As with the heat model, the ansatz enforces the IC/BCs exactly, so IC/BC
    terms would be structurally ~0 and contribute no gradient. Only the PDE
    residual trains the network, so that is all this loss computes.
    """
    x_c, t_c = collocation_points[:, 0], collocation_points[:, 1]
    vmap_residual = jax.vmap(burgers_residual, in_axes=(None, 0, 0))
    residuals = vmap_residual(model, x_c, t_c)
    return jnp.mean(residuals ** 2)


def generate_burgers_data(key, num_collocation=2000, num_bc=100, num_ic=100):
    """
    Collocation, IC and BC points for Burgers'. Same pattern as
    train.generate_training_data but every point is just [x, t] (nu is fixed).
    Domains: x in [-1, 1], t in [0, 1].
    """
    k_x, k_t = jr.split(key, 2)
    x_c = jr.uniform(k_x, (num_collocation, 1), minval=-1.0, maxval=1.0)
    t_c = jr.uniform(k_t, (num_collocation, 1), minval=0.0, maxval=1.0)
    collocation_points = jnp.hstack([x_c, t_c])

    x_ic = jr.uniform(jr.PRNGKey(24), (num_ic, 1), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros((num_ic, 1))
    X_ic = jnp.hstack([x_ic, t_ic])
    u_ic = -jnp.sin(jnp.pi * x_ic)
    ic_points = (X_ic, u_ic)

    t_bc = jr.uniform(jr.PRNGKey(26), (num_bc, 1), minval=0.0, maxval=1.0)
    x_bc = jnp.where(jr.bernoulli(jr.PRNGKey(28), 0.5, (num_bc, 1)), 1.0, -1.0)
    X_bc = jnp.hstack([x_bc, t_bc])
    u_bc = jnp.zeros((num_bc, 1))
    bc_points = (X_bc, u_bc)

    return collocation_points, ic_points, bc_points


def burgers_reference(nu=None, nx=512, nt=100, nx_out=100):
    """
    Method-of-lines reference solution: the ground truth for validation.

    The semi-discretisation of u_t = nu u_xx - u u_x on a fine spatial grid of
    `nx` points (central differences, zero Dirichlet boundaries) gives an ODE
    system du/dt = f(u, t), which jax.experimental.ode.odeint integrates from
    t = 0 to t = 1. The result is then interpolated onto a coarser output grid
    of `nx_out` points so it can be embedded compactly in the exported JSON.

    Returns (x_grid, t_grid, u_grid) with u_grid of shape (nt, nx_out); rows are
    time, columns space. nu defaults to NU.
    """
    if nu is None:
        nu = NU

    x = jnp.linspace(-1.0, 1.0, nx)
    dx = x[1] - x[0]
    u0 = -jnp.sin(jnp.pi * x)

    def rhs(u, _t):
        # Hold the Dirichlet boundaries at 0; roll wraps at the edges but those
        # rows are overwritten below, and the interior neighbours of the
        # boundary correctly see the zero boundary value.
        u = u.at[0].set(0.0).at[-1].set(0.0)
        u_x = (jnp.roll(u, -1) - jnp.roll(u, 1)) / (2.0 * dx)
        u_xx = (jnp.roll(u, -1) - 2.0 * u + jnp.roll(u, 1)) / dx ** 2
        du = nu * u_xx - u * u_x
        return du.at[0].set(0.0).at[-1].set(0.0)

    t_grid = jnp.linspace(0.0, 1.0, nt)
    sol = odeint(rhs, u0, t_grid, rtol=1e-7, atol=1e-9)

    x_out = jnp.linspace(-1.0, 1.0, nx_out)
    u_grid = jax.vmap(lambda row: jnp.interp(x_out, x, row))(sol)
    return x_out, t_grid, u_grid


@eqx.filter_jit
def train_burgers_step(model, opt_state, optimizer, collocation_points):
    """Executes a single compiled optimisation step."""
    loss_val, grads = eqx.filter_value_and_grad(compute_burgers_loss)(
        model, collocation_points
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


def train_burgers(key, epochs=20000, lr=1e-3, num_collocation=2000):
    """Cosine-annealed Adam training loop for Burgers'.

    There is no closed form to validate against, so the printed log reports the
    training loss only; correctness is checked afterwards in evaluate_burgers
    against the method-of-lines reference.
    """
    model = BurgersPINN(key)

    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=epochs)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # IC/BC points are enforced exactly by the ansatz, so training only needs
    # the collocation points (see compute_burgers_loss).
    collocation_points, _ic_points, _bc_points = generate_burgers_data(
        key, num_collocation=num_collocation
    )

    for epoch in range(epochs):
        model, opt_state, loss = train_burgers_step(
            model, opt_state, optimizer, collocation_points
        )

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:04d} | Loss: {loss:.6f}")

    return model


def evaluate_burgers(model, nu=None, nx=100, nt=100):
    """Evaluate the PINN against the method-of-lines reference on a shared grid.

    Returns a dict with relative L2 and L-infinity errors plus the predicted and
    reference fields (both shape (nt, nx)) and the grid axes.
    """
    x, t, u_ref = burgers_reference(nu=nu, nt=nt, nx_out=nx)

    X, T = jnp.meshgrid(x, t, indexing="xy")
    inputs = jnp.hstack([X.reshape(-1, 1), T.reshape(-1, 1)])
    u_pred = jax.vmap(model)(inputs).reshape(nt, nx)

    num = jnp.linalg.norm(u_pred - u_ref)
    den = jnp.linalg.norm(u_ref)
    rel_l2 = float(num / den)
    linf = float(jnp.max(jnp.abs(u_pred - u_ref)))

    return {
        "rel_l2": rel_l2,
        "linf": linf,
        "u_pred": u_pred,
        "u_ref": u_ref,
        "x": x,
        "t": t,
    }


def export_burgers_to_json(model, filepath, nu=None, nx=100, nt=100):
    """
    Export the trained Burgers' MLP plus the embedded reference field.

    Same weight layout as train.export_to_json (a list of Linear layers, tanh
    after every layer except the last). The consumer must replicate the input
    normalisation and the "burgers_dirichlet_negsin" ansatz
    (u = -sin(pi x) + (1 - x^2) * t * N); see frontend/src/lib/inference.ts.
    The method-of-lines reference is embedded under "reference" as the ground
    truth, with rel_l2 / linf measured against it.
    """
    if nu is None:
        nu = NU

    metrics = evaluate_burgers(model, nu=nu, nx=nx, nt=nt)

    layers = []
    for layer in model.mlp.layers:
        weight = np.asarray(layer.weight, dtype=np.float64)
        bias = layer.bias
        bias = (np.zeros(weight.shape[0]) if bias is None
                else np.asarray(bias, dtype=np.float64))
        layers.append({
            "weight": weight.tolist(),
            "bias": bias.tolist(),
        })

    payload = {
        "format": "tanh-mlp-burgers-v1",
        "in_size": 2,
        "out_size": 1,
        "activation": "tanh",
        "input_names": ["x", "t"],
        "output_names": ["u"],
        "input_center": list(model.input_center),
        "input_scale": list(model.input_scale),
        "ansatz": "burgers_dirichlet_negsin",
        "nu": float(nu),
        "reference": {
            "x": np.asarray(metrics["x"], dtype=np.float64).tolist(),
            "t": np.asarray(metrics["t"], dtype=np.float64).tolist(),
            "u": np.asarray(metrics["u_ref"], dtype=np.float64).tolist(),
        },
        "rel_l2": metrics["rel_l2"],
        "linf": metrics["linf"],
        "layers": layers,
    }

    print(f"Exporting Burgers' model to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print("Export complete.")


def run_burgers_demo(key=None, epochs=20000, export_path=None):
    """
    End-to-end Burgers' demo: train, validate against the method-of-lines
    reference, print a report, and optionally export for the frontend.
    """
    if key is None:
        key = jr.PRNGKey(42)

    print("\n--- Burgers' equation: nonlinear PDE, numerical reference ---")
    print(f"Training BurgersPINN (nu = {float(NU):.6e})...")
    model = train_burgers(key, epochs=epochs)

    metrics = evaluate_burgers(model)
    print("\nBurgers' report (vs. method-of-lines reference):")
    print(f"    relative L2 : {metrics['rel_l2']:.3e}")
    print(f"    L-infinity  : {metrics['linf']:.3e}")

    if export_path is not None:
        export_burgers_to_json(model, export_path)

    return model, metrics


if __name__ == "__main__":
    run_burgers_demo()
