"""
The inverse problem: recover an unknown thermal diffusivity alpha from sparse,
noisy temperature observations.

The forward model (model.ParametricPINN) takes alpha as an input and predicts u.
Here we flip it: alpha is unknown, and all we are given is a handful of noisy
measurements u_i at scattered (x_i, t_i) points. We fit a network u(x, t) to
those measurements while simultaneously treating alpha as a *trainable scalar*
of the module. The PDE residual u_t - alpha * u_xx ties the field and alpha
together, so minimising it alongside the data forces alpha toward the value that
makes the observed field a valid heat-equation solution. This is the kind of
parameter-estimation task classical forward solvers cannot do directly.

InversePINN is deliberately separate from ParametricPINN: its MLP takes only
[x, t] and it carries its own alpha leaf, so none of the forward training code is
touched.
"""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from analytical import u_exact

# Inputs here are only [x, t] (alpha is no longer an input but an unknown), so we
# normalise both to ~[-1, 1] before the MLP exactly as the forward model does.
INPUT_CENTER = (0.0, 0.5)
INPUT_SCALE = (1.0, 0.5)


class InversePINN(eqx.Module):
    """
    Physics-Informed Neural Network for the inverse heat-equation problem.

    Inputs (shape: (2,)):
        - x: Spatial coordinate in [-1, 1]
        - t: Time               in [0, 1]

    Outputs (shape: (1,)):
        - u: Temperature

    Unlike the forward model, alpha is not an input. It is a trainable scalar leaf
    (a JAX array, so it participates in autodiff) initialised to a deliberately
    wrong prior; training drives it toward the true diffusivity. The IC/BCs are
    enforced softly through the loss here (see compute_inverse_loss) rather than
    baked in by an ansatz, so the raw MLP output is the temperature.
    """
    mlp: eqx.nn.MLP
    alpha: jnp.ndarray
    input_center: tuple = eqx.field(static=True)
    input_scale: tuple = eqx.field(static=True)

    def __init__(self, key: jr.PRNGKey, alpha_init: float = 0.05,
                 width_size: int = 32, depth: int = 3):
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        self.alpha = jnp.asarray(alpha_init)
        self.input_center = INPUT_CENTER
        self.input_scale = INPUT_SCALE

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        center = jnp.asarray(self.input_center)
        scale = jnp.asarray(self.input_scale)
        x_norm = (x - center) / scale
        u = self.mlp(x_norm)[0]
        return jnp.reshape(u, (1,))


def generate_observations(key, alpha_true, n_obs=50, noise_sigma=0.01):
    """
    Synthesise sparse, noisy measurements of the true field.

    Samples (x, t) uniformly in the domain, evaluates the closed-form solution at
    alpha_true, and adds Gaussian noise. Returns (X_obs, u_obs) with shapes
    (n_obs, 2) and (n_obs, 1) -- the only data the inverse solver is allowed to see.
    """
    k_x, k_t, k_noise = jr.split(key, 3)
    x = jr.uniform(k_x, (n_obs, 1), minval=-1.0, maxval=1.0)
    t = jr.uniform(k_t, (n_obs, 1), minval=0.0, maxval=1.0)
    X_obs = jnp.hstack([x, t])
    u_clean = u_exact(x, t, alpha_true)
    u_obs = u_clean + noise_sigma * jr.normal(k_noise, (n_obs, 1))
    return X_obs, u_obs


def generate_inverse_data(key, num_collocation=1000, num_bc=100, num_ic=100):
    """
    Collocation, IC and BC points for the inverse problem.

    Same domains and IC/BC profile as the forward problem (IC u(x,0)=sin(pi x),
    zero Dirichlet BCs), but every point is just [x, t] -- alpha is not an input.
    """
    k_x, k_t = jr.split(key, 2)
    x_c = jr.uniform(k_x, (num_collocation, 1), minval=-1.0, maxval=1.0)
    t_c = jr.uniform(k_t, (num_collocation, 1), minval=0.0, maxval=1.0)
    collocation_points = jnp.hstack([x_c, t_c])

    x_ic = jr.uniform(jr.PRNGKey(14), (num_ic, 1), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros((num_ic, 1))
    X_ic = jnp.hstack([x_ic, t_ic])
    u_ic = jnp.sin(jnp.pi * x_ic)
    ic_points = (X_ic, u_ic)

    t_bc = jr.uniform(jr.PRNGKey(16), (num_bc, 1), minval=0.0, maxval=1.0)
    x_bc = jnp.where(jr.bernoulli(jr.PRNGKey(18), 0.5, (num_bc, 1)), 1.0, -1.0)
    X_bc = jnp.hstack([x_bc, t_bc])
    u_bc = jnp.zeros((num_bc, 1))
    bc_points = (X_bc, u_bc)

    return collocation_points, ic_points, bc_points


def inverse_residual(model, x, t):
    """
    PDE residual u_t - alpha * u_xx at a single (x, t), using the model's own
    trainable alpha. Mirrors physics.heat_equation_residual but for the 2-input
    model, so gradients of the loss flow into alpha through this term.
    """
    alpha = model.alpha

    def u_fn(x_val, t_val):
        inputs = jnp.stack([x_val, t_val])
        return model(inputs)[0]

    u_t_fn = jax.grad(u_fn, argnums=1)
    u_x_fn = jax.grad(u_fn, argnums=0)
    u_xx_fn = jax.grad(u_x_fn, argnums=0)

    u_t = u_t_fn(x, t)
    u_xx = u_xx_fn(x, t)
    return u_t - alpha * u_xx


def compute_inverse_loss(model, collocation_points, ic_points, bc_points,
                         obs_points, w_ic=10.0, w_bc=10.0, w_data=100.0):
    """
    Total inverse loss: PDE residual + soft IC + soft BC + data misfit.

    The data term is weighted heavily because it is what breaks the degeneracy:
    the PDE/IC/BC alone admit a family of (u, alpha) pairs, and only the
    observations pin the field -- and therefore alpha -- to the true solution.
    """
    x_c, t_c = collocation_points[:, 0], collocation_points[:, 1]
    vmap_residual = jax.vmap(inverse_residual, in_axes=(None, 0, 0))
    residuals = vmap_residual(model, x_c, t_c)
    loss_pde = jnp.mean(residuals ** 2)

    X_ic, u_ic_true = ic_points
    u_ic_pred = jax.vmap(model)(X_ic)
    loss_ic = jnp.mean((u_ic_pred - u_ic_true) ** 2)

    X_bc, u_bc_true = bc_points
    u_bc_pred = jax.vmap(model)(X_bc)
    loss_bc = jnp.mean((u_bc_pred - u_bc_true) ** 2)

    X_obs, u_obs = obs_points
    u_obs_pred = jax.vmap(model)(X_obs)
    loss_data = jnp.mean((u_obs_pred - u_obs) ** 2)

    return loss_pde + w_ic * loss_ic + w_bc * loss_bc + w_data * loss_data


@eqx.filter_jit
def inverse_train_step(model, opt_state, optimizer, collocation_points,
                       ic_points, bc_points, obs_points):
    """Single compiled optimisation step for the inverse problem."""
    loss_val, grads = eqx.filter_value_and_grad(compute_inverse_loss)(
        model, collocation_points, ic_points, bc_points, obs_points
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


def _split_keys(key):
    """Deterministic key split shared by train_inverse and run_inverse_demo so
    both reconstruct the *same* observations from one seed (needed for export)."""
    return jr.split(key, 3)  # -> (k_model, k_obs, k_data)


def train_inverse(alpha_true=0.042, key=None, epochs=2000, lr=1e-3,
                  alpha_init=0.05, n_obs=50, noise_sigma=0.01,
                  num_collocation=1000, decay_steps=None):
    """
    Fit InversePINN to noisy observations and recover alpha.

    Prints the alpha estimate and data-anchoring loss every 100 epochs. Returns
    (trained_model, history) where history is a list of
    {'epoch', 'loss', 'alpha_est', 'alpha_error'} dicts. Adam's per-parameter
    scaling lets the tiny-magnitude alpha move at roughly `lr` per step despite
    its small gradient, so a cosine-annealed run converges it cleanly.

    `decay_steps` sets the cosine annealing horizon and defaults to `epochs`.
    Early training briefly overshoots alpha while the field forms; annealing the
    LR to zero exactly at `epochs` can freeze a short run mid-transit, so a short
    run can pass a larger `decay_steps` to keep the LR alive long enough to land.
    """
    if key is None:
        key = jr.PRNGKey(0)
    if decay_steps is None:
        decay_steps = epochs
    k_model, k_obs, k_data = _split_keys(key)

    model = InversePINN(k_model, alpha_init=alpha_init)
    obs_points = generate_observations(k_obs, alpha_true, n_obs=n_obs,
                                       noise_sigma=noise_sigma)
    collocation_points, ic_points, bc_points = generate_inverse_data(
        k_data, num_collocation=num_collocation
    )

    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=decay_steps)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    history = []
    for epoch in range(epochs):
        model, opt_state, loss = inverse_train_step(
            model, opt_state, optimizer, collocation_points,
            ic_points, bc_points, obs_points
        )

        if epoch % 100 == 0 or epoch == epochs - 1:
            alpha_est = float(model.alpha)
            alpha_error = abs(alpha_est - alpha_true)
            history.append({
                "epoch": epoch,
                "loss": float(loss),
                "alpha_est": alpha_est,
                "alpha_error": alpha_error,
            })
            print(f"Epoch {epoch:04d} | Loss: {loss:.6f} "
                  f"| alpha_est: {alpha_est:.5f} | alpha_err: {alpha_error:.5f}")

    return model, history


def export_inverse_to_json(model, obs_points, alpha_true,
                           filepath="inverse_model.json"):
    """
    Export the inverse result for the frontend: the true and recovered alpha plus
    the noisy (x, t, u) observations, so the UI can show the readouts and scatter
    the measurement points over the heatmap without retraining.
    """
    X_obs, u_obs = obs_points
    X_obs = np.asarray(X_obs, dtype=np.float64)
    u_obs = np.asarray(u_obs, dtype=np.float64).reshape(-1)
    observations = [
        {"x": float(X_obs[i, 0]), "t": float(X_obs[i, 1]), "u": float(u_obs[i])}
        for i in range(X_obs.shape[0])
    ]

    payload = {
        "format": "inverse-heat-v1",
        "alpha_true": float(alpha_true),
        "alpha_est": float(model.alpha),
        "observations": observations,
    }

    print(f"Exporting inverse result to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print("Export complete.")


def run_inverse_demo(alpha_true=0.042, epochs=2000, seed=0, export_path=None):
    """
    End-to-end inverse demo: train from a wrong prior, print a final report, and
    optionally export the result for the frontend. Reconstructs the observations
    from the same seed used inside train_inverse so the exported scatter matches
    exactly what the network was trained on.
    """
    print("\n--- Inverse Problem: recovering alpha from sparse, noisy data ---")
    key = jr.PRNGKey(seed)
    model, history = train_inverse(alpha_true=alpha_true, key=key, epochs=epochs)

    alpha_est = float(model.alpha)
    abs_err = abs(alpha_est - alpha_true)
    rel_err = abs_err / alpha_true

    print("\nInverse problem report:")
    print(f"    true alpha      : {alpha_true:.5f}")
    print(f"    estimated alpha : {alpha_est:.5f}")
    print(f"    absolute error  : {abs_err:.5f}")
    print(f"    relative error  : {rel_err * 100:.2f}%")

    if export_path is not None:
        _, k_obs, _ = _split_keys(key)
        obs_points = generate_observations(k_obs, alpha_true)
        export_inverse_to_json(model, obs_points, alpha_true, filepath=export_path)

    return model, history


if __name__ == "__main__":
    run_inverse_demo()
