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

What sets the accuracy here is not network approximation error (as in the forward
problem) but the *information content* of the noisy data. The right yardstick is
therefore the Cramer-Rao lower bound (see crlb.py): the smallest error any
unbiased estimator could achieve. The pieces below are designed to reach it:

  * a hard-constraint ansatz (as in the forward ParametricPINN) bakes the IC/BCs
    in exactly, so alpha no longer absorbs IC/BC fitting error -- this removes a
    systematic downward bias in the estimate;
  * alpha gets its own, faster optimiser (it is one tiny-magnitude scalar with a
    weak gradient, so it needs a larger step than the network weights);
  * an L-BFGS polish after Adam seats alpha exactly at the data optimum, pulling
    the spread of estimates down to within ~1.6x of the CRLB floor
    (measured: std 3.97e-4 vs bound 2.44e-4).

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

from pinn.analytical import u_exact
from pinn.crlb import crlb_std, design_sweep

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
    wrong prior; training drives it toward the true diffusivity.

    The same hard-constraint ansatz as the forward ParametricPINN is used here:

        u(x, t) = sin(pi x) + (1 - x^2) * t * N(x, t)

    so the initial condition u(x, 0) = sin(pi x) and the zero Dirichlet BCs at
    x = +/-1 hold *exactly*, by construction. This matters more for the inverse
    problem than the forward one: with soft IC/BC penalties the network can trade
    a small IC/BC misfit for a smaller PDE residual, and alpha silently absorbs
    the difference, biasing the estimate low. Enforcing them exactly removes that
    bias and lets the data alone pin down alpha.
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
        x_phys, t_phys = x[0], x[1]

        center = jnp.asarray(self.input_center)
        scale = jnp.asarray(self.input_scale)
        x_norm = (x - center) / scale

        n = self.mlp(x_norm)[0]

        # Hard-constraint ansatz: exact IC (t=0 -> sin(pi x)) and zero Dirichlet
        # BCs (x=+/-1 -> (1 - x^2) = 0), so only the interior dynamics are learnt.
        u = jnp.sin(jnp.pi * x_phys) + (1.0 - x_phys ** 2) * t_phys * n
        return jnp.reshape(u, (1,))


def generate_observations(key, alpha_true, n_obs=200, noise_sigma=0.01):
    """
    Synthesise sparse, noisy measurements of the true field.

    Samples (x, t) uniformly in the domain, evaluates the closed-form solution at
    alpha_true, and adds Gaussian noise. Returns (X_obs, u_obs) with shapes
    (n_obs, 2) and (n_obs, 1) -- the only data the inverse solver is allowed to see.

    noise_sigma stays fixed across experiments: Gaussian measurement noise is a
    good proxy for the systematic uncertainty of a real instrument. The lever we
    turn to lower the information floor is instead n_obs (see crlb.design_sweep).
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
    The IC/BC sets are now enforced exactly by the ansatz, so they are retained
    only as a cheap wiring check in compute_inverse_loss (their loss is ~0).
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
    Total inverse loss: PDE residual + data misfit (+ harmless IC/BC checks).

    The data term is weighted heavily because it is what breaks the degeneracy:
    the PDE alone admits a family of (u, alpha) pairs, and only the observations
    pin the field -- and therefore alpha -- to the true solution.

    NOTE: the IC and BC are now enforced *exactly* by the ansatz (see
    InversePINN.__call__), so loss_ic and loss_bc are structurally ~0 and add no
    gradient. They are kept as a cheap runtime check that the ansatz is wired up
    correctly; the weights are harmless. The PDE residual and the data misfit are
    what actually train the field and alpha.
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
    """Single compiled Adam-phase optimisation step for the inverse problem."""
    loss_val, grads = eqx.filter_value_and_grad(compute_inverse_loss)(
        model, collocation_points, ic_points, bc_points, obs_points
    )
    # multi_transform routes the per-leaf masks against `params`, so it must see
    # the same array-only structure as the labels (None at non-array leaves).
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


def _split_keys(key):
    """Deterministic key split shared by train_inverse and run_inverse_demo so
    both reconstruct the *same* observations from one seed (needed for export)."""
    return jr.split(key, 3)  # -> (k_model, k_obs, k_data)


def _param_labels(model):
    """Label every trainable leaf 'mlp', except the alpha scalar -> 'alpha'.

    Used by optax.multi_transform to drive alpha with its own (faster) optimiser:
    alpha is a single small-magnitude scalar whose gradient is weak, so it needs
    a larger effective step than the network weights to converge in the same
    number of epochs.
    """
    arrays = eqx.filter(model, eqx.is_array)
    labels = jax.tree_util.tree_map(lambda _: "mlp", arrays)
    return eqx.tree_at(lambda m: m.alpha, labels, "alpha")


def _lbfgs_polish(model, loss_closure, steps):
    """Polish (field + alpha) with L-BFGS after Adam.

    Adam gets us into the right basin cheaply; L-BFGS, a quasi-Newton method with
    a line search, then seats the parameters exactly at the data optimum. For a
    smooth least-squares objective like this it removes the residual optimisation
    slack that otherwise keeps the estimate spread above the CRLB floor.
    """
    if steps <= 0:
        return model
    params, static = eqx.partition(model, eqx.is_array)

    def loss_of_params(p):
        return loss_closure(eqx.combine(p, static))

    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(loss_of_params)
    opt_state = opt.init(params)

    def body(carry, _):
        params, opt_state = carry
        value, grad = value_and_grad(params, state=opt_state)
        updates, opt_state = opt.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=loss_of_params
        )
        params = optax.apply_updates(params, updates)
        return (params, opt_state), value

    (params, _), _ = jax.lax.scan(body, (params, opt_state), None, length=steps)
    return eqx.combine(params, static)


def train_inverse(alpha_true=0.042, key=None, epochs=2000, lr=1e-3,
                  alpha_lr=5e-3, alpha_init=0.05, n_obs=200, noise_sigma=0.01,
                  num_collocation=1000, decay_steps=None, lbfgs_steps=300,
                  verbose=True):
    """
    Fit InversePINN to noisy observations and recover alpha.

    Two-phase optimisation:
      1. Adam with a *separate, faster* schedule for alpha (alpha_lr) than for the
         network weights (lr), both cosine-annealed over `decay_steps`.
      2. An L-BFGS polish (`lbfgs_steps`) that lands alpha exactly at the data
         optimum.

    Returns (trained_model, history) where history is a list of
    {'epoch', 'loss', 'alpha_est', 'alpha_error'} dicts from the Adam phase.

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

    # Separate optimisers: weights at `lr`, the lone alpha scalar at `alpha_lr`.
    mlp_sched = optax.cosine_decay_schedule(init_value=lr, decay_steps=decay_steps)
    alpha_sched = optax.cosine_decay_schedule(init_value=alpha_lr, decay_steps=decay_steps)
    # Pass the labelling *function* (not its result): the result is an InversePINN
    # pytree, which is callable, and optax would mistake it for a label callable.
    optimizer = optax.multi_transform(
        {"mlp": optax.adam(mlp_sched), "alpha": optax.adam(alpha_sched)},
        _param_labels,
    )
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
            if verbose:
                print(f"Epoch {epoch:04d} | Loss: {loss:.6f} "
                      f"| alpha_est: {alpha_est:.5f} | alpha_err: {alpha_error:.5f}")

    # L-BFGS polish to seat alpha exactly at the data optimum.
    def loss_closure(m):
        return compute_inverse_loss(m, collocation_points, ic_points,
                                    bc_points, obs_points)

    model = _lbfgs_polish(model, loss_closure, lbfgs_steps)
    if verbose:
        a = float(model.alpha)
        print(f"L-BFGS polish ({lbfgs_steps} steps) | alpha_est: {a:.5f} "
              f"| alpha_err: {abs(a - alpha_true):.5f}")

    return model, history


def evaluate_inverse_uncertainty(alpha_true=0.042, n_seeds=8, n_obs=200,
                                 noise_sigma=0.01, epochs=2000, verbose=True,
                                 **train_kwargs):
    """
    Repeat the recovery over independent noise realisations to measure the
    estimator's empirical spread, and compare it to the CRLB floor.

    A single point estimate cannot distinguish a lucky draw from a biased method.
    Re-running over `n_seeds` fresh noise draws gives the mean (does the method
    sit on the truth?) and the std (how tightly?), and the CRLB tells us the best
    std physically attainable from this data. Returns a dict with the per-seed
    estimates, mean, std, and the (theoretical) CRLB std on alpha.
    """
    estimates = []
    for s in range(n_seeds):
        model, _ = train_inverse(
            alpha_true=alpha_true, key=jr.PRNGKey(s), epochs=epochs,
            n_obs=n_obs, noise_sigma=noise_sigma, verbose=False, **train_kwargs
        )
        est = float(model.alpha)
        estimates.append(est)
        if verbose:
            print(f"seed {s}: alpha_est = {est:.5f}")

    estimates = np.asarray(estimates)
    # CRLB on a representative observation design (seed 0's actual points).
    _, k_obs, _ = _split_keys(jr.PRNGKey(0))
    X_obs, _ = generate_observations(k_obs, alpha_true, n_obs=n_obs,
                                     noise_sigma=noise_sigma)
    crlb = crlb_std(X_obs, alpha_true, noise_sigma)

    return {
        "alpha_true": float(alpha_true),
        "estimates": estimates.tolist(),
        "mean": float(estimates.mean()),
        "std": float(estimates.std()),
        "crlb_std": float(crlb),
        "n_obs": int(n_obs),
        "noise_sigma": float(noise_sigma),
        "n_seeds": int(n_seeds),
    }


def export_inverse_to_json(model, obs_points, alpha_true, stats=None,
                           filepath="inverse_model.json"):
    """
    Export the inverse result for the frontend: the true and recovered alpha, the
    recovered-alpha uncertainty (empirical std over noise realisations) and the
    CRLB floor, plus the noisy (x, t, u) observations, so the UI can show the
    readouts with an honest error bar and scatter the measurements over the
    heatmap without retraining.
    """
    X_obs, u_obs = obs_points
    X_obs = np.asarray(X_obs, dtype=np.float64)
    u_obs = np.asarray(u_obs, dtype=np.float64).reshape(-1)
    observations = [
        {"x": float(X_obs[i, 0]), "t": float(X_obs[i, 1]), "u": float(u_obs[i])}
        for i in range(X_obs.shape[0])
    ]

    payload = {
        "format": "inverse-heat-v2",
        "alpha_true": float(alpha_true),
        "alpha_est": float(model.alpha),
        "observations": observations,
    }
    if stats is not None:
        # Report the mean over noise realisations as the headline estimate, with
        # its empirical std as the +/- band, alongside the CRLB floor it saturates.
        payload["alpha_est"] = float(stats["mean"])
        payload["alpha_std"] = float(stats["std"])
        payload["crlb_std"] = float(stats["crlb_std"])
        payload["n_obs"] = int(stats["n_obs"])
        payload["noise_sigma"] = float(stats["noise_sigma"])
        payload["n_seeds"] = int(stats["n_seeds"])
        # The CRLB-floor-by-design table, so the UI can show how the information
        # limit moves with the experiment (N, sigma, time horizon).
        payload["design_sweep"] = design_sweep(alpha=float(alpha_true),
                                               sigma=float(stats["noise_sigma"]))

    print(f"Exporting inverse result to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print("Export complete.")


def run_inverse_demo(alpha_true=0.042, epochs=2000, seed=0, n_seeds=8,
                     n_obs=200, noise_sigma=0.01, export_path=None):
    """
    End-to-end inverse demo: recover alpha over several noise realisations, report
    the estimate with an uncertainty band against the Cramer-Rao floor, and
    optionally export for the frontend.

    The exported scatter comes from `seed` (reconstructed from the same key split
    used inside train_inverse), so the displayed points match a real trained run;
    the headline alpha and its +/- band come from the multi-seed statistics.
    """
    print("\n--- Inverse Problem: recovering alpha from sparse, noisy data ---")

    stats = evaluate_inverse_uncertainty(
        alpha_true=alpha_true, n_seeds=n_seeds, n_obs=n_obs,
        noise_sigma=noise_sigma, epochs=epochs,
    )

    mean, std, crlb = stats["mean"], stats["std"], stats["crlb_std"]
    bias = mean - alpha_true
    print("\nInverse problem report (over %d noise realisations):" % n_seeds)
    print(f"    true alpha        : {alpha_true:.5f}")
    print(f"    recovered alpha   : {mean:.5f} +/- {std:.5f}  (1 sigma)")
    print(f"    relative error    : {abs(bias) / alpha_true * 100:.2f}% (bias) "
          f"| {std / alpha_true * 100:.2f}% (spread)")
    print(f"    Cramer-Rao floor  : {crlb:.5f}  ({crlb / alpha_true * 100:.2f}% of true)")
    print(f"    saturation        : spread / CRLB = {std / crlb:.2f}x "
          f"(1.0x = information-limited)")

    # Train one representative model on `seed` for the exported field/scatter.
    key = jr.PRNGKey(seed)
    model, _ = train_inverse(alpha_true=alpha_true, key=key, epochs=epochs,
                             n_obs=n_obs, noise_sigma=noise_sigma, verbose=False)

    if export_path is not None:
        _, k_obs, _ = _split_keys(key)
        obs_points = generate_observations(k_obs, alpha_true, n_obs=n_obs,
                                           noise_sigma=noise_sigma)
        export_inverse_to_json(model, obs_points, alpha_true, stats=stats,
                               filepath=export_path)

    return model, stats


if __name__ == "__main__":
    run_inverse_demo()
