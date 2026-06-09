import json

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import optax
from model import ParametricPINN
from physics import compute_loss, compute_loss_components
from analytical import relative_l2_error

def generate_training_data(key, num_collocation=1000, num_bc=100, num_ic=100):
    """
    Generates synthetic training data using uniform random sampling.
    Domains: x in [-1, 1], t in [0, 1], alpha in [0.01, 0.1].
    """
    k1, k2, k3 = jr.split(key, 3)
    
    # 1. Collocation Points
    # Scale uniform [0, 1] to specific ranges
    x_c = jr.uniform(k1, (num_collocation, 1), minval=-1.0, maxval=1.0)
    t_c = jr.uniform(k2, (num_collocation, 1), minval=0.0, maxval=1.0)
    alpha_c = jr.uniform(k3, (num_collocation, 1), minval=0.01, maxval=0.1)
    collocation_points = jnp.hstack([x_c, t_c, alpha_c])
    
    # 2. Initial Condition Points (t = 0)
    # Using the same domains for x and alpha
    x_ic = jr.uniform(jr.PRNGKey(4), (num_ic, 1), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros((num_ic, 1))
    alpha_ic = jr.uniform(jr.PRNGKey(5), (num_ic, 1), minval=0.01, maxval=0.1)
    X_ic = jnp.hstack([x_ic, t_ic, alpha_ic])
    # Non-trivial initial profile u(x, 0) = sin(pi * x).
    # This vanishes at x = +/-1 so it is consistent with the zero boundary
    # conditions below, and it gives the heat equation something real to
    # diffuse. (A zero IC would make u == 0 the exact solution everywhere,
    # so the network would just learn a flat field.)
    u_ic = jnp.sin(jnp.pi * x_ic)
    ic_points = (X_ic, u_ic)
    
    # 3. Boundary Condition Points (x = -1 and x = 1)
    t_bc = jr.uniform(jr.PRNGKey(6), (num_bc, 1), minval=0.0, maxval=1.0)
    alpha_bc = jr.uniform(jr.PRNGKey(7), (num_bc, 1), minval=0.01, maxval=0.1)
    
    # Half points at x=-1, half at x=1
    x_bc = jnp.where(jr.bernoulli(jr.PRNGKey(8), 0.5, (num_bc, 1)), 1.0, -1.0)
    X_bc = jnp.hstack([x_bc, t_bc, alpha_bc])
    u_bc = jnp.zeros((num_bc, 1)) # Assuming u(boundary, t) = 0
    bc_points = (X_bc, u_bc)
    
    return collocation_points, ic_points, bc_points

@eqx.filter_jit
def train_step(model, opt_state, optimizer, collocation_points, ic_points, bc_points):
    """Executes a single compiled optimization step."""
    loss_val, grads = eqx.filter_value_and_grad(compute_loss)(
        model, collocation_points, ic_points, bc_points
    )
    # Calculate updates and apply them
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

def train(model, key, epochs=20000, lr=1e-3, validate=True,
          val_alphas=(0.01, 0.05, 0.1), num_collocation=4000,
          log_callback=None):
    """Main training loop using Optax.

    When `validate` is set, the printed log also reports the mean relative L2
    error against the analytical solution (averaged over `val_alphas`). The
    training loss alone is not a meaningful measure of correctness for a PINN --
    the residual can be small while the field is wrong -- so we track the error
    versus ground truth as the real progress signal.

    The learning rate follows a cosine decay from `lr` to ~0 over `epochs`. A
    high constant LR plateaus early on the residual; annealing it lets Adam keep
    sharpening the fit in late training, which is where most of the accuracy on
    a smooth problem like this comes from.

    `log_callback` is an optional, backward-compatible instrumentation hook. When
    provided, it is called as `log_callback(epoch, metrics_dict)` at the same
    `epoch % 100` cadence as the printed log, where `metrics_dict` carries
    `total_loss`, `loss_pde`, `loss_ic`, `loss_bc`, and `mean_rel_l2`. It does not
    touch the gradient step; existing callers that pass nothing behave identically.
    """
    # Cosine-annealed Adam: start at `lr`, decay smoothly toward 0 by the last
    # epoch so late steps fine-tune rather than bounce around the minimum.
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=epochs)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Generate static dataset (for dynamic PINNs, one might resample per epoch).
    # More collocation points give denser coverage of the 3D (x, t, alpha) domain,
    # which the residual needs to constrain the solution across the alpha range.
    collocation_points, ic_points, bc_points = generate_training_data(
        key, num_collocation=num_collocation
    )

    def mean_rel_l2(m):
        errs = [relative_l2_error(m, a) for a in val_alphas]
        return float(sum(errs) / len(errs))

    for epoch in range(epochs):
        model, opt_state, loss = train_step(
            model, opt_state, optimizer, collocation_points, ic_points, bc_points
        )

        if epoch % 100 == 0 or epoch == epochs - 1:
            mrl2 = mean_rel_l2(model) if (validate or log_callback is not None) else None
            if validate:
                print(f"Epoch {epoch:04d} | Loss: {loss:.6f} "
                      f"| mean rel L2: {mrl2:.3e}")
            else:
                print(f"Epoch {epoch:04d} | Loss: {loss:.6f}")

            if log_callback is not None:
                metrics = compute_loss_components(
                    model, collocation_points, ic_points, bc_points
                )
                metrics["mean_rel_l2"] = mrl2
                log_callback(epoch, metrics)

    return model

def export_to_json(model, filepath="pinn_model.json"):
    """
    Exports the trained Equinox MLP to a plain JSON file of weights/biases.

    The core is a tanh-MLP (in=3, out=1), so the browser can run the forward pass
    directly in a few lines of TypeScript -- no ONNX runtime, TensorFlow or
    tf2onnx toolchain required. Each entry in "layers" is a Linear layer with
    `weight` of shape (out, in) and `bias` of shape (out,). `tanh` activation is
    applied after every layer except the last.

    Two wrappers around the MLP must be replicated by the consumer (see
    frontend/src/lib/inference.ts), so they are exported as metadata:

    - "input_center"/"input_scale": normalise raw [x, t, alpha] via
      (raw - center) / scale before the MLP.
    - "ansatz" = "heat_dirichlet_sin": reconstruct the temperature from the MLP
      output N as  u = sin(pi x) + (1 - x^2) * t * N,  which makes the IC/BCs
      exact. The format string is bumped accordingly so stale consumers fail loudly.
    """
    layers = []
    for layer in model.mlp.layers:
        # eqx.nn.Linear stores weight as (out_features, in_features) and an
        # optional bias of shape (out_features,).
        weight = np.asarray(layer.weight, dtype=np.float64)
        bias = layer.bias
        bias = (np.zeros(weight.shape[0]) if bias is None
                else np.asarray(bias, dtype=np.float64))
        layers.append({
            "weight": weight.tolist(),
            "bias": bias.tolist(),
        })

    payload = {
        "format": "tanh-mlp-heat-v2",
        "in_size": 3,
        "out_size": 1,
        "activation": "tanh",
        # Inputs are ordered [x, t, alpha]; output is [u].
        "input_names": ["x", "t", "alpha"],
        "output_names": ["u"],
        # Input normalisation: norm = (raw - center) / scale, applied before the MLP.
        "input_center": list(model.input_center),
        "input_scale": list(model.input_scale),
        # Hard-constraint reconstruction: u = sin(pi x) + (1 - x^2) * t * N.
        "ansatz": "heat_dirichlet_sin",
        "layers": layers,
    }

    print(f"Exporting model weights to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print("Export complete.")