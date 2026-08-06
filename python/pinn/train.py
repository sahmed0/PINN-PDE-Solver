import json

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from pinn.analytical import relative_l2_error
from pinn.json_forward import forward_from_payload
from pinn.physics import compute_loss, compute_loss_components

# Fixed [x, t, alpha] rows whose float64 reference outputs are embedded in every
# exported model as "test_vectors". They pin the TS forward pass and the scoring
# endpoint to this JSON contract (see json_forward and the parity tests). The rows
# cover t=0, x=+/-1, the alpha edges, the gate's OOD alphas, and irregular interior
# points; keep them stable so old exports stay comparable.
PARITY_INPUTS = [
    [-0.7, 0.0, 0.055], [0.3, 0.0, 0.01],
    [1.0, 0.5, 0.05], [-1.0, 0.25, 0.1],
    [0.5, 1.0, 0.01], [-0.5, 1.0, 0.1],
    [0.25, 0.5, 0.007], [0.25, 0.5, 0.12],
    [0.123, 0.456, 0.033], [-0.987, 0.001, 0.099],
    [0.001, 0.999, 0.0123], [0.777, 0.333, 0.071],
    [-0.333, 0.667, 0.047], [0.9, 0.9, 0.089],
    [-0.6, 0.1, 0.023], [0.05, 0.55, 0.055],
    [-0.25, 0.75, 0.06], [0.65, 0.2, 0.085],
    [-0.85, 0.85, 0.015], [0.45, 0.05, 0.095],
]


def generate_training_data(key, num_collocation=1000, num_bc=100, num_ic=100):
    """
    Generates synthetic training data using uniform random sampling.
    Domains: x in [-1, 1], t in [0, 1], alpha in [0.01, 0.1].
    """
    # Keep the collocation stream keyed exactly as before (split(key, 3)) so runs
    # made before the IC/BC seeding fix stay bit-for-bit reproducible. The IC/BC
    # keys are derived from a folded-in copy of the seed so --seed now controls
    # ALL sampling (previously they used hardcoded PRNGKeys).
    k1, k2, k3 = jr.split(key, 3)
    k_ic_x, k_ic_a, k_bc_t, k_bc_a, k_bc_side = jr.split(jr.fold_in(key, 1), 5)

    # 1. Collocation Points
    # Scale uniform [0, 1] to specific ranges
    x_c = jr.uniform(k1, (num_collocation, 1), minval=-1.0, maxval=1.0)
    t_c = jr.uniform(k2, (num_collocation, 1), minval=0.0, maxval=1.0)
    alpha_c = jr.uniform(k3, (num_collocation, 1), minval=0.01, maxval=0.1)
    collocation_points = jnp.hstack([x_c, t_c, alpha_c])

    # 2. Initial Condition Points (t = 0)
    # Using the same domains for x and alpha
    x_ic = jr.uniform(k_ic_x, (num_ic, 1), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros((num_ic, 1))
    alpha_ic = jr.uniform(k_ic_a, (num_ic, 1), minval=0.01, maxval=0.1)
    X_ic = jnp.hstack([x_ic, t_ic, alpha_ic])
    # Non-trivial initial profile u(x, 0) = sin(pi * x).
    # This vanishes at x = +/-1 so it is consistent with the zero boundary
    # conditions below, and it gives the heat equation something real to
    # diffuse. (A zero IC would make u == 0 the exact solution everywhere,
    # so the network would just learn a flat field.)
    u_ic = jnp.sin(jnp.pi * x_ic)
    ic_points = (X_ic, u_ic)
    
    # 3. Boundary Condition Points (x = -1 and x = 1)
    t_bc = jr.uniform(k_bc_t, (num_bc, 1), minval=0.0, maxval=1.0)
    alpha_bc = jr.uniform(k_bc_a, (num_bc, 1), minval=0.01, maxval=0.1)

    # Half points at x=-1, half at x=1
    x_bc = jnp.where(jr.bernoulli(k_bc_side, 0.5, (num_bc, 1)), 1.0, -1.0)
    X_bc = jnp.hstack([x_bc, t_bc, alpha_bc])
    u_bc = jnp.zeros((num_bc, 1)) # Assuming u(boundary, t) = 0
    bc_points = (X_bc, u_bc)
    
    return collocation_points, ic_points, bc_points

@eqx.filter_jit
def train_step(model, opt_state, optimizer, collocation_points):
    """Executes a single compiled optimization step."""
    loss_val, grads = eqx.filter_value_and_grad(compute_loss)(
        model, collocation_points
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
            model, opt_state, optimizer, collocation_points
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

    A "test_vectors" block is also embedded: the float64 reference outputs of a
    fixed set of inputs (PARITY_INPUTS), computed from these very weights. Both
    consumers are tested against it so any drift in a re-implementation is caught.
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

    # Golden parity vectors: float64 reference outputs computed from the payload
    # above, so any consumer of this JSON can be pinned to it.
    inputs = np.asarray(PARITY_INPUTS, dtype=np.float64)
    outputs = forward_from_payload(payload, inputs)
    payload["test_vectors"] = {
        "dtype": "float64",
        "note": "reference outputs from a float64 NumPy forward pass over this file's weights",
        "inputs": inputs.tolist(),
        "outputs": outputs.tolist(),
    }

    print(f"Exporting model weights to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print("Export complete.")