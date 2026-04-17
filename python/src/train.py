import json

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import optax
from model import ParametricPINN
from physics import compute_loss

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

def train(model, key, epochs=1000, lr=1e-3):
    """Main training loop using Optax."""
    # Initialize the Adam optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Generate static dataset (for dynamic PINNs, one might resample per epoch)
    collocation_points, ic_points, bc_points = generate_training_data(key)
    
    for epoch in range(epochs):
        model, opt_state, loss = train_step(
            model, opt_state, optimizer, collocation_points, ic_points, bc_points
        )
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:04d} | Loss: {loss:.6f}")
            
    return model

def export_to_json(model, filepath="pinn_model.json"):
    """
    Exports the trained Equinox MLP to a plain JSON file of weights/biases.

    The model is just a tanh-MLP (in=3, out=1), so the browser can run the
    forward pass directly in a few lines of TypeScript -- no ONNX runtime,
    TensorFlow or tf2onnx toolchain required. Each entry in "layers" is a
    Linear layer with `weight` of shape (out, in) and `bias` of shape (out,).
    `tanh` activation is applied after every layer except the last.
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
        "format": "tanh-mlp",
        "in_size": 3,
        "out_size": 1,
        "activation": "tanh",
        # Inputs are ordered [x, t, alpha]; output is [u].
        "input_names": ["x", "t", "alpha"],
        "output_names": ["u"],
        "layers": layers,
    }

    print(f"Exporting model weights to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print("Export complete.")