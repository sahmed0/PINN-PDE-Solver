import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from model import ParametricPINN
from physics import compute_loss

def test_compute_loss_and_gradients():
    # 1. Initialize Model
    key = jr.PRNGKey(42)
    model = ParametricPINN(key)
    
    # 2. Create Dummy Data
    # Collocation points: 10 points of shape (3,) -> (x, t, alpha)
    collocation_points = jax.random.uniform(jr.PRNGKey(0), (10, 3))
    
    # IC points: 5 points, targets are 0.0 for testing
    X_ic = jax.random.uniform(jr.PRNGKey(1), (5, 3))
    u_ic = jnp.zeros((5, 1))
    ic_points = (X_ic, u_ic)
    
    # BC points: 5 points, targets are 0.0 for testing
    X_bc = jax.random.uniform(jr.PRNGKey(2), (5, 3))
    u_bc = jnp.zeros((5, 1))
    bc_points = (X_bc, u_bc)
    
    # 3. Define the value and gradient function
    # Equinox filters out non-differentiable parts of the model (like static configs)
    loss_and_grad_fn = eqx.filter_value_and_grad(compute_loss)
    
    # 4. Execute
    loss_val, grads = loss_and_grad_fn(model, collocation_points, ic_points, bc_points)
    
    # 5. Assertions
    # Check that loss is a valid scalar
    assert loss_val.ndim == 0, f"Loss should be a scalar, got shape {loss_val.shape}"
    assert not jnp.isnan(loss_val), "Loss returned NaN."
    
    # Verify gradients flowed correctly
    # 'grads' is a PyTree matching the model architecture. We iterate through its leaves.
    gradients_computed = False
    for leaf in jax.tree_util.tree_leaves(grads):
        if leaf is not None:
            gradients_computed = True
            assert not jnp.isnan(leaf).any(), "Gradients contain NaN values."
            
    assert gradients_computed, "No gradients were computed for the model parameters."