import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from pinn.model import ParametricPINN
from pinn.physics import compute_loss, compute_loss_components
from pinn.train import generate_training_data


def test_compute_loss_and_gradients():
    # 1. Initialize Model
    key = jr.PRNGKey(42)
    model = ParametricPINN(key)

    # 2. Create Dummy Data
    # Collocation points: 10 points of shape (3,) -> (x, t, alpha)
    collocation_points = jax.random.uniform(jr.PRNGKey(0), (10, 3))

    # 3. Define the value and gradient function
    # Equinox filters out non-differentiable parts of the model (like static configs)
    loss_and_grad_fn = eqx.filter_value_and_grad(compute_loss)

    # 4. Execute
    loss_val, grads = loss_and_grad_fn(model, collocation_points)

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


def test_ansatz_makes_ic_bc_exact():
    # The hard-constraint ansatz enforces the IC/BCs exactly, so the IC and BC
    # loss components must be ~0 for ANY model (even an untrained one). This turns
    # the dropped-loss-term rationale into an actual assertion.
    model = ParametricPINN(jr.PRNGKey(7))
    collocation_points, ic_points, bc_points = generate_training_data(
        jr.PRNGKey(0), num_collocation=50, num_bc=40, num_ic=40
    )
    components = compute_loss_components(model, collocation_points, ic_points, bc_points)
    assert components["loss_ic"] < 1e-10
    assert components["loss_bc"] < 1e-10