import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from pinn.model import ParametricPINN
from pinn.physics import compute_loss, compute_loss_components, heat_equation_residual
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


def test_residual_of_exact_solution_is_zero():
    """The autodiff PDE operator annihilates the closed-form solution.

    This checks the physics loss independently of any trained model: if the grad
    wiring in heat_equation_residual were wrong (wrong argnums, a missing second
    derivative, a sign slip), u = sin(pi x) exp(-alpha pi^2 t) would leave a nonzero
    residual. Everything the PINN learns is measured against this operator, so this
    is the test that makes the rest of the training loss trustworthy.
    """

    def exact(inp):
        x, t, a = inp[0], inp[1], inp[2]
        return jnp.array([jnp.sin(jnp.pi * x) * jnp.exp(-a * jnp.pi**2 * t)])

    xs = jnp.linspace(-1.0, 1.0, 20)
    ts = jnp.linspace(0.0, 1.0, 20)
    alphas = jnp.array([0.01, 0.05, 0.1])
    X, T, A = (g.ravel() for g in jnp.meshgrid(xs, ts, alphas, indexing="ij"))

    residuals = jax.vmap(lambda x, t, a: heat_equation_residual(exact, x, t, a))(X, T, A)
    # JAX runs in float32 and this nests two grads, so the bound is float32 noise on a
    # second derivative, not a physics tolerance.
    assert jnp.max(jnp.abs(residuals)) < 1e-4
