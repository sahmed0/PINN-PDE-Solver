import jax
import jax.numpy as jnp


def heat_equation_residual(model, x, t, alpha):
    """
    Computes the PDE residual for the 1D Heat Equation.
    """
    # 1. Define a scalar-to-scalar wrapper for jax.grad
    # jax.grad requires the function to output a single scalar.
    def u_fn(x_val, t_val, alpha_val):
        inputs = jnp.stack([x_val, t_val, alpha_val])
        return model(inputs)[0] 
    
    # 2. Compute first derivatives
    # argnums=1 is t, argnums=0 is x
    u_t_fn = jax.grad(u_fn, argnums=1)
    u_x_fn = jax.grad(u_fn, argnums=0)
    
    # 3. Compute second derivative with respect to x
    u_xx_fn = jax.grad(u_x_fn, argnums=0)
    
    # 4. Evaluate the derivatives at the given points
    u_t = u_t_fn(x, t, alpha)
    u_xx = u_xx_fn(x, t, alpha)
    
    # 5. Return the residual: u_t - alpha * u_xx
    return u_t - alpha * u_xx

def compute_loss(model, collocation_points):
    """
    Computes the training loss: the PDE residual MSE.

    The model enforces the initial and boundary conditions *exactly* via a
    hard-constraint ansatz (see model.ParametricPINN.__call__), so IC/BC terms
    would be structurally ~0 and contribute no gradient. Only the PDE residual
    actually trains the network, so that is all this loss computes.
    (compute_loss_components still reports the IC/BC terms for logging.)
    """
    # Unpack the collocation points (Shape: N, 3)
    x_c, t_c, alpha_c = collocation_points[:, 0], collocation_points[:, 1], collocation_points[:, 2]

    # We vmap over the spatial, temporal, and alpha arrays.
    # in_axes=(None, 0, 0, 0) means we don't map over the model, but we do map over x, t, and alpha.
    vmap_residual = jax.vmap(heat_equation_residual, in_axes=(None, 0, 0, 0))
    residuals = vmap_residual(model, x_c, t_c, alpha_c)
    return jnp.mean(residuals ** 2)


def compute_loss_components(model, collocation_points, ic_points, bc_points,
                            w_ic=10.0, w_bc=10.0):
    """Return the individual loss terms as a dict of Python floats.

    Same arithmetic as `compute_loss`, but exposes the breakdown for logging
    (PDE residual, IC, BC, and the weighted total). This is a read-only helper
    for instrumentation -- the gradient step still uses `compute_loss` so the
    training dynamics are unchanged.
    """
    x_c, t_c, alpha_c = (collocation_points[:, 0],
                         collocation_points[:, 1],
                         collocation_points[:, 2])
    vmap_residual = jax.vmap(heat_equation_residual, in_axes=(None, 0, 0, 0))
    residuals = vmap_residual(model, x_c, t_c, alpha_c)
    loss_pde = jnp.mean(residuals ** 2)

    X_ic, u_ic_true = ic_points
    loss_ic = jnp.mean((jax.vmap(model)(X_ic) - u_ic_true) ** 2)

    X_bc, u_bc_true = bc_points
    loss_bc = jnp.mean((jax.vmap(model)(X_bc) - u_bc_true) ** 2)

    total = loss_pde + w_ic * loss_ic + w_bc * loss_bc
    return {
        "total_loss": float(total),
        "loss_pde": float(loss_pde),
        "loss_ic": float(loss_ic),
        "loss_bc": float(loss_bc),
    }