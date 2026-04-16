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

def compute_loss(model, collocation_points, ic_points, bc_points,
                 w_ic=10.0, w_bc=10.0):
    """
    Computes the total loss: PDE residual MSE + IC MSE + BC MSE.

    ic_points and bc_points are expected to be tuples of (inputs, true_values).

    The IC and BC terms are weighted (default 10x) because for a PINN the data
    constraints have to dominate early training -- otherwise the optimiser
    happily drives the PDE residual to zero with a trivial / unconstrained
    solution that ignores the initial and boundary conditions.
    """
    # --- 1. Physics Loss (PDE Residual) ---
    # Unpack the collocation points (Shape: N, 3)
    x_c, t_c, alpha_c = collocation_points[:, 0], collocation_points[:, 1], collocation_points[:, 2]
    
    # We vmap over the spatial, temporal, and alpha arrays. 
    # in_axes=(None, 0, 0, 0) means we don't map over the model, but we do map over x, t, and alpha.
    vmap_residual = jax.vmap(heat_equation_residual, in_axes=(None, 0, 0, 0))
    residuals = vmap_residual(model, x_c, t_c, alpha_c)
    loss_pde = jnp.mean(residuals ** 2)
    
    # --- 2. Initial Condition (IC) Loss ---
    X_ic, u_ic_true = ic_points
    # vmap the model to handle batched inputs for the IC
    u_ic_pred = jax.vmap(model)(X_ic)
    loss_ic = jnp.mean((u_ic_pred - u_ic_true) ** 2)
    
    # --- 3. Boundary Condition (BC) Loss ---
    X_bc, u_bc_true = bc_points
    u_bc_pred = jax.vmap(model)(X_bc)
    loss_bc = jnp.mean((u_bc_pred - u_bc_true) ** 2)
    
    # --- Total Loss ---
    return loss_pde + w_ic * loss_ic + w_bc * loss_bc