import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

# Physical input domains used during training:
#   x     in [-1, 1]
#   t     in [0,  1]
#   alpha in [0.01, 0.1]
#
# Raw alpha is a narrow band of small numbers, which a tanh-MLP resolves poorly:
# the input has tiny variance so the gradient signal w.r.t. alpha is weak and the
# network collapses toward an alpha-averaged solution (accurate in the middle of
# the range, wrong at the edges). We therefore normalise every input to roughly
# [-1, 1] before the network sees it, via  norm = (raw - center) / scale.
INPUT_CENTER = (0.0, 0.5, 0.055)
INPUT_SCALE = (1.0, 0.5, 0.045)


class ParametricPINN(eqx.Module):
    """
    A Parametric Physics-Informed Neural Network for the 1D heat equation.

    Inputs (shape: (3,)):
        - x: Spatial coordinate          in [-1, 1]
        - t: Time                        in [0, 1]
        - alpha: Thermal diffusivity     in [0.01, 0.1]

    Outputs (shape: (1,)):
        - u: Temperature

    The raw inputs are normalised to ~[-1, 1] before the MLP, and a hard-constraint
    ansatz bakes the initial and boundary conditions into the output exactly so the
    network only has to learn the interior dynamics (see __call__).
    """
    mlp: eqx.nn.MLP
    input_center: tuple = eqx.field(static=True)
    input_scale: tuple = eqx.field(static=True)

    def __init__(self, key: jr.PRNGKey, width_size: int = 32, depth: int = 3):
        self.mlp = eqx.nn.MLP(
            in_size=3,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        self.input_center = INPUT_CENTER
        self.input_scale = INPUT_SCALE

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the PINN.

        Two transforms wrap the raw MLP:

        1. Input normalisation: raw [x, t, alpha] -> ~[-1, 1] before the network,
           which is essential for the network to resolve the small-magnitude alpha.

        2. Hard-constraint ansatz, which makes the IC and BCs exact by construction:

               u(x, t) = sin(pi x) + (1 - x^2) * t * N(x, t, alpha)

           - At t = 0:      u = sin(pi x)                 -> initial condition u(x,0)
           - At x = +/- 1:  (1 - x^2) = 0 and sin(pi x)=0 -> zero Dirichlet BCs

           The network N only has to learn the interior dynamics through the PDE
           residual. This is far easier (and far more accurate) than enforcing the
           IC/BC softly through weighted loss terms, where the optimiser trades
           them off against the residual.
        """
        x_phys, t_phys = x[0], x[1]

        center = jnp.asarray(self.input_center)
        scale = jnp.asarray(self.input_scale)
        x_norm = (x - center) / scale

        n = self.mlp(x_norm)[0]

        u = jnp.sin(jnp.pi * x_phys) + (1.0 - x_phys ** 2) * t_phys * n
        return jnp.reshape(u, (1,))
