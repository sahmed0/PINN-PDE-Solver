import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from pinn.forward import heat_ansatz, normalised_mlp

# Physical input domains used during training:
#   x     in [-1, 1]
#   t     in [0,  1]
#   alpha in [0.01, 0.1]
#
# alpha spans a band two orders of magnitude narrower than x and t, so un-normalised
# it enters the first layer at a far smaller scale than the other inputs. Every input
# is therefore normalised to roughly [-1, 1] via  norm = (raw - center) / scale.
# Measured (scripts/ablate_normalisation.py, identical seed, init and collocation
# points, 2000 epochs): mean rel-L2 6.83e-3 normalised vs 1.45e-2 raw (2.1x), with the
# raw model 1.3-3.7x worse at every alpha tested. The raw model does NOT collapse toward
# an alpha-averaged solution: it reproduces 96.4% of the analytic alpha-sensitivity
# spread against 94.7% for the normalised model. Normalisation buys accuracy, not the
# ability to resolve alpha.
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

    `input_center` / `input_scale` are exposed so the normalisation ablation
    (`scripts/ablate_normalisation.py`) can build an otherwise-identical raw-input model.
    """

    mlp: eqx.nn.MLP
    input_center: tuple = eqx.field(static=True)
    input_scale: tuple = eqx.field(static=True)

    def __init__(
        self,
        key: jr.PRNGKey,
        width_size: int = 32,
        depth: int = 3,
        input_center=INPUT_CENTER,
        input_scale=INPUT_SCALE,
    ):
        self.mlp = eqx.nn.MLP(
            in_size=3,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        self.input_center = tuple(input_center)
        self.input_scale = tuple(input_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the PINN.

        Two transforms wrap the raw MLP:

        1. Input normalisation: raw [x, t, alpha] -> ~[-1, 1] before the network,
           which roughly halves the error at a fixed training budget (see above).

        2. Hard-constraint ansatz, which makes the IC and BCs exact by construction:

               u(x, t) = sin(pi x) + (1 - x^2) * t * N(x, t, alpha)

           - At t = 0:      u = sin(pi x)                 -> initial condition u(x,0)
           - At x = +/- 1:  (1 - x^2) = 0 and sin(pi x)=0 -> zero Dirichlet BCs

           The network N only has to learn the interior dynamics through the PDE
           residual. This is far easier (and far more accurate) than enforcing the
           IC/BC softly through weighted loss terms, where the optimiser trades
           them off against the residual.
        """
        n = normalised_mlp(self.mlp, self.input_center, self.input_scale, x)
        u = heat_ansatz(x[0], x[1], n)
        return jnp.reshape(u, (1,))
