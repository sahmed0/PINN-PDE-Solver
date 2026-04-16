import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

class ParametricPINN(eqx.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for a Parametric Physics-Informed Neural Network.
    
    Inputs (shape: (3,)):
        - x: Spatial coordinate
        - t: Time
        - alpha: Thermal diffusivity
        
    Outputs (shape: (1,)):
        - u: Temperature
    """
    mlp: eqx.nn.MLP

    def __init__(self, key: jr.PRNGKey):
        # We define a standard MLP. You can adjust width_size and depth 
        # as needed for the complexity of your PDE.
        self.mlp = eqx.nn.MLP(
            in_size=3,
            out_size=1,
            width_size=32,
            depth=3,
            activation=jax.nn.tanh,
            key=key
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the PINN."""
        # Equinox modules are callable just like standard functions
        return self.mlp(x)