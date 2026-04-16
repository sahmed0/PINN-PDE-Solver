import pytest
import jax.numpy as jnp
import jax.random as jr
from model import ParametricPINN

def test_pinn_output_shape():
    # 1. Initialize a JAX random key for model weight initialization
    key = jr.PRNGKey(42)
    
    # 2. Instantiate the model
    model = ParametricPINN(key)
    
    # 3. Define a dummy input of shape (3,) 
    # Let's say: space (x) = 1.0, time (t) = 0.5, diffusivity (alpha) = 0.01
    dummy_input = jnp.array([1.0, 0.5, 0.01])
    
    # 4. Execute the forward pass
    output = model(dummy_input)
    
    # 5. Assert the output is a JAX array with shape (1,)
    assert isinstance(output, jnp.ndarray), "Output must be a JAX array."
    assert output.shape == (1,), f"Expected output shape (1,), but got {output.shape}."

    # Additional sanity check: ensure output contains valid float numbers (no NaNs)
    assert not jnp.isnan(output).any(), "Model output contains NaN values."