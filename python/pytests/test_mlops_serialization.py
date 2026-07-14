"""Round-trip test for the model artefact contract."""

import jax.numpy as jnp
import jax.random as jr

from mlops import serialization
from pinn.model import ParametricPINN


def test_save_load_roundtrip(tmp_path):
    key = jr.PRNGKey(7)
    width, depth = 16, 2
    model = ParametricPINN(key, width_size=width, depth=depth)

    out_dir = tmp_path / "model_dir"
    serialization.save_model(model, str(out_dir), width, depth)

    # All three artefact files written.
    assert (out_dir / "model.eqx").exists()
    assert (out_dir / "architecture.json").exists()
    assert (out_dir / "pinn_model.json").exists()

    loaded = serialization.load_model(str(out_dir))

    # Identical predictions on a batch of test inputs.
    inputs = jnp.array([
        [0.0, 0.0, 0.05],
        [0.3, 0.5, 0.02],
        [-0.7, 0.9, 0.09],
    ])
    orig = jnp.stack([model(x) for x in inputs])
    again = jnp.stack([loaded(x) for x in inputs])
    assert jnp.allclose(orig, again, atol=1e-6)
