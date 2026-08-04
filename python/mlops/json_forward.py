"""Rebuild a live ParametricPINN from an exported heat-model JSON payload.

The float64 reference forward pass lives in ``json_forward`` under ``python/pinn``
(framework-free, shared with the exporter). This module holds the piece that needs the
Equinox model class: it reconstructs a ``ParametricPINN`` whose weights come straight from
a committed ``pinn_model.json``, so the scoring endpoint can be exercised in CI without any
local ``model.eqx`` checkpoint.
"""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from pinn.model import ParametricPINN


def model_from_heat_payload(payload: dict) -> ParametricPINN:
    """Build a ParametricPINN carrying the payload's weights (cast to float32).

    ``width_size``/``depth`` are inferred from the layer shapes: the MLP has ``depth + 1``
    Linear layers and the first layer's output dimension is ``width_size``.
    """
    layers = payload["layers"]
    width_size = len(layers[0]["weight"])
    depth = len(layers) - 1

    skeleton = ParametricPINN(jr.PRNGKey(0), width_size=width_size, depth=depth)

    replacements = []
    for layer in layers:
        replacements.append(jnp.asarray(layer["weight"], dtype=jnp.float32))
        replacements.append(jnp.asarray(layer["bias"], dtype=jnp.float32))

    def leaves(m):
        out = []
        for lin in m.mlp.layers:
            out.append(lin.weight)
            out.append(lin.bias)
        return out

    return eqx.tree_at(leaves, skeleton, replacements)
