"""Model (de)serialization for the heat-equation PINN.

A trained model persists as a directory containing three files:
  - model.eqx        : eqx.tree_serialise_leaves output (canonical Python artefact)
  - architecture.json: {"width_size", "depth", "in_size": 3, "out_size": 1} — needed
                       to rebuild the ParametricPINN skeleton before deserialising.
  - pinn_model.json  : the existing frontend JSON export (train.export_to_json).

Reload rebuilds the skeleton from architecture.json, then deserialises into it.
"""

import json
import os

import equinox as eqx
import jax.random as jr

from pinn import train as _train_mod
from pinn.model import ParametricPINN

from . import config


def save_model(model, out_dir, width_size, depth):
    """Persist ``model`` as the three-file artefact dir. Returns out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    eqx_path = os.path.join(out_dir, config.MODEL_FILENAME)
    eqx.tree_serialise_leaves(eqx_path, model)

    arch = {
        "width_size": int(width_size),
        "depth": int(depth),
        "in_size": 3,
        "out_size": 1,
    }
    with open(os.path.join(out_dir, config.ARCH_FILENAME), "w", encoding="utf-8") as f:
        json.dump(arch, f, indent=2)

    # Frontend JSON export (tanh-mlp-heat-v2) — the React app still consumes this.
    _train_mod.export_to_json(model, os.path.join(out_dir, config.FRONTEND_JSON_FILENAME))

    return out_dir


def load_model(model_dir, key=None):
    """Rebuild a ParametricPINN from ``model_dir``.

    Reads architecture.json to size the skeleton, then deserialises model.eqx into
    it. ``key`` only seeds the throwaway skeleton; deserialised leaves overwrite it.
    """
    with open(os.path.join(model_dir, config.ARCH_FILENAME), encoding="utf-8") as f:
        arch = json.load(f)

    if key is None:
        key = jr.PRNGKey(0)
    skeleton = ParametricPINN(
        key, width_size=arch["width_size"], depth=arch["depth"]
    )

    eqx_path = os.path.join(model_dir, config.MODEL_FILENAME)
    return eqx.tree_deserialise_leaves(eqx_path, skeleton)
