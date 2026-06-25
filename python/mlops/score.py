"""Azure ML online-endpoint scoring script for the heat-equation PINN.

The managed online deployment loads this file and calls ``init()`` once, then
``run(raw_data)`` per request. The registered model artefact is a
directory with ``model.eqx`` + ``architecture.json`` + ``pinn_model.json`` — but it
carries **no source code**. ``eqx.tree_deserialise_leaves`` rebuilds the skeleton
from the live ``ParametricPINN`` class, so ``model.py`` (and ``serialization.py``)
must be importable at scoring time.

That is why the deployment sets ``code_configuration.code: ../python`` (uploads
``src/`` + ``mlops/``) and ``scoring_script: mlops/score.py``. This file lives under
the ``mlops`` package but does the same ``sys.path`` bootstrap as the other
entrypoints and uses **absolute** imports, so it works both as Azure's standalone
scoring script and as ``mlops.score`` in tests.

Request shapes — ``run`` accepts either:

    {"inputs": [[x, t, alpha], ...]}            -> {"predictions": [u, ...]}
    {"grid": {"alpha": a, "nx": n, "nt": m}}    -> {"x": [...], "t": [...],
                                                    "u": [[...]]}   # shape (nt, nx)

Malformed input returns ``{"error": "<message>"}`` rather than raising.
"""

import json
import os
import sys

# Script-run bootstrap: ensure python/ is importable so `from mlops import ...`
# resolves; the package __init__ then adds python/src for `import model` etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from mlops import config, serialization
import analytical

# Populated by init(); reused across requests.
_MODEL = None

# Input domain bounds for light validation (match the training physics).
_X_LO, _X_HI = config.X_RANGE
_T_LO, _T_HI = config.T_RANGE


def _find_model_dir(root):
    """Return the directory under ``root`` that holds the model artefact.

    Azure mounts the registered model under ``AZUREML_MODEL_DIR``; depending on how
    the model was registered the files may sit directly in that dir or one level
    down in a named subfolder. Return ``root`` if it contains ``architecture.json``,
    otherwise go for the first subdir that does.
    """
    if os.path.exists(os.path.join(root, config.ARCH_FILENAME)):
        return root
    for dirpath, _dirnames, filenames in os.walk(root):
        if config.ARCH_FILENAME in filenames:
            return dirpath
    raise FileNotFoundError(
        f"No {config.ARCH_FILENAME} found under {root!r} — cannot locate model dir."
    )


def init():
    """Load the registered model once at deployment start.

    Reads ``AZUREML_MODEL_DIR`` (set by the Azure ML inference server to the mount
    point of the registered model), resolves the artefact dir, and deserialises the
    PINN via ``serialization.load_model``.
    """
    global _MODEL
    root = os.environ["AZUREML_MODEL_DIR"]
    model_dir = _find_model_dir(root)
    _MODEL = serialization.load_model(model_dir)


def _predict_points(rows):
    """Run the model on an (N, 3) list of [x, t, alpha] rows -> list of floats."""
    inputs = jnp.asarray(rows, dtype=jnp.float32)
    preds = jax.vmap(_MODEL)(inputs).reshape(-1)
    return [float(v) for v in preds]


def _validate_points(rows):
    """Return an error string if ``rows`` is not a clean (N, 3) numeric grid, else None."""
    if not isinstance(rows, list) or len(rows) == 0:
        return "'inputs' must be a non-empty list of [x, t, alpha] rows."
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return f"Row {i} must have exactly 3 numbers [x, t, alpha]."
        if not all(isinstance(v, (int, float)) for v in row):
            return f"Row {i} contains a non-numeric value."
        x, t, alpha = row
        if not (_X_LO <= x <= _X_HI):
            return f"Row {i}: x={x} out of range [{_X_LO}, {_X_HI}]."
        if not (_T_LO <= t <= _T_HI):
            return f"Row {i}: t={t} out of range [{_T_LO}, {_T_HI}]."
        if alpha <= 0:
            return f"Row {i}: alpha={alpha} must be positive."
    return None


def _predict_grid(spec):
    """Build a full (nt, nx) field for one alpha. Returns a response dict or an error."""
    if not isinstance(spec, dict) or "alpha" not in spec:
        return {"error": "'grid' must be an object with at least 'alpha'."}
    try:
        alpha = float(spec["alpha"])
        nx = int(spec.get("nx", config.TEST_GRID_NX))
        nt = int(spec.get("nt", config.TEST_GRID_NT))
    except (TypeError, ValueError):
        return {"error": "'grid' alpha/nx/nt must be numbers."}
    if alpha <= 0:
        return {"error": f"alpha={alpha} must be positive."}
    if nx < 2 or nt < 2:
        return {"error": "nx and nt must each be >= 2."}

    u_pred, _u_ref = analytical.predict_on_grid(_MODEL, nx, nt, alpha)
    x_axis = jnp.linspace(_X_LO, _X_HI, nx)
    t_axis = jnp.linspace(_T_LO, _T_HI, nt)
    return {
        "x": [float(v) for v in x_axis],
        "t": [float(v) for v in t_axis],
        "u": [[float(v) for v in row] for row in u_pred],  # (nt, nx)
    }


def run(raw_data):
    """Entry point per request. ``raw_data`` is a JSON string (Azure) or a dict (tests)."""
    if _MODEL is None:
        return {"error": "Model not initialised; init() did not run."}

    try:
        data = json.loads(raw_data) if isinstance(raw_data, (str, bytes, bytearray)) else raw_data
    except (ValueError, TypeError) as exc:
        return {"error": f"Could not parse request JSON: {exc}"}

    if not isinstance(data, dict):
        return {"error": "Request body must be a JSON object."}

    if "inputs" in data:
        err = _validate_points(data["inputs"])
        if err is not None:
            return {"error": err}
        return {"predictions": _predict_points(data["inputs"])}

    if "grid" in data:
        return _predict_grid(data["grid"])

    return {"error": "Request must contain either 'inputs' or 'grid'."}
