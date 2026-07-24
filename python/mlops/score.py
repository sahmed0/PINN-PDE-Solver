"""Azure ML online-endpoint scoring script for the heat-equation PINN.

The managed online deployment loads this file and calls ``init()`` once, then
``run(raw_data)`` per request. The registered model artefact is a
directory with ``model.eqx`` + ``architecture.json`` + ``pinn_model.json`` — but it
carries **no source code**. ``eqx.tree_deserialise_leaves`` rebuilds the skeleton
from the live ``ParametricPINN`` class, so ``model.py`` (and ``serialization.py``)
must be importable at scoring time.

That is why the deployment sets ``code_configuration.code: ../python`` (uploads
``pinn/`` + ``mlops/``) and ``scoring_script: mlops/score.py``. The Azure inference
container mounts that code directory but pip-installs nothing, so this file keeps a
single ``sys.path`` bootstrap (see below) and uses **absolute** imports, letting it
work both as Azure's standalone scoring script and as ``mlops.score`` in tests (where
the ``pinn``/``mlops`` packages are installed).

Request shapes — ``run`` accepts either:

    {"inputs": [[x, t, alpha], ...]}            -> {"predictions": [u, ...]}
    {"grid": {"alpha": a, "nx": n, "nt": m}}    -> {"x": [...], "t": [...],
                                                    "u": [[...]]}   # shape (nt, nx)
    {"health": true}                            -> {"status": "ok", ...}

Serving-time input policy (see ``config``): ``alpha`` must lie inside
``ALPHA_SERVING_RANGE`` and a grid must satisfy ``nx*nt <= MAX_GRID_POINTS``,
otherwise an error naming the limit is returned. When any requested ``alpha`` is
outside the trained ``ALPHA_RANGE`` the (successful) response also carries
``"ood": true`` and an ``"ood_note"``; in-distribution responses omit both keys.
The ``health`` echo works before and after ``init()``.

Malformed input returns ``{"error": "<message>"}`` rather than raising.
"""

import json
import os
import sys

# Azure inference-container bootstrap (the ONLY remaining sys.path insert in the
# codebase): the managed deployment mounts this code dir but pip-installs nothing, so
# put python/ on the path to make `pinn` and `mlops` importable. Local/test use goes
# through the installed package, where this insert is a harmless no-op.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from mlops import config, serialization
from pinn import analytical

# Populated by init(); reused across requests.
_MODEL = None

# Input domain bounds for light validation (match the training physics).
_X_LO, _X_HI = config.X_RANGE
_T_LO, _T_HI = config.T_RANGE
# Serving-time alpha policy: reject outside the band, flag outside the trained range.
_ALPHA_LO, _ALPHA_HI = config.ALPHA_SERVING_RANGE
_ALPHA_TRAIN_LO, _ALPHA_TRAIN_HI = config.ALPHA_RANGE

_OOD_NOTE = (
    f"alpha outside trained range [{_ALPHA_TRAIN_LO}, {_ALPHA_TRAIN_HI}]; "
    "accuracy verified only to gate OOD thresholds"
)


def _is_ood(alphas):
    """True if any alpha lies outside the trained ``config.ALPHA_RANGE``."""
    return any(a < _ALPHA_TRAIN_LO or a > _ALPHA_TRAIN_HI for a in alphas)


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
        if not (_ALPHA_LO <= alpha <= _ALPHA_HI):
            return f"Row {i}: alpha={alpha} outside serving range [{_ALPHA_LO}, {_ALPHA_HI}]."
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
    if not (_ALPHA_LO <= alpha <= _ALPHA_HI):
        return {"error": f"alpha={alpha} outside serving range [{_ALPHA_LO}, {_ALPHA_HI}]."}
    if nx < 2 or nt < 2:
        return {"error": "nx and nt must each be >= 2."}
    if nx * nt > config.MAX_GRID_POINTS:
        return {"error": f"grid nx*nt={nx * nt} exceeds the cap of {config.MAX_GRID_POINTS} points."}

    u_pred, _u_ref = analytical.predict_on_grid(_MODEL, nx, nt, alpha)
    x_axis = jnp.linspace(_X_LO, _X_HI, nx)
    t_axis = jnp.linspace(_T_LO, _T_HI, nt)
    resp = {
        "x": [float(v) for v in x_axis],
        "t": [float(v) for v in t_axis],
        "u": [[float(v) for v in row] for row in u_pred],  # (nt, nx)
    }
    if _is_ood([alpha]):
        resp["ood"] = True
        resp["ood_note"] = _OOD_NOTE
    return resp


def run(raw_data):
    """Entry point per request. ``raw_data`` is a JSON string (Azure) or a dict (tests)."""
    try:
        data = json.loads(raw_data) if isinstance(raw_data, (str, bytes, bytearray)) else raw_data
    except (ValueError, TypeError) as exc:
        return {"error": f"Could not parse request JSON: {exc}"}

    if not isinstance(data, dict):
        return {"error": "Request body must be a JSON object."}

    # Health echo — answerable before init(), so it precedes the _MODEL guard.
    if "health" in data:
        return {"status": "ok", "model_loaded": _MODEL is not None, "format": "tanh-mlp-heat-v2"}

    if _MODEL is None:
        return {"error": "Model not initialised; init() did not run."}

    if "inputs" in data:
        err = _validate_points(data["inputs"])
        if err is not None:
            return {"error": err}
        resp = {"predictions": _predict_points(data["inputs"])}
        if _is_ood([row[2] for row in data["inputs"]]):
            resp["ood"] = True
            resp["ood_note"] = _OOD_NOTE
        return resp

    if "grid" in data:
        return _predict_grid(data["grid"])

    return {"error": "Request must contain either 'inputs' or 'grid'."}
