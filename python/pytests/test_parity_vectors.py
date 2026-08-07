"""Golden-vector parity tests pinning the exported JSON contract.

The committed ``frontend/public/pinn_model.json`` embeds ``test_vectors``: float64
reference outputs of a fixed set of inputs. These tests assert that

  1. the JSON is self-consistent (a float64 forward pass over its own weights
     reproduces the stored outputs) — guards against hand-edits of the file; and
  2. the JAX float32 scoring endpoint (mlops/score.py) reproduces those same outputs
     to within float32 accumulation noise.

Both use only committed files, so they run in CI with no local ``outputs/`` checkpoint.
"""

import json
import os

import numpy as np

from mlops import config, score, serialization
from mlops.json_forward import model_from_heat_payload
from pinn.json_forward import forward_from_payload

HEAT_JSON = os.path.join(config.REPO_ROOT, "frontend", "public", "pinn_model.json")


def _load_heat_payload():
    with open(HEAT_JSON, encoding="utf-8") as f:
        return json.load(f)


def test_heat_json_vectors_selfconsistent():
    """A float64 forward pass over the committed weights matches its stored outputs."""
    payload = _load_heat_payload()
    tv = payload["test_vectors"]
    inputs = np.asarray(tv["inputs"], dtype=np.float64)
    expected = np.asarray(tv["outputs"], dtype=np.float64)

    got = forward_from_payload(payload, inputs)

    assert got.shape == expected.shape
    assert np.max(np.abs(got - expected)) < 1e-12


def test_score_matches_export(tmp_path, monkeypatch):
    """The float32 endpoint reproduces the float64 reference vectors to < 1e-4."""
    payload = _load_heat_payload()
    tv = payload["test_vectors"]
    inputs = tv["inputs"]
    expected = np.asarray(tv["outputs"], dtype=np.float64)

    # Rebuild a scoring artefact directly from the committed JSON (no checkpoint).
    model = model_from_heat_payload(payload)
    width = len(payload["layers"][0]["weight"])
    depth = len(payload["layers"]) - 1
    model_dir = tmp_path / "model_dir"
    serialization.save_model(model, str(model_dir), width, depth)

    monkeypatch.setenv("AZUREML_MODEL_DIR", str(model_dir))
    score.init()
    resp = score.run({"inputs": inputs})

    assert "predictions" in resp, resp
    got = np.asarray(resp["predictions"], dtype=np.float64)
    assert np.max(np.abs(got - expected)) < 1e-4
