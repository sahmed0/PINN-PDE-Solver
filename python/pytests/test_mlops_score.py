"""Fast, offline tests for the online-endpoint scoring script (mlops/score.py).

No Azure: we save a tiny model to a temp dir, point AZUREML_MODEL_DIR at it, run
score.init(), and exercise run() for both payload shapes.

The hard-constraint ansatz (model.py) makes u(x, 0) = sin(pi x) EXACTLY for any
model, trained or not — so the t=0 predictions give a deterministic, training-free
ground truth to assert against.
"""

import json
import math

import jax.random as jr

from mlops import score, serialization
from pinn.model import ParametricPINN


def _init_with_tiny_model(tmp_path, monkeypatch):
    """Save an untrained tiny model and point score.init() at it."""
    key = jr.PRNGKey(0)
    width, depth = 16, 2
    model = ParametricPINN(key, width_size=width, depth=depth)
    model_dir = tmp_path / "model_dir"
    serialization.save_model(model, str(model_dir), width, depth)
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(model_dir))
    score.init()


def test_init_finds_model_in_subfolder(tmp_path, monkeypatch):
    # Azure may mount the artefact one level down — _find_model_dir must locate it.
    key = jr.PRNGKey(1)
    sub = tmp_path / "outer" / "pinn-heat"
    serialization.save_model(ParametricPINN(key, width_size=8, depth=2), str(sub), 8, 2)
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path / "outer"))
    score.init()
    resp = score.run(json.dumps({"inputs": [[0.5, 0.0, 0.05]]}))
    assert math.isclose(resp["predictions"][0], 1.0, abs_tol=1e-5)


def test_points_mode_exact_at_t0(tmp_path, monkeypatch):
    _init_with_tiny_model(tmp_path, monkeypatch)
    # At t=0 the ansatz forces u = sin(pi x) regardless of the (untrained) network.
    rows = [[0.5, 0.0, 0.05], [0.0, 0.0, 0.05], [-0.5, 0.0, 0.1]]
    resp = score.run(json.dumps({"inputs": rows}))
    assert "predictions" in resp
    preds = resp["predictions"]
    assert len(preds) == 3
    assert all(isinstance(v, float) for v in preds)
    for (x, _t, _a), u in zip(rows, preds):
        assert math.isclose(u, math.sin(math.pi * x), abs_tol=1e-5)


def test_points_mode_accepts_dict_payload(tmp_path, monkeypatch):
    # Azure passes a JSON string; tests may pass a dict directly.
    _init_with_tiny_model(tmp_path, monkeypatch)
    resp = score.run({"inputs": [[0.5, 0.0, 0.05]]})
    assert math.isclose(resp["predictions"][0], 1.0, abs_tol=1e-5)


def test_grid_mode_shape_and_t0_row(tmp_path, monkeypatch):
    _init_with_tiny_model(tmp_path, monkeypatch)
    nx, nt = 8, 8
    resp = score.run(json.dumps({"grid": {"alpha": 0.05, "nx": nx, "nt": nt}}))
    assert set(resp) == {"x", "t", "u"}
    assert len(resp["x"]) == nx
    assert len(resp["t"]) == nt
    assert len(resp["u"]) == nt and all(len(row) == nx for row in resp["u"])
    # Row 0 is t=0 -> exact sin(pi x) across the x axis.
    for x, u in zip(resp["x"], resp["u"][0]):
        assert math.isclose(u, math.sin(math.pi * x), abs_tol=1e-5)


def test_malformed_inputs_return_error(tmp_path, monkeypatch):
    _init_with_tiny_model(tmp_path, monkeypatch)
    # Wrong row width.
    assert "error" in score.run(json.dumps({"inputs": [[0.5, 0.0]]}))
    # Out-of-range x.
    assert "error" in score.run(json.dumps({"inputs": [[5.0, 0.0, 0.05]]}))
    # Neither key present.
    assert "error" in score.run(json.dumps({"foo": 1}))
    # Not even JSON.
    assert "error" in score.run("not json {")
    # Non-object body.
    assert "error" in score.run(json.dumps([1, 2, 3]))


def test_grid_mode_rejects_bad_spec(tmp_path, monkeypatch):
    _init_with_tiny_model(tmp_path, monkeypatch)
    assert "error" in score.run(json.dumps({"grid": {"nx": 10}}))          # no alpha
    assert "error" in score.run(json.dumps({"grid": {"alpha": -1.0}}))     # alpha <= 0
    assert "error" in score.run(json.dumps({"grid": {"alpha": 0.05, "nx": 1}}))  # nx < 2
