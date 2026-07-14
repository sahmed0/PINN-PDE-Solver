"""Fast, offline tests for the gate registration path.

No real Azure calls: the Azure ML client is mocked. Assert the registry
``create_or_update`` is invoked on a PASS and NOT invoked on a FAIL, and that the
gate metrics ride along as tags/properties on the registered model.
"""

from unittest.mock import MagicMock

import jax.random as jr

from mlops import config, gate, serialization, test_set
from pinn.model import ParametricPINN


def _save_untrained(tmp_path):
    key = jr.PRNGKey(0)
    width, depth = 16, 2
    model = ParametricPINN(key, width_size=width, depth=depth)
    out_dir = tmp_path / "model_dir"
    serialization.save_model(model, str(out_dir), width, depth)
    return out_dir


def _good_metrics(model):
    interp = tuple(float(a) for a in config.TEST_ALPHAS_INTERP)
    ood = tuple(float(a) for a in config.TEST_ALPHAS_OOD)
    return {
        "per_alpha_rel_l2": {f"{a}": 1e-4 for a in interp + ood},
        "per_alpha_linf": {f"{a}": 1e-4 for a in interp + ood},
        "mean_rel_l2": 1e-4,
        "mean_rel_l2_ood": 1e-3,
        "alphas_interp": interp,
        "alphas_ood": ood,
    }


def _mock_client(version="1"):
    client = MagicMock()
    client.models.create_or_update.return_value = MagicMock(version=version)
    return client


def test_register_called_on_pass(tmp_path, monkeypatch):
    out_dir = _save_untrained(tmp_path)
    monkeypatch.setattr(test_set, "held_out_metrics", _good_metrics)
    client = _mock_client(version="7")
    monkeypatch.setattr(gate, "_make_ml_client", lambda: client)

    rc = gate.main([
        "--model-dir", str(out_dir),
        "--output", str(tmp_path / "gate_result.json"),
        "--register",
        "--config-name", "baseline",
    ])

    assert rc == 0
    client.models.create_or_update.assert_called_once()
    # The registered Model carries the gate provenance as tags.
    registered = client.models.create_or_update.call_args.args[0]
    assert registered.name == config.REGISTERED_MODEL_NAME
    assert registered.tags["passed"] == "True"
    assert registered.tags["config_name"] == "baseline"
    assert registered.tags["source_run_id"] == "local"
    # And the metrics as (string) properties.
    assert registered.properties["mean_rel_l2"] == str(1e-4)


def test_register_not_called_on_fail(tmp_path, monkeypatch):
    out_dir = _save_untrained(tmp_path)  # untrained -> fails the gate
    client = _mock_client()
    monkeypatch.setattr(gate, "_make_ml_client", lambda: client)

    rc = gate.main([
        "--model-dir", str(out_dir),
        "--output", str(tmp_path / "gate_result.json"),
        "--register",
    ])

    assert rc != 0
    client.models.create_or_update.assert_not_called()
