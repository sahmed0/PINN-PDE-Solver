"""Fast, offline tests for the evaluation gate.

No training to convergence and no Azure calls: the fail case uses an untrained
model, the pass case monkeypatches the metrics.
"""

import jax.random as jr

from mlops import config, gate, serialization, test_set
from model import ParametricPINN


def _save_untrained(tmp_path):
    key = jr.PRNGKey(0)
    width, depth = 16, 2
    model = ParametricPINN(key, width_size=width, depth=depth)
    out_dir = tmp_path / "model_dir"
    serialization.save_model(model, str(out_dir), width, depth)
    return out_dir


def test_untrained_model_fails_gate(tmp_path):
    out_dir = _save_untrained(tmp_path)

    result = gate.run_gate(
        str(out_dir),
        config.MEAN_REL_L2_THRESHOLD,
        config.MEAN_REL_L2_OOD_THRESHOLD,
    )

    # Ansatz makes an untrained model exact at t=0, so rel_l2 is O(1)-ish (not
    # astronomical) — but it comfortably exceeds the 1e-2 threshold.
    assert result["passed"] is False
    assert result["mean_rel_l2"] > config.MEAN_REL_L2_THRESHOLD
    # Contract shape: both means + both thresholds + per-alpha dicts present.
    assert set(result["per_alpha_rel_l2"]) == {
        f"{a}" for a in config.TEST_ALPHAS_INTERP + config.TEST_ALPHAS_OOD
    }


def test_cli_returns_nonzero_on_fail(tmp_path):
    out_dir = _save_untrained(tmp_path)
    rc = gate.main([
        "--model-dir", str(out_dir),
        "--output", str(tmp_path / "gate_result.json"),
    ])
    assert rc != 0
    assert (tmp_path / "gate_result.json").exists()


def test_good_metrics_pass_gate(tmp_path, monkeypatch):
    out_dir = _save_untrained(tmp_path)

    def fake_metrics(model):
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

    monkeypatch.setattr(test_set, "held_out_metrics", fake_metrics)

    result = gate.run_gate(
        str(out_dir),
        config.MEAN_REL_L2_THRESHOLD,
        config.MEAN_REL_L2_OOD_THRESHOLD,
    )
    assert result["passed"] is True
