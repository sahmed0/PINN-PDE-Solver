"""Fast smoke test for the training entrypoint (kept under ~30s)."""

import json

from mlops import train_entry


def test_train_entry_smoke(tmp_path):
    out_dir = tmp_path / "run_out"
    tracking = "file:///" + str(tmp_path / "mlruns").replace("\\", "/")

    run_id, returned_dir = train_entry.main([
        "--epochs", "50",
        "--num-collocation", "200",
        "--output-dir", str(out_dir),
        "--tracking-uri", tracking,
        "--config-name", "smoke",
    ])

    assert run_id
    # Artefacts exist.
    for name in ("model.eqx", "architecture.json", "pinn_model.json",
                 "loss_curve.png", "solution.png", "metrics.json"):
        assert (out_dir / name).exists(), f"missing {name}"

    # metrics.json is the final analytical.evaluate output.
    with open(out_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    for key in ("rel_l2", "linf", "mean_rel_l2", "mean_linf"):
        assert key in metrics
