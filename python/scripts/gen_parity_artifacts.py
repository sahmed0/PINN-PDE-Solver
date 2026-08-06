"""Generate the golden parity artifacts for the committed frontend models.

Run once from ``python/``::

    uv run python scripts/gen_parity_artifacts.py

Heat: attempts a provenance check — re-export every local ``outputs/baseline-*``
checkpoint and compare its weights to the committed ``frontend/public/pinn_model.json``.
On a match the committed file is rewritten from that checkpoint (identical weights, now
with ``test_vectors``). If nothing matches, it falls back to deriving ``test_vectors`` from
the committed file's own float64 weights and adds them in place — the TS/endpoint parity
guarantee holds either way; only the "traceable to a checkpoint" claim is dropped.

Burgers: no checkpoint exists, so the reference vectors are derived from the committed
``burgers_model.json`` and written to a separate fixture; the model file is left untouched.
"""

import json
import os
import tempfile

import numpy as np

from mlops import config, serialization
from pinn import train as train_mod
from pinn.json_forward import forward_from_payload

FRONTEND_PUBLIC = os.path.join(config.REPO_ROOT, "frontend", "public")
HEAT_JSON = os.path.join(FRONTEND_PUBLIC, "pinn_model.json")
BURGERS_JSON = os.path.join(FRONTEND_PUBLIC, "burgers_model.json")
FIXTURES_DIR = os.path.join(config.REPO_ROOT, "frontend", "src", "lib", "__fixtures__")
BURGERS_FIXTURE = os.path.join(FIXTURES_DIR, "burgers_test_vectors.json")


def _weights_match(a, b):
    """True if two payloads have identical weight layout and normalisation."""
    return (
        a["layers"] == b["layers"]
        and a["input_center"] == b["input_center"]
        and a["input_scale"] == b["input_scale"]
    )


def _find_matching_checkpoint(committed):
    """Re-export each local baseline checkpoint and return the dir whose weights match."""
    outputs_dir = config.DEFAULT_OUTPUTS_DIR
    if not os.path.isdir(outputs_dir):
        return None
    for name in sorted(os.listdir(outputs_dir)):
        model_dir = os.path.join(outputs_dir, name)
        if not os.path.exists(os.path.join(model_dir, config.ARCH_FILENAME)):
            continue
        try:
            model = serialization.load_model(model_dir)
        except Exception as exc:  # provenance probe: report and move on
            print(f"  {name}: load failed ({exc})")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_json = os.path.join(tmp, "reexport.json")
            train_mod.export_to_json(model, tmp_json)
            with open(tmp_json, encoding="utf-8") as f:
                fresh = json.load(f)
        if _weights_match(fresh, committed):
            print(f"  {name}: MATCH")
            return model_dir
        print(f"  {name}: no match")
    return None


def generate_heat():
    with open(HEAT_JSON, encoding="utf-8") as f:
        committed = json.load(f)
    old_layers = committed["layers"]

    print("Heat provenance check (re-export each outputs/baseline-* vs committed JSON):")
    match_dir = _find_matching_checkpoint(committed)

    if match_dir is not None:
        print(f"Provenance PASS — rewriting {HEAT_JSON} from {match_dir}")
        model = serialization.load_model(match_dir)
        train_mod.export_to_json(model, HEAT_JSON)
    else:
        print("Provenance FALLBACK — no checkpoint matches the committed model.")
        print("  Deriving test_vectors from the committed JSON's own weights (in place).")
        inputs = np.asarray(train_mod.PARITY_INPUTS, dtype=np.float64)
        outputs = forward_from_payload(committed, inputs)
        committed["test_vectors"] = {
            "dtype": "float64",
            "note": "reference outputs from a float64 NumPy forward pass over this file's weights",
            "inputs": inputs.tolist(),
            "outputs": outputs.tolist(),
        }
        with open(HEAT_JSON, "w", encoding="utf-8") as f:
            json.dump(committed, f)

    with open(HEAT_JSON, encoding="utf-8") as f:
        new = json.load(f)
    assert new["layers"] == old_layers, "layers changed — expected weights to be identical!"
    assert "test_vectors" in new, "test_vectors missing after generation"
    print(f"  Self-check OK: layers identical, {len(new['test_vectors']['inputs'])} "
          f"heat test_vectors embedded.\n")


def generate_burgers():
    with open(BURGERS_JSON, encoding="utf-8") as f:
        model = json.load(f)

    from pinn.burgers import BURGERS_PARITY_INPUTS

    inputs = np.asarray(BURGERS_PARITY_INPUTS, dtype=np.float64)
    outputs = forward_from_payload(model, inputs)
    fixture = {
        "dtype": "float64",
        "note": "reference outputs from a float64 NumPy forward pass over this file's weights",
        "source": "derived from public/burgers_model.json",
        "inputs": inputs.tolist(),
        "outputs": outputs.tolist(),
    }
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    with open(BURGERS_FIXTURE, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)
    print(f"Burgers: wrote {len(inputs)} test_vectors to {BURGERS_FIXTURE}")
    print("  (burgers_model.json left byte-identical — no checkpoint to regenerate.)\n")


if __name__ == "__main__":
    generate_heat()
    generate_burgers()
    print("Done.")
