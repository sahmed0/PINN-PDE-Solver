"""Generate the golden parity fixture for the committed Burgers model.

Run from python/:

    uv run python scripts/gen_parity_artifacts.py

The heat model's parity vectors are embedded by `export_to_json` whenever the model is shipped
(see scripts/ship_frontend_model.py). Burgers has no checkpoint in the repo, so its reference
vectors are derived from the committed burgers_model.json and written to a separate fixture;
the model file is left byte-identical.
"""

import json
import os

import numpy as np

from mlops import config
from pinn.json_forward import forward_from_payload

FRONTEND_PUBLIC = os.path.join(config.REPO_ROOT, "frontend", "public")
BURGERS_JSON = os.path.join(FRONTEND_PUBLIC, "burgers_model.json")
FIXTURES_DIR = os.path.join(config.REPO_ROOT, "frontend", "src", "lib", "__fixtures__")
BURGERS_FIXTURE = os.path.join(FIXTURES_DIR, "burgers_test_vectors.json")


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
    generate_burgers()
    print("Done.")
