import os
import pytest
import jax.random as jr
import json

from model import ParametricPINN
from train import train, export_to_json

def test_training_loop_and_export(tmp_path):
    key = jr.PRNGKey(99)
    model = ParametricPINN(key)

    # 1. Run a tiny training loop (2 epochs)
    # We just want to ensure it doesn't crash and returns a valid model
    trained_model = train(model, key, epochs=2, lr=1e-3)

    # Ensure the returned object is still our Equinox module
    assert isinstance(trained_model, ParametricPINN), "Training did not return a valid model instance."

    # 2. Test JSON weights export
    test_filepath = str(tmp_path / "test_pinn_model.json")
    export_to_json(trained_model, filepath=test_filepath)

    # 3. Assert file exists and has size > 0
    assert os.path.exists(test_filepath), "JSON file was not created."
    assert os.path.getsize(test_filepath) > 0, "JSON file is empty."

    # 4. Assert the payload is well-formed and matches the architecture
    with open(test_filepath, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["in_size"] == 3
    assert payload["out_size"] == 1
    assert len(payload["layers"]) > 0
    # First layer maps the 3 inputs; last layer maps to the single output.
    assert len(payload["layers"][0]["weight"][0]) == 3
    assert len(payload["layers"][-1]["weight"]) == 1