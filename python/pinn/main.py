"""Run the three research demos end to end: heat, inverse, and Burgers'.

This is the core-package entrypoint. It trains the parametric heat PINN, validates it against
the closed-form solution, and exports all three models as JSON for the frontend.

`mlops/train_entry.py` is the MLOps entrypoint for the same heat model: it adds MLflow
instrumentation, a CLI, and the `.eqx` artefact directory the evaluation gate consumes. Both
derive the model and data keys the same way from `--seed`, so the same seed gives the same
weights from either path.
"""

import os

import jax.random as jr

from pinn.analytical import evaluate, format_report
from pinn.burgers import run_burgers_demo
from pinn.inverse import run_inverse_demo
from pinn.model import ParametricPINN
from pinn.train import export_to_json, train


def main():
    print("--- Starting PINN Backend Pipeline ---")

    # 1. Initialise random seed for reproducibility
    # Using a fixed seed ensures our model initializes the same way every time we run it
    seed = 42
    key = jr.PRNGKey(seed)
    model_key, train_key = jr.split(key)

    # 2. Instantiate the model
    print("Initialising ParametricPINN...")
    model = ParametricPINN(model_key)

    # 3. Train the model
    print("\nStarting optimisation...")
    # You can easily adjust hyperparameters here based on your PDE's complexity.
    trained_model = train(model, train_key, epochs=20000, lr=1e-3)

    # 3b. Validate against the closed-form solution u = sin(pi x) exp(-alpha pi^2 t).
    print("\nValidation against the analytical solution:")
    metrics = evaluate(trained_model)
    print(format_report(metrics))

    # 4. Export the trained model as JSON weights for the frontend.
    # Write straight into the React app's public/ folder so it is served
    # at /pinn_model.json with no extra copy step.
    print("\nExporting model for the frontend...")
    public_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "public")
    json_filepath = os.path.normpath(os.path.join(public_dir, "pinn_model.json"))
    export_to_json(trained_model, filepath=json_filepath)

    # 5. Inverse problem: recover an unknown alpha from sparse, noisy data.
    # This is the scientific headline -- a parameter-estimation task a classical
    # forward solver cannot do directly. Exports its own JSON next to the forward
    # model so the frontend can display the recovered alpha and the observations.
    inverse_filepath = os.path.normpath(os.path.join(public_dir, "inverse_model.json"))
    run_inverse_demo(epochs=2000, export_path=inverse_filepath)

    # 6. Burgers' equation: a second, nonlinear PDE validated against a
    # method-of-lines numerical reference (no closed form). Proves the same
    # architecture solves more than the heat equation.
    burgers_filepath = os.path.normpath(os.path.join(public_dir, "burgers_model.json"))
    run_burgers_demo(export_path=burgers_filepath)

    print("\n--- Pipeline Complete! ---")
    print(f"Successfully saved: {json_filepath}")
    print(f"Successfully saved: {inverse_filepath}")
    print(f"Successfully saved: {burgers_filepath}")
    print("Ready to be loaded into your React application.")


if __name__ == "__main__":
    main()
