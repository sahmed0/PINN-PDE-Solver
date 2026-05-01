import os
# '0' = all logs (default)
# '1' = filter out INFO logs (this hides the oneDNN message)
# '2' = filter out INFO and WARNING logs
# '3' = filter out INFO, WARNING, and ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import os

import jax.random as jr
from model import ParametricPINN
from train import train, export_to_json
from analytical import evaluate, format_report

def main():
    print("--- Starting PINN Backend Pipeline ---")
    
    # 1. Initialise random seed for reproducibility
    # Using a fixed seed ensures our model initializes the same way every time we run it
    seed = 42
    key = jr.PRNGKey(seed)
    
    # 2. Instantiate the model
    print("Initialising ParametricPINN...")
    model = ParametricPINN(key)
    
    # 3. Train the model
    print("\nStarting optimisation...")
    # You can easily adjust hyperparameters here based on your PDE's complexity.
    # The model uses normalised inputs + a hard-constraint ansatz (exact IC/BCs),
    # so the cosine-annealed run below converges to a much tighter fit than the
    # old 1000-epoch soft-constraint training.
    trained_model = train(model, key, epochs=20000, lr=1e-3)

    # 3b. Validate against the closed-form solution u = sin(pi x) exp(-alpha pi^2 t).
    print("\nValidation against the analytical solution:")
    metrics = evaluate(trained_model)
    print(format_report(metrics))

    # 4. Export the trained model as JSON weights for the frontend.
    # Write straight into the React app's public/ folder so it is served
    # at /pinn_model.json with no extra copy step.
    print("\nExporting model for the frontend...")
    public_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "frontend", "public"
    )
    json_filepath = os.path.normpath(os.path.join(public_dir, "pinn_model.json"))
    export_to_json(trained_model, filepath=json_filepath)

    print(f"\n--- Pipeline Complete! ---")
    print(f"Successfully saved: {json_filepath}")
    print("Ready to be loaded into your React application.")

if __name__ == "__main__":
    main()
