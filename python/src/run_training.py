"""
import jax
from jax import random
from model import ParametricPINN
from train import train, export_to_onnx # Adjust if your filename is different

def main():
    # 1. Initialize Random Key
    key = random.PRNGKey(42)
    model_key, data_key = random.split(key)

    # 2. Initialize Model (3 inputs: x, t, alpha | 1 output: u)
    print("Initializing Parametric PINN...")
    model = ParametricPINN(model_key) 

    # 3. Train the Model
    # Start with 5000 epochs to get a decent result. 
    # If it's too slow, drop to 1000 for a quick test.
    print("Starting training loop...")
    trained_model = train(model, data_key, epochs=1000, lr=1e-3)

    # 4. Export to ONNX
    # This creates the 'pinn_model.onnx' file in your current folder
    export_to_onnx(trained_model, "pinn_model.onnx")
    print("Success! Model is ready for the browser.")

if __name__ == "__main__":
    main()
"""