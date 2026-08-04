"""Float64 NumPy forward pass over an exported model JSON payload.

This is the *reference* implementation of the exported contract: given the plain
weights-and-metadata dict that ``train.export_to_json`` / ``burgers.export_burgers_to_json``
write, it reproduces the model output in double precision with nothing but NumPy.

Two independent consumers are tested against it (see the parity tests):
  - the browser forward pass (frontend/src/lib/inference.ts), which must match to 1e-9;
  - the Azure scoring endpoint (mlops/score.py, JAX float32), which must match to 1e-4.

Because it depends only on the JSON payload, it needs no trained checkpoint and no ML
framework, so it doubles as the generator for the embedded ``test_vectors``.
"""

import numpy as np


def forward_from_payload(payload: dict, inputs: np.ndarray) -> np.ndarray:
    """Evaluate the exported tanh-MLP + ansatz on ``inputs`` in float64.

    ``inputs`` is (N, in_size): [x, t, alpha] for the heat model, [x, t] for Burgers.
    Returns a (N,) array of scalar outputs u. Weights are stored ``(out, in)`` and tanh
    is applied after every layer except the last, matching the exporter.
    """
    x = np.asarray(inputs, dtype=np.float64)

    center = np.asarray(payload["input_center"], dtype=np.float64)
    scale = np.asarray(payload["input_scale"], dtype=np.float64)
    a = (x - center) / scale

    layers = payload["layers"]
    for i, layer in enumerate(layers):
        w = np.asarray(layer["weight"], dtype=np.float64)   # (out, in)
        b = np.asarray(layer["bias"], dtype=np.float64)     # (out,)
        a = a @ w.T + b
        if i < len(layers) - 1:
            a = np.tanh(a)
    n = a[:, 0]

    x_phys = x[:, 0]
    t_phys = x[:, 1]
    ansatz = payload["ansatz"]
    if ansatz == "heat_dirichlet_sin":
        return np.sin(np.pi * x_phys) + (1.0 - x_phys ** 2) * t_phys * n
    if ansatz == "burgers_dirichlet_negsin":
        return -np.sin(np.pi * x_phys) + (1.0 - x_phys ** 2) * t_phys * n
    raise ValueError(f"Unknown ansatz {ansatz!r} in payload.")
