"""Shared constants & contracts for the heat-equation MLOps pipeline.

Define once here. Do not hardcode these values elsewhere.
"""

import os

# --- Identity -------------------------------------------------------------
EXPERIMENT_NAME = "pinn-heat-equation"
REGISTERED_MODEL_NAME = "pinn-heat"

# --- Gate — BOTH thresholds must be cleared to pass -----------------
MEAN_REL_L2_THRESHOLD = 1e-2        # in-distribution mean rel-L2 must be BELOW this
MEAN_REL_L2_OOD_THRESHOLD = 5e-2    # OOD mean rel-L2 must be BELOW this.
#                                     Tuned empirically: the 20k-epoch
#                                     baseline clears it ~8x under (OOD=6.1e-3),
#                                     while the weak config breaches it (OOD=1.7e-1).
#                                     Looser than the interp threshold because
#                                     extrapolation is intrinsically harder.

# --- Held-out test set — two slices ----------------------------------
# (a) in-distribution interpolation: alphas INSIDE [0.01, 0.1], chosen NOT to
#     coincide with the training-time validation alphas (0.01, 0.05, 0.1).
#     Training samples alpha continuously/uniformly over the whole range, so these
#     are interpolation, not "unmemorised anchors". Primary promotion metric.
TEST_ALPHAS_INTERP = (0.023, 0.047, 0.071, 0.089)
# (b) out-of-distribution extrapolation: alphas JUST OUTSIDE [0.01, 0.1] — a regime
#     training never covered. Ground truth still holds (u_exact for any alpha > 0).
TEST_ALPHAS_OOD = (0.007, 0.12)
TEST_GRID_NX = 100
TEST_GRID_NT = 100

# --- Training domains (match existing code; do not change physics) --------
ALPHA_RANGE = (0.01, 0.1)
X_RANGE = (-1.0, 1.0)
T_RANGE = (0.0, 1.0)

# --- Serving-time input policy (see score.py) -----------------------------
MAX_GRID_POINTS = 250_000            # nx*nt cap: bounds memory on a DS2_v2
ALPHA_SERVING_RANGE = (0.005, 0.15)  # reject outside; trained range + gated-OOD margin

# --- MLflow logging cadence -----------------------------------------------
LOG_EVERY = 100  # log metrics every N epochs (matches existing print cadence)

# --- Repo-stable paths ----------------------------------------------------
PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python/
REPO_ROOT = os.path.dirname(PYTHON_DIR)
# Default local MLflow tracking store (used when neither MLFLOW_TRACKING_URI env
# nor an explicit --tracking-uri is supplied). Repo-stable regardless of CWD.
DEFAULT_MLRUNS_DIR = os.path.join(PYTHON_DIR, "mlruns")
DEFAULT_OUTPUTS_DIR = os.path.join(PYTHON_DIR, "outputs")

# --- Model artefact contract -----------------------------------------
MODEL_FILENAME = "model.eqx"
ARCH_FILENAME = "architecture.json"
FRONTEND_JSON_FILENAME = "pinn_model.json"
