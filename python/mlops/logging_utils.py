"""MLflow setup and artefact (plot) helpers for the heat-equation pipeline."""

import os

from . import _bootstrap  # noqa: F401  (puts python/src on sys.path)

import matplotlib

matplotlib.use("Agg")  # headless: no display on CI / Azure compute
import matplotlib.pyplot as plt

import mlflow

import analytical

from . import config


def setup_mlflow(tracking_uri=None, experiment_name=config.EXPERIMENT_NAME):
    """Configure the MLflow tracking store and active experiment.

    Precedence:
      1. If ``MLFLOW_TRACKING_URI`` is already set in the environment (Azure ML
         injects this inside a job), honour it and do NOT override.
      2. Else if ``tracking_uri`` was passed explicitly, use it.
      3. Else default to a repo-stable local ``mlruns`` dir.

    Returns the effective tracking URI string.
    """
    # MLflow 3.x gates the local filesystem store behind this opt-in. We rely on
    # the file-based ./mlruns layout locally (so `mlflow ui` reads it directly);
    # # harmless for db/cloud backends.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        effective = env_uri  # already active; do not call set_tracking_uri
    else:
        if tracking_uri is None:
            os.makedirs(config.DEFAULT_MLRUNS_DIR, exist_ok=True)
            # file:// URI so it works regardless of CWD and on Windows paths
            tracking_uri = "file:///" + config.DEFAULT_MLRUNS_DIR.replace("\\", "/")
        mlflow.set_tracking_uri(tracking_uri)
        effective = tracking_uri

    mlflow.set_experiment(experiment_name)
    return effective


def plot_loss_curve(history, out_path):
    """Plot total loss + mean rel-L2 vs epoch on a twin (log-y) axis.

    ``history`` is a dict with parallel lists: ``epoch``, ``total_loss``,
    ``mean_rel_l2``.
    """
    epochs = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("total loss", color="tab:blue")
    ax1.semilogy(epochs, history["total_loss"], color="tab:blue", label="total loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("mean rel L2", color="tab:red")
    ax2.semilogy(epochs, history["mean_rel_l2"], color="tab:red", label="mean rel L2")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Training history")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_solution(model, alpha, out_path):
    """3-panel figure: predicted field, exact field, abs-error heatmap."""
    u_pred, u_ref = analytical.predict_on_grid(
        model, config.TEST_GRID_NX, config.TEST_GRID_NT, alpha
    )
    err = abs(u_pred - u_ref)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    extent = [config.X_RANGE[0], config.X_RANGE[1], config.T_RANGE[0], config.T_RANGE[1]]
    for ax, field, title in (
        (axes[0], u_pred, "predicted u"),
        (axes[1], u_ref, "exact u"),
        (axes[2], err, "abs error"),
    ):
        im = ax.imshow(
            field, origin="lower", aspect="auto", extent=extent, cmap="viridis"
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"Heat-equation PINN solution (alpha={alpha})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
