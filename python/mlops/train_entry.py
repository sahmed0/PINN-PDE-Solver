"""Parametrised heat-equation training entrypoint with MLflow instrumentation.

Reproduces the existing heat-equation training as a CLI that logs hyperparameters,
per-epoch metrics, and output artefacts to MLflow. Runs identically locally (local
`mlruns/`) and inside an Azure ML job (workspace tracking URI injected via env) --
the precedence lives in `logging_utils.setup_mlflow`.

Usage (from python/):
    python mlops/train_entry.py --epochs 20000              # baseline
    python mlops/train_entry.py --epochs 300 --num-collocation 150 --config-name weak

The "weak" config above is the deliberately-failing config: 300 epochs
on 150 collocation points trains too little to clear the gate. Gating its output
dir breaches at least one threshold (interp or OOD), so it is NOT promoted.
"""

import argparse
import datetime
import json
import os

import jax.random as jr
import mlflow

from mlops import config, logging_utils, serialization
from pinn import analytical
from pinn import train as train_mod
from pinn.model import ParametricPINN


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the heat-equation PINN with MLflow.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--width-size", type=int, default=32)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-collocation", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Defaults to a timestamped dir under python/outputs/.")
    p.add_argument("--tracking-uri", type=str, default=None,
                   help="MLflow tracking URI. None -> local mlruns (unless "
                        "MLFLOW_TRACKING_URI is set in the env, which wins).")
    p.add_argument("--experiment-name", type=str, default=config.EXPERIMENT_NAME)
    p.add_argument("--config-name", type=str, default="baseline",
                   help="Label for this run (e.g. 'baseline'/'weak'); logged as a tag.")
    # NOTE: --num-bc / --num-ic are intentionally NOT exposed. The existing
    # train() only accepts num_collocation; IC/BC counts are hardcoded in
    # generate_training_data and contribute no gradient under the hard-constraint
    # ansatz (physics.py).
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.output_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(config.DEFAULT_OUTPUTS_DIR, f"{args.config_name}-{stamp}")
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    logging_utils.setup_mlflow(args.tracking_uri, args.experiment_name)

    # Reproducibility caveat: --seed controls model init + collocation sampling,
    # but IC/BC points use hardcoded PRNGKeys in generate_training_data. Pre-existing
    # and harmless (IC/BC contribute no gradient).
    key = jr.PRNGKey(args.seed)
    model_key, train_key = jr.split(key)
    model = ParametricPINN(model_key, width_size=args.width_size, depth=args.depth)

    history = {"epoch": [], "total_loss": [], "loss_pde": [],
               "loss_ic": [], "loss_bc": [], "mean_rel_l2": []}

    def log_callback(epoch, metrics):
        for k, v in metrics.items():
            history.setdefault(k, [])
            if k != "epoch":
                history[k].append(v)
        history["epoch"].append(epoch)
        mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if v is not None}, step=epoch
        )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("config_name", args.config_name)
        mlflow.log_params({
            "lr": args.lr,
            "epochs": args.epochs,
            "width_size": args.width_size,
            "depth": args.depth,
            "num_collocation": args.num_collocation,
            "seed": args.seed,
            "config_name": args.config_name,
        })

        trained = train_mod.train(
            model, train_key,
            epochs=args.epochs, lr=args.lr,
            num_collocation=args.num_collocation,
            validate=True, log_callback=log_callback,
        )

        # Final evaluation against the analytical solution (training val_alphas).
        metrics = analytical.evaluate(trained)
        mlflow.log_metrics({
            "final_mean_rel_l2": metrics["mean_rel_l2"],
            "final_mean_linf": metrics["mean_linf"],
        })

        # Persist the model artefact dir (model.eqx + architecture.json + json).
        serialization.save_model(trained, out_dir, args.width_size, args.depth)

        # Render plots.
        loss_png = os.path.join(out_dir, "loss_curve.png")
        sol_png = os.path.join(out_dir, "solution.png")
        logging_utils.plot_loss_curve(history, loss_png)
        logging_utils.plot_solution(trained, alpha=0.05, out_path=sol_png)

        # metrics.json == final analytical.evaluate output
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Log every artefact to MLflow. These files already live in out_dir,
        # which Azure ML auto-captures from outputs/, so a logging failure
        # (e.g. an mlflow/azureml-mlflow artifact-builder version mismatch)
        # must NOT fail an otherwise-successful training run.
        artifacts = [os.path.join(out_dir, name) for name in (
            config.MODEL_FILENAME, config.ARCH_FILENAME,
            config.FRONTEND_JSON_FILENAME)]
        artifacts += [loss_png, sol_png, metrics_path]
        for path in artifacts:
            try:
                mlflow.log_artifact(path)
            except Exception as exc:  # noqa: BLE001 -- never fail the run on logging
                print(f"WARNING: mlflow.log_artifact failed for {path}: {exc!r} "
                      "(file is still in outputs/ and captured by Azure ML)")

    print(f"\nRun id:    {run_id}")
    print(f"Output dir: {out_dir}")
    print(f"Final mean rel L2: {metrics['mean_rel_l2']:.3e}")
    return run_id, out_dir


if __name__ == "__main__":
    main()
