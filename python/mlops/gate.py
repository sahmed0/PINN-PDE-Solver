"""Evaluation gate for the heat-equation PINN.

The gate is the pipeline decision point. It loads a saved model, evaluates it on the held-out test
set (``test_set.held_out_metrics``) and passes ONLY if BOTH thresholds clear:

    passed = (mean_rel_l2 < threshold)        # in-distribution interpolation
             AND
             (mean_rel_l2_ood < threshold_ood)  # OOD extrapolation (looser)

It writes ``gate_result.json`` and exits 0 on pass / non-zero
on fail so the terminal can branch on the result. On Windows branch
with ``$LASTEXITCODE``, not bash ``&&``:

    python mlops/gate.py --model-dir <dir>
    if ($LASTEXITCODE -ne 0) { Write-Host "GATE FAILED — not promoting" }

With ``--register`` a passing model is promoted to the Azure ML Model
Registry via ``MLClient.models.create_or_update`` with its gate metrics attached
as tags/properties; a failing model is NOT registered. Requires Azure auth
(``az login`` + a workspace ``config.json``). The local path (no ``--register``)
is unchanged and needs no Azure SDK.

    python mlops/gate.py --model-dir <dir> --register --config-name baseline
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime

import mlflow

from mlops import config, logging_utils, serialization, test_set


def run_gate(model_dir, threshold, threshold_ood, source_run_id="local"):
    """Evaluate the model in ``model_dir`` and build the gate-result dict.

    ``passed`` is the AND of both threshold checks. All metric values come from
    ``test_set.held_out_metrics`` already ``float()``-cast.
    """
    model = serialization.load_model(model_dir)
    metrics = test_set.held_out_metrics(model)

    passed = bool(
        metrics["mean_rel_l2"] < threshold
        and metrics["mean_rel_l2_ood"] < threshold_ood
    )

    return {
        "passed": passed,
        "mean_rel_l2": metrics["mean_rel_l2"],
        "threshold": float(threshold),
        "mean_rel_l2_ood": metrics["mean_rel_l2_ood"],
        "threshold_ood": float(threshold_ood),
        "per_alpha_rel_l2": metrics["per_alpha_rel_l2"],
        "per_alpha_linf": metrics["per_alpha_linf"],
        "source_run_id": source_run_id,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


def _resolve_model_dir(run_id, dst):
    """Download a run's model artefacts from MLflow into ``dst``; return that dir.

    The training entrypoint logs model.eqx / architecture.json / pinn_model.json
    at the run's artifact root, so the downloaded root is a valid model dir.
    """
    logging_utils.setup_mlflow(None, config.EXPERIMENT_NAME)
    return mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=dst)


def _make_ml_client():
    """Build an Azure ML client, resolving the workspace two ways.

    Lazy-imported so the local gate path (no ``--register``) and the offline tests
    never need the Azure SDK installed/loaded. Auth = ``DefaultAzureCredential``
    (``az login`` locally; the job's managed identity in the cloud).

    Inside an Azure ML job the workspace coordinates are injected as env vars and
    ``config.json`` is NOT present. So prefer the env vars when present,
    and fall back to ``config.json`` in mlops/ for the local path.
    """
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    cred = DefaultAzureCredential()

    sub = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    rg = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    ws = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    if sub and rg and ws:
        return MLClient(cred, sub, rg, ws)

    # Local path: config.json (gitignored) in mlops/ supplies subscription/RG/workspace.
    # Point from_config at mlops directory explicitly so it works regardless of CWD.
    mlops_dir = os.path.join(config.REPO_ROOT, "mlops")
    return MLClient.from_config(cred, path=mlops_dir)


def register_model(result, model_path, model_name, config_name, ml_client=None):
    """Register the model dir at ``model_path`` in the Azure ML Model Registry.

    Only call this when ``result["passed"]`` is True. Attaches the gate metrics as
    tags + properties so the registry records why the model cleared the bar and
    which run produced it. Returns the new model version.

    ``model_path`` is a local directory (the gate already has it, or downloaded it
    from a run via ``_resolve_model_dir``). Registering from a path is uniform
    across local + cloud; run lineage is preserved via the ``source_run_id``
    tag/property. (For tighter portal lineage one could instead point at the
    Azure-only job-output URI ``azureml://jobs/<run-id>/outputs/artifacts/paths/``
    — see mlops/README.md.)
    """
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model
    from azure.core.exceptions import ClientAuthenticationError

    if ml_client is None:
        ml_client = _make_ml_client()

    # Azure ML property values must be strings; result values are already float-cast.
    properties = {
        "mean_rel_l2": str(result["mean_rel_l2"]),
        "threshold": str(result["threshold"]),
        "mean_rel_l2_ood": str(result["mean_rel_l2_ood"]),
        "threshold_ood": str(result["threshold_ood"]),
        "source_run_id": str(result["source_run_id"]),
        "config_name": str(config_name),
        "passed": str(result["passed"]),
    }
    for alpha, val in result["per_alpha_rel_l2"].items():
        properties[f"rel_l2_{alpha}"] = str(val)

    model = Model(
        path=model_path,
        name=model_name,
        type=AssetTypes.CUSTOM_MODEL,
        description="Heat-equation PINN that cleared the evaluation gate.",
        tags={
            "passed": str(result["passed"]),
            "config_name": str(config_name),
            "source_run_id": str(result["source_run_id"]),
        },
        properties=properties,
    )
    try:
        registered = ml_client.models.create_or_update(model)
    except ClientAuthenticationError as e:
        raise RuntimeError(
            "Model registration failed to authenticate to Azure.\n"
            "  - Local run: have you run `az login` (and is mlops/config.json present)?\n"
            "  - Azure ML job: the compute cluster needs a managed identity with an\n"
            "    RBAC role (AzureML Data Scientist / Contributor) on the workspace,\n"
            "    and the job must request it via `identity: { type: managed }`.\n"
            "    See mlops/gate_job.yml.\n"
            f"Underlying error: {e}"
        ) from e
    return registered.version


def _verdict_text(result):
    interp_ok = result["mean_rel_l2"] < result["threshold"]
    ood_ok = result["mean_rel_l2_ood"] < result["threshold_ood"]
    lines = [
        "=" * 56,
        f"GATE: {'PASS' if result['passed'] else 'FAIL'}",
        "-" * 56,
        f"  in-distribution mean rel-L2 : {result['mean_rel_l2']:.4e} "
        f"{'<' if interp_ok else '>='} {result['threshold']:.1e}  "
        f"[{'ok' if interp_ok else 'FAIL'}]",
        f"  OOD mean rel-L2             : {result['mean_rel_l2_ood']:.4e} "
        f"{'<' if ood_ok else '>='} {result['threshold_ood']:.1e}  "
        f"[{'ok' if ood_ok else 'FAIL'}]",
        "=" * 56,
    ]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluation gate for the heat-equation PINN.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", type=str, help="A training output dir.")
    src.add_argument("--run-id", type=str,
                     help="Resolve the model artefact from an MLflow run.")
    p.add_argument("--threshold", type=float, default=config.MEAN_REL_L2_THRESHOLD,
                   help="In-distribution mean rel-L2 must be below this.")
    p.add_argument("--threshold-ood", type=float, default=config.MEAN_REL_L2_OOD_THRESHOLD,
                   help="OOD mean rel-L2 must be below this (looser).")
    p.add_argument("--output", type=str, default="gate_result.json",
                   help="Where to write gate_result.json.")
    p.add_argument("--register", action="store_true",
                   help="On PASS, register the model in the Azure ML Model "
                        "Registry (needs az login + config.json). On FAIL, "
                        "register nothing.")
    p.add_argument("--model-name", type=str, default=config.REGISTERED_MODEL_NAME,
                   help="Registered model name (Azure ML Model Registry).")
    p.add_argument("--config-name", type=str, default="unknown",
                   help="Label recorded as a registry tag/property "
                        "(e.g. 'baseline'/'weak').")
    args = p.parse_args(argv)

    # ExitStack owns the temp dir (only created for --run-id) so it is cleaned up
    # even on error, while keeping model_dir alive through registration below.
    with contextlib.ExitStack() as stack:
        if args.run_id:
            tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="gate_model_"))
            model_dir = _resolve_model_dir(args.run_id, tmp)
            source_run_id = args.run_id
        else:
            model_dir = args.model_dir
            source_run_id = "local"

        result = run_gate(model_dir, args.threshold, args.threshold_ood, source_run_id)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        print(_verdict_text(result))
        print(f"Wrote {args.output}")

        # If invoked inside an active MLflow run, record
        # the gate metrics + outcome on that run. Standalone local runs have none.
        if mlflow.active_run() is not None:
            mlflow.log_metrics({
                "gate_mean_rel_l2": result["mean_rel_l2"],
                "gate_mean_rel_l2_ood": result["mean_rel_l2_ood"],
            })
            mlflow.set_tag("gate_passed", str(result["passed"]).lower())

        # Promote only on PASS. The registry holds only models that
        # cleared the gate — a failing run logs its failure and registers nothing.
        if args.register:
            if result["passed"]:
                version = register_model(
                    result, model_dir, args.model_name, args.config_name
                )
                print(f"Registered '{args.model_name}' version {version}")
            else:
                print("GATE FAILED — not registering (registry holds passing models only).")

        return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
