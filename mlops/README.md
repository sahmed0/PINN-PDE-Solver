# MLOps & Azure ML – train, gate, register, deploy

This directory holds the **Azure ML assets** (environment, job, endpoint and deployment
YAML) and the runbook for the heat-equation PINN's MLOps workflow. For the project
overview, the science, and the result screenshots, see the root [README.md](../README.md).

The workflow wraps the PINN forward model with the MLOps concept that matters: **a gate,
not just a training loop.** A model is only promoted if it clears a real, physical quality
bar measured against the closed-form analytical solution.

```
train (train_entry.py) ──► MLflow ──► gate (gate.py) ──► registry ──► endpoint
                                          │
                                   decision point
```

All Python commands run from `python/`. (mlflow lives in the `mlops` dependency group:
prefix with `uv run --group mlops` if you are not in the activated venv.)

## Local train → gate

```powershell
# 1. Train a baseline (CPU, a few minutes).
uv run mlops/train_entry.py --epochs 20000 --config-name baseline

# 2. Gate the resulting output dir (printed by train_entry as "Output dir").
uv run mlops/gate.py --model-dir <output-dir>
if ($LASTEXITCODE -ne 0) { Write-Host "GATE FAILED – not promoting" }
```

The gate passes only if **both** thresholds clear (defined in
[python/mlops/config.py](../python/mlops/config.py)):

| metric | meaning | threshold (`config.py`) |
|--------|---------|-------------------------|
| `mean_rel_l2` | in-distribution interpolation (primary) | `MEAN_REL_L2_THRESHOLD` (`1e-2`) |
| `mean_rel_l2_ood` | out-of-distribution extrapolation (looser) | `MEAN_REL_L2_OOD_THRESHOLD` (`5e-2`) |

The OOD threshold is looser because extrapolation is intrinsically harder. The gate writes
`gate_result.json` and exits `0` on pass / non-zero on fail. On Windows, branch with
`$LASTEXITCODE` (above), not bash `&&`.

You can also gate a model straight from an MLflow run id:

```powershell
uv run mlops/gate.py --run-id <mlflow-run-id>
```

## Demonstrating the gate (the deliberately-failing config)

To prove the gate discriminates, train a deliberately-weak config with too few epochs on
too few collocation points. It breaches at least one threshold and is **not** promoted:

```powershell
uv run mlops/train_entry.py --epochs 300 --num-collocation 150 --config-name weak
uv run mlops/gate.py --model-dir <weak-output-dir>   # -> FAIL, $LASTEXITCODE = 1
```

The 20k-epoch baseline clears the gate ~8× under (OOD ≈ `6.1e-3`), while the weak config
breaches it (OOD ≈ `1.7e-1`) – the thresholds sit comfortably between the two.

---

## Run on Azure ML

Run the **exact same** `train_entry.py` as a command job on an Azure ML compute cluster.
Inside an Azure ML job, `azureml-mlflow` injects `MLFLOW_TRACKING_URI` pointing at the
workspace, and `setup_mlflow()` honours it before any local fallback – so params, per-epoch
metrics, and artefacts land in the Azure ML portal automatically, with **zero Azure-specific
code** in the script.

### Prerequisites

- An Azure ML **workspace** and a **CPU compute cluster** (scale-to-zero, `min_instances=0`,
  so an idle cluster costs nothing).
- Azure CLI with the ML extension: `az extension add -n ml`.

### Files

| file | role |
|------|------|
| `config.json` | Use `config.json.example` template or download the pre-built one for your workspace from Azure ML portal |
| `environment.yml` | Azure ML training environment asset (image + conda) |
| `conda_train.yml` | conda spec referenced by `environment.yml` |
| `train_job.yml` | command job: runs `mlops/train_entry.py` on the cluster |
| `python/.amlignore` | keeps `mlruns/`, `outputs/`, `.venv/` out of the code upload |

### 0. Docker smoke-test the environment locally first (recommended)

Catch a dependency typo before a 20–40 min cloud provision. Build the conda env into a
local image off the same base and import every dep:

```powershell
# From directory root. Requires Docker Desktop.
docker run --rm -v "${PWD}\mlops:/w" -w /w mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04 bash -lc "conda env create -f conda_train.yml -n smoke && conda run -n smoke python -c 'import jax, equinox, optax, mlflow, azureml.mlflow, numpy; print(jax.numpy.zeros(3).sum())'"
```

A clean `0.0` (and no `ImportError`) means the env is good.

### 1. Authenticate & point at the workspace

```powershell
# Requires workspace config.json
az login
az configure --defaults group=<RG> workspace=<WS>
```

### 2. Register the environment

```powershell
az ml environment create -f environment.yml
```

### 3. Submit the training job

```powershell
az ml job create -f train_job.yml --web
```

`--web` opens the run in the portal. Override hyperparameters inline, e.g. the
deliberately-weak config:

```powershell
az ml job create -f train_job.yml `
  --set inputs.epochs=300 inputs.num_collocation=150 inputs.config_name=weak
```

### 4. Watch it / find the results

- **Portal → Jobs**: the run appears under experiment **`pinn-heat-equation`**;
  status goes `Queued → Running → Completed`. The first run wakes the scaled-to-zero
  cluster (a few minutes of provisioning).
- **Metrics** tab: per-epoch `total_loss`, `loss_pde`, `mean_rel_l2` curves, plus
  the final `final_mean_rel_l2`.
- **Outputs + logs / Images** tab: `model.eqx`, `architecture.json`,
  `pinn_model.json`, `loss_curve.png`, `solution.png`, `metrics.json`.

Note the run id – the gate job (below) resolves the model artefact from it.

---

## Promote to the Model Registry

The gate becomes the *promotion* decision point: on **PASS** it registers the model in the
**Azure ML Model Registry** (name `pinn-heat`) with its gate metrics attached; on **FAIL**
it registers **nothing**. So the registry only ever contains models that cleared the bar –
that is the whole point.

Add `--register` to `gate.py`. It needs Azure auth (`az login` + the workspace `config.json`)
and the `azure-ai-ml` / `azure-identity` SDKs (in the `mlops` dependency group and
`conda_train.yml`):

```powershell
# Local promote from the Azure CLI (recommended path).
uv run mlops/gate.py --model-dir <output-dir> --register --config-name baseline
if ($LASTEXITCODE -ne 0) { Write-Host "GATE FAILED – not promoting" }
```

On PASS, the model carries:

- **tags:** `passed`, `config_name`, `source_run_id`.
- **properties:** `mean_rel_l2`, `threshold`, `mean_rel_l2_ood`, `threshold_ood`,
  `source_run_id`, `config_name`, `passed`, and `rel_l2_<alpha>` per test alpha.

### Checking if the gate really works

```powershell
# Baseline clears the gate -> registers pinn-heat v1.
uv run mlops/train_entry.py --epochs 20000 --config-name baseline
uv run mlops/gate.py --model-dir <baseline-dir> --register --config-name baseline

# Weak config FAILS -> NOT registered (no version is created – the gap is the demo).
uv run mlops/train_entry.py --epochs 300 --num-collocation 150 --config-name weak
uv run mlops/gate.py --model-dir <weak-dir> --register --config-name weak   # FAIL, no register

# Train a fresh baseline again -> registers pinn-heat v2.
uv run mlops/train_entry.py --epochs 20000 --config-name baseline
uv run mlops/gate.py --model-dir <baseline-dir-2> --register --config-name baseline
```

The registry shows only the passing versions; the missing version where the weak run
*would* have landed is visible evidence the gate did its job.

### Model source: local path vs job-output URI

`--register` uploads the **local model directory** (the dir from `--model-dir`, or the temp
dir downloaded from `--run-id`). This is uniform across local and cloud and easy to test;
run lineage is preserved via the `source_run_id` tag/property. For tighter portal lineage
you can instead point the registration at the Azure-only job-output URI
`azureml://jobs/<run-id>/outputs/artifacts/paths/` (swap `model_path` in
`gate.register_model`) – but that URI is invalid against a local `mlruns/` store, which is
why the path form is the default.

### As an Azure command job (`gate_job.yml`)

If you want to run the gate as an Azure command job:
```powershell
az ml job create -f gate_job.yml --set inputs.run_id=<training-run-id>
```

> ⚠️ **Auth caveat:** no `config.json` is uploaded with the job, so `--register` from inside a job relies on the workspace/compute managed identity (`DefaultAzureCredential`). This takes extra setup.
>Therefore, the recommended default path is the **local** promote above against the Azure run id.
> `gate_job.yml` is provided as the cloud variant.

---

## Serve the model – managed online endpoint

Deploy a gate-passed model from the registry as a REST API and prove it works with a live call
(both payload shapes).

> 💸 **Cost warning.** A managed online endpoint runs a dedicated VM and **bills for as long
> as it exists** – it does *not* scale to zero like the training cluster. The first cloud
> deploy takes **20–40 min** to provision. **To learn how to delete the endpoint see [Teardown](#6-teardown) below.**

### Files

| file | role |
|------|------|
| `python/mlops/score.py` | scoring script – `init()` loads the model, `run()` predicts |
| `environment_inference.yml` | **dedicated** inference env asset (image + conda) |
| `conda_inference.yml` | lean conda spec + `azureml-inference-server-http` |
| `endpoint.yml` | managed online endpoint (`auth_mode: key`) |
| `deployment.yml` | deployment: registered model + inference env + `score.py` |
| `sample_request_points.json` | point-list payload `{"inputs": [[x,t,alpha], ...]}` |
| `sample_request_grid.json` | grid payload `{"grid": {"alpha", "nx", "nt"}}` |

The scoring script ([python/mlops/score.py](../python/mlops/score.py)) accepts **both**
request shapes:

```jsonc
// point list -> {"predictions": [u, ...]}
{"inputs": [[0.5, 0.0, 0.05], [0.0, 0.5, 0.05]]}
// grid (alpha, nx, nt) -> {"x": [...], "t": [...], "u": [[...]]}   // u is (nt, nx)
{"grid": {"alpha": 0.05, "nx": 100, "nt": 100}}
```

**Source-packaging note:** the registered `.eqx` artefact carries **no code**, but
`eqx.tree_deserialise_leaves` rebuilds the skeleton from the live `ParametricPINN` class.
So the deployment sets `code: ../python` (uploads `src/` + `mlops/`) and
`scoring_script: mlops/score.py`; `score.py` does the same `sys.path` bootstrap as the other
entrypoints so `import model` / `from mlops import serialization` resolve at inference time.

### 0. Docker smoke-test the deployment locally first (recommended)

A wrong dep or a missing inference server only shows up *after* a 20–40 min cloud provision.
Catch it locally. First confirm the inference env imports cleanly:

```powershell
# Requires Docker Desktop.
docker run --rm -v ${PWD}:/w -w /w mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04 bash -lc `
  "conda env create -f conda_inference.yml -n inf && conda run -n inf python -c 'import jax, equinox, optax, numpy, azureml_inference_server_http; print(jax.numpy.zeros(3).sum())'"
```

Then run the **whole deployment locally in Docker** and invoke it before touching the cloud
(needs `az login` + `config.json`, see Run on Azure ML step 1; `--local` builds the image
and runs the container on your machine):

```powershell
az ml online-endpoint create -f endpoint.yml --local
az ml online-deployment create -f deployment.yml --local --all-traffic
# Local invoke – both payload shapes:
az ml online-endpoint invoke --name pinn-heat-endpoint --local --request-file sample_request_points.json
az ml online-endpoint invoke --name pinn-heat-endpoint --local --request-file sample_request_grid.json
az ml online-endpoint delete --name pinn-heat-endpoint --local --yes   # clean up the local container
```

> ⚠️ **`--local` can't resolve `@latest` labels, so pin both model and environment.**
> `deployment.yml` references `model: azureml:pinn-heat@latest` and
> `environment: azureml:pinn-heat-inference-env@latest`, but a `--local` deploy does
> **not** call the workspace registry to resolve those labels. You get a `ResourceNotFoundError`
> / `System.Net.Http.HttpConnectionResponseContent` error on the environment lookup.
> This bites only on `--local`; the cloud deploy (step 3) resolves `@latest` fine.
>
> Look up the registered versions:
> ```powershell
> az ml model list --name pinn-heat --query "[].version" -o tsv
> az ml environment list --name pinn-heat-inference-env --query "[].version" -o tsv
> ```
>
> Then pin both explicitly:
> ```powershell
> az ml online-deployment create -f deployment.yml --local --all-traffic `
>   --set model=azureml:pinn-heat:<version> `
>   --set environment=azureml:pinn-heat-inference-env:<version>
> ```

Sanity-check the field against the analytical solution: at `t = 0` every prediction must
equal `sin(pi x)` exactly (the hard-constraint ansatz), and interior values should track
`u_exact = sin(pi x) exp(-alpha pi^2 t)`.

### 1. Register the inference environment

```powershell
az ml environment create -f environment_inference.yml
```

### 2. Create the endpoint

```powershell
az ml online-endpoint create -f endpoint.yml
```

> The endpoint `name` (`pinn-heat-endpoint`) must be **unique per region** within the
> subscription. If it's taken, change it in `endpoint.yml`, `deployment.yml`
> (`endpoint_name`), and the commands below.

### 3. Create the deployment

```powershell
az ml online-deployment create -f deployment.yml --all-traffic
```

`deployment.yml` serves `azureml:pinn-heat@latest` (the most recent gate-passing model).
Pin a specific version with:

```powershell
az ml online-deployment create -f deployment.yml --all-traffic `
  --set model=azureml:pinn-heat:<version>
```

### 4. Invoke it live

```powershell
az ml online-endpoint invoke --name pinn-heat-endpoint --request-file sample_request_points.json
az ml online-endpoint invoke --name pinn-heat-endpoint --request-file sample_request_grid.json
```

### 5. Consume from your own code

The endpoint is a standard key-authenticated REST API. Get its URI and key with:

```powershell
az ml online-endpoint show --name pinn-heat-endpoint --query scoring_uri -o tsv
az ml online-endpoint get-credentials --name pinn-heat-endpoint --query primaryKey -o tsv
```

POST either payload shape with a `Bearer <key>` header. See the
[docs/live-endpoint-info/](../docs/live-endpoint-info/) directory for ready-made
Python, JavaScript, and C# consumption snippets. These snippets are the default consumption examples provided by Azure.

### 6. Teardown

To delete the live endpoint, run:

```powershell
az ml online-endpoint delete --name pinn-heat-endpoint --yes
```

Confirm it's gone (`az ml online-endpoint list -o table` – it should not appear).
