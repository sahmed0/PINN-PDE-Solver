# Local Docker smoke test of the Azure INFERENCE environment + score path + latency.
#
# Requires Docker Desktop. Run it from anywhere (it resolves the repo root itself):
#     pwsh python/scripts/docker_smoke.ps1
#
# It builds the conda_inference env inside the same AzureML base image the managed
# deployment uses, mounts the whole repo at /w (so python/ and frontend/public/ are
# visible), and — crucially — pip-installs nothing, exactly like the managed online
# deployment. With python/ on PYTHONPATH it runs scripts/smoke_score.py, which rebuilds a
# scoring artefact from the committed pinn_model.json, exercises score.init()/run() plus a
# {"health": true} echo, and prints p50/p95 scoring latency.
#
# A clean run ending in "SMOKE OK" means the inference env resolves and the score path
# works end-to-end. First run pulls the base image and solves the conda env (~10-20 min).

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot/../..").Path
$image = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"

Write-Host "Repo root: $repoRoot"
Write-Host "Image:     $image"

docker run --rm -v "${repoRoot}:/w" -w /w $image bash -lc @'
set -e
conda env create -f mlops/conda_inference.yml -n inf
export PYTHONPATH=/w/python
conda run --no-capture-output -n inf python /w/python/scripts/smoke_score.py
'@
