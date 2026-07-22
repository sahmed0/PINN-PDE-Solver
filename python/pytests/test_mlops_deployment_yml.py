"""Guards on the Azure ML endpoint/deployment YAML, parsed with yaml.safe_load:

  - deployment.yml `model` references config.REGISTERED_MODEL_NAME.
  - `scoring_script` resolves to an existing file under `code` (python/mlops/score.py).
  - deployment.yml `endpoint_name` equals endpoint.yml `name`.
  - single instance (cap costs).
  - endpoint auth_mode is `key`.
"""

import os

import yaml

from mlops import config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLOPS_DIR = os.path.join(REPO_ROOT, "mlops")
DEPLOYMENT_YML = os.path.join(MLOPS_DIR, "deployment.yml")
ENDPOINT_YML = os.path.join(MLOPS_DIR, "endpoint.yml")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_deployment_model_references_registered_name():
    dep = _load(DEPLOYMENT_YML)
    # e.g. "azureml:pinn-heat@latest" must reference the registered model name.
    assert config.REGISTERED_MODEL_NAME in dep["model"]


def test_scoring_script_resolves_to_existing_file():
    code_cfg = _load(DEPLOYMENT_YML)["code_configuration"]
    # `code` is relative to mlops/ directory; join then check the script exists.
    code_dir = os.path.normpath(os.path.join(MLOPS_DIR, code_cfg["code"]))
    script_path = os.path.join(code_dir, code_cfg["scoring_script"].replace("/", os.sep))
    assert os.path.isfile(script_path), f"scoring script missing: {script_path}"


def test_endpoint_name_matches_between_files():
    assert _load(DEPLOYMENT_YML)["endpoint_name"] == _load(ENDPOINT_YML)["name"]


def test_single_instance():
    assert _load(DEPLOYMENT_YML)["instance_count"] == 1


def test_endpoint_auth_mode_is_key():
    assert _load(ENDPOINT_YML)["auth_mode"] == "key"
