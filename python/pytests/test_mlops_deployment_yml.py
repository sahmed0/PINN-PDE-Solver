"""Guards on the Azure ML endpoint/deployment YAML. Pure text checks
(no yaml dep), mirroring test_mlops_train_job_yml.py:

  - deployment.yml `model` references config.REGISTERED_MODEL_NAME.
  - `scoring_script` resolves to an existing file under `code` (python/mlops/score.py).
  - deployment.yml `endpoint_name` equals endpoint.yml `name`.
  - single instance (cap costs).
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops import config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLOPS_DIR = os.path.join(REPO_ROOT, "mlops")
DEPLOYMENT_YML = os.path.join(MLOPS_DIR, "deployment.yml")
ENDPOINT_YML = os.path.join(MLOPS_DIR, "endpoint.yml")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _field(text, key):
    # Tolerate leading indentation (e.g. fields under code_configuration).
    m = re.search(rf"^\s*{re.escape(key)}:\s*(\S+)", text, re.MULTILINE)
    assert m is not None, f"{key} not found"
    return m.group(1)


def test_deployment_model_references_registered_name():
    text = _read(DEPLOYMENT_YML)
    model = _field(text, "model")
    # e.g. "azureml:pinn-heat@latest" must reference the registered model name.
    assert config.REGISTERED_MODEL_NAME in model


def test_scoring_script_resolves_to_existing_file():
    text = _read(DEPLOYMENT_YML)
    code = _field(text, "code")              # under code_configuration (indented)
    script = _field(text, "scoring_script")
    # `code` is relative to mlops/ directory; join then check the script exists.
    code_dir = os.path.normpath(os.path.join(MLOPS_DIR, code))
    script_path = os.path.join(code_dir, script.replace("/", os.sep))
    assert os.path.isfile(script_path), f"scoring script missing: {script_path}"


def test_endpoint_name_matches_between_files():
    dep_ep = _field(_read(DEPLOYMENT_YML), "endpoint_name")
    ep_name = _field(_read(ENDPOINT_YML), "name")
    assert dep_ep == ep_name


def test_single_instance():
    assert _field(_read(DEPLOYMENT_YML), "instance_count") == "1"


def test_endpoint_auth_mode_is_key():
    assert _field(_read(ENDPOINT_YML), "auth_mode") == "key"
