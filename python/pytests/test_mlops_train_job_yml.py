"""Guard: the Azure ML training job's experiment_name must equal
config.EXPERIMENT_NAME, otherwise Azure splits runs across two experiments.
"""

import os

import yaml

from mlops import config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_JOB_YML = os.path.join(REPO_ROOT, "mlops", "train_job.yml")


def test_train_job_experiment_name_matches_config():
    with open(TRAIN_JOB_YML, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    assert doc["experiment_name"] == config.EXPERIMENT_NAME
