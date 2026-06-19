"""Guard: the Azure ML training job's experiment_name must equal
config.EXPERIMENT_NAME, otherwise Azure splits runs across two experiments.
Pure text check — no yaml dep.
"""

import os
import re

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops import config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_JOB_YML = os.path.join(REPO_ROOT, "mlops", "train_job.yml")


def test_train_job_experiment_name_matches_config():
    with open(TRAIN_JOB_YML, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^experiment_name:\s*(\S+)", text, re.MULTILINE)
    assert m is not None, "train_job.yml has no experiment_name"
    assert m.group(1) == config.EXPERIMENT_NAME
