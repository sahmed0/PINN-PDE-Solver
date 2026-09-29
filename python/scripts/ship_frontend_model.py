"""Promote a checkpoint directory to the browser model, with provenance and parity vectors.

Run from ``python/``::

    uv run --group mlops python scripts/ship_frontend_model.py --model-dir <path> \\
        [--seed 42] [--command "<the training command>"]

It re-scores the checkpoint on the real gate slices (``test_set.held_out_metrics``), refuses
to ship if either threshold is missed, then writes ``frontend/public/pinn_model.json`` via
``export_to_json`` with a ``provenance`` block recording the source checkpoint, the seed and
command that trained it, and the gate metrics just measured. Pick the directory with
``scripts/audit_checkpoints.py``.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime

from mlops import config, serialization, test_set
from pinn import train as train_mod

HEAT_JSON = os.path.join(config.REPO_ROOT, "frontend", "public", "pinn_model.json")
DEFAULT_COMMAND = (
    "uv run --group mlops python mlops/train_entry.py --epochs 20000 --config-name baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True, help="checkpoint directory to ship")
    parser.add_argument("--seed", type=int, default=42, help="seed the checkpoint was trained with")
    parser.add_argument(
        "--command", default=DEFAULT_COMMAND, help="training command that produced it"
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    model = serialization.load_model(model_dir)
    metrics = test_set.held_out_metrics(model)
    mean_rel_l2 = metrics["mean_rel_l2"]
    mean_rel_l2_ood = metrics["mean_rel_l2_ood"]

    if not (
        mean_rel_l2 < config.MEAN_REL_L2_THRESHOLD
        and mean_rel_l2_ood < config.MEAN_REL_L2_OOD_THRESHOLD
    ):
        sys.exit(
            f"Refusing to ship {model_dir}: gate FAILED "
            f"(mean_rel_l2={mean_rel_l2:.4e} vs < {config.MEAN_REL_L2_THRESHOLD}, "
            f"mean_rel_l2_ood={mean_rel_l2_ood:.4e} vs < {config.MEAN_REL_L2_OOD_THRESHOLD})."
        )

    provenance = {
        "source_checkpoint": os.path.basename(model_dir),
        "seed": args.seed,
        "command": args.command,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gate": {
            "mean_rel_l2": mean_rel_l2,
            "mean_rel_l2_ood": mean_rel_l2_ood,
            "threshold": config.MEAN_REL_L2_THRESHOLD,
            "threshold_ood": config.MEAN_REL_L2_OOD_THRESHOLD,
            "alphas_interp": list(config.TEST_ALPHAS_INTERP),
            "alphas_ood": list(config.TEST_ALPHAS_OOD),
            "grid": [config.TEST_GRID_NX, config.TEST_GRID_NT],
            "passed": True,
        },
        "note": "Gate metrics recomputed locally from this checkpoint by scripts/ship_frontend_model.py.",
    }

    train_mod.export_to_json(model, HEAT_JSON, provenance=provenance)

    with open(HEAT_JSON, encoding="utf-8") as f:
        shipped = json.load(f)
    n_params = sum(
        len(layer["bias"]) + sum(len(row) for row in layer["weight"]) for layer in shipped["layers"]
    )
    print(f"Shipped {provenance['source_checkpoint']} -> {HEAT_JSON}")
    print(f"  parameters      : {n_params}")
    print(f"  mean_rel_l2     : {shipped['provenance']['gate']['mean_rel_l2']:.4e}")
    print(f"  mean_rel_l2_ood : {shipped['provenance']['gate']['mean_rel_l2_ood']:.4e}")
    print(f"  test_vectors    : {len(shipped['test_vectors']['inputs'])}")


if __name__ == "__main__":
    main()
