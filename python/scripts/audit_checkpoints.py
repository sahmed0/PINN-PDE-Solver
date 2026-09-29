"""Audit every local checkpoint, plus the committed browser model, against the real gate.

Run from ``python/``::

    uv run --group mlops python scripts/audit_checkpoints.py [--json <path>]

Read-only: it loads each ``outputs/*`` directory that holds an ``architecture.json``, scores
it with ``test_set.held_out_metrics`` (the exact slices and metric the gate uses), and does
the same for the committed ``frontend/public/pinn_model.json`` (weights cast to float32, the
precision the gate sees when it loads a ``.eqx``). Nothing under ``frontend/`` or
``outputs/`` is written.

The ``VERDICT`` line names the gate-passing checkpoint with the lowest in-distribution
``mean_rel_l2``; the committed model is reported for comparison but never eligible.
"""

import argparse
import json
import os

from mlops import config, serialization, test_set
from mlops.json_forward import model_from_heat_payload

HEAT_JSON = os.path.join(config.REPO_ROOT, "frontend", "public", "pinn_model.json")
COMMITTED = "committed"

# Gate metrics of the Azure-registered baseline model v2, as shown on
# docs/screenshots/azure-baseline-model-v2-properties.png.
REGISTERED_V2_MEAN_REL_L2 = 2.48e-4
REGISTERED_V2_MEAN_REL_L2_OOD = 6.09e-3


def _sig3(value):
    return float(f"{value:.2e}")


def _matches_registered_v2(metrics):
    return (
        _sig3(metrics["mean_rel_l2"]) == REGISTERED_V2_MEAN_REL_L2
        and _sig3(metrics["mean_rel_l2_ood"]) == REGISTERED_V2_MEAN_REL_L2_OOD
    )


def _passes(metrics):
    return (
        metrics["mean_rel_l2"] < config.MEAN_REL_L2_THRESHOLD
        and metrics["mean_rel_l2_ood"] < config.MEAN_REL_L2_OOD_THRESHOLD
    )


def _candidates():
    """Yield (name, loader) for every checkpoint directory, then the committed JSON."""
    outputs_dir = config.DEFAULT_OUTPUTS_DIR
    if os.path.isdir(outputs_dir):
        for name in sorted(os.listdir(outputs_dir)):
            model_dir = os.path.join(outputs_dir, name)
            if os.path.exists(os.path.join(model_dir, config.ARCH_FILENAME)):
                yield name, lambda d=model_dir: serialization.load_model(d)

    def load_committed():
        with open(HEAT_JSON, encoding="utf-8") as f:
            return model_from_heat_payload(json.load(f))

    yield COMMITTED, load_committed


def audit():
    rows = []
    for name, load in _candidates():
        try:
            model = load()
        except Exception as exc:  # report and keep auditing the rest
            print(f"  {name}: load failed ({exc})")
            rows.append({"name": name, "error": str(exc)})
            continue
        metrics = test_set.held_out_metrics(model)
        rows.append(
            {
                "name": name,
                "mean_rel_l2": metrics["mean_rel_l2"],
                "mean_rel_l2_ood": metrics["mean_rel_l2_ood"],
                "per_alpha_rel_l2": metrics["per_alpha_rel_l2"],
                "passed": _passes(metrics),
                "registered_v2_match": _matches_registered_v2(metrics),
            }
        )

    eligible = [r for r in rows if r["name"] != COMMITTED and r.get("passed")]
    winner = min(eligible, key=lambda r: r["mean_rel_l2"]) if eligible else None
    return {
        "threshold": config.MEAN_REL_L2_THRESHOLD,
        "threshold_ood": config.MEAN_REL_L2_OOD_THRESHOLD,
        "candidates": rows,
        "verdict": winner["name"] if winner else None,
    }


def print_table(result):
    header = f"{'candidate':<26} {'mean_rel_l2':>12} {'mean_rel_l2_ood':>16} {'gate':>5}  REGISTERED-V2 MATCH"
    print(header)
    print("-" * len(header))
    for r in result["candidates"]:
        if "error" in r:
            print(f"{r['name']:<26} {'load failed':>12}")
            continue
        print(
            f"{r['name']:<26} {r['mean_rel_l2']:>12.4e} {r['mean_rel_l2_ood']:>16.4e} "
            f"{'PASS' if r['passed'] else 'FAIL':>5}  {'yes' if r['registered_v2_match'] else 'no'}"
        )
    print(
        f"\nThresholds: mean_rel_l2 < {result['threshold']:.0e}, "
        f"mean_rel_l2_ood < {result['threshold_ood']:.0e}"
    )
    if result["verdict"] is None:
        print("VERDICT: none — retrain required")
    else:
        print(f"VERDICT: {result['verdict']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="also write the full result to this path")
    args = parser.parse_args()

    result = audit()
    print_table(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
