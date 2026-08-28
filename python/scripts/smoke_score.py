"""In-container score-path smoke test for the Azure inference environment.

Run by ``scripts/docker_smoke.ps1`` inside the conda_inference env, with ``python/`` on
``PYTHONPATH`` (the container mounts the repo but pip-installs nothing — same situation as
the managed online deployment). It:

  1. rebuilds a scoring artefact from the committed ``frontend/public/pinn_model.json``
     (via ``mlops.json_forward.model_from_heat_payload`` + ``serialization.save_model``) —
     no ``model.eqx`` checkpoint needed;
  2. points ``AZUREML_MODEL_DIR`` at it and exercises ``score.init()`` + ``score.run(...)``
     for a points request and a ``{"health": true}`` echo;
  3. measures local-container scoring latency: 3 warmups, then 50 timed points requests,
     printing p50/p95 in ms.

A clean run (predictions returned, health "ok", latency printed) means the inference env
and the score path are wired correctly.
"""

import json
import os
import statistics
import tempfile
import time

from mlops import config, score, serialization
from mlops.json_forward import model_from_heat_payload

HEAT_JSON = os.path.join(config.REPO_ROOT, "frontend", "public", "pinn_model.json")
POINTS_REQUEST = {"inputs": [[0.5, 0.5, 0.05]]}


def main():
    with open(HEAT_JSON, encoding="utf-8") as f:
        payload = json.load(f)

    with tempfile.TemporaryDirectory() as model_dir:
        # 1. Rebuild the scoring artefact straight from the committed JSON.
        model = model_from_heat_payload(payload)
        width = len(payload["layers"][0]["weight"])
        depth = len(payload["layers"]) - 1
        serialization.save_model(model, model_dir, width, depth)
        os.environ["AZUREML_MODEL_DIR"] = model_dir

        # 2. Exercise the score path.
        score.init()
        resp = score.run(POINTS_REQUEST)
        assert "predictions" in resp, resp
        print(f"score.run(points) -> {resp}")

        health = score.run({"health": True})
        assert health.get("status") == "ok", health
        print(f"score.run(health) -> {health}")

        # 3. Latency: warm up, then time 50 points requests.
        for _ in range(3):
            score.run(POINTS_REQUEST)
        timings_ms = []
        for _ in range(50):
            t0 = time.perf_counter()
            score.run(POINTS_REQUEST)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

        timings_ms.sort()
        p50 = statistics.median(timings_ms)
        p95 = timings_ms[int(0.95 * (len(timings_ms) - 1))]
        print(f"latency (local container, 50 reqs): p50={p50:.2f} ms  p95={p95:.2f} ms")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
