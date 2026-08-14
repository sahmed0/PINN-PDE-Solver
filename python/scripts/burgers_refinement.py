"""Quantify the Burgers reference solution's own discretisation error.

Run once from ``python/`` (~5-15 min CPU)::

    uv run python scripts/burgers_refinement.py

The shipped ``frontend/public/burgers_model.json`` embeds a method-of-lines
reference integrated on an nx=512 spatial grid, and quotes the PINN's rel-L2
against it. This script recomputes that reference on a 4x-finer nx=2048 grid and
reports how far the two differ (rel-L2 and L-infinity on the shared output grid).
That difference is the reference's *own* error bar — the fraction of the quoted
rel-L2 that reflects reference discretisation rather than the model.

The result is written back into ``burgers_model.json`` **in place and additively**
as a top-level ``reference_uncertainty`` key. The model weights and every other
field are left byte-identical (only a new key is added), so the frontend forward
pass and the vitest parity vectors are unaffected.
"""

import json
import os

from mlops import config
from pinn.burgers import reference_uncertainty

BURGERS_JSON = os.path.join(config.REPO_ROOT, "frontend", "public", "burgers_model.json")


def main():
    print("Computing Burgers reference grid-refinement error (nx=512 vs nx=2048)...")
    print("(the nx=2048 method-of-lines integration is the slow part)")
    unc = reference_uncertainty(nx_coarse=512, nx_fine=2048, nx_out=100, nt=100)

    print(f"    rel L2 (512 vs 2048) : {unc['rel_l2_512_vs_2048']:.3e}")
    print(f"    L-infinity           : {unc['linf_512_vs_2048']:.3e}")

    with open(BURGERS_JSON, encoding="utf-8") as f:
        payload = json.load(f)

    payload["reference_uncertainty"] = unc

    with open(BURGERS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"\nUpdated {BURGERS_JSON} (added reference_uncertainty; weights unchanged).")

    # Machine-readable line so the README numbers can be filled from the console log.
    print(json.dumps({"reference_uncertainty": unc}))


if __name__ == "__main__":
    main()
