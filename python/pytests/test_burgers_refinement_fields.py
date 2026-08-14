"""Schema guard for the Burgers reference-uncertainty block.

``scripts/burgers_refinement.py`` embeds a ``reference_uncertainty`` key into the
committed ``frontend/public/burgers_model.json`` (the grid-refinement error bar of
the nx=512 method-of-lines reference). This test validates that block when it is
present. It is deliberately tolerant of the key's absence so CI stays green both
before and after that user-run script has been executed.
"""

import json
import os

from mlops import config

BURGERS_JSON = os.path.join(config.REPO_ROOT, "frontend", "public", "burgers_model.json")


def _load_burgers_payload():
    with open(BURGERS_JSON, encoding="utf-8") as f:
        return json.load(f)


def test_reference_uncertainty_schema_when_present():
    """If reference_uncertainty is embedded, its schema and values are sane."""
    payload = _load_burgers_payload()
    unc = payload.get("reference_uncertainty")
    if unc is None:
        return  # not yet generated — tolerated (see module docstring)

    rel_l2 = unc["rel_l2_512_vs_2048"]
    linf = unc["linf_512_vs_2048"]
    assert isinstance(rel_l2, (int, float))
    assert isinstance(linf, (int, float))
    assert isinstance(unc["note"], str) and unc["note"]
    assert 0 < rel_l2 < 0.1
    assert linf > 0
