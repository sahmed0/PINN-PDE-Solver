"""Slow accuracy-regression guard for the forward heat-equation PINN.

Every other test pins *structure* (the ansatz identities, the export contract, the
gate wiring) but nothing asserts the model actually *trains to accuracy*. This does:
a seeded, reduced-budget run must reach a mean relative-L2 below a fixed threshold,
so a regression that quietly wrecks convergence (a broken optimiser schedule, a bad
loss change) fails the build instead of shipping.

It is marked ``slow`` and runs in the nightly workflow, not the push CI job.
"""

import jax.random as jr
import pytest

from pinn.analytical import evaluate
from pinn.model import ParametricPINN
from pinn.train import train

# Threshold tuned empirically, the project's usual way (cf. config.py gate
# thresholds): this exact seeded 2000-epoch / 1000-collocation run measured
# mean rel-L2 = 9.33e-3 (data seed 0) and 8.92e-3 (data seed 1). The threshold
# is set at ~3x the larger measurement so the pinned run clears it comfortably
# and normal float32 run-to-run wobble never flakes, while a real convergence
# regression (which lands one to two orders of magnitude worse) still trips it.
ACCURACY_THRESHOLD = 2.8e-2


@pytest.mark.slow
def test_training_reaches_expected_accuracy():
    model = ParametricPINN(jr.PRNGKey(42))
    model = train(model, jr.PRNGKey(0), epochs=2000,
                  num_collocation=1000, validate=False)
    mean_rel_l2 = evaluate(model)["mean_rel_l2"]
    assert mean_rel_l2 < ACCURACY_THRESHOLD, (
        f"mean rel-L2 {mean_rel_l2:.3e} exceeds regression threshold "
        f"{ACCURACY_THRESHOLD:.3e}"
    )
