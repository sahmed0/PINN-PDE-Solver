"""
Convergence study: how does the PINN's accuracy depend on the number of
collocation points used to enforce the PDE residual?

We train the same network from the same seed for a range of collocation counts
and measure the relative L2 error against the analytical solution. The expected
behaviour is that error decreases as we sample the residual more densely, until
it plateaus where the bottleneck becomes optimisation / network capacity rather
than collocation density. The resulting log-log plot is saved to
`figures/convergence.png`.

This is a *reduced-budget* study: each point is trained for 2000 epochs, whereas
the production headline model runs for 20000. The finding here is the *slope*
(how error scales with collocation density), not the absolute error level, which
would be lower at the full budget. Each model is trained through the exact same
`train()` production loop (cosine-annealed Adam), so the trend reflects how the
shipped model is actually optimised.

Run from the `python/` directory:

    python -m pinn.convergence_study
"""

import json
import os

import jax.random as jr
import matplotlib.pyplot as plt

from pinn.analytical import evaluate
from pinn.model import ParametricPINN
from pinn.train import train


def run_study(collocation_counts=(100, 250, 500, 1000, 2000, 4000),
              epochs=2000, lr=1e-3, seed=42):
    """Train one model per collocation count and return (counts, mean rel L2).

    Uses the production `train()` loop (cosine-annealed Adam) so the measured
    trend reflects the real optimiser, not a stand-in. Validation printing is
    disabled per step; final accuracy is measured once via `evaluate`.
    """
    errors = []
    for n in collocation_counts:
        print(f"\n=== Collocation points: {n} ===")
        key = jr.PRNGKey(seed)
        model_key, data_key = jr.split(key)
        model = ParametricPINN(model_key)
        model = train(model, data_key, epochs=epochs, lr=lr,
                      num_collocation=n, validate=False)
        metrics = evaluate(model)
        errors.append(metrics["mean_rel_l2"])
        print(f"    -> mean rel L2 = {metrics['mean_rel_l2']:.3e}")
    return list(collocation_counts), errors


def plot(counts, errors, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(counts, errors, "o-", color="#2563eb", linewidth=2, markersize=7)
    ax.set_xlabel("Number of collocation points")
    ax.set_ylabel("Mean relative $L_2$ error")
    ax.set_title("PINN convergence vs. collocation density\n"
                 "(1D heat equation, reduced-budget 2000-epoch study)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure to {out_path}")


def main():
    counts, errors = run_study()
    out_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "figures", "convergence.png")
    )
    plot(counts, errors, out_path)
    # Machine-readable line so README numbers can be filled from the console log.
    print(json.dumps({"counts": counts, "errors": errors}))


if __name__ == "__main__":
    main()
