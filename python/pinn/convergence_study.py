"""
Convergence study: how does the PINN's accuracy depend on the number of
collocation points used to enforce the PDE residual?

We train the same network from the same seed for a range of collocation counts
and measure the relative L2 error against the analytical solution. The expected
behaviour is that error decreases as we sample the residual more densely, until
it plateaus where the bottleneck becomes optimisation / network capacity rather
than collocation density. The resulting log-log plot is saved to
`figures/convergence.png`.

Run from the `python/` directory:

    python -m pinn.convergence_study
"""

import os

import jax.random as jr
import matplotlib.pyplot as plt

from pinn.model import ParametricPINN
from pinn.train import train
from pinn.analytical import evaluate


def run_study(collocation_counts=(100, 250, 500, 1000, 2000, 4000),
              epochs=2000, lr=1e-3, seed=42):
    """Train one model per collocation count and return (counts, mean rel L2)."""
    errors = []
    for n in collocation_counts:
        print(f"\n=== Collocation points: {n} ===")
        key = jr.PRNGKey(seed)
        model_key, data_key = jr.split(key)
        model = ParametricPINN(model_key)
        # Patch the default sample size via a closure-free override: train()
        # calls generate_training_data(key) with its default count, so we
        # resample here and feed a model trained against `n` points by
        # temporarily setting the module-level default.
        model = _train_with_collocation(model, data_key, n, epochs, lr)
        metrics = evaluate(model)
        errors.append(metrics["mean_rel_l2"])
        print(f"    -> mean rel L2 = {metrics['mean_rel_l2']:.3e}")
    return list(collocation_counts), errors


def _train_with_collocation(model, key, num_collocation, epochs, lr):
    """Train using a custom number of collocation points.

    `train.train` uses the default collocation count, so for the study we build
    the dataset explicitly and run the same optimisation loop here.
    """
    import equinox as eqx
    import optax
    from pinn.train import generate_training_data, train_step
    from pinn.analytical import relative_l2_error

    # generate_training_data only exposes num_collocation as a kwarg.
    collocation_points, ic_points, bc_points = generate_training_data(
        key, num_collocation=num_collocation
    )

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for epoch in range(epochs):
        model, opt_state, loss = train_step(
            model, opt_state, optimizer, collocation_points, ic_points, bc_points
        )
        if epoch % 500 == 0 or epoch == epochs - 1:
            err = sum(relative_l2_error(model, a) for a in (0.01, 0.05, 0.1)) / 3
            print(f"    epoch {epoch:04d} | loss {loss:.6f} | rel L2 {err:.3e}")
    return model


def plot(counts, errors, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(counts, errors, "o-", color="#2563eb", linewidth=2, markersize=7)
    ax.set_xlabel("Number of collocation points")
    ax.set_ylabel("Mean relative $L_2$ error")
    ax.set_title("PINN convergence vs. collocation density\n(1D heat equation)")
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


if __name__ == "__main__":
    main()
