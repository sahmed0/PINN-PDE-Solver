"""Paired ablation: does input normalisation matter, and is the failure an alpha-average?

Run from ``python/`` (~10-20 min CPU)::

    uv run python scripts/ablate_normalisation.py

Two arms share seed 42, the same initial weights and the same collocation points. The only
difference is the input normalisation:

- normalised: ``ParametricPINN(model_key)`` (the defaults, as shipped);
- raw: the same model with ``input_center = (0, 0, 0)`` and ``input_scale = (1, 1, 1)``.

Two metrics:

1. Accuracy vs alpha: relative L2 against the analytical solution at five alphas.
2. Alpha-sensitivity spread: on a fixed interior probe grid, S(x, t) = max_alpha u - min_alpha u
   over the same five alphas, for each arm and for the analytical solution. A model that has
   collapsed toward an alpha-averaged solution has a mean S well below the analytic one.

Writes ``figures/normalisation_ablation.png`` and prints a markdown table plus a KEY NUMBERS
block with the figures quoted in the README.
"""

import os

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from pinn.analytical import relative_l2_error, u_exact
from pinn.model import ParametricPINN
from pinn.train import train

SEED = 42
EPOCHS = 2000
NUM_COLLOCATION = 4000
LR = 1e-3
ALPHAS = (0.01, 0.025, 0.05, 0.075, 0.1)
PROBE_N = 21
PROBE_X = (-0.9, 0.9)
PROBE_T = (0.05, 1.0)

ARMS = {
    "normalised": {},
    "raw": {"input_center": (0.0, 0.0, 0.0), "input_scale": (1.0, 1.0, 1.0)},
}
COLORS = {"analytic": "#6b7280", "normalised": "#2563eb", "raw": "#dc2626"}

FIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "figures", "normalisation_ablation.png")
)


def probe_grid():
    x = jnp.linspace(*PROBE_X, PROBE_N)
    t = jnp.linspace(*PROBE_T, PROBE_N)
    X, T = jnp.meshgrid(x, t, indexing="xy")
    return X.reshape(-1), T.reshape(-1)


def mean_spread(field_fn):
    """Mean over the probe grid of max_alpha u - min_alpha u."""
    x, t = probe_grid()
    fields = jnp.stack([field_fn(x, t, a) for a in ALPHAS])
    return float(jnp.mean(jnp.max(fields, axis=0) - jnp.min(fields, axis=0)))


def model_field(model):
    def field(x, t, alpha):
        inputs = jnp.stack([x, t, jnp.full_like(x, alpha)], axis=1)
        return jax.vmap(model)(inputs).reshape(-1)

    return field


def run(epochs=EPOCHS, fig_path=FIG_PATH):
    model_key, train_key = jr.split(jr.PRNGKey(SEED))

    results = {}
    for name, kwargs in ARMS.items():
        print(f"\n=== Training arm: {name} ({epochs} epochs, {NUM_COLLOCATION} collocation) ===")
        model = ParametricPINN(model_key, **kwargs)
        model = train(
            model,
            train_key,
            epochs=epochs,
            lr=LR,
            validate=False,
            num_collocation=NUM_COLLOCATION,
        )
        rel_l2 = [float(relative_l2_error(model, a, 100, 100)) for a in ALPHAS]
        results[name] = {
            "rel_l2": rel_l2,
            "mean_rel_l2": sum(rel_l2) / len(rel_l2),
            "spread": mean_spread(model_field(model)),
        }

    analytic_spread = mean_spread(u_exact)
    for r in results.values():
        r["spread_pct"] = 100.0 * r["spread"] / analytic_spread

    print_report(results, analytic_spread)
    plot(results, analytic_spread, fig_path, epochs)
    return results, analytic_spread


def print_report(results, analytic_spread):
    norm, raw = results["normalised"], results["raw"]

    print("\n## Metric 1 - relative L2 vs alpha\n")
    print("| alpha | normalised | raw | raw / normalised |")
    print("|---|---|---|---|")
    for a, en, er in zip(ALPHAS, norm["rel_l2"], raw["rel_l2"], strict=True):
        print(f"| {a:g} | {en:.3e} | {er:.3e} | {er / en:.1f}x |")
    ratio = raw["mean_rel_l2"] / norm["mean_rel_l2"]
    print(f"| **mean** | {norm['mean_rel_l2']:.3e} | {raw['mean_rel_l2']:.3e} | {ratio:.1f}x |")

    print(f"\n## Metric 2 - alpha-sensitivity spread ({PROBE_N}x{PROBE_N} interior probe grid)\n")
    print("| field | mean S | % of analytic |")
    print("|---|---|---|")
    print(f"| analytic | {analytic_spread:.4e} | 100.0% |")
    for name, r in results.items():
        print(f"| {name} | {r['spread']:.4e} | {r['spread_pct']:.1f}% |")

    print("\nKEY NUMBERS")
    print(f"  mean rel-L2 normalised           : {norm['mean_rel_l2']:.3e}")
    print(f"  mean rel-L2 raw                  : {raw['mean_rel_l2']:.3e}")
    print(f"  raw / normalised mean rel-L2     : {ratio:.1f}x")
    print(f"  analytic mean spread             : {analytic_spread:.4e}")
    print(f"  normalised spread (% of analytic): {norm['spread_pct']:.1f}%")
    print(f"  raw spread (% of analytic)       : {raw['spread_pct']:.1f}%")


def plot(results, analytic_spread, out_path, epochs=EPOCHS):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

    for name, r in results.items():
        ax_l.semilogy(
            ALPHAS, r["rel_l2"], "o-", color=COLORS[name], linewidth=2, markersize=7, label=name
        )
    ax_l.set_xlabel(r"Thermal diffusivity $\alpha$")
    ax_l.set_ylabel("Relative $L_2$ error")
    ax_l.set_title(r"Accuracy vs $\alpha$")
    ax_l.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_l.legend()

    names = ["analytic", "normalised", "raw"]
    values = [analytic_spread] + [results[n]["spread"] for n in names[1:]]
    pcts = [100.0] + [results[n]["spread_pct"] for n in names[1:]]
    bars = ax_r.bar(names, values, color=[COLORS[n] for n in names])
    for bar, pct in zip(bars, pcts, strict=True):
        ax_r.annotate(
            f"{pct:.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
        )
    ax_r.set_ylabel(r"Mean spread $\max_\alpha u - \min_\alpha u$")
    ax_r.set_title(r"$\alpha$-sensitivity reproduced (% of analytic)")
    ax_r.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax_r.set_ylim(0, max(values) * 1.15)

    fig.suptitle(
        f"Input normalisation ablation (seed {SEED}, same init and collocation, "
        f"{epochs} epochs each)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    run()
