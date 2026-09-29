"""Regenerate every headline figure quoted in the README and the app from the shipped artefacts.

Run from ``python/``::

    uv run --group mlops python scripts/report_headline_metrics.py

Read-only and training-free: it loads the three committed browser payloads under
``frontend/public/`` and prints one markdown table per section, then a ``KEY NUMBERS`` block
listing exactly the values the README quotes.

1. Shipped heat model on the gate's slices (the exact metric ``test_set.held_out_metrics`` uses).
2. A dense 19-point alpha sweep over the whole training band.
3. The training-time validation mean over ``train()``'s default ``val_alphas``.
4. The PDE residual at uniformly sampled collocation points.
5. Burgers, from the reference field embedded in ``burgers_model.json`` (no re-solve).
6. The inverse demo's estimate, spread and Cramér-Rao bound.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from mlops import config, test_set
from mlops.json_forward import model_from_heat_payload
from pinn import analytical
from pinn.json_forward import forward_from_payload
from pinn.physics import heat_equation_residual

PUBLIC_DIR = os.path.join(config.REPO_ROOT, "frontend", "public")

DENSE_ALPHAS = np.linspace(0.01, 0.1, 19)
TRAIN_VAL_ALPHAS = (0.01, 0.05, 0.1)  # train()'s default val_alphas
RESIDUAL_POINTS = 2000
RESIDUAL_SEED = 0
SHOCK_HALF_WIDTH = 0.1  # the Burgers "away from the front" mask is |x| >= this


def _load(name):
    with open(os.path.join(PUBLIC_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    print("\n".join(lines))


def heat_gate(model, payload):
    metrics = test_set.held_out_metrics(model)
    print("## 1. Shipped heat model - gate slices\n")
    print(f"Source checkpoint: `{payload['provenance']['source_checkpoint']}`\n")
    rows = []
    for a in metrics["alphas_interp"] + metrics["alphas_ood"]:
        slice_name = "in-dist" if a in metrics["alphas_interp"] else "OOD"
        rows.append(
            [
                f"{a}",
                slice_name,
                f"{metrics['per_alpha_rel_l2'][f'{a}']:.4e}",
                f"{metrics['per_alpha_linf'][f'{a}']:.4e}",
            ]
        )
    _table(["alpha", "slice", "rel L2", "L-inf"], rows)
    print(f"\nIn-dist mean rel L2: {metrics['mean_rel_l2']:.4e}")
    print(f"OOD mean rel L2:     {metrics['mean_rel_l2_ood']:.4e}\n")
    return metrics


def dense_sweep(model):
    errs = [float(analytical.relative_l2_error(model, float(a), 100, 100)) for a in DENSE_ALPHAS]
    print("## 2. Dense alpha sweep (19 alphas over [0.01, 0.1], 100 x 100 grid)\n")
    _table(
        ["alpha", "rel L2"],
        [[f"{a:.4f}", f"{e:.4e}"] for a, e in zip(DENSE_ALPHAS, errs, strict=True)],
    )
    worst, best = int(np.argmax(errs)), int(np.argmin(errs))
    result = {
        "mean": float(np.mean(errs)),
        "worst_alpha": float(DENSE_ALPHAS[worst]),
        "worst": errs[worst],
        "best_alpha": float(DENSE_ALPHAS[best]),
        "best": errs[best],
    }
    print(f"\nMean: {result['mean']:.4e}")
    print(f"Worst: {result['worst']:.4e} at alpha = {result['worst_alpha']:.4f}")
    print(f"Best:  {result['best']:.4e} at alpha = {result['best_alpha']:.4f}\n")
    return result


def training_validation(model):
    errs = [float(analytical.relative_l2_error(model, a, 100, 100)) for a in TRAIN_VAL_ALPHAS]
    print("## 3. Training-time validation mean (train()'s val_alphas)\n")
    _table(
        ["alpha", "rel L2"],
        [[f"{a}", f"{e:.4e}"] for a, e in zip(TRAIN_VAL_ALPHAS, errs, strict=True)],
    )
    mean = float(np.mean(errs))
    print(f"\nMean: {mean:.4e}\n")
    return mean


def pde_residual(model):
    rng = np.random.default_rng(RESIDUAL_SEED)
    x = rng.uniform(-1.0, 1.0, RESIDUAL_POINTS)
    t = rng.uniform(0.0, 1.0, RESIDUAL_POINTS)
    alpha = rng.uniform(0.01, 0.1, RESIDUAL_POINTS)
    res = jax.vmap(heat_equation_residual, in_axes=(None, 0, 0, 0))(
        model,
        jnp.asarray(x, jnp.float32),
        jnp.asarray(t, jnp.float32),
        jnp.asarray(alpha, jnp.float32),
    )
    mse = float(jnp.mean(res**2))
    rms = float(np.sqrt(mse))
    print(
        f"## 4. PDE residual ({RESIDUAL_POINTS} uniform (x, t, alpha) points, seed {RESIDUAL_SEED})\n"
    )
    _table(["RMS", "MSE"], [[f"{rms:.4e}", f"{mse:.4e}"]])
    print()
    return {"rms": rms, "mse": mse}


def burgers(payload):
    ref = payload["reference"]
    x = np.asarray(ref["x"], dtype=np.float64)
    t = np.asarray(ref["t"], dtype=np.float64)
    u_ref = np.asarray(ref["u"], dtype=np.float64)  # indexed [t][x]
    X, T = np.meshgrid(x, t, indexing="xy")  # (nt, nx): rows are time
    inputs = np.stack([X.ravel(), T.ravel()], axis=1)
    u_pinn = forward_from_payload(payload, inputs).reshape(len(t), len(x))

    diff = u_pinn - u_ref
    rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(u_ref))
    linf = float(np.max(np.abs(diff)))
    away = np.abs(x) >= SHOCK_HALF_WIDTH
    rel_l2_away = float(np.linalg.norm(diff[:, away]) / np.linalg.norm(u_ref[:, away]))
    ref_rel_l2 = payload["reference_uncertainty"]["rel_l2_512_vs_2048"]
    ref_share = ref_rel_l2 / payload["rel_l2"] * 100

    print("## 5. Burgers (committed burgers_model.json, embedded reference)\n")
    _table(
        ["quantity", "value"],
        [
            ["stored rel L2", f"{payload['rel_l2']:.5e}"],
            ["recomputed rel L2 (self-check)", f"{rel_l2:.5e}"],
            ["stored L-inf", f"{payload['linf']:.5e}"],
            ["recomputed L-inf (self-check)", f"{linf:.5e}"],
            [f"rel L2 excluding |x| < {SHOCK_HALF_WIDTH}", f"{rel_l2_away:.4e}"],
            ["reference rel L2 (nx 512 vs 2048)", f"{ref_rel_l2:.4e}"],
            [
                "reference L-inf (nx 512 vs 2048)",
                f"{payload['reference_uncertainty']['linf_512_vs_2048']:.4e}",
            ],
            ["reference share of PINN rel L2", f"{ref_share:.2f}%"],
        ],
    )
    drift = max(abs(rel_l2 - payload["rel_l2"]), abs(linf - payload["linf"]))
    print(
        f"\nSelf-check max |recomputed - stored|: {drift:.2e} ({'OK' if drift < 1e-5 else 'MISMATCH'})\n"
    )
    return {
        "rel_l2": payload["rel_l2"],
        "linf": payload["linf"],
        "rel_l2_away": rel_l2_away,
        "ref_rel_l2": ref_rel_l2,
        "ref_share": ref_share,
        "self_check_ok": drift < 1e-5,
    }


def inverse(payload):
    ratio = payload["alpha_std"] / payload["crlb_std"]
    ratio_se = ratio / np.sqrt(2 * (payload["n_seeds"] - 1))
    print("## 6. Inverse (committed inverse_model.json)\n")
    _table(
        ["quantity", "value"],
        [
            ["alpha_true", f"{payload['alpha_true']}"],
            ["alpha_est", f"{payload['alpha_est']:.6f}"],
            ["alpha_std", f"{payload['alpha_std']:.4e}"],
            ["crlb_std", f"{payload['crlb_std']:.4e}"],
            ["n_seeds", f"{payload['n_seeds']}"],
            ["alpha_std / crlb_std", f"{ratio:.3f}"],
            ["standard error of ratio", f"{ratio_se:.3f}"],
        ],
    )
    print()
    return {"ratio": ratio, "ratio_se": float(ratio_se)}


def main():
    heat_payload = _load("pinn_model.json")
    model = model_from_heat_payload(heat_payload)

    gate = heat_gate(model, heat_payload)
    dense = dense_sweep(model)
    val_mean = training_validation(model)
    residual = pde_residual(model)
    burg = burgers(_load("burgers_model.json"))
    inv = inverse(_load("inverse_model.json"))

    print("## KEY NUMBERS\n")
    print("```")
    print(f"heat gate in-dist mean rel L2      {gate['mean_rel_l2']:.2e}")
    print(f"heat gate OOD mean rel L2          {gate['mean_rel_l2_ood']:.2e}")
    print(f"heat dense-sweep mean rel L2       {dense['mean']:.2e}")
    print(
        f"heat dense-sweep worst rel L2      {dense['worst']:.2e} at alpha = {dense['worst_alpha']:.3f}"
    )
    print(
        f"heat dense-sweep best rel L2       {dense['best']:.2e} at alpha = {dense['best_alpha']:.3f}"
    )
    print(f"heat training-val mean rel L2      {val_mean:.2e}")
    print(f"heat PDE residual RMS              {residual['rms']:.2e}")
    print(f"burgers rel L2                     {burg['rel_l2']:.3f}")
    print(f"burgers L-inf                      {burg['linf']:.2f}")
    print(f"burgers rel L2 away from front     {burg['rel_l2_away']:.2e}")
    print(f"burgers reference rel L2           {burg['ref_rel_l2']:.2e}")
    print(f"burgers reference share            {burg['ref_share']:.1f}%")
    print(f"inverse alpha_std / crlb_std       {inv['ratio']:.2f} +/- {inv['ratio_se']:.2f}")
    print("```")
    if not burg["self_check_ok"]:
        raise SystemExit("Burgers self-check failed: the [t][x] orientation is likely transposed.")


if __name__ == "__main__":
    main()
