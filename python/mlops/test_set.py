"""Held-out test set for the heat-equation gate.

The gate evaluates a trained PINN against the closed-form analytical solution on
two fixed slices of alpha (each on a (nx, nt) grid). Both slices use the SAME
metric the training loop reports (`analytical.relative_l2_error`), so "passes the
gate" means exactly what the training-time validation number means.

* ``TEST_ALPHAS_INTERP`` — alphas INSIDE the training band [0.01, 0.1]. Training
  samples ``alpha`` *continuously and uniformly* across that band, so these are
  genuine **in-distribution interpolation** points; they were chosen merely so as
  NOT to coincide with the training-time validation alphas (0.01, 0.05, 0.1).
  They are NOT "unmemorised anchors" — the model never trained on discrete
  alphas. This is the primary promotion metric.

* ``TEST_ALPHAS_OOD`` — alphas JUST OUTSIDE [0.01, 0.1]. This is the **only
  genuinely held-out regime**: a band of diffusivities the training data never
  covered. The analytical solution still provides exact ground truth there
  (``u_exact`` holds for any alpha > 0), so we can still measure a real error;
  extrapolation is intrinsically harder, hence the looser OOD threshold.
"""

from . import _bootstrap  # noqa: F401  (puts python/src on sys.path)

import analytical

from . import config


def held_out_metrics(model):
    """Evaluate ``model`` on both held-out slices against the analytical solution.

    Returns a dict with per-alpha relative-L2 / L-infinity errors (string alpha
    keys, both slices combined) plus separate means for the in-distribution
    interpolation slice (``mean_rel_l2``, the primary metric) and the
    out-of-distribution extrapolation slice (``mean_rel_l2_ood``).

    Every value is ``float()``-cast: ``analytical.relative_l2_error`` /
    ``max_abs_error`` return JAX arrays, which ``json.dump`` / the Azure SDK
    cannot serialise.
    """
    interp = tuple(float(a) for a in config.TEST_ALPHAS_INTERP)
    ood = tuple(float(a) for a in config.TEST_ALPHAS_OOD)
    nx, nt = config.TEST_GRID_NX, config.TEST_GRID_NT

    per_alpha_rel_l2 = {}
    per_alpha_linf = {}
    for a in interp + ood:
        per_alpha_rel_l2[f"{a}"] = float(analytical.relative_l2_error(model, a, nx, nt))
        per_alpha_linf[f"{a}"] = float(analytical.max_abs_error(model, a, nx, nt))

    mean_rel_l2 = sum(per_alpha_rel_l2[f"{a}"] for a in interp) / len(interp)
    mean_rel_l2_ood = sum(per_alpha_rel_l2[f"{a}"] for a in ood) / len(ood)

    return {
        "per_alpha_rel_l2": per_alpha_rel_l2,
        "per_alpha_linf": per_alpha_linf,
        "mean_rel_l2": float(mean_rel_l2),
        "mean_rel_l2_ood": float(mean_rel_l2_ood),
        "alphas_interp": interp,
        "alphas_ood": ood,
    }
