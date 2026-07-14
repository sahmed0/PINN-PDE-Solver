"""
Cramer-Rao lower bound (CRLB) for the inverse heat-equation problem.

The inverse solver recovers a single scalar -- the diffusivity alpha -- from a
handful of noisy temperature measurements. Unlike the forward problem, whose
error is just how well a network approximates a *known, noise-free* function, the
inverse error is fundamentally limited by how much information about alpha the
noisy data actually carries. The CRLB makes that limit precise: for *any*
unbiased estimator, the variance of the estimate cannot beat

    Var(alpha_hat) >= 1 / I(alpha)

where I(alpha) is the Fisher information of the measurement model. So the right
yardstick for the inverse PINN is not the forward problem's ~0.1%, but this
bound: a method that recovers alpha to within ~sqrt(1/I) is doing as well as the
physics permits.

Measurement model
-----------------
Each observation is a noisy sample of the exact field at a scattered (x_i, t_i):

    u_i = g(x_i, t_i; alpha) + eps_i,   eps_i ~ N(0, sigma^2)  i.i.d.
    g(x, t; alpha) = sin(pi x) * exp(-alpha * pi^2 * t).

For a Gaussian likelihood with known sigma and a scalar parameter, the Fisher
information reduces to a sum of squared sensitivities:

    I(alpha) = (1 / sigma^2) * sum_i ( d g(x_i, t_i; alpha) / d alpha )^2,

with the sensitivity (how fast the signal moves when alpha moves)

    dg/dalpha = -pi^2 * t * sin(pi x) * exp(-alpha * pi^2 * t) = -pi^2 * t * g.

Two things fall straight out of this and explain the inverse problem's behaviour:

  * The t factor means late-time points carry far more information than early
    ones -- you only learn a decay rate by watching it decay. With alpha small,
    the field barely decays over t in [0, 1] (less than half a time-constant
    tau = 1/(alpha pi^2)), so the information per point is modest.
  * I(alpha) grows linearly with the number of points and as 1/sigma^2, so the
    floor on the *relative* error shrinks like sigma / sqrt(N).
"""

import jax.numpy as jnp
import jax.random as jr


def sensitivity(x, t, alpha):
    """d/dalpha of g(x, t; alpha) = sin(pi x) exp(-alpha pi^2 t).

    This is the per-point signal sensitivity that drives the Fisher information:
    dg/dalpha = -pi^2 t sin(pi x) exp(-alpha pi^2 t). Accepts scalars or arrays.
    """
    return -(jnp.pi ** 2) * t * jnp.sin(jnp.pi * x) * jnp.exp(-alpha * (jnp.pi ** 2) * t)


def fisher_information(X_obs, alpha, sigma):
    """Fisher information I(alpha) for the measurement set X_obs.

    X_obs has shape (N, 2) with columns [x, t] -- the *actual* observation
    locations, so the bound reflects the real experiment rather than an average.
    """
    x = X_obs[:, 0]
    t = X_obs[:, 1]
    dg = sensitivity(x, t, alpha)
    return jnp.sum(dg ** 2) / sigma ** 2


def crlb_std(X_obs, alpha, sigma):
    """Cramer-Rao lower bound on the *standard deviation* of alpha_hat.

    Returns sqrt(1 / I(alpha)): the smallest standard error any unbiased
    estimator of alpha could achieve from exactly these noisy measurements.
    """
    return float(1.0 / jnp.sqrt(fisher_information(X_obs, alpha, sigma)))


def crlb_relative(X_obs, alpha, sigma):
    """CRLB std expressed as a fraction of alpha (the headline 'floor' figure)."""
    return crlb_std(X_obs, alpha, sigma) / alpha


def expected_crlb_std(alpha, sigma, n_obs, t_max=1.0, n_draws=200, seed=0):
    """Monte-Carlo expected CRLB std for *random* uniform measurement designs.

    crlb_std above is conditioned on one specific set of points; this averages
    the bound over many random draws of N points (x ~ U[-1, 1], t ~ U[0, t_max]),
    which is the right quantity when comparing experiment designs (N, sigma,
    t_max) before any data is collected. Used for the design sweep / floor table.
    """
    keys = jr.split(jr.PRNGKey(seed), n_draws)
    stds = []
    for k in keys:
        k_x, k_t = jr.split(k, 2)
        x = jr.uniform(k_x, (n_obs, 1), minval=-1.0, maxval=1.0)
        t = jr.uniform(k_t, (n_obs, 1), minval=0.0, maxval=t_max)
        X = jnp.hstack([x, t])
        stds.append(crlb_std(X, alpha, sigma))
    return float(sum(stds) / len(stds))


#: (n_obs, sigma, t_max) designs shown in the floor table. The first matching the
#: live experiment (N=200, sigma=0.01, t<=1) is the design actually used.
_SWEEP_DESIGNS = (
    (50, 0.01, 1.0),
    (200, 0.01, 1.0),
    (500, 0.01, 1.0),
    (50, 0.003, 1.0),
    (50, 0.01, 3.0),
)


def design_sweep(alpha=0.042, sigma=0.01, n_draws=200):
    """Reproduce the floor table used to choose the experiment design.

    Returns a list of row dicts -- one per (n_obs, sigma, t_max) design -- each
    with the expected relative CRLB floor (% of alpha). Structured (rather than a
    label->value map) so the frontend can render it as a table and highlight the
    design actually used. Handy to re-run while studying the trade-offs.
    """
    rows = []
    for n, sig, tmax in _SWEEP_DESIGNS:
        rel = expected_crlb_std(alpha, sig, n, t_max=tmax, n_draws=n_draws) / alpha
        rows.append({
            "n_obs": int(n),
            "sigma": float(sig),
            "t_max": float(tmax),
            "rel_pct": float(rel * 100.0),
        })
    return rows


if __name__ == "__main__":
    print("Expected relative CRLB floor by experiment design (alpha=0.042):\n")
    print(f"    {'N':>4s} | {'sigma':>6s} | {'t<=':>4s} | CRLB std (% of true)")
    print(f"    {'-' * 4}-+-{'-' * 6}-+-{'-' * 4}-+--------------------")
    for r in design_sweep():
        print(f"    {r['n_obs']:>4d} | {r['sigma']:>6.3f} | {r['t_max']:>4.1f} "
              f"| {r['rel_pct']:6.2f}%")
