"""Unit tests for the Cramer-Rao lower bound (pinn/crlb.py).

The inverse demo quotes its error against this floor, so the bound itself is checked
against its defining properties: I(alpha) is additive over observations, scales as
1/sigma^2, matches the closed form for a single point, and the sensitivity it is
built from is the true derivative of the exact solution.
"""

import math

import jax.numpy as jnp
import jax.random as jr
import pytest

from pinn import crlb

ALPHA = 0.042
SIGMA = 0.01


def _obs(n=20, seed=0):
    """N scattered [x, t] observation locations over the training domain."""
    k_x, k_t = jr.split(jr.PRNGKey(seed))
    x = jr.uniform(k_x, (n, 1), minval=-1.0, maxval=1.0)
    t = jr.uniform(k_t, (n, 1), minval=0.0, maxval=1.0)
    return jnp.hstack([x, t])


def _g(x, t, alpha):
    """Exact field sin(pi x) exp(-alpha pi^2 t), in float64."""
    return math.sin(math.pi * x) * math.exp(-alpha * math.pi**2 * t)


def test_fisher_information_scales_with_n():
    X = _obs()
    i_n = float(crlb.fisher_information(X, ALPHA, SIGMA))
    i_2n = float(crlb.fisher_information(jnp.vstack([X, X]), ALPHA, SIGMA))
    assert i_2n == pytest.approx(2 * i_n, rel=1e-6)


def test_fisher_information_scales_with_inverse_sigma_squared():
    X = _obs()
    i_sigma = float(crlb.fisher_information(X, ALPHA, SIGMA))
    i_half = float(crlb.fisher_information(X, ALPHA, SIGMA / 2))
    assert i_half == pytest.approx(4 * i_sigma, rel=1e-6)


def test_single_point_closed_form():
    x, t = 0.3, 0.8
    expected = (math.pi**2 * t * math.sin(math.pi * x) * math.exp(-ALPHA * math.pi**2 * t)) ** 2
    expected /= SIGMA**2
    got = float(crlb.fisher_information(jnp.array([[x, t]]), ALPHA, SIGMA))
    assert got == pytest.approx(expected, rel=1e-5)


def test_crlb_std_is_inverse_sqrt_fisher():
    X = _obs()
    i = float(crlb.fisher_information(X, ALPHA, SIGMA))
    assert crlb.crlb_std(X, ALPHA, SIGMA) == pytest.approx(1 / math.sqrt(i))


def test_crlb_relative_is_std_over_alpha():
    X = _obs()
    assert crlb.crlb_relative(X, ALPHA, SIGMA) == pytest.approx(
        crlb.crlb_std(X, ALPHA, SIGMA) / ALPHA
    )


def test_sensitivity_matches_numerical_derivative():
    # The central difference runs in float64 via math: at h=1e-6 a float32 difference
    # would be dominated by rounding, not by the derivative.
    h = 1e-6
    for x, t in [(0.3, 0.8), (-0.7, 0.5), (0.5, 1.0)]:
        numerical = (_g(x, t, ALPHA + h) - _g(x, t, ALPHA - h)) / (2 * h)
        assert float(crlb.sensitivity(x, t, ALPHA)) == pytest.approx(numerical, rel=1e-4)
