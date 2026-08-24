"""Shared forward-pass cores for the three PINN models.

The input normalisation and the two hard-constraint ansatze were copy-pasted into
``ParametricPINN``, ``BurgersPINN`` and ``InversePINN`` (and mirrored again in the
browser and in the float64 JSON reference). This module holds the single source of
truth for the JAX side so a future edit can't silently diverge one copy.

Why plain functions and not a shared ``eqx.Module`` (e.g. a ``NormalisedMLP`` layer):
trained models persist via ``eqx.tree_deserialise_leaves`` into a *live* skeleton,
so the pytree structure of the three classes must stay byte-for-byte what the
existing ``.eqx`` checkpoints were written with (local baselines, MLflow runs, the
Azure registry artifact). Wrapping the MLP in a new module would change every
class's field layout and break deserialisation. Refactoring the ``__call__`` bodies
into free functions keeps the modules structurally identical while removing the
duplication.

The float64 NumPy reference (json_forward.forward_from_payload) deliberately keeps
its own NumPy ansatz two-liner: it must run with nothing but NumPy over a JSON
payload, so it does not import this jnp-based module.
"""

import jax.numpy as jnp


def normalised_mlp(mlp, input_center, input_scale, x):
    """Normalise raw inputs to ~[-1, 1] and return the scalar MLP output N.

    ``norm = (raw - center) / scale`` before the network sees it (the small-band
    alpha input is unresolvable otherwise); the MLP has a single output.
    """
    center = jnp.asarray(input_center)
    scale = jnp.asarray(input_scale)
    x_norm = (x - center) / scale
    return mlp(x_norm)[0]


def heat_ansatz(x_phys, t_phys, n):
    """Hard-constraint reconstruction for the heat IC/BCs: u = sin(pi x) + (1 - x^2) t N.

    At t=0 -> sin(pi x) (initial condition); at x=+/-1 -> (1 - x^2)=0 and sin=0
    (zero Dirichlet BCs). Shared by the forward and inverse heat models.
    """
    return jnp.sin(jnp.pi * x_phys) + (1.0 - x_phys ** 2) * t_phys * n


def burgers_ansatz(x_phys, t_phys, n):
    """Hard-constraint reconstruction for Burgers': u = -sin(pi x) + (1 - x^2) t N.

    Same structure as heat_ansatz but with the -sin initial profile u(x, 0) = -sin(pi x).
    """
    return -jnp.sin(jnp.pi * x_phys) + (1.0 - x_phys ** 2) * t_phys * n
