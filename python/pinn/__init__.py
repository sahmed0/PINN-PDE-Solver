"""Core physics-informed neural-network library for the heat and Burgers equations.

Holds the model definitions, the PDE residual/loss physics, the training loop, the
closed-form analytical solution used for scoring, the inverse-problem and CRLB tooling,
and the JSON export/forward-pass contract shared with the browser frontend. Installed as
the ``pinn`` package (see ``python/pyproject.toml``); the ``mlops`` package builds the
Azure ML pipeline on top of it.
"""
