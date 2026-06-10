"""MLOps pipeline package for the heat-equation PINN.

Importing the package runs the path bootstrap so the flat core imports
(``import model`` etc.) and the ``mlops`` package both resolve.
"""

from . import _bootstrap  # noqa: F401  (sets up sys.path as a side effect)
