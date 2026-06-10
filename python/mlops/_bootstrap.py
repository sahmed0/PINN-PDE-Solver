"""Path bootstrap for the MLOps package.

The existing core in ``python/src`` uses flat imports (``from model import ...``),
mirrored by ``python/conftest.py`` which prepends ``src/`` to ``sys.path``. Importing
this module does the same so ``import model``, ``import train``, ``import physics``,
``import analytical`` resolve from anywhere, and it also puts ``python/`` on the path
so the ``mlops`` package itself is importable (``from mlops import ...``) when a script
under ``mlops/`` is run directly.
"""

import os
import sys

_PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python/
_SRC_DIR = os.path.join(_PY_DIR, "src")                                # python/src

for _p in (_SRC_DIR, _PY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
