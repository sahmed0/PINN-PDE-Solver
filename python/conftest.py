import os
import sys

# Put the package source on the import path so both the source modules
# (which use flat imports like `from model import ...`) and the tests can
# resolve `model`, `physics`, and `train` consistently.
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
