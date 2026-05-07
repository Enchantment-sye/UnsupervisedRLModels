"""Legacy compatibility shim for the PPO meta-controller utilities."""

import os
import sys


_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


from legacy.meta_controler import *  # noqa: F401,F403
