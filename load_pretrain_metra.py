"""Legacy compatibility shim for pretrained METRA loading helpers."""

import os
import sys


_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


from legacy.load_pretrain_metra import *  # noqa: F401,F403
