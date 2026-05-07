"""Legacy compatibility shim for the standalone DADS variant."""

import importlib.util
import os
import sys


_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
_LEGACY_DADS = os.path.join(_ROOT, "scripts", "legacy", "dads.py")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


_SPEC = importlib.util.spec_from_file_location("_legacy_dads", _LEGACY_DADS)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load legacy DADS module from {_LEGACY_DADS}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

DADS = _MODULE.DADS

__all__ = ["DADS"]
