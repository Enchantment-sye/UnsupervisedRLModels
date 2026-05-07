"""Legacy compatibility shim for the original METRA agent API.

This module restores ``import metra`` for legacy scripts by exposing the
reference implementation from ``src/core/metra.py``.
"""

import os
import sys


_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


from core.metra import DRQ_METRAAgent, DictBatchDataset, MeasureAndAccTime, _StopGradEncoder, compute_total_norm


__all__ = [
    "DRQ_METRAAgent",
    "DictBatchDataset",
    "MeasureAndAccTime",
    "_StopGradEncoder",
    "compute_total_norm",
]
