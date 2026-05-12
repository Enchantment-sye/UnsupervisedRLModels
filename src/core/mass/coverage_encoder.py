"""Backward-compatible import surface for coverage encoder components.

The coverage encoder and its ResNet teacher are algorithm-agnostic now. New
code should import from `core.cov_encoder`; this module keeps older MASS tests
and callers working without coupling distillation to MASS.
"""

from core.cov_encoder.coverage_encoder import (
    CoverageEncoder,
    DirectCoverageEncoder,
    ResNet101Teacher,
    effective_rank,
    load_coverage_encoder_checkpoint,
    save_coverage_encoder_checkpoint,
)

__all__ = [
    "CoverageEncoder",
    "DirectCoverageEncoder",
    "ResNet101Teacher",
    "effective_rank",
    "load_coverage_encoder_checkpoint",
    "save_coverage_encoder_checkpoint",
]
