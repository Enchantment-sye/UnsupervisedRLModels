"""Algorithm-agnostic coverage encoder distillation components."""

from .config import CovEncoderDistillConfig, build_arg_parser, parse_args
from .coverage_encoder import (
    CoverageEncoder,
    DirectCoverageEncoder,
    ResNet101Teacher,
    effective_rank,
    load_coverage_encoder_checkpoint,
    save_coverage_encoder_checkpoint,
)
from .distill import CoverageEncoderDistillTrainer

__all__ = [
    "CovEncoderDistillConfig",
    "CoverageEncoder",
    "CoverageEncoderDistillTrainer",
    "DirectCoverageEncoder",
    "ResNet101Teacher",
    "build_arg_parser",
    "effective_rank",
    "load_coverage_encoder_checkpoint",
    "parse_args",
    "save_coverage_encoder_checkpoint",
]
