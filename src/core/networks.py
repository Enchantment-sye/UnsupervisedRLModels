# Facade for backward compatibility after refactoring into models/
from models import (
    SpectralNorm, MultiHeadedMLPModule, MLPModule, _NonLinearity, NormLayer, CNN,
    BaseHuggingFaceEncoder, DINOEncoder, ResNetEncoder, Encoder, StopGradEncoder, WithEncoder,
    PolicyEx,
    GaussianMLPBaseModule, GaussianMLPModule, GaussianMLPIndependentStdModule,
    GaussianMLPTwoHeadedModule, GaussianMLPModuleEx, GaussianMLPIndependentStdModuleEx,
    GaussianMLPTwoHeadedModuleEx, GaussianMixtureMLPModule, get_gaussian_module_construction,
    ContinuousMLPQFunctionEx
)
from core.encoder_factory import EncoderFactory
