from __future__ import annotations

import torch
import torch.nn as nn


class FallbackRMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"FallbackRMSNorm expected trailing dim {self.normalized_shape}, got {x.shape[-1]}"
            )
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x * rms
        if self.weight is not None:
            y = y * self.weight
        return y


def build_traj_latent_normalizer(kind: str, dim: int, eps: float) -> nn.Module | None:
    kind = str(kind).lower()
    dim = int(dim)
    eps = float(eps)

    if kind == 'off':
        return None
    if kind == 'layernorm':
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
    if kind == 'rmsnorm':
        if hasattr(nn, 'RMSNorm'):
            return nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        return FallbackRMSNorm(dim, eps=eps, elementwise_affine=True)
    raise ValueError(f"Unsupported traj_latent_norm: {kind}")
