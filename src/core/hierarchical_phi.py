from __future__ import annotations

from typing import Optional

import torch


def resolve_hierarchical_phi_depth(cfg) -> int:
    depth = int(getattr(cfg.algo, 'hierarchical_phi_depth', 0) or 0)
    if depth <= 0:
        depth = int(getattr(cfg.algo, 'num_skill_levels', 1))
    return depth


def resolve_hierarchical_phi_dim(cfg) -> int:
    return int(getattr(cfg.algo, 'dim_skill', 0))


def resolve_total_phi_dim(cfg) -> int:
    if not getattr(cfg.cascade, 'use_cascade', False):
        return int(cfg.algo.dim_skill)
    if not getattr(cfg.algo, 'use_hierarchical_phi', False):
        return int(cfg.algo.dim_skill)
    return resolve_hierarchical_phi_depth(cfg) * resolve_hierarchical_phi_dim(cfg)


def split_hierarchical_phi(phi: torch.Tensor, depth: int, level_dim: int) -> torch.Tensor:
    if phi.dim() != 2:
        raise ValueError(f"Expected phi tensor with shape (B, D), got {tuple(phi.shape)}")
    expected_dim = depth * level_dim
    if phi.shape[-1] != expected_dim:
        raise ValueError(
            f"Hierarchical phi dim mismatch: got {phi.shape[-1]}, expected {expected_dim} "
            f"(depth={depth}, level_dim={level_dim})"
        )
    return phi.reshape(phi.shape[0], depth, level_dim)


def split_hierarchical_skill(skill: torch.Tensor, depth: int, level_dim: int) -> torch.Tensor:
    if skill.dim() == 3:
        if skill.shape[1] != depth or skill.shape[2] != level_dim:
            raise ValueError(
                f"Hierarchical skill shape mismatch: got {tuple(skill.shape)}, "
                f"expected (*, {depth}, {level_dim})"
            )
        return skill
    if skill.dim() == 2 and skill.shape[-1] == depth * level_dim:
        return skill.reshape(skill.shape[0], depth, level_dim)
    raise ValueError(
        f"Expected hierarchical skill with shape (B, {depth}, {level_dim}) or "
        f"(B, {depth * level_dim}), got {tuple(skill.shape)}"
    )


def build_hierarchical_beta(
        depth: int,
        mode: str,
        rho: float,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    dtype = dtype or torch.float32
    device = device or torch.device('cpu')

    if mode == 'uniform':
        return torch.full((depth,), 1.0 / float(depth), device=device, dtype=dtype)

    if mode not in ('exp_unnorm', 'exp_norm'):
        raise ValueError(f"Unknown beta_mode={mode!r}, expected uniform/exp_unnorm/exp_norm")

    rho = float(rho)
    if not (0.0 < rho < 1.0):
        raise ValueError(f"beta_rho must be in (0, 1) for {mode}, got {rho}")

    exponents = torch.arange(depth, device=device, dtype=dtype)
    log_rho = torch.log(torch.as_tensor(rho, device=device, dtype=dtype))
    weights = torch.exp(exponents * log_rho)

    if mode == 'exp_unnorm':
        return weights
    return weights / weights.sum().clamp_min(torch.finfo(dtype).tiny)
