"""
simba.py

SimBa networks (ICLR 2025): RSNorm + Residual Feedforward Blocks + Post-LN.

This file intentionally contains only pure PyTorch modules so it can be reused
by SAC/METRA and other off-policy actor-critic code.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils import utils  # for TanhNormal


@dataclass(frozen=True)
class SimBaConfig:
    """Configuration for a SimBa trunk."""
    input_dim: int
    hidden_dim: int
    num_blocks: int
    mlp_ratio: int = 4
    rsnorm_momentum: float = 0.999
    rsnorm_eps: float = 1e-5
    ln_eps: float = 1e-5


# simba.py
class RSNorm(nn.Module):
    """
    Running Statistics Normalization (RSNorm) from SimBa (ICLR 2025).

    Paper Eq.(3)-(4) exact online statistics (NO EMA / momentum).
    """

    def __init__(self, dim: int, *, eps: float = 1e-8, momentum=None):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros(self.dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(self.dim, dtype=torch.float32))

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"RSNorm expects [B,{self.dim}], got {tuple(x.shape)}.")

        x64 = x.to(torch.float64)
        b = x64.shape[0]
        if b == 0:
            return

        batch_mean = x64.mean(dim=0)
        batch_var = x64.var(dim=0, unbiased=False)  # population variance

        n = int(self.count.item())
        if n == 0:
            new_mean = batch_mean
            new_var = batch_var
            new_n = b
        else:
            mean = self.mean.to(torch.float64)
            var = self.var.to(torch.float64)

            total = n + b
            delta = batch_mean - mean
            new_mean = mean + delta * (b / total)

            m_a = var * n
            m_b = batch_var * b
            m2 = m_a + m_b + (delta * delta) * (n * b / total)
            new_var = m2 / total
            new_n = total

        self.count.fill_(new_n)
        self.mean.copy_(new_mean.to(torch.float32))
        self.var.copy_(torch.clamp(new_var.to(torch.float32), min=0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"RSNorm expects [B,D], got {tuple(x.shape)}.")
        if self.training:
            self._update(x)
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class ResidualFFNBlock(nn.Module):
    """Pre-LN residual FFN block: x <- x + MLP(LN(x)).

    MLP uses inverted bottleneck: D -> (ratio*D) -> D with ReLU.
    """
    def __init__(self, hidden_dim: int, mlp_ratio: int = 4, ln_eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, eps=ln_eps)
        inner = int(hidden_dim * mlp_ratio)

        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)
        self.act = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        return x + y


class SimBaTrunk(nn.Module):
    """SimBa trunk up to the post-layernorm output."""
    def __init__(self, cfg: SimBaConfig):
        super().__init__()
        self.cfg = cfg

        self.rsnorm = RSNorm(cfg.input_dim, momentum=cfg.rsnorm_momentum, eps=cfg.rsnorm_eps)
        self.in_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)

        self.blocks = nn.ModuleList([
            ResidualFFNBlock(cfg.hidden_dim, mlp_ratio=cfg.mlp_ratio, ln_eps=cfg.ln_eps)
            for _ in range(cfg.num_blocks)
        ])
        self.post_ln = nn.LayerNorm(cfg.hidden_dim, eps=cfg.ln_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rsnorm(x)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.post_ln(x)


class SimBaGaussianPolicy(nn.Module):
    """SimBa actor head producing a TanhNormal distribution."""
    def __init__(
            self,
            trunk_cfg: SimBaConfig,
            action_dim: int,
            *,
            max_log_std: float = 2.0,
            min_log_std: float = -10.0,
            init_log_std: float = 0.0,
            normal_distribution_cls=utils.TanhNormal,
    ):
        super().__init__()
        self.trunk = SimBaTrunk(trunk_cfg)
        self.action_dim = int(action_dim)

        self.max_log_std = float(max_log_std)
        self.min_log_std = float(min_log_std)
        self.normal_distribution_cls = normal_distribution_cls

        self.mean_head = nn.Linear(trunk_cfg.hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(trunk_cfg.hidden_dim, self.action_dim)

        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

        nn.init.xavier_uniform_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, float(init_log_std))

    def forward(self, obs: torch.Tensor):
        z = self.trunk(obs)
        mean = self.mean_head(z)
        log_std = self.log_std_head(z).clamp(self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)
        return self.normal_distribution_cls(mean, std)

    def forward_mode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).mean

    def forward_with_transform(self, obs: torch.Tensor, *, transform):
        # For compatibility with existing PolicyEx API; current training code does not rely on this.
        dist = self.forward(obs)
        return dist, dist


class SimBaQFunction(nn.Module):
    """SimBa critic Q(s,a): scalar output."""
    def __init__(self, trunk_cfg: SimBaConfig, action_dim: int):
        super().__init__()
        trunk_cfg2 = SimBaConfig(
            input_dim=int(trunk_cfg.input_dim) + int(action_dim),
            hidden_dim=int(trunk_cfg.hidden_dim),
            num_blocks=int(trunk_cfg.num_blocks),
            mlp_ratio=int(trunk_cfg.mlp_ratio),
            rsnorm_momentum=float(trunk_cfg.rsnorm_momentum),
            rsnorm_eps=float(trunk_cfg.rsnorm_eps),
            ln_eps=float(trunk_cfg.ln_eps),
        )
        self.trunk = SimBaTrunk(trunk_cfg2)
        self.out = nn.Linear(trunk_cfg2.hidden_dim, 1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observations, actions], dim=-1)
        z = self.trunk(x)
        return self.out(z)
