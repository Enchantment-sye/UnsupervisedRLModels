"""
actor_critic_factory.py

Factory for actor/critic core modules (MLP or SimBa).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Literal, Optional
import functools

import numpy as np
import torch
import torch.nn as nn

from utils import utils
from core.networks import ContinuousMLPQFunctionEx, GaussianMLPTwoHeadedModuleEx
from core.simba import SimBaConfig, SimBaGaussianPolicy, SimBaQFunction

BackboneType = Literal["mlp", "simba"]


@dataclass(frozen=True)
class SimBaActorCriticHP:
    """SimBa hyperparameters (paper defaults are actor:128x1, critic:512x2)."""
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 1

    critic_hidden_dim: int = 512
    critic_num_blocks: int = 2

    mlp_ratio: int = 4
    rsnorm_momentum: float = 0.999
    rsnorm_eps: float = 1e-5
    ln_eps: float = 1e-5


class ActorCriticModuleFactory:
    """Creates policy/critic *core* modules (without encoder wrappers)."""

    @staticmethod
    def create_actor_core(
            *,
            backbone: BackboneType,
            input_dim: int,
            action_dim: int,
            mlp_hidden_sizes: Sequence[int],
            mlp_nonlinearity: Callable,
            actor_init_std: float,
            actor_max_log_std: float,
            simba_hp: Optional[SimBaActorCriticHP] = None,
    ) -> nn.Module:
        if backbone == "mlp":
            return GaussianMLPTwoHeadedModuleEx(
                input_dim=int(input_dim),
                output_dim=int(action_dim),
                hidden_sizes=list(mlp_hidden_sizes),
                hidden_nonlinearity=mlp_nonlinearity,
                layer_normalization=False,
                max_std=np.exp(float(actor_max_log_std)),
                normal_distribution_cls=utils.TanhNormal,
                output_w_init=functools.partial(utils.xavier_normal_ex, gain=1.),
                init_std=float(actor_init_std),
            )

        if backbone == "simba":
            hp = simba_hp or SimBaActorCriticHP()
            trunk_cfg = SimBaConfig(
                input_dim=int(input_dim),
                hidden_dim=int(hp.actor_hidden_dim),
                num_blocks=int(hp.actor_num_blocks),
                mlp_ratio=int(hp.mlp_ratio),
                rsnorm_momentum=float(hp.rsnorm_momentum),
                rsnorm_eps=float(hp.rsnorm_eps),
                ln_eps=float(hp.ln_eps),
            )
            init_log_std = float(np.log(max(float(actor_init_std), 1e-6)))
            return SimBaGaussianPolicy(
                trunk_cfg,
                action_dim=int(action_dim),
                max_log_std=float(actor_max_log_std),
                min_log_std=-10.0,
                init_log_std=init_log_std,
                normal_distribution_cls=utils.TanhNormal,
            )

        raise ValueError(f"Unknown backbone={backbone!r} (expected 'mlp' or 'simba').")

    @staticmethod
    def create_critic_core(
            *,
            backbone: BackboneType,
            obs_dim: int,
            action_dim: int,
            mlp_hidden_sizes: Sequence[int],
            mlp_nonlinearity: Callable,
            simba_hp: Optional[SimBaActorCriticHP] = None,
    ) -> nn.Module:
        if backbone == "mlp":
            return ContinuousMLPQFunctionEx(
                obs_dim=int(obs_dim),
                action_dim=int(action_dim),
                hidden_sizes=list(mlp_hidden_sizes),
                hidden_nonlinearity=mlp_nonlinearity or torch.relu,
                hidden_w_init=torch.nn.init.xavier_uniform_,
                hidden_b_init=torch.nn.init.zeros_,
                output_w_init=torch.nn.init.xavier_uniform_,
                output_b_init=torch.nn.init.zeros_,
                layer_normalization=True,
            )

        if backbone == "simba":
            hp = simba_hp or SimBaActorCriticHP()
            trunk_cfg = SimBaConfig(
                input_dim=int(obs_dim),
                hidden_dim=int(hp.critic_hidden_dim),
                num_blocks=int(hp.critic_num_blocks),
                mlp_ratio=int(hp.mlp_ratio),
                rsnorm_momentum=float(hp.rsnorm_momentum),
                rsnorm_eps=float(hp.rsnorm_eps),
                ln_eps=float(hp.ln_eps),
            )
            return SimBaQFunction(trunk_cfg, action_dim=int(action_dim))

        raise ValueError(f"Unknown backbone={backbone!r} (expected 'mlp' or 'simba').")