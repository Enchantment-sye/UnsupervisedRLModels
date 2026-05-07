import copy
import functools
import logging
import os
import time
from collections import defaultdict
from typing import List, Iterable

import numpy as np
import torch
import tqdm
from math import inf, sqrt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core import sac_utils
from utils import utils
from utils.utils import _finalize_lr, OptimizerGroupWrapper
from core.isolation_kernel import SoftIsolationKernel
from core.networks import PolicyEx, ContinuousMLPQFunctionEx, GaussianMLPIndependentStdModuleEx, \
    GaussianMLPTwoHeadedModuleEx, Encoder, WithEncoder
from data_structs.trajectory_batch import TrajectoryBatch
from workers.rollout import SkillRolloutWorker
from torch.nn import functional as F

class SacTrainer:

    def __init__(
            self,
            discount,
            alpha,
            device,
            scale_reward,
            target_entropy,
            tau,
            critic1,
            critic2,
            actor,
            lr_op,
            sac_lr_q,
            sac_lr_a,
            *,
            policy_delay: int = 1,
            actor_start_steps: int = 0,
            safe_action_distill_weight: float = 0.0,
    ):
        self._reward_scale_factor = float(scale_reward)
        self.device = device
        self.discount = float(discount)
        self.alpha = float(alpha)
        self.tau = float(tau)

        # --- Update schedule (warmup + delayed actor updates) ---
        self.policy_delay = int(policy_delay)
        self.actor_start_steps = int(actor_start_steps)
        self.safe_action_distill_weight = float(safe_action_distill_weight)

        # Modules
        self.skill_policy = actor.to(self.device)
        self.qf1 = critic1.to(self.device)
        self.qf2 = critic2.to(self.device)

        # For compatibility with sac_utils._clip_actions (even if you use TanhNormal).
        self._env_spec = getattr(self.skill_policy, "env_spec", None)

        # Determine whether we are in encoder mode.
        policy_core = self._unwrap_policy_core(self.skill_policy)
        self.use_encoder = self._is_with_encoder_like(policy_core)
        if (not self.use_encoder) and hasattr(policy_core, 'encoder'):
            # Fail fast: we are likely mis-detecting and would accidentally run the
            # old non-encoder SAC update path, which can build unnecessary graphs
            # (policy->encoder->critic) and/or update encoder in unintended ways.
            raise RuntimeError(
                "SACTrainer mis-detected encoder usage: policy looks encoder-wrapped, but use_encoder=False. "
                "Please check policy wrappers and detection logic."
            )
        if self.use_encoder:
            # policy_core is WithEncoder(encoder, policy_mlp)
            self.encoder = policy_core.encoder
            self.encoder_trainable = any(p.requires_grad for p in self.encoder.parameters())

            self.skill_policy_mlp = policy_core.module
            self.qf1_mlp = self.qf1.module if self._is_with_encoder_like(self.qf1) else self.qf1
            self.qf2_mlp = self.qf2.module if self._is_with_encoder_like(self.qf2) else self.qf2

            self.target_qf1_mlp = copy.deepcopy(self.qf1_mlp).to(self.device)
            self.target_qf2_mlp = copy.deepcopy(self.qf2_mlp).to(self.device)
            for p in self.target_qf1_mlp.parameters():
                p.requires_grad_(False)
            for p in self.target_qf2_mlp.parameters():
                p.requires_grad_(False)

            if self.encoder_trainable:
                self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
                self.target_encoder.eval()
                for p in self.target_encoder.parameters():
                    p.requires_grad_(False)
            else:
                # share the frozen encoder; it's already eval() in BaseHuggingFaceEncoder when finetune=False
                self.target_encoder = self.encoder

            # Optimizers: actor updates ONLY policy head; critic updates Q heads + encoder ONCE.
            actor_params = [{'params': self.skill_policy_mlp.parameters(), 'lr': _finalize_lr(lr_op)}]

            qf_params = [
                {'params': self.qf1_mlp.parameters(), 'lr': _finalize_lr(sac_lr_q)},
                {'params': self.qf2_mlp.parameters(), 'lr': _finalize_lr(sac_lr_q)},
            ]

            if self.encoder_trainable:
                qf_params.append({
                    'params': [p for p in self.encoder.parameters() if p.requires_grad],
                    'lr': _finalize_lr(sac_lr_q),
                })
        else:
            # Standard SAC: target critics are full copies.
            self.target_qf1 = copy.deepcopy(self.qf1).to(self.device)
            self.target_qf2 = copy.deepcopy(self.qf2).to(self.device)
            for p in self.target_qf1.parameters():
                p.requires_grad_(False)
            for p in self.target_qf2.parameters():
                p.requires_grad_(False)

            actor_params = [{'params': self.skill_policy.parameters(), 'lr': _finalize_lr(lr_op)}]

            # Safety: de-duplicate parameters in case of accidental sharing.
            q_params = list(self.qf1.parameters()) + list(self.qf2.parameters())
            q_params = self._unique_params(q_params)
            qf_params = [{'params': q_params, 'lr': _finalize_lr(sac_lr_q)}]

        # Temperature
        log_alpha = utils.ParameterModule(torch.tensor([np.log(self.alpha)], dtype=torch.float32))
        self.log_alpha = log_alpha.to(self.device)
        self._target_entropy = float(target_entropy)

        optimizers_dict = {
            'actor': torch.optim.Adam(actor_params),
            'qf': torch.optim.Adam(qf_params),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(sac_lr_a)},
            ]),
        }

        self.optimizer = OptimizerGroupWrapper(
            optimizers=optimizers_dict,
            max_optimization_epochs=None,
        )

        # For external hooks (e.g., gradient clipping wrappers)
        self.param_modules = {
            'actor': self.skill_policy,
            'qf1': self.qf1,
            'qf2': self.qf2,
            'log_alpha': self.log_alpha,
        }

    @staticmethod
    def _unwrap_policy_core(policy):
        """If policy is PolicyEx, return its underlying module; else return itself."""
        return getattr(policy, '_module', policy)

    @staticmethod
    def _is_with_encoder_like(module) -> bool:
        # Primary check: exact WithEncoder type.
        if isinstance(module, WithEncoder):
            return True
        # Fallback: duck-typing for older / wrapped WithEncoder implementations.
        return (
                hasattr(module, 'encoder') and
                hasattr(module, 'module') and
                callable(getattr(module, 'get_rep', None))
        )

    @staticmethod
    def _unique_params(params: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
        """Remove duplicate parameter objects (by identity) to avoid double-optimizing."""
        seen = set()
        uniq = []
        for p in params:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            uniq.append(p)
        return uniq

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
        for p in module.parameters():
            p.requires_grad_(requires_grad)

    def _soft_update_(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        """Polyak update: target = (1-tau)*target + tau*source."""
        with torch.no_grad():
            for t_param, param in zip(target.parameters(), source.parameters()):
                t_param.data.mul_(1.0 - self.tau)
                t_param.data.add_(self.tau * param.data)

    def _gradient_descent(self, loss, optimizer_keys):
        self.optimizer.zero_grad(keys=optimizer_keys)
        loss.backward()
        self._record_grad_norms(getattr(self, "_active_metrics", None), optimizer_keys)
        self.optimizer.step(keys=optimizer_keys)

    def _record_grad_norms(self, metrics, optimizer_keys):
        if metrics is None:
            return
        name_map = {
            'actor': 'TotalGradNormOptionPolicy',
            'qf': 'TotalGradNormQf',
            'log_alpha': 'TotalGradNormLogAlpha',
        }
        params = []
        for key in optimizer_keys:
            key_params = list(self.optimizer.target_parameters(keys=[key]))
            params.extend(key_params)
            metric_key = name_map.get(key)
            if metric_key is not None:
                metrics[metric_key] = utils.compute_total_norm(key_params).detach()
        if params:
            metrics['TotalGradNormAll'] = utils.compute_total_norm(params).detach()

    def get_resume_state(self, include_policy: bool = True):
        state = {}
        if include_policy:
            state['skill_policy_state_dict'] = self.skill_policy.state_dict()

        module_names = [
            'qf1',
            'qf2',
            'target_qf1',
            'target_qf2',
            'target_qf1_mlp',
            'target_qf2_mlp',
            'target_encoder',
            'log_alpha',
        ]
        for name in module_names:
            module = getattr(self, name, None)
            if module is not None:
                state[f'{name}_state_dict'] = module.state_dict()
        return state

    def load_resume_state(self, state, include_policy: bool = True):
        if include_policy:
            policy_state = state.get('skill_policy_state_dict')
            if policy_state is None:
                raise KeyError("Resume checkpoint is missing skill_policy_state_dict")
            self.skill_policy.load_state_dict(policy_state)

        required = ['qf1', 'qf2', 'log_alpha']
        for name in required:
            key = f'{name}_state_dict'
            if key not in state:
                raise KeyError(f"Resume checkpoint is missing {key}")

        module_names = [
            'qf1',
            'qf2',
            'target_qf1',
            'target_qf2',
            'target_qf1_mlp',
            'target_qf2_mlp',
            'target_encoder',
            'log_alpha',
        ]
        for name in module_names:
            key = f'{name}_state_dict'
            module = getattr(self, name, None)
            module_state = state.get(key)
            if module is None or module_state is None:
                continue
            module.load_state_dict(module_state)

    def _optimize_once(self, metrics, v, processed_cat_obs, next_processed_cat_obs, step):
        self._active_metrics = metrics
        obs = v['obs']
        next_obs = v['next_obs']
        skills = v['skills']
        next_skills = v['next_skills']
        actions = v['actions']
        rewards = v['rewards']
        dones = v['dones']
        zero = rewards.detach().new_zeros(())
        metrics.setdefault('LossSacp', zero)
        metrics.setdefault('SacpNewActionLogProbMean', zero)
        metrics.setdefault('Alpha', self.log_alpha.param.detach().exp())
        metrics.setdefault('LogAlpha', self.log_alpha.param.detach())
        metrics.setdefault(
            'AlphaLr',
            torch.as_tensor(
                self.optimizer._optimizers['log_alpha'].param_groups[0].get('lr', 0.0),
                device=self.log_alpha.param.device,
                dtype=self.log_alpha.param.dtype,
            ),
        )
        metrics.setdefault('LossAlpha', zero)
        metrics.setdefault('TotalGradNormAll', zero)
        metrics.setdefault('TotalGradNormOptionPolicy', zero)
        metrics.setdefault('TotalGradNormQf', zero)
        metrics.setdefault('TotalGradNormLogAlpha', zero)

        # 1. Determine Networks and Inputs
        if self.use_encoder:
            # --- Encoder Forward ---
            if self.encoder_trainable:
                z = self.encoder(obs)
            else:
                with torch.no_grad():
                    z = self.encoder(obs)
            
            with torch.no_grad():
                z_next = self.target_encoder(next_obs)
            
            obs_in = utils.get_torch_concat_obs(z, skills)
            next_obs_in = utils.get_torch_concat_obs(z_next, next_skills)
            
            qf1, qf2 = self.qf1_mlp, self.qf2_mlp
            t_qf1, t_qf2 = self.target_qf1_mlp, self.target_qf2_mlp
            policy_net = self.skill_policy_mlp
            
            # For actor update, we need detached Z
            z_det = z.detach()
            actor_obs_in = utils.get_torch_concat_obs(z_det, skills)
            
            # Target update pairs
            target_pairs = [
                (t_qf1, qf1), (t_qf2, qf2)
            ]
            if self.encoder_trainable and (self.target_encoder is not self.encoder):
                target_pairs.append((self.target_encoder, self.encoder))

        else:
            # --- Default Forward ---
            obs_in = processed_cat_obs
            next_obs_in = next_processed_cat_obs
            actor_obs_in = obs_in
            
            qf1, qf2 = self.qf1, self.qf2
            t_qf1, t_qf2 = self.target_qf1, self.target_qf2
            policy_net = self.skill_policy
            
            target_pairs = [
                (t_qf1, qf1), (t_qf2, qf2)
            ]

        # 2. Critic Update
        sac_utils.update_loss_qf(
            self, metrics, v,
            obs=obs_in, actions=actions, next_obs=next_obs_in,
            dones=dones, rewards=rewards * self._reward_scale_factor,
            policy=policy_net,
            qf1=qf1, qf2=qf2, target_qf1=t_qf1, target_qf2=t_qf2
        )
        self._gradient_descent(metrics['LossQf1'] + metrics['LossQf2'], optimizer_keys=['qf'])

        # 3. Actor + Alpha Update
        should_update_actor = (step >= self.actor_start_steps) and (step % self.policy_delay == 0)
        if should_update_actor:
            # Freeze critic for actor update (optimization)
            self._set_requires_grad(qf1, False)
            self._set_requires_grad(qf2, False)
            
            sac_utils.update_loss_sacp(
                self, metrics, v,
                obs=actor_obs_in,
                policy=policy_net,
                qf1=qf1, qf2=qf2
            )
            
            self._set_requires_grad(qf1, True)
            self._set_requires_grad(qf2, True)

            self._gradient_descent(metrics['LossSacp'], optimizer_keys=['actor'])

            # Alpha
            sac_utils.update_loss_alpha(self, metrics, v)
            self._gradient_descent(metrics['LossAlpha'], optimizer_keys=['log_alpha'])

        # 4. Target Update
        sac_utils.update_targets(self, pairs=target_pairs)
        self._active_metrics = None
