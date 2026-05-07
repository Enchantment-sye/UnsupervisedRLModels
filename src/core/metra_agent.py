import copy
import os
import time
from math import sqrt
from typing import List, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from utils import utils
from utils.utils import _finalize_lr, OptimizerGroupWrapper
from core.cascade_actor import CascadeActor
from core.metra_config import MetraConfig
from core.metra_builder import MetraAgentBuilder
from core.stage_contract import get_base_algo_name, should_update_target_traj_encoder, should_use_kme, uses_external_reward
from utils import agent_utils
from core.kme_module import KMEModule
from models.encoders import WithEncoder
from utils.checkpointing import safe_torch_load


class MetraAgent:
    """
    Refactored DrQ agent with METRA skill discovery integration.
    """
    def __init__(self, config: MetraConfig, env, replay_buffer):
        self.cfg = config
        self._env = env
        self.replay_buffer = replay_buffer
        
        # Use Builder to construct components
        builder = MetraAgentBuilder(config, env, replay_buffer)
        components = builder.build()
        
        # Unpack components
        self.device = components['device']
        self.shared_encoder = components['shared_encoder']
        self.module_obs_dim = components['module_obs_dim']
        self.make_encoder_fn = components['make_encoder_fn']
        self.with_encoder_fn = components['with_encoder_fn']
        
        self.traj_encoder = components['traj_encoder']
        self.target_traj_encoder = components['target_traj_encoder']
        self.traj_latent_normalizer = components.get('traj_latent_normalizer')
        self.target_traj_latent_normalizer = components.get('target_traj_latent_normalizer')
        self.dist_predictor = components['dist_predictor']
        self.dual_lam = components['dual_lam']
        self.skill_dynamics = components.get('skill_dynamics')
        self.sd_input_bn = components.get('sd_input_bn')
        self.sd_target_bn = components.get('sd_target_bn')
        
        self.sac_trainer = components['sac_trainer']
        self._optimizer = components['optimizer']
        self.param_modules = components['param_modules']
        self.variant = components['variant']
        
        # KME components
        self.kernel = components.get('kernel', None)
        self.kme_module = None
        if should_use_kme(self.cfg):
            # Helper to get the correct encoder
            def traj_encoder_getter():
                return self.target_traj_encoder if self.cfg.algo.use_target_traj_encoder else self.traj_encoder

            def phi_from_obs_getter(obs):
                return self.encode_phi(obs, use_target=self.cfg.algo.use_target_traj_encoder)
                
            self.kme_module = KMEModule(
                config=self.cfg,
                device=self.device,
                kernel=self.kernel,
                replay_buffer=self.replay_buffer,
                traj_encoder_getter=traj_encoder_getter,
                phi_from_obs_getter=phi_from_obs_getter,
            )
            
            # For backward compatibility if variant accesses these directly
            # Though variant usually accesses agent.kernel, which we kept
            # But variant might access agent.kme_vector
            self.kme_vector = None 
            
        # Counters
        self.step_itr = 0
        self.total_train_steps = 0
        self.current_skill_level = 1
        self.best_skill = None
        self._last_replay_transfer_seconds = 0.0

    @property
    def kme_vector(self):
        if self.kme_module:
            return self.kme_module.get_kme_vector()
        return None
        
    @kme_vector.setter
    def kme_vector(self, value):
        # Allow setting if needed, or ignore?
        # KMEModule manages it. 
        if self.kme_module:
            self.kme_module.kme_vector = value

    @property
    def init_kme(self):
        if self.kme_module:
            return self.kme_module.init_kme
        return False
    
    @init_kme.setter
    def init_kme(self, value):
        if self.kme_module:
            self.kme_module.init_kme = value
            
    @property
    def path_datas(self):
        if self.kme_module:
            return self.kme_module.path_datas
        return []
    
    @path_datas.setter
    def path_datas(self, value):
        if self.kme_module:
            self.kme_module.path_datas = value

    def _extract_state_from_obs(self, obs):
        return agent_utils.extract_state_from_obs(obs)

    def _get_phi_normalizer(self, use_target: bool = False):
        if use_target and self.target_traj_latent_normalizer is not None:
            return self.target_traj_latent_normalizer
        return self.traj_latent_normalizer

    def normalize_phi_tensor(self, phi: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        normalizer = self._get_phi_normalizer(use_target=use_target)
        if normalizer is None:
            return phi
        return normalizer(phi)

    def phi_from_encoder_output(self, encoded, use_target: bool = False) -> torch.Tensor:
        if torch.is_tensor(encoded):
            phi = encoded
        else:
            phi = getattr(encoded, 'mean', None)
            if phi is None or callable(phi):
                raise TypeError('traj_encoder must expose phi(s) via distribution.mean or direct tensor output')
        return self.normalize_phi_tensor(phi, use_target=use_target)

    def encode_phi(self, obs, use_target: bool = False) -> torch.Tensor:
        traj_encoder = self.target_traj_encoder if use_target and self.target_traj_encoder is not None else self.traj_encoder
        if traj_encoder is None:
            raise ValueError("Current config does not build a trajectory encoder, so phi(s) is unavailable.")
        encoded = traj_encoder(obs)
        return self.phi_from_encoder_output(encoded, use_target=use_target)

    def update(self, epoch_data):
        metrics = {}
        self._ensure_train_diagnostic_defaults(metrics)
        total_replay_transfer = 0.0
        for _ in range(self.cfg.train.trans_optimization_epochs):
            self.total_train_steps += 1
            
            # 1. Sample
            if self.replay_buffer is None:
                v = agent_utils.get_mini_tensors(epoch_data, self.cfg.train.trans_minibatch_size)
            else:
                v = self._sample_replay_buffer()
                total_replay_transfer += float(getattr(self, "_last_replay_transfer_seconds", 0.0))

            self._normalize_sac_scalars(v)
                
            # 2. Optimize Trajectory Encoder (if pretraining)
            if self.cfg.log.stage == 'pre_training':
                self._optimize_te(metrics, v)
                
            # 3. Compute Rewards
            if uses_external_reward(self.cfg):
                self._update_external_rewards(metrics, v)
            else:
                self._update_rewards(metrics, v)
            
            # 4. Optimize SAC
            self._optimize_op(metrics, v, self.total_train_steps)
            
            # 5. Update Target TE
            if should_update_target_traj_encoder(self.cfg):
                self.update_target_traj_encoder()

        if self.replay_buffer is not None:
            metrics["TimeReplayTransfer"] = total_replay_transfer
                
        return metrics

    def _update_rewards(self, metrics, v):
        self.variant.compute_intrinsic_reward(self, v, metrics)

    def _normalize_sac_scalars(self, batch):
        for key in ('rewards', 'dones'):
            value = batch.get(key)
            if value is None or not torch.is_tensor(value):
                continue
            if value.dim() == 1:
                continue
            value = value.reshape(value.shape[0], -1)
            if value.shape[1] != 1:
                raise ValueError(f"Expected scalar {key} per transition, got shape={tuple(value.shape)}")
            batch[key] = value[:, 0]

    def _update_external_rewards(self, metrics, v):
        rewards = v.get('rewards')
        if rewards is None:
            raise KeyError("Downstream training requires environment rewards in batch['rewards']")
        if not torch.is_tensor(rewards):
            rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        else:
            rewards = rewards.to(self.device).float()
        if rewards.dim() > 1:
            rewards = rewards.reshape(rewards.shape[0], -1)
            if rewards.shape[1] != 1:
                raise ValueError(f"Expected scalar environment rewards, got shape={tuple(rewards.shape)}")
            rewards = rewards[:, 0]
        v['rewards'] = rewards
        metrics.update({
            'RewardMean': rewards.mean().item(),
            'RewardStd': rewards.std().item(),
            'ExternalRewardMean': rewards.mean().item(),
            'ExternalRewardStd': rewards.std().item(),
        })

    def _optimize_op(self, metrics, internal_vars, step):
        B = internal_vars['obs'].shape[0]
        dev = internal_vars['obs'].device
        
        # Ensure skills
        if 'skills' not in internal_vars or internal_vars['skills'] is None:
             internal_vars['skills'] = torch.zeros((B, 0), device=dev)
        if 'next_skills' not in internal_vars or internal_vars['next_skills'] is None:
             internal_vars['next_skills'] = torch.zeros((B, 0), device=dev)
             
        # Process obs (normalization/augmentations if inside policy)
        processed_obs = self.sac_trainer.skill_policy.process_observations(internal_vars['obs'])
        processed_next = self.sac_trainer.skill_policy.process_observations(internal_vars['next_obs'])
        
        processed_cat_obs = utils.get_torch_concat_obs(processed_obs, internal_vars['skills'])
        next_processed_cat_obs = utils.get_torch_concat_obs(processed_next, internal_vars['next_skills'])
        
        self.sac_trainer._optimize_once(metrics, internal_vars, processed_cat_obs, next_processed_cat_obs, step)

    def _optimize_te(self, metrics, internal_vars):
        self._ensure_train_diagnostic_defaults(metrics, internal_vars)
        # Always re-compute rewards for TE loss calculation to maintain gradient flow
        # Replay buffer stores environment rewards which are detached from the computational graph
        # But we don't need to update metrics['RewardMean'] again if it's already done in _update_rewards
        # However, _optimize_te is called BEFORE _update_rewards in update() loop usually?
        # In update(): 1. Sample, 2. Optimize TE, 3. Compute Rewards, 4. Optimize SAC.
        
        # If we are in Representation Phase (only TE update), we call _optimize_te directly.
        # compute_te_loss usually recalculates rewards with online encoder.
        
        loss_te = self.variant.compute_te_loss(self, internal_vars, metrics)
        
        # DADS-style algorithms optimize skill_dynamics for TE; other variants optimize traj_encoder.
        opt_key = 'skill_dynamics' if get_base_algo_name(self.cfg) == 'dads' else 'traj_encoder'
        self._gradient_descent(loss_te, optimizer_keys=[opt_key], metrics=metrics)
        
        self.variant.update_auxiliary(self, internal_vars, metrics)

    def _gradient_descent(self, loss, optimizer_keys, metrics=None):
        self._optimizer.zero_grad(keys=optimizer_keys)
        loss.backward()
        self._record_grad_norms(metrics, optimizer_keys)
        self._optimizer.step(keys=optimizer_keys)

    def _ensure_train_diagnostic_defaults(self, metrics, batch=None):
        if batch:
            zero = None
            for value in batch.values():
                if torch.is_tensor(value):
                    zero = value.detach().new_zeros(())
                    break
            if zero is None:
                zero = torch.zeros((), device=self.device)
        else:
            zero = torch.zeros((), device=self.device)
        metrics.setdefault('TotalGradNormAll', zero)
        metrics.setdefault('TotalGradNormTrajEncoder', zero)
        metrics.setdefault('TotalGradNormDualLam', zero)

    def _record_grad_norms(self, metrics, optimizer_keys):
        if metrics is None:
            return
        self._ensure_train_diagnostic_defaults(metrics)
        name_map = {
            'traj_encoder': 'TotalGradNormTrajEncoder',
            'dual_lam': 'TotalGradNormDualLam',
        }
        params = []
        for key in optimizer_keys:
            key_params = list(self._optimizer.target_parameters(keys=[key]))
            params.extend(key_params)
            metric_key = name_map.get(key)
            if metric_key is not None:
                metrics[metric_key] = utils.compute_total_norm(key_params).detach()
        if params:
            metrics['TotalGradNormAll'] = utils.compute_total_norm(params).detach()

    def _update_replay_buffer(self, data):
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    elif cur_list.ndim > 2:
                        cur_list = cur_list.reshape(cur_list.shape[0], -1)
                    path[key] = cur_list
                
                # Handle KME augmentation if needed
                if should_use_kme(self.cfg) and self.init_kme:
                    traj_obs = path['obs']
                    if traj_obs.dtype in [np.uint8, np.float32, np.float64]:
                         traj_obs = torch.from_numpy(traj_obs).float().to(self.device)
                    else:
                         traj_obs = traj_obs.to(self.device)
                    
                    path['skill_kme'] = self.kme_module.compute_skill_kme(traj_obs)
                
                self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self):
        transfer_started = time.perf_counter()
        samples = self.replay_buffer.sample_transitions(self.cfg.train.trans_minibatch_size)
        data = {}
        non_blocking = bool(torch.cuda.is_available() and str(self.device).startswith("cuda"))
        for key, value in samples.items():
            if value.shape[1] == 1 and 'skill' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = agent_utils.numpy_batch_to_torch(
                value,
                self.device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
        self._last_replay_transfer_seconds = time.perf_counter() - transfer_started
        return data

    @torch.no_grad()
    def update_target_traj_encoder(self):
        if not self.target_traj_encoder: return
        tau = self.cfg.train.sac_tau
        # ... Polyak update ...
        for p, tp in zip(self.traj_encoder.parameters(), self.target_traj_encoder.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)
        if self.traj_latent_normalizer is not None and self.target_traj_latent_normalizer is not None:
            for p, tp in zip(self.traj_latent_normalizer.parameters(), self.target_traj_latent_normalizer.parameters()):
                tp.data.mul_(1.0 - tau)
                tp.data.add_(tau * p.data)

    def save(self, path):
        torch.save({
            'discrete': self.cfg.algo.discrete,
            'dim_skill': self.cfg.algo.dim_skill,
            'policy': self.sac_trainer.skill_policy,
        }, path)

    def load_component_checkpoints(self, skill_policy_path: str, traj_encoder_path: Optional[str] = None, strict: bool = True):
        if not skill_policy_path:
            raise ValueError("skill_policy_path is required for component checkpoint loading")

        skill_policy = self._load_module_checkpoint(skill_policy_path, keys=('policy', 'skill_policy', 'module'))
        checkpoint_policy_core = self._unwrap_policy_core_from(skill_policy)
        current_policy_core = self._unwrap_policy_core()
        if isinstance(checkpoint_policy_core, CascadeActor):
            if not isinstance(current_policy_core, CascadeActor):
                raise ValueError(
                    f"Checkpoint {os.path.abspath(os.path.expanduser(skill_policy_path))} contains a CascadeActor "
                    f"with {len(checkpoint_policy_core.stages)} stages, but current policy is not a CascadeActor. "
                    "Check --use_cascade / --algo configuration."
                )
            # Component loading starts from a fresh policy, so grow cascade stages to
            # match the checkpoint before loading weights.
            self.ensure_policy_stage_count(len(checkpoint_policy_core.stages))
        elif isinstance(current_policy_core, CascadeActor):
            raise ValueError(
                f"Checkpoint {os.path.abspath(os.path.expanduser(skill_policy_path))} contains a non-cascade policy, "
                f"but current policy is a CascadeActor with {len(current_policy_core.stages)} stage(s). "
                "Check eval config or restore training args from the source run."
            )
        self.sac_trainer.skill_policy.load_state_dict(skill_policy.state_dict(), strict=strict)
        self.sac_trainer.skill_policy.to(self.device)
        self.sac_trainer.skill_policy.eval()

        if traj_encoder_path is None:
            return

        if self.traj_encoder is None:
            raise ValueError(
                "Current config does not build a trajectory encoder, but traj_encoder_path was provided. "
                "Use stage=pre_training for trajectory analysis."
            )

        traj_checkpoint = safe_torch_load(os.path.abspath(os.path.expanduser(traj_encoder_path)), map_location=self.device)
        traj_encoder = self._resolve_module_from_checkpoint(
            traj_checkpoint,
            keys=('traj_encoder', 'encoder', 'module'),
            path=traj_encoder_path,
            allow_raw_module=True,
        )
        self.traj_encoder.load_state_dict(traj_encoder.state_dict(), strict=strict)
        self.traj_encoder.to(self.device)
        self.traj_encoder.eval()

        traj_latent_normalizer = self._resolve_module_from_checkpoint(
            traj_checkpoint,
            keys=('traj_latent_normalizer',),
            path=traj_encoder_path,
            required=False,
            allow_raw_module=False,
        )
        if self.traj_latent_normalizer is not None and traj_latent_normalizer is not None:
            self.traj_latent_normalizer.load_state_dict(traj_latent_normalizer.state_dict(), strict=strict)
            self.traj_latent_normalizer.to(self.device)
            self.traj_latent_normalizer.eval()

        if self.target_traj_encoder is not None:
            self.target_traj_encoder.load_state_dict(traj_encoder.state_dict(), strict=strict)
            self.target_traj_encoder.to(self.device)
            self.target_traj_encoder.eval()
        if self.target_traj_latent_normalizer is not None and self.traj_latent_normalizer is not None:
            self.target_traj_latent_normalizer.load_state_dict(self.traj_latent_normalizer.state_dict(), strict=strict)
            self.target_traj_latent_normalizer.to(self.device)
            self.target_traj_latent_normalizer.eval()

    def _load_module_checkpoint(self, path: str, keys):
        resolved_path = os.path.abspath(os.path.expanduser(path))
        raw = safe_torch_load(resolved_path, map_location=self.device)
        module = self._resolve_module_from_checkpoint(raw, keys=keys, path=resolved_path, allow_raw_module=True)
        return module.to(self.device)

    @staticmethod
    def _resolve_module_from_checkpoint(raw, keys, path: str, required: bool = True, allow_raw_module: bool = True):
        if isinstance(raw, torch.nn.Module):
            if not allow_raw_module:
                if not required:
                    return None
                raise ValueError(
                    f"Checkpoint {os.path.abspath(os.path.expanduser(path))} stores a bare module, but "
                    f"keys={tuple(keys)!r} were required."
                )
            return raw
        if isinstance(raw, dict):
            for key in keys:
                candidate = raw.get(key)
                if isinstance(candidate, torch.nn.Module):
                    return candidate
            if not required:
                return None
            raise ValueError(
                f"Checkpoint {os.path.abspath(os.path.expanduser(path))} does not contain a module under "
                f"keys={tuple(keys)!r}; available keys={list(raw.keys())}"
            )
        if not required:
            return None
        raise ValueError(f"Unsupported checkpoint type at {os.path.abspath(os.path.expanduser(path))}: {type(raw)!r}")

    def all_parameters(self):
        for m in self.param_modules.values():
            for p in m.parameters():
                yield p

    @staticmethod
    def _unwrap_policy_core_from(policy_like):
        policy_module = getattr(policy_like, '_module', policy_like)
        if isinstance(policy_module, WithEncoder):
            return policy_module.module
        return policy_module

    def _unwrap_policy_core(self):
        return self._unwrap_policy_core_from(self.sac_trainer.skill_policy)

    def _get_cascade_stage_count(self):
        policy_core = self._unwrap_policy_core()
        if isinstance(policy_core, CascadeActor):
            return len(policy_core.stages)
        return 1

    def add_policy_stage(self):
        """Append one cascade stage and register its parameters with the actor optimizer."""
        policy_core = self._unwrap_policy_core()
        if not isinstance(policy_core, CascadeActor):
            raise TypeError("add_policy_stage requires the policy core to be a CascadeActor")

        actor_opt = self.sac_trainer.optimizer._optimizers['actor']
        actor_lr = actor_opt.param_groups[0]['lr']
        new_stage = policy_core.add_stage(init_from_prev=self.cfg.cascade.cascade_init_from_prev)
        actor_opt.add_param_group({'params': new_stage.parameters(), 'lr': actor_lr})
        return new_stage

    def ensure_policy_stage_count(self, target_stage_count):
        if target_stage_count is None:
            return

        policy_core = self._unwrap_policy_core()
        if not isinstance(policy_core, CascadeActor):
            if int(target_stage_count) not in (0, 1):
                raise ValueError(
                    f"Resume checkpoint expects cascade stage count {target_stage_count}, "
                    "but current policy is not a CascadeActor."
                )
            return

        target_stage_count = int(target_stage_count)
        if target_stage_count < len(policy_core.stages):
            raise ValueError(
                f"Cannot shrink CascadeActor from {len(policy_core.stages)} to {target_stage_count} during resume."
            )

        while len(policy_core.stages) < target_stage_count:
            self.add_policy_stage()

    def get_resume_state(self):
        state = {
            'policy_state_dict': self.sac_trainer.skill_policy.state_dict(),
            'traj_encoder_state_dict': None if self.traj_encoder is None else self.traj_encoder.state_dict(),
            'target_traj_encoder_state_dict': None
            if self.target_traj_encoder is None
            else self.target_traj_encoder.state_dict(),
            'traj_latent_normalizer_state_dict': None
            if self.traj_latent_normalizer is None
            else self.traj_latent_normalizer.state_dict(),
            'target_traj_latent_normalizer_state_dict': None
            if self.target_traj_latent_normalizer is None
            else self.target_traj_latent_normalizer.state_dict(),
            'dual_lam_state_dict': None if self.dual_lam is None else self.dual_lam.state_dict(),
            'dist_predictor_state_dict': None
            if self.dist_predictor is None
            else self.dist_predictor.state_dict(),
            'skill_dynamics_state_dict': None
            if self.skill_dynamics is None
            else self.skill_dynamics.state_dict(),
            'sac_trainer_state': self.sac_trainer.get_resume_state(include_policy=False),
            'total_train_steps': int(self.total_train_steps),
            'current_skill_level': int(self.current_skill_level),
            'best_skill': None if self.best_skill is None else np.asarray(self.best_skill, dtype=np.float32),
            'structure': {
                'cascade_stage_count': int(self._get_cascade_stage_count()),
            },
        }

        if self.kme_module is not None:
            state['kme_state'] = {
                'init_kme': bool(self.init_kme),
                'idk_step_counter': int(self.kme_module.idk_step_counter),
                'kme_vector': None if self.kme_vector is None else self.kme_vector.detach().cpu(),
                'kernel_state_dict': None if self.kernel is None else self.kernel.state_dict(),
            }
        return state

    def load_resume_state(self, state):
        policy_state = state.get('policy_state_dict')
        if policy_state is None:
            raise KeyError("Resume checkpoint is missing policy_state_dict")

        if self.traj_encoder is not None and state.get('traj_encoder_state_dict') is None:
            raise KeyError("Resume checkpoint is missing traj_encoder_state_dict")

        self.ensure_policy_stage_count(state.get('structure', {}).get('cascade_stage_count'))

        self.sac_trainer.skill_policy.load_state_dict(policy_state)
        if self.traj_encoder is not None:
            self.traj_encoder.load_state_dict(state['traj_encoder_state_dict'])

        optional_pairs = [
            ('target_traj_encoder', 'target_traj_encoder_state_dict'),
            ('traj_latent_normalizer', 'traj_latent_normalizer_state_dict'),
            ('target_traj_latent_normalizer', 'target_traj_latent_normalizer_state_dict'),
            ('dual_lam', 'dual_lam_state_dict'),
            ('dist_predictor', 'dist_predictor_state_dict'),
            ('skill_dynamics', 'skill_dynamics_state_dict'),
        ]
        for attr, key in optional_pairs:
            module = getattr(self, attr, None)
            module_state = state.get(key)
            if module is None or module_state is None:
                continue
            module.load_state_dict(module_state)

        sac_state = state.get('sac_trainer_state')
        if sac_state is None:
            raise KeyError("Resume checkpoint is missing sac_trainer_state")
        self.sac_trainer.load_resume_state(sac_state, include_policy=False)

        self.total_train_steps = int(state.get('total_train_steps', self.total_train_steps))
        self.current_skill_level = int(state.get('current_skill_level', self.current_skill_level))
        best_skill = state.get('best_skill')
        if best_skill is not None:
            self.best_skill = np.asarray(best_skill, dtype=np.float32)

        if self.kme_module is not None:
            kme_state = state.get('kme_state', {})
            kernel_state = kme_state.get('kernel_state_dict')
            if self.kernel is not None and kernel_state is not None:
                self.kernel.load_state_dict(kernel_state)
            kme_vector = kme_state.get('kme_vector')
            if kme_vector is not None:
                self.kme_vector = kme_vector.to(self.device)
            self.init_kme = bool(kme_state.get('init_kme', self.init_kme))
            self.kme_module.idk_step_counter = int(kme_state.get('idk_step_counter', self.kme_module.idk_step_counter))
    
    # KME delegation
    def _maybe_refresh_idk_from_replay(self, metrics: dict=None):
        if self.kme_module:
            self.kme_module.maybe_refresh_from_replay(metrics)
            
    def _build_idk_initial(self, metrics: dict=None):
        if self.kme_module:
            self.kme_module.build_initial(metrics)
