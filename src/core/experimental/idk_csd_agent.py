from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from core.hierarchical_phi import resolve_total_phi_dim
from core.isolation_kernel import SoftIsolationKernel
from core.metra_agent import MetraAgent
from core.stage_contract import should_update_target_traj_encoder, uses_external_reward
from utils import agent_utils


class IdkCsdAgent(MetraAgent):
    """
    IDK-CSD agent with joint phi/policy updates and a contrastive-only IK map.
    """

    def __init__(self, config, env, replay_buffer):
        super().__init__(config, env, replay_buffer)
        self.kernel = self.kernel or self._build_shared_kernel()
        self._shared_kernel_initialized = bool(self.init_kme) if self.kme_module is not None else False
        self.contrastive_kernel = self._build_contrastive_kernel()
        self.contrastive_kernel_initialized = False

    def update_joint(self, epoch_data, live_paths: Optional[List[Dict]] = None, mix_lambda: float = 0.0):
        metrics = {}

        if self.replay_buffer is None:
            batch = agent_utils.get_mini_tensors(epoch_data, self.cfg.train.trans_minibatch_size)
        else:
            batch = self._sample_replay_buffer()

        self._normalize_sac_scalars(batch)
        batch['contrastive_rollouts'] = self._sample_contrastive_rollouts(live_paths)

        if hasattr(self.variant, 'set_phase'):
            self.variant.set_phase('joint')
        if hasattr(self.variant, 'set_mix_lambda'):
            self.variant.set_mix_lambda(mix_lambda)

        self._optimize_te(metrics, batch)

        if uses_external_reward(self.cfg):
            self._update_external_rewards(metrics, batch)
        else:
            self._update_rewards(metrics, batch)

        self.total_train_steps += 1
        self._optimize_op(metrics, batch, self.total_train_steps)

        if should_update_target_traj_encoder(self.cfg):
            self.update_target_traj_encoder()

        metrics['MixLambda'] = float(mix_lambda)
        return metrics

    def update_te(self, epoch_data, live_paths=None):
        metrics = {}
        if self.replay_buffer is None:
            batch = agent_utils.get_mini_tensors(epoch_data, self.cfg.train.trans_minibatch_size)
        else:
            batch = self._sample_replay_buffer()
        self._normalize_sac_scalars(batch)
        batch['contrastive_rollouts'] = self._sample_contrastive_rollouts(live_paths)

        if hasattr(self.variant, 'set_phase'):
            self.variant.set_phase('representation')
        self._optimize_te(metrics, batch)

        if should_update_target_traj_encoder(self.cfg):
            self.update_target_traj_encoder()
        return metrics

    def update_sac(self, epoch_data):
        metrics = {}
        if self.replay_buffer is None:
            batch = agent_utils.get_mini_tensors(epoch_data, self.cfg.train.trans_minibatch_size)
        else:
            batch = self._sample_replay_buffer()

        self._normalize_sac_scalars(batch)
        if uses_external_reward(self.cfg):
            self._update_external_rewards(metrics, batch)
        else:
            self._update_rewards(metrics, batch)

        self.total_train_steps += 1
        self._optimize_op(metrics, batch, self.total_train_steps)
        return metrics

    def _build_idk_initial(self, metrics: dict = None):
        if self.kme_module is not None:
            super()._build_idk_initial(metrics)
            self._shared_kernel_initialized = bool(self.init_kme)
        else:
            self._refresh_shared_kernel_from_replay(metrics)
        self._refresh_contrastive_kernel_from_replay(metrics)

    def _sample_contrastive_rollouts(self, live_paths=None):
        rollout_batch_size = self._contrastive_rollout_batch_size()
        rollouts: List[Dict] = []

        if self.replay_buffer is not None and hasattr(self.replay_buffer, 'sample_paths'):
            try:
                replay_rollouts = self.replay_buffer.sample_paths(rollout_batch_size, replace=False)
                rollouts.extend(self._normalize_rollout(rollout) for rollout in replay_rollouts)
            except RuntimeError:
                pass

        if len(rollouts) >= rollout_batch_size:
            return rollouts[:rollout_batch_size]

        if live_paths is None:
            return rollouts

        for path in live_paths:
            rollouts.append(self._convert_live_path(path))
            if len(rollouts) >= rollout_batch_size:
                break
        return rollouts

    def _refresh_contrastive_kernel_from_replay(self, metrics: dict = None):
        if self.contrastive_kernel is None:
            return
        if self.replay_buffer is None or not hasattr(self.replay_buffer, 'sample_paths'):
            return

        rollout_batch_size = self._contrastive_rollout_batch_size()
        try:
            rollouts = self.replay_buffer.sample_paths(rollout_batch_size, replace=False)
        except RuntimeError:
            return

        normalized_rollouts = [self._normalize_rollout(rollout) for rollout in rollouts]
        token_batches = []
        with torch.no_grad():
            for rollout in normalized_rollouts:
                tokens = self.variant.encode_rollout_tokens(self, rollout)
                if tokens.numel() == 0:
                    continue
                token_batches.append(tokens)

        if not token_batches:
            return

        token_tensor = torch.cat(token_batches, dim=0)
        self.contrastive_kernel.fit(token_tensor)
        self.contrastive_kernel_initialized = True

        if metrics is not None:
            metrics['ContrastiveKernelNumTokens'] = int(token_tensor.shape[0])
            metrics['ContrastiveKernelInitialized'] = 1.0

    def _refresh_shared_kernel_from_replay(self, metrics: dict = None):
        if self.kernel is None or self.replay_buffer is None:
            return

        n_transitions = int(self.replay_buffer.n_transitions_stored)
        if n_transitions <= 0:
            return

        sample_size = min(max(int(self.cfg.algo.idk_subsample_size), 1), n_transitions)
        batch = self.replay_buffer.sample_transitions(sample_size)
        obs = torch.from_numpy(batch['obs']).float().to(self.device)
        anchors = self._shared_kernel_input_from_obs(obs)
        if anchors.numel() == 0:
            return

        self.kernel.fit(anchors)
        self._shared_kernel_initialized = True

        if metrics is not None:
            metrics['SharedKernelNumAnchors'] = int(anchors.shape[0])
            metrics['SharedKernelInitialized'] = 1.0

    def get_resume_state(self):
        state = super().get_resume_state()
        state['idk_csd_state'] = {
            'shared_kernel_initialized': bool(self.shared_kernel_initialized),
            'shared_kernel_state_dict': None
            if self.kme_module is not None or self.kernel is None
            else self.kernel.state_dict(),
            'contrastive_kernel_initialized': bool(self.contrastive_kernel_initialized),
            'contrastive_kernel_state_dict': None
            if self.contrastive_kernel is None
            else self.contrastive_kernel.state_dict(),
        }
        return state

    def load_resume_state(self, state):
        super().load_resume_state(state)
        idk_csd_state = state.get('idk_csd_state', {})
        shared_kernel_state = idk_csd_state.get('shared_kernel_state_dict')
        if self.kme_module is None and self.kernel is not None and shared_kernel_state is not None:
            self.kernel.load_state_dict(shared_kernel_state)
        self._shared_kernel_initialized = bool(
            idk_csd_state.get('shared_kernel_initialized', self._shared_kernel_initialized)
        )
        kernel_state = idk_csd_state.get('contrastive_kernel_state_dict')
        if self.contrastive_kernel is not None and kernel_state is not None:
            self.contrastive_kernel.load_state_dict(kernel_state)
        self.contrastive_kernel_initialized = bool(
            idk_csd_state.get('contrastive_kernel_initialized', self.contrastive_kernel_initialized)
        )

    @property
    def shared_kernel_initialized(self) -> bool:
        if self.kme_module is not None:
            return bool(self.init_kme)
        return bool(self._shared_kernel_initialized)

    def _build_shared_kernel(self):
        return SoftIsolationKernel(
            input_dim=self._shared_kernel_input_dim(),
            ensemble_size=100,
            subsample_size=self.cfg.algo.idk_subsample_size,
            temperature=0.0001,
            device=self.device,
        ).to(self.device)

    def _build_contrastive_kernel(self):
        if self.kernel is None:
            return None

        return SoftIsolationKernel(
            input_dim=resolve_total_phi_dim(self.cfg),
            ensemble_size=self.kernel.ensemble_size,
            subsample_size=self.kernel.subsample_size,
            temperature=self.kernel.temperature,
            device=self.device,
        ).to(self.device)

    def _shared_kernel_input_dim(self) -> int:
        if self.cfg.algo.idk_from == 'traj':
            return resolve_total_phi_dim(self.cfg)
        return self.module_obs_dim

    def _contrastive_rollout_batch_size(self) -> int:
        rollout_batch_size = int(self.cfg.algo.contrastive_rollout_batch_size)
        if rollout_batch_size <= 0:
            rollout_batch_size = int(self.cfg.train.traj_batch_size)
        return rollout_batch_size

    def _shared_kernel_input_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.cfg.algo.idk_from != 'traj':
            return obs

        with torch.no_grad():
            return self.encode_phi(obs, use_target=self.cfg.algo.use_target_traj_encoder)

    @staticmethod
    def _normalize_rollout(rollout: Dict) -> Dict:
        normalized = {
            'obs': np.asarray(rollout['obs']),
            'next_obs': np.asarray(rollout['next_obs']),
        }

        if 'skills' in rollout:
            normalized['skills'] = np.asarray(rollout['skills'])

        traj_len = int(normalized['obs'].shape[0])
        time_idxs = rollout.get('time_idxs')
        next_time_idxs = rollout.get('next_time_idxs')

        if time_idxs is None:
            normalized['time_idxs'] = np.arange(traj_len, dtype=np.float32)
        else:
            normalized['time_idxs'] = np.asarray(time_idxs, dtype=np.float32).reshape(-1)

        if next_time_idxs is None:
            normalized['next_time_idxs'] = np.arange(1, traj_len + 1, dtype=np.float32)
        else:
            normalized['next_time_idxs'] = np.asarray(next_time_idxs, dtype=np.float32).reshape(-1)

        return normalized

    @staticmethod
    def _convert_live_path(path: Dict) -> Dict:
        skill = path['agent_infos'].get('skill')
        if skill is None:
            raise KeyError("IDK-CSD contrastive rollout requires path['agent_infos']['skill']")

        traj_len = int(len(path['observations']))
        return {
            'obs': np.asarray(path['observations']),
            'next_obs': np.asarray(path['next_observations']),
            'skills': np.asarray(skill),
            'time_idxs': np.arange(traj_len, dtype=np.float32),
            'next_time_idxs': np.arange(1, traj_len + 1, dtype=np.float32),
        }
