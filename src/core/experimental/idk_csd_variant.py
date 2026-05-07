from __future__ import annotations

from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.metra_variants import IksdVariant


class IdkCsdVariant(IksdVariant):
    name = "idk_csd"

    def __init__(self, config):
        super().__init__(config)
        self._is_warmup = True
        self._phase = "warmup"
        self._mix_lambda = 1.0
        self._positional_encoding_cache = {}

    def set_warmup(self, is_warmup: bool):
        self.set_phase("warmup" if is_warmup else "policy")

    def set_phase(self, phase: str):
        if phase not in ("warmup", "representation", "policy", "joint"):
            raise ValueError(f"Unsupported IDK-CSD phase: {phase}")
        self._phase = phase
        self._is_warmup = phase == "warmup"

    def set_mix_lambda(self, mix_lambda: float):
        self._mix_lambda = float(mix_lambda)

    def compute_intrinsic_reward(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        if self._phase not in ("warmup", "policy", "joint"):
            return super().compute_intrinsic_reward(agent, batch, metrics)

        rewards, reward_metrics = self._compute_warmup_style_reward(agent, batch)
        batch['skill_rewards'] = rewards
        batch['rewards'] = rewards
        metrics.update(reward_metrics)
        return rewards

    def compute_te_loss(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        if self._phase == "warmup":
            return self._compute_warmup_te_loss(agent, batch, metrics)
        if self._phase == "joint":
            return self._compute_joint_te_loss(agent, batch, metrics)
        return self._compute_contrastive_te_loss(agent, batch, metrics)

    def _compute_warmup_style_reward(self, agent, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        use_target = self.cfg.use_target_traj_encoder
        cur_z = agent.encode_phi(batch['obs'], use_target=use_target)
        next_z = agent.encode_phi(batch['next_obs'], use_target=use_target)
        batch['cur_phi'] = cur_z
        batch['next_phi'] = next_z

        if self._use_kernel_features(agent):
            batch['kernel_cur_z'] = agent.kernel(cur_z) / sqrt(agent.kernel.ensemble_size)
            batch['kernel_next_z'] = agent.kernel(next_z) / sqrt(agent.kernel.ensemble_size)

        rewards = ((next_z - cur_z) * batch['skills']).sum(dim=1)
        delta = next_z - cur_z
        return rewards, self._build_reward_metrics(batch, rewards, delta)

    def _compute_joint_te_loss(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        batch_online = batch.copy()
        self._populate_embeddings(agent, batch_online, use_target=False)

        cur_z = batch_online['cur_phi']
        next_z = batch_online['next_phi']
        metrics['currentStateMean'] = torch.square(cur_z).mean().item()

        loss_metra_main, warmup_cst, warmup_metrics = self._compute_warmup_terms(batch_online)
        loss_nce, contrastive_cst, contrastive_metrics = self._compute_contrastive_terms(
            agent,
            batch_online,
            batch_online.get('contrastive_rollouts', []),
        )

        metrics.update(warmup_metrics)
        metrics.update(contrastive_metrics)

        if self.cfg.algo.dual_reg:
            dual_lam = agent.dual_lam.param.exp()
            loss_metra = loss_metra_main - (dual_lam.detach() * warmup_cst).mean()
            loss_contrastive = loss_nce - (dual_lam.detach() * contrastive_cst).mean()
            mixed_cst = self._mix_lambda * warmup_cst + (1.0 - self._mix_lambda) * contrastive_cst
            batch['cst_penalty'] = mixed_cst
            metrics['DualCstPenalty'] = mixed_cst.mean().item()
        else:
            loss_metra = loss_metra_main
            loss_contrastive = loss_nce
            batch['cst_penalty'] = torch.zeros_like(cur_z[:, 0])

        loss_te = self._mix_lambda * loss_metra + (1.0 - self._mix_lambda) * loss_contrastive
        metrics.update({
            'MixLambda': float(self._mix_lambda),
            'LossTeMetra': loss_metra.item(),
            'LossTeContrastive': loss_contrastive.item(),
            'LossTeMixed': loss_te.item(),
            'LossTe': loss_te,
        })
        return loss_te

    def _compute_warmup_te_loss(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        batch_online = batch.copy()
        self._populate_embeddings(agent, batch_online, use_target=False)

        cur_z = batch_online['cur_phi']
        metrics['currentStateMean'] = torch.square(cur_z).mean().item()

        loss_main, cst_penalty, warmup_metrics = self._compute_warmup_terms(batch_online)
        metrics.update(warmup_metrics)

        if self.cfg.algo.dual_reg:
            dual_lam = agent.dual_lam.param.exp()
            loss_te = loss_main - (dual_lam.detach() * cst_penalty).mean()
            batch['cst_penalty'] = cst_penalty
            metrics['DualCstPenalty'] = cst_penalty.mean().item()
        else:
            batch['cst_penalty'] = torch.zeros_like(cur_z[:, 0])
            loss_te = loss_main

        metrics.update({
            'LossTeMetra': loss_te.item(),
            'LossTe': loss_te,
            'MixLambda': 1.0,
        })
        return loss_te

    def _compute_contrastive_te_loss(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        batch_online = batch.copy()
        self._populate_embeddings(agent, batch_online, use_target=False)

        cur_z = batch_online['cur_phi']
        metrics['currentStateMean'] = torch.square(cur_z).mean().item()

        loss_nce, cst_penalty, contrastive_metrics = self._compute_contrastive_terms(
            agent,
            batch_online,
            batch_online.get('contrastive_rollouts', []),
        )
        metrics.update(contrastive_metrics)

        if self.cfg.algo.dual_reg:
            dual_lam = agent.dual_lam.param.exp()
            loss_te = loss_nce - (dual_lam.detach() * cst_penalty).mean()
            batch['cst_penalty'] = cst_penalty
            metrics['DualCstPenalty'] = cst_penalty.mean().item()
        else:
            batch['cst_penalty'] = torch.zeros_like(cur_z[:, 0])
            loss_te = loss_nce

        metrics.update({
            'LossTeContrastive': loss_te.item(),
            'LossTe': loss_te,
            'MixLambda': 0.0,
        })
        return loss_te

    def _compute_warmup_terms(self, batch_online: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        cur_z = batch_online['cur_phi']
        next_z = batch_online['next_phi']

        rewards = ((next_z - cur_z) * batch_online['skills']).sum(dim=1)
        loss_main = -rewards.mean()
        cst_penalty = torch.zeros_like(rewards)
        metrics = {
            'WarmupReward': rewards.mean().item(),
        }

        if self.cfg.algo.dual_reg:
            phi_diff_sq = torch.square(next_z - cur_z).mean(dim=1)
            cst_penalty = torch.clamp(
                1.0 - phi_diff_sq,
                max=self.cfg.algo.dual_slack,
            )
            metrics['WarmupConstraintMean'] = cst_penalty.mean().item()

        return loss_main, cst_penalty, metrics

    def _compute_contrastive_terms(
        self,
        agent,
        batch_online: Dict,
        rollouts: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        cur_z = batch_online['cur_phi']
        next_z = batch_online['next_phi']

        loss_nce, nce_metrics = self._compute_rollout_nce_loss(agent, rollouts)
        metrics = dict(nce_metrics)
        cst_penalty = torch.zeros_like(cur_z[:, 0])

        if self.cfg.algo.dual_reg:
            cst_base = self._compute_base_constraint_penalty(agent, batch_online)
            cst_temp = self._compute_temporal_constraint_penalty(cur_z, next_z)
            cst_penalty = cst_base + cst_temp
            metrics.update({
                'BaseConstraintMean': cst_base.mean().item(),
                'TemporalConstraintMean': cst_temp.mean().item(),
                'ContrastiveConstraintMean': cst_penalty.mean().item(),
            })

        diff_sq = torch.square(next_z - cur_z).mean(dim=1)
        budget = float(self.cfg.algo.contrastive_temporal_budget)
        metrics.update({
            'LossNCE': loss_nce.item(),
            'TemporalViolationMean': torch.relu(diff_sq - budget).mean().item(),
        })
        return loss_nce, cst_penalty, metrics

    def _compute_rollout_nce_loss(self, agent, rollouts: List[Dict]) -> Tuple[torch.Tensor, Dict]:
        metrics = {
            'ContrastiveAcc': 0.0,
            'NumContrastiveSkills': 0,
            'NumContrastiveRollouts': 0,
        }

        skill_to_embeddings = {}
        valid_rollouts = 0
        for rollout in rollouts:
            if rollout is None:
                continue
            skill_key = self._rollout_skill_key(rollout)
            if skill_key is None:
                continue
            embedding = self._encode_rollout(agent, rollout)
            skill_to_embeddings.setdefault(skill_key, []).append(embedding)
            valid_rollouts += 1

        metrics['NumContrastiveRollouts'] = valid_rollouts

        anchors = []
        positives = []
        for embeddings in skill_to_embeddings.values():
            if len(embeddings) < 2:
                continue
            anchors.append(embeddings[0])
            positives.append(embeddings[1])

        metrics['NumContrastiveSkills'] = len(anchors)
        if len(anchors) < 2:
            zero_ref = next(iter(skill_to_embeddings.values()), None)
            if zero_ref is None:
                return agent.dual_lam.param.sum() * 0.0, metrics
            return self._graph_zero(zero_ref[0]), metrics

        anchor_tensor = torch.stack(anchors, dim=0)
        positive_tensor = torch.stack(positives, dim=0)
        logits = torch.matmul(anchor_tensor, positive_tensor.T)
        logits = logits / self.cfg.algo.contrastive_temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_nce = F.cross_entropy(logits, labels)

        metrics['ContrastiveAcc'] = (logits.argmax(dim=1) == labels).float().mean().item()
        return loss_nce, metrics

    def encode_rollout_tokens(self, agent, rollout: Dict) -> torch.Tensor:
        states = self._rollout_states_tensor(agent, rollout)
        latents = agent.encode_phi(states, use_target=False)
        positions = self._rollout_positions_tensor(agent, rollout, latents.shape[0])
        return self._apply_rollout_positional_encoding(latents, positions)

    def _encode_rollout(self, agent, rollout: Dict) -> torch.Tensor:
        tokens = self.encode_rollout_tokens(agent, rollout)
        if self._use_contrastive_kernel(agent):
            tokens = agent.contrastive_kernel(tokens) / sqrt(agent.contrastive_kernel.ensemble_size)
        return tokens.mean(dim=0)

    def _rollout_states_tensor(self, agent, rollout: Dict) -> torch.Tensor:
        obs = self._to_device_tensor(agent, rollout['obs'])
        next_obs = self._to_device_tensor(agent, rollout['next_obs'])
        return torch.cat([obs, next_obs[-1:]], dim=0)

    def _rollout_positions_tensor(self, agent, rollout: Dict, seq_len: int) -> torch.Tensor:
        time_idxs = rollout.get('time_idxs')
        next_time_idxs = rollout.get('next_time_idxs')
        if time_idxs is None or next_time_idxs is None:
            return torch.arange(seq_len, dtype=torch.float32, device=agent.device)

        time_tensor = self._to_device_tensor(agent, time_idxs).reshape(-1).float()
        next_time_tensor = self._to_device_tensor(agent, next_time_idxs).reshape(-1).float()

        if time_tensor.numel() == seq_len:
            return time_tensor
        if time_tensor.numel() == seq_len - 1 and next_time_tensor.numel() >= 1:
            return torch.cat([time_tensor, next_time_tensor[-1:]], dim=0)
        return torch.arange(seq_len, dtype=torch.float32, device=agent.device)

    def _apply_rollout_positional_encoding(self, latents: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        mode = getattr(self.cfg.algo, 'traj_pos_encoding', 'rotary')
        if mode == 'off':
            return latents
        if mode == 'rotary':
            return self._apply_rotary_positional_encoding(latents, positions)
        if mode == 'sinusoidal':
            return self._apply_sinusoidal_positional_encoding(latents, positions)
        raise ValueError(f"Unsupported traj_pos_encoding: {mode}")

    def _apply_rotary_positional_encoding(self, latents: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        dim = latents.shape[-1]
        pair_dim = (dim // 2) * 2
        if pair_dim == 0:
            return latents

        sin, cos = self._lookup_or_compute_angle_tables(
            mode='rotary',
            latents=latents,
            positions=positions,
            pair_dim=pair_dim,
            exponent_dim=pair_dim,
        )

        even = latents[:, 0:pair_dim:2]
        odd = latents[:, 1:pair_dim:2]
        rotated = latents.clone()
        rotated[:, 0:pair_dim:2] = even * cos - odd * sin
        rotated[:, 1:pair_dim:2] = even * sin + odd * cos
        return rotated

    def _apply_sinusoidal_positional_encoding(self, latents: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        dim = latents.shape[-1]
        pair_dim = (dim // 2) * 2
        if pair_dim == 0:
            return latents

        sin, cos = self._lookup_or_compute_angle_tables(
            mode='sinusoidal',
            latents=latents,
            positions=positions,
            pair_dim=pair_dim,
            exponent_dim=dim,
        )

        pe = torch.zeros_like(latents)
        pe[:, 0:pair_dim:2] = sin
        pe[:, 1:pair_dim:2] = cos
        return latents + pe

    def _lookup_or_compute_angle_tables(
        self,
        mode: str,
        latents: torch.Tensor,
        positions: torch.Tensor,
        pair_dim: int,
        exponent_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position_indices = self._positions_to_cache_indices(positions)
        if position_indices is not None and position_indices.numel() > 0:
            sin_table, cos_table = self._get_cached_angle_tables(
                mode=mode,
                latents=latents,
                pair_dim=pair_dim,
                exponent_dim=exponent_dim,
                max_position=int(position_indices.max().item()),
            )
            return (
                sin_table.index_select(0, position_indices),
                cos_table.index_select(0, position_indices),
            )

        inv_freq = self._get_cached_inv_freq(
            mode=mode,
            latents=latents,
            pair_dim=pair_dim,
            exponent_dim=exponent_dim,
        )
        angles = positions.to(latents.dtype).unsqueeze(-1) * inv_freq.unsqueeze(0)
        return torch.sin(angles), torch.cos(angles)

    def _positions_to_cache_indices(self, positions: torch.Tensor):
        if positions.numel() == 0:
            return positions.new_empty((0,), dtype=torch.long)

        rounded = positions.round()
        if torch.any(rounded < 0):
            return None
        if not torch.allclose(positions, rounded, atol=1e-4, rtol=0.0):
            return None
        return rounded.to(dtype=torch.long)

    def _get_cached_angle_tables(
        self,
        mode: str,
        latents: torch.Tensor,
        pair_dim: int,
        exponent_dim: int,
        max_position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self._positional_cache_key(
            mode=mode,
            latents=latents,
            pair_dim=pair_dim,
            exponent_dim=exponent_dim,
        )
        cache_entry = self._positional_encoding_cache.get(key)
        cached_max = -1 if cache_entry is None else int(cache_entry['max_position'])

        if cache_entry is None or cached_max < max_position:
            inv_freq = self._get_cached_inv_freq(
                mode=mode,
                latents=latents,
                pair_dim=pair_dim,
                exponent_dim=exponent_dim,
            )
            positions = torch.arange(
                max_position + 1,
                device=latents.device,
                dtype=latents.dtype,
            )
            angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
            cache_entry = {
                'inv_freq': inv_freq,
                'sin': torch.sin(angles),
                'cos': torch.cos(angles),
                'max_position': int(max_position),
            }
            self._positional_encoding_cache[key] = cache_entry

        return cache_entry['sin'], cache_entry['cos']

    def _get_cached_inv_freq(
        self,
        mode: str,
        latents: torch.Tensor,
        pair_dim: int,
        exponent_dim: int,
    ) -> torch.Tensor:
        key = self._positional_cache_key(
            mode=mode,
            latents=latents,
            pair_dim=pair_dim,
            exponent_dim=exponent_dim,
        )
        cache_entry = self._positional_encoding_cache.get(key)
        if cache_entry is not None and 'inv_freq' in cache_entry:
            return cache_entry['inv_freq']

        base = float(getattr(self.cfg.algo, 'traj_pos_encoding_base', 10000.0))
        freq_idx = torch.arange(0, pair_dim, 2, device=latents.device, dtype=latents.dtype)
        inv_freq = torch.pow(torch.as_tensor(base, device=latents.device, dtype=latents.dtype), -freq_idx / exponent_dim)

        if cache_entry is None:
            cache_entry = {}
            self._positional_encoding_cache[key] = cache_entry
        cache_entry['inv_freq'] = inv_freq
        return inv_freq

    @staticmethod
    def _positional_cache_key(
        mode: str,
        latents: torch.Tensor,
        pair_dim: int,
        exponent_dim: int,
    ):
        return (
            mode,
            latents.device.type,
            latents.device.index,
            latents.dtype,
            int(pair_dim),
            int(exponent_dim),
        )

    def _compute_base_constraint_penalty(self, agent, batch_online: Dict) -> torch.Tensor:
        phi_x = batch_online.get('kernel_cur_z', batch_online['cur_phi'])
        phi_y = batch_online.get('kernel_next_z', batch_online['next_phi'])
        cst_dist = self._compute_cst_dist(agent, batch_online)

        if self.cfg.algo.dual_dist != 'kernel_sim':
            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
        else:
            cst_penalty = (phi_y - phi_x).sum(dim=1) - cst_dist
        return torch.clamp(cst_penalty, max=self.cfg.algo.dual_slack)

    def _compute_temporal_constraint_penalty(self, cur_z: torch.Tensor, next_z: torch.Tensor) -> torch.Tensor:
        diff_sq = torch.square(next_z - cur_z).mean(dim=1)
        return torch.clamp(
            float(self.cfg.algo.contrastive_temporal_budget) - diff_sq,
            max=self.cfg.algo.dual_slack,
        )

    def _compute_cst_dist(self, agent, batch: Dict) -> torch.Tensor:
        dual_dist = self.cfg.algo.dual_dist
        if dual_dist == 'one':
            return torch.ones_like(batch['obs'][:, 0])
        if dual_dist == 'l2':
            cur_z = batch['cur_phi']
            next_z = batch['next_phi']
            return torch.square(next_z - cur_z).mean(dim=1)
        if dual_dist == 'kernel_sim_dist':
            return 1 - (batch['kernel_next_z'] * batch['kernel_cur_z']).sum(dim=1)
        if dual_dist == 'kernel_mmd':
            return torch.square(batch['kernel_next_z'] - batch['kernel_cur_z']).mean(dim=1)
        if dual_dist == 'kernel_sim':
            return (batch['kernel_next_z'] * batch['kernel_cur_z']).sum(dim=1)
        if dual_dist == 'skill_kme':
            if 'skill_kme' not in batch:
                raise ValueError("dual_dist=skill_kme requires skill_kme in batch")
            phi_x = batch.get('kernel_cur_z', batch['cur_phi'])
            phi_y = batch.get('kernel_next_z', batch['next_phi'])
            return 1e-6 * torch.einsum('ij,ij->i', (phi_x + phi_y) / 2.0, batch['skill_kme'])
        if dual_dist == 's2_from_s':
            s2_dist = agent.dist_predictor(batch['obs'])
            s2_dist_mean = s2_dist.mean
            s2_dist_std = s2_dist.stddev
            scaling_factor = 1. / s2_dist_std
            geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
            normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
            return torch.mean(
                torch.square((batch['next_obs'] - batch['obs']) - s2_dist_mean) * normalized_scaling_factor,
                dim=1,
            )
        raise ValueError(f"Unsupported dual_dist for IDK-CSD: {dual_dist}")

    def _rollout_skill_key(self, rollout: Dict):
        skills = rollout.get('skills')
        if skills is None:
            return None
        skill_array = np.asarray(skills, dtype=np.float32)
        if skill_array.ndim == 1:
            skill_array = skill_array.reshape(-1, 1)
        if skill_array.shape[0] < 2:
            return None
        reference = skill_array[0].reshape(-1)
        if not np.allclose(skill_array.reshape(skill_array.shape[0], -1), reference[None, :]):
            return None
        return tuple(reference.tolist())

    def _use_kernel_features(self, agent) -> bool:
        if not hasattr(agent, 'kernel') or agent.kernel is None:
            return False
        return self.cfg.algo.dual_dist in ('kernel_sim_dist', 'kernel_sim', 'kernel_mmd', 'skill_kme')

    def _use_contrastive_kernel(self, agent) -> bool:
        if not hasattr(agent, 'contrastive_kernel') or agent.contrastive_kernel is None:
            return False
        return bool(getattr(agent, 'contrastive_kernel_initialized', False))

    @staticmethod
    def _to_device_tensor(agent, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.float().to(agent.device)
        return torch.as_tensor(value, dtype=torch.float32, device=agent.device)

    @staticmethod
    def _as_tensor_mean(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, torch.distributions.Distribution):
            return value.mean
        return value.mean

    @staticmethod
    def _graph_zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _build_reward_metrics(
        self,
        batch: Dict,
        rewards: torch.Tensor,
        delta: torch.Tensor,
        eps: float = 1e-4,
    ) -> Dict[str, float]:
        metrics = {
            'RewardMean': rewards.mean().item(),
            'RewardStd': rewards.std().item(),
            'RewardAbsMean': rewards.abs().mean().item(),
            'RewardNearZeroFrac': (rewards.abs() < eps).float().mean().item(),
            'RewardPositiveFrac': (rewards > 0).float().mean().item(),
            'RewardDeltaSqMean': torch.square(delta).mean().item(),
            'RewardModeWarmupAligned': 1.0,
        }

        kernel_cur_z = batch.get('kernel_cur_z')
        kernel_next_z = batch.get('kernel_next_z')
        if kernel_cur_z is not None and kernel_next_z is not None:
            kernel_cur_z = self._as_tensor_mean(kernel_cur_z)
            kernel_next_z = self._as_tensor_mean(kernel_next_z)
            metrics['KernelDeltaSqMean'] = torch.square(kernel_next_z - kernel_cur_z).mean().item()

        skills = batch.get('skills')
        if self.cfg.algo.discrete and skills is not None and torch.is_tensor(skills) and skills.numel() > 0:
            flat_skills = skills.reshape(skills.shape[0], -1)
            if flat_skills.shape[1] > 0:
                skill_ids = flat_skills.argmax(dim=1)
                dim_skill = int(self.cfg.algo.dim_skill)
                for skill_idx in range(dim_skill):
                    mask = skill_ids == skill_idx
                    count = float(mask.sum().item())
                    metrics[f'RewardCount/Skill{skill_idx}'] = count
                    if count > 0:
                        skill_rewards = rewards[mask]
                        metrics[f'RewardMean/Skill{skill_idx}'] = skill_rewards.mean().item()
                        metrics[f'RewardAbsMean/Skill{skill_idx}'] = skill_rewards.abs().mean().item()
                    else:
                        metrics[f'RewardMean/Skill{skill_idx}'] = 0.0
                        metrics[f'RewardAbsMean/Skill{skill_idx}'] = 0.0

        return metrics
