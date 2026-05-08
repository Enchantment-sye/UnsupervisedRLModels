import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Protocol, Any, Union
from math import sqrt
import numpy as np
from utils import utils
from core.hierarchical_phi import (
    build_hierarchical_beta,
    resolve_hierarchical_phi_depth,
    resolve_hierarchical_phi_dim,
    split_hierarchical_phi,
    split_hierarchical_skill,
)
from core.stage_contract import get_base_algo_name

# Protocol definition for Algorithm Variants
class AlgoVariant(Protocol):
    name: str
    
    def compute_intrinsic_reward(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        """
        Calculates skill rewards (and novelty rewards if applicable).
        Returns the final reward tensor (B,).
        Updates metrics dict in-place.
        """
        ...
        
    def compute_te_loss(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        """
        Calculates the Trajectory Encoder loss (to minimize).
        Returns loss tensor.
        Updates metrics dict in-place.
        """
        ...
        
    def update_auxiliary(self, agent, batch: Dict, metrics: Dict):
        """
        Updates auxiliary components (Dual Lambda, Dist Predictor, etc.).
        Updates metrics dict in-place.
        """
        ...

# Base Implementation with common logic
class BaseVariant:
    def __init__(self, config):
        self.cfg = config

    def _zero_metric(self, agent, batch=None):
        if batch:
            for value in batch.values():
                if torch.is_tensor(value):
                    return value.detach().new_zeros(())
        return torch.zeros((), device=agent.device)

    def _record_reward_diagnostics(self, metrics: Dict, skill_rewards: torch.Tensor):
        skill_rewards_detached = skill_rewards.detach()
        metrics.update({
            'PureRewardMean': skill_rewards_detached.mean(),
            'PureRewardStd': skill_rewards_detached.std(unbiased=False),
            'PureRewardMin': skill_rewards_detached.min(),
            'PureRewardMax': skill_rewards_detached.max(),
        })

    def _record_delta_phi_diagnostics(self, metrics: Dict, cur_phi: torch.Tensor, next_phi: torch.Tensor):
        if not torch.is_tensor(cur_phi) or not torch.is_tensor(next_phi):
            return
        delta_phi_norm = torch.linalg.vector_norm((next_phi - cur_phi).detach(), dim=1)
        metrics.update({
            'DeltaPhiNormMean': delta_phi_norm.mean(),
            'DeltaPhiNormStd': delta_phi_norm.std(unbiased=False),
            'DeltaPhiNormMax': delta_phi_norm.max(),
        })

    def _record_dual_disabled_diagnostics(self, agent, batch: Dict, metrics: Dict):
        zero = self._zero_metric(agent, batch)
        dual_lam_param = getattr(getattr(agent, 'dual_lam', None), 'param', None)
        dual_lam = dual_lam_param.detach().exp() if dual_lam_param is not None else zero
        metrics.update({
            'DualLam': dual_lam,
            'LossDualLam': zero,
            'DualCstPenalty': zero,
            'TemporalViolationMean': zero,
            'TemporalViolationFrac': zero,
        })

    def compute_intrinsic_reward(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        # 1. Get embeddings (respecting use_target_traj_encoder config)
        self._populate_embeddings(agent, batch, use_target=self.cfg.use_target_traj_encoder)
        
        # 2. Compute skill rewards
        skill_rewards = self._calculate_skill_rewards(agent, batch)
        
        # 3. Novelty rewards
        if self.cfg.use_kme and self.cfg.use_novelty_reward:
            self._update_distributional_novelty_rewards(agent, batch, metrics)
             
        rewards = skill_rewards * batch.get('novelty_rewards', 1.0)
        
        # 4. Update batch & metrics
        batch['skill_rewards'] = skill_rewards # Keep pure skill reward separate if needed
        batch['rewards'] = rewards
        self._record_reward_diagnostics(metrics, skill_rewards)
        metrics.update({'RewardMean': rewards.mean().item(), 'RewardStd': rewards.std().item()})
        self._maybe_log_hierarchical_beta(batch, metrics)
        return rewards

    def compute_te_loss(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        # CRITICAL: For TE optimization, we MUST use the ONLINE encoder and maintain gradients.
        # We work on a shallow copy of batch to avoid polluting the main batch with online embeddings 
        # if the main batch is supposed to hold target-based embeddings (for RL).
        batch_online = batch.copy()
        
        # 1. Force Online Embeddings
        self._populate_embeddings(agent, batch_online, use_target=False)
        
        # 2. Re-calculate rewards using online embeddings
        rewards = self._calculate_skill_rewards(agent, batch_online)
        
        # 3. Stats
        cur_phi = batch_online['cur_phi']
        next_phi = batch_online['next_phi']
        self._record_delta_phi_diagnostics(metrics, cur_phi, next_phi)
        metrics.update({'currentStateMean': torch.square(cur_phi).mean().item()})
        
        # 4. Dual Regularization Logic
        if self.cfg.dual_reg:
            dual_lam = agent.dual_lam.param.exp()
            
            # Helper to get mean tensors for cst calculation
            phi_x_mean = batch_online.get('kernel_cur_z', batch_online['cur_phi'])
            phi_y_mean = batch_online.get('kernel_next_z', batch_online['next_phi'])
            
            cst_dist = self._compute_cst_dist(agent, batch_online)
            
            if self.cfg.dual_dist != 'kernel_sim':
                cst_penalty = cst_dist - torch.square(phi_y_mean - phi_x_mean).mean(dim=1)
            else:
                cst_penalty = (phi_y_mean - phi_x_mean).sum(dim=1) - cst_dist
            temporal_violation = cst_penalty
            cst_penalty = torch.clamp(cst_penalty, max=self.cfg.dual_slack)
            
            te_obj = rewards + dual_lam.detach() * cst_penalty
            metrics.update({
                'DualCstPenalty': cst_penalty.detach().mean(),
                'TemporalViolationMean': temporal_violation.detach().mean(),
                'TemporalViolationFrac': (temporal_violation.detach() > 0).float().mean(),
            })
            
            # Store cst_penalty in main batch for dual_lam update (optional, but safe)
            # But dual_lam update usually happens after this using `batch`.
            # If dual_lam update uses `cst_penalty`, it should come from here.
            # However, `update_auxiliary` is called separately. 
            # We should probably store it in the main batch or return it.
            # Let's store in main batch, assuming it's detached anyway later.
            batch['cst_penalty'] = cst_penalty
        else:
            te_obj = rewards
            self._record_dual_disabled_diagnostics(agent, batch_online, metrics)

        loss_te = -te_obj.mean()
        metrics.update({'TeObjMean': te_obj.mean().item(), 'LossTe': loss_te})
        return loss_te

    def update_auxiliary(self, agent, batch: Dict, metrics: Dict):
        if self.cfg.dual_reg:
            self._update_loss_dual_lam(agent, batch, metrics)
            agent._gradient_descent(metrics['LossDualLam'], optimizer_keys=['dual_lam'], metrics=metrics)

    # --- Helpers ---

    def _populate_embeddings(self, agent, batch, use_target):
        """
        Computes embeddings (cur_z, next_z, and optionally kernel maps) 
        and populates the batch dict.
        """
        # 1. Get Base Embeddings
        traj_encoder = agent.target_traj_encoder if use_target else agent.traj_encoder
        
        # For TE loss (use_target=False), we expect gradients.
        # For Reward (use_target=True), gradients might be blocked if target encoder is frozen.
        
        obs = batch['obs']
        next_obs = batch['next_obs']
        
        cur_z = traj_encoder(obs)
        next_z = traj_encoder(next_obs)
        
        batch['cur_z'] = cur_z
        batch['next_z'] = next_z
        batch['cur_phi'] = agent.phi_from_encoder_output(cur_z, use_target=use_target)
        batch['next_phi'] = agent.phi_from_encoder_output(next_z, use_target=use_target)
        
        # 2. Kernel Maps (if needed)
        need_kernel = (self.cfg.use_kme and self.cfg.kernel_map) or \
                      (self.cfg.dual_dist in ('kernel_sim_dist', 'kernel_mmd', 'kernel_sim', 'skill_kme')) or \
                      (getattr(self, 'name', '') == 'iksd')
                      
        if need_kernel:
            # Ensure kernel exists
            if not hasattr(agent, 'kernel') or agent.kernel is None:
                 # Fallback or skip? Assuming initialized.
                 pass
            else:
                # Check for skills
                skills = batch.get('skills')
                if skills is None:
                     # Should not happen in training usually
                     skills = torch.zeros_like(batch['cur_phi'])
                elif skills.dim() > 2:
                    skills = skills.reshape(skills.shape[0], -1)

                kernel_cur_z = agent.kernel(batch['cur_phi']) / sqrt(agent.kernel.ensemble_size)
                kernel_next_z = agent.kernel(batch['next_phi']) / sqrt(agent.kernel.ensemble_size)
                kernel_skills =  agent.kernel(skills) / sqrt(agent.kernel.ensemble_size)
                
                batch['kernel_cur_z'] = kernel_cur_z
                batch['kernel_next_z'] = kernel_next_z
                batch['kernel_skills'] = kernel_skills

    def _calculate_skill_rewards(self, agent, batch):
        cur_z = batch['cur_z']
        next_z = batch['next_z']
        cur_phi = batch['cur_phi']
        next_phi = batch['next_phi']
        skills = batch['skills']
        
        # Handle kernel vs encoder z
        # Logic from original: if inner, uses z. If not inner, uses dist logic.
        # Kernel variants usually use 'kernel_cur_z' for dual dist but 'z' for reward?
        # Let's check original _update_skill_rewards logic.
        # It used _get_batch_emb_vectors which returned (kernel_cur_z if need_kernel else cur_z).
        
        need_kernel = (self.cfg.use_kme and self.cfg.kernel_map) or \
                      (self.cfg.dual_dist in ('kernel_sim_dist', 'kernel_mmd', 'kernel_sim', 'skill_kme')) or \
                      (getattr(self, 'name', '') == 'iksd')
                      
        use_hierarchical_phi = self.cfg.cascade.use_cascade and self.cfg.use_hierarchical_phi

        if need_kernel and not self.cfg.use_hierarchical_skill:
            phi_cur = batch['kernel_cur_z']
            phi_next = batch['kernel_next_z']
            phi_skills = batch['kernel_skills']
        else:
            phi_cur = cur_phi
            phi_next = next_phi
            phi_skills = skills

        if self.cfg.inner:
            target_z = phi_next - phi_cur
            if use_hierarchical_phi:
                depth = resolve_hierarchical_phi_depth(self.cfg)
                level_dim = resolve_hierarchical_phi_dim(self.cfg)
                phi_cur_levels = split_hierarchical_phi(phi_cur, depth, level_dim)
                phi_next_levels = split_hierarchical_phi(phi_next, depth, level_dim)
                skill_levels = split_hierarchical_skill(phi_skills, depth, level_dim)
                beta = build_hierarchical_beta(
                    depth=depth,
                    mode=self.cfg.beta_mode,
                    rho=self.cfg.beta_rho,
                    device=phi_cur_levels.device,
                    dtype=phi_cur_levels.dtype,
                )
                delta_phi = phi_next_levels - phi_cur_levels
                level_rewards = (delta_phi * skill_levels).sum(dim=-1)
                rewards = (level_rewards * beta.view(1, -1)).sum(dim=1)
                batch['hierarchical_level_rewards'] = level_rewards
                batch['hierarchical_beta'] = beta
                return rewards
            if self.cfg.use_hierarchical_skill:
                if phi_skills.dim() == 2:
                    phi_skills = phi_skills.reshape(phi_skills.shape[0], self.cfg.num_skill_levels, self.cfg.dim_skill)
                rewards = (target_z.unsqueeze(1) * phi_skills).sum(dim=-1).sum(dim=1)
                return rewards
            if self.cfg.discrete:
                masks = (phi_skills - phi_skills.mean(dim=1, keepdim=True)) * self.cfg.dim_skill / (self.cfg.dim_skill - 1 if self.cfg.dim_skill != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                rewards = (target_z * phi_skills).sum(dim=1)
        else:
            # dist based (DIAYN style or direct log prob)
            target_dists = next_z # Use distribution object directly
            if self.cfg.discrete:
                logits = target_dists.mean
                rewards = -F.cross_entropy(logits, skills.argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(skills)
                
        return rewards

    def _maybe_log_hierarchical_beta(self, batch, metrics):
        if not (self.cfg.cascade.use_cascade and self.cfg.use_hierarchical_phi):
            return
        if not getattr(self.cfg, 'log_beta_values', True):
            return
        beta = batch.get('hierarchical_beta')
        if beta is None:
            return
        for level, value in enumerate(beta.detach().cpu().tolist(), start=1):
            metrics[f'HierarchicalBeta/L{level}'] = value

    def _update_distributional_novelty_rewards(self, agent, batch, metrics):
        # Always uses kernel maps
        # Check if they exist, if not populate?
        if 'kernel_next_z' not in batch:
             # Should be populated by _populate_embeddings if config set correctly
             pass
             
        next_z = batch['kernel_next_z']
        kme_map = agent.kme_vector
        rewards = torch.matmul(next_z , kme_map)
        rewards = torch.clamp(- torch.log(rewards), min = 1, max=5)
        metrics.update({
            'NoveltyRewardMean': rewards.mean().item(),
            'NoveltyRewardStd': rewards.std().item(),
        })
        batch['novelty_rewards'] = rewards

    def _update_loss_dual_lam(self, agent, batch, metrics):
        log_dual_lam = agent.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (batch['cst_penalty'].detach()).mean()
        metrics.update({'DualLam': dual_lam.detach(), 'LossDualLam': loss_dual_lam}) # keep tensor for backward

    def _compute_cst_dist(self, agent, batch):
        # Default fallback
        return torch.ones_like(batch['obs'][:, 0])

# Specific Variants

class MetraVariant(BaseVariant):
    name = "metra"
    def _compute_cst_dist(self, agent, batch):
        return torch.ones_like(batch['obs'][:, 0])

class LsdVariant(BaseVariant):
    name = "lsd"
    def _compute_cst_dist(self, agent, batch):
        cur_z = batch['cur_phi']
        next_z = batch['next_phi']
        return torch.square(next_z - cur_z).mean(dim=1)

class IksdVariant(BaseVariant):
    name = "iksd"
    def _compute_cst_dist(self, agent, batch):
        return 1 - (batch['kernel_next_z'] * batch['kernel_cur_z']).sum(dim=1)

class CsdVariant(BaseVariant):
    name = "csd"
    def _compute_cst_dist(self, agent, batch):
        s2_dist = agent.dist_predictor(batch['obs'])
        s2_dist_mean = s2_dist.mean
        s2_dist_std = s2_dist.stddev
        scaling_factor = 1. / s2_dist_std
        geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
        normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
        cst_dist = torch.mean(torch.square((batch['next_obs'] - batch['obs']) - s2_dist_mean) * normalized_scaling_factor, dim=1)
        return cst_dist

    def update_auxiliary(self, agent, batch: Dict, metrics: Dict):
        super().update_auxiliary(agent, batch, metrics)
        if agent.dist_predictor:
             s2_dist = agent.dist_predictor(batch['obs'])
             diff = batch['next_obs'] - batch['obs']
             loss_dp = -s2_dist.log_prob(diff).mean()
             metrics['LossDp'] = loss_dp
             agent._gradient_descent(loss_dp, optimizer_keys=['dist_predictor'], metrics=metrics)

class DiaynVariant(BaseVariant):
    name = "diayn"
    
    def compute_intrinsic_reward(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        original_inner = self.cfg.inner
        self.cfg.inner = False
        
        self._populate_embeddings(agent, batch, use_target=self.cfg.use_target_traj_encoder)
        rewards = self._calculate_skill_rewards(agent, batch)
        
        self.cfg.inner = original_inner
        
        batch['rewards'] = rewards
        batch['skill_rewards'] = rewards
        self._record_reward_diagnostics(metrics, rewards)
        metrics.update({'RewardMean': rewards.mean().item(), 'RewardStd': rewards.std().item()})
        return rewards

    def compute_te_loss(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        # For DIAYN, maximizing reward is minimizing negative reward
        # We need online gradients here too
        batch_online = batch.copy()
        self._populate_embeddings(agent, batch_online, use_target=False)
        self._record_delta_phi_diagnostics(metrics, batch_online['cur_phi'], batch_online['next_phi'])
        
        # Temporarily force inner=False for calculation
        original_inner = self.cfg.inner
        self.cfg.inner = False
        rewards = self._calculate_skill_rewards(agent, batch_online)
        self.cfg.inner = original_inner
        
        loss_te = -rewards.mean()
        self._record_dual_disabled_diagnostics(agent, batch_online, metrics)
        metrics.update({'TeObjMean': rewards.mean().item(), 'LossTe': loss_te.item()})
        return loss_te
        
    def update_auxiliary(self, agent, batch: Dict, metrics: Dict):
        pass

class DadsVariant(BaseVariant):
    name = "dads"

    def compute_intrinsic_reward(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        # Ensure BN is in eval mode
        if agent.sd_input_bn:
            agent.sd_input_bn.eval()
            agent.sd_target_bn.eval()
            
        with torch.no_grad():
            obs = batch['obs']
            next_obs = batch['next_obs']
            skills = batch['skills']
            
            # Encode if needed
            if agent.traj_encoder:
                obs = agent.encode_phi(obs, use_target=self.cfg.use_target_traj_encoder)
                next_obs = agent.encode_phi(next_obs, use_target=self.cfg.use_target_traj_encoder)
            
            diff = next_obs - obs
            target = self._process_sd_target(agent, diff)
            
            # Repeat logic for diversity reward
            num_alt_samples = self.cfg.algo.num_alt_samples
            
            obs_repeated = torch.cat([obs] * num_alt_samples, dim=0)
            next_obs_repeated = torch.cat([target] * num_alt_samples, dim=0)
            
            B = obs.size(0)
            dim_skill = self.cfg.algo.dim_skill
            alt_options_shape = (obs_repeated.size(0), dim_skill)
            
            if self.cfg.algo.discrete:
                alt_options = torch.randint(dim_skill, (obs_repeated.size(0),), device=agent.device)
                alt_options = torch.eye(dim_skill, device=agent.device)[alt_options]
            else:
                alt_options = torch.normal(mean=torch.zeros(alt_options_shape), std=torch.ones(alt_options_shape)).to(agent.device)
                
            sd_input = utils.get_torch_concat_obs(self._process_sd_input(agent, obs), skills)
            next_obs_log_probs = agent.skill_dynamics(sd_input).log_prob(target)
            
            split_group = self.cfg.algo.split_group
            next_obs_alt_log_probs = []
            
            # Calculate in chunks to avoid OOM
            total_alt = alt_options_shape[0]
            for i in range((total_alt + split_group - 1) // split_group):
                start_idx = i * split_group
                end_idx = min((i + 1) * split_group, total_alt)
                
                chunk_obs = obs_repeated[start_idx:end_idx]
                chunk_alt = alt_options[start_idx:end_idx]
                chunk_next = next_obs_repeated[start_idx:end_idx]
                
                chunk_input = utils.get_torch_concat_obs(self._process_sd_input(agent, chunk_obs), chunk_alt)
                next_obs_alt_log_probs.append(
                    agent.skill_dynamics(chunk_input).log_prob(chunk_next)
                )
                
            next_obs_alt_log_probs = torch.cat(next_obs_alt_log_probs, dim=0).view(num_alt_samples, -1)
            
            # DADS reward formula
            # log(L+1) - log(1 + sum(exp(log_p_alt - log_p)))
            # The clip is for numerical stability
            rewards = (np.log(num_alt_samples + 1) - torch.log(1 + torch.exp(torch.clip(
                next_obs_alt_log_probs - next_obs_log_probs.view(1, -1), -50, 50)).sum(dim=0)))
            
            metrics.update({
                'DadsSdLogProbMean': next_obs_log_probs.mean().item(),
                'DadsSdAltLogProbMean': next_obs_alt_log_probs.mean().item(),
                'RewardMean': rewards.mean().item(),
                'RewardStd': rewards.std().item()
            })
            self._record_reward_diagnostics(metrics, rewards)
            
            batch['rewards'] = rewards
            batch['skill_rewards'] = rewards
            return rewards

    def compute_te_loss(self, agent, batch: Dict, metrics: Dict) -> torch.Tensor:
        # Ensure BN is in train mode
        if agent.sd_input_bn:
            agent.sd_input_bn.train()
            agent.sd_target_bn.train()
            
        obs = batch['obs']
        next_obs = batch['next_obs']
        skills = batch['skills']
        
        # Encode if needed
        if agent.traj_encoder:
            obs = agent.encode_phi(obs, use_target=False)
            next_obs = agent.encode_phi(next_obs, use_target=False)
        self._record_delta_phi_diagnostics(metrics, obs, next_obs)
            
        diff = next_obs - obs
        target = self._process_sd_target(agent, diff)
        
        sd_input = utils.get_torch_concat_obs(self._process_sd_input(agent, obs), skills)
        next_obs_dists = agent.skill_dynamics(sd_input)
        next_obs_log_probs = next_obs_dists.log_prob(target)
        
        # Mask out done steps? dads.py does: next_obs_log_probs * (1. - v['dones'])
        # If 'dones' is in batch (float 0 or 1).
        if 'dones' in batch: # metra agent might name it 'is_terminal' or 'dones'?
             # agent_utils.get_mini_tensors returns 'dones' if in epoch_data?
             # _sample_replay_buffer usually returns 'terminals' or 'dones'.
             # Let's check replay buffer. PathBufferEx returns 'dones'.
             mask = (1. - batch['dones'])
             next_obs_log_probs = next_obs_log_probs * mask
             denom = mask.sum() + 1e-12
        else:
             denom = next_obs_log_probs.numel() / next_obs_log_probs.shape[0] # Just mean?
             denom = batch['obs'].shape[0] # Just batch size
        
        # dads.py: next_obs_log_prob_mean = next_obs_log_probs.sum() / ((1. - v['dones']).sum() + 1e-12)
        next_obs_log_prob_mean = next_obs_log_probs.sum() / denom
        
        loss_sd = -next_obs_log_prob_mean
        
        metrics.update({
            'LossSd': loss_sd.item(),
        })
        self._record_dual_disabled_diagnostics(agent, batch, metrics)
        
        if agent.sd_target_bn:
            metrics.update({
                'SdTargetRunningMean': agent.sd_target_bn.running_mean.mean().item(),
                'SdTargetRunningVar': agent.sd_target_bn.running_var.mean().item(),
            })
            
        return loss_sd

    def update_auxiliary(self, agent, batch: Dict, metrics: Dict):
        # Nothing specific for auxiliary update in standard DADS, optimization happens in _optimize_te
        # But maybe we need to update running stats? Done implicitly by forward in train mode.
        pass

    def _process_sd_input(self, agent, sd_input):
        if agent.sd_input_bn:
            return agent.sd_input_bn(sd_input)
        return sd_input

    def _process_sd_target(self, agent, sd_target):
        if agent.sd_target_bn:
            return agent.sd_target_bn(sd_target)
        return sd_target

# Factory
class VariantFactory:
    @staticmethod
    def create(config) -> AlgoVariant:
        base_algo = get_base_algo_name(config)

        if base_algo == 'dads':
            return DadsVariant(config)
        if base_algo == 'idk_csd':
            from core.experimental.idk_csd_variant import IdkCsdVariant
            return IdkCsdVariant(config)
        if base_algo == 'diayn':
            return DiaynVariant(config)
        if base_algo == 'lsd':
            return LsdVariant(config)
        if base_algo == 'iksd':
            return IksdVariant(config)
        if base_algo == 'csd':
            return CsdVariant(config)
        if base_algo == 'metra':
            return MetraVariant(config)

        raise ValueError(f"Unknown algorithm variant for base_algo={base_algo!r}")
