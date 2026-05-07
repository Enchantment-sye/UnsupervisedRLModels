import time
import os
import logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

from utils import utils
from utils import agent_utils
from data_structs.trajectory_batch import TrajectoryBatch
from core.metra_agent import MetraAgent
from core.metra_config import MetraConfig
from core.auto_branch import AutoBranchController
from core.task_adapter import SkillDiscoveryTaskAdapter
from core.cascade_actor import CascadeActor
from models.encoders import WithEncoder
from core.hierarchical_phi import (
    build_hierarchical_beta,
    resolve_hierarchical_phi_depth,
    resolve_hierarchical_phi_dim,
)
from core.stage_contract import is_pretraining_stage, requires_best_skill_search, should_use_kme
from utils.checkpointing import safe_torch_load
from safety.metrics import aggregate_safety_metrics
from iod.coverage_tracker import CoverageTracker

class MetraTrainer:
    def __init__(self, config: MetraConfig, agent: MetraAgent, env, replay_buffer, work_dir):
        self.cfg = config
        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.work_dir = work_dir
        
        self.step_itr = 0
        self.total_env_steps = 0
        self.total_epoch = 0
        self.start_epoch = 0
        self._start_time = time.time()
        self._itr_start_time = time.time()
        
        # Logging
        self.logger = logging.getLogger('MetraTrainer')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        fh = logging.FileHandler(os.path.join(work_dir, 'debug.log'), mode='a')
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        self.logger.addHandler(ch)
            
        self.writer = SummaryWriter(os.path.join(work_dir, 'tb'))
        env_name = getattr(config, 'task', None) or getattr(getattr(config, 'env', None), 'task', None)
        self.coverage_tracker = CoverageTracker(env_name)
        
        # Task Adapter
        self.task_adapter = SkillDiscoveryTaskAdapter(config, env, agent, work_dir, self.logger)
        self.task_adapter.coverage_tracker = self.coverage_tracker
        self.auto_branch = AutoBranchController(config, agent, self.task_adapter, work_dir, self.logger, self.writer)
        self._log_hierarchical_phi_setup()

    def _log_hierarchical_phi_setup(self):
        if not (self.cfg.cascade.use_cascade and self.cfg.use_hierarchical_phi):
            return
        depth = resolve_hierarchical_phi_depth(self.cfg)
        level_dim = resolve_hierarchical_phi_dim(self.cfg)
        beta = build_hierarchical_beta(depth, self.cfg.beta_mode, self.cfg.beta_rho)
        self.logger.info(
            "[HierarchicalPhi] enabled depth=%d level_dim=%d beta_mode=%s beta_rho=%.4f beta=%s",
            depth,
            level_dim,
            self.cfg.beta_mode,
            self.cfg.beta_rho,
            [round(float(x), 6) for x in beta.tolist()],
        )

    def train(self):
        last_return = None
        
        # Task Start Hook
        self.task_adapter.on_train_start()

        if self.start_epoch >= self.cfg.n_epochs:
            self.logger.info(
                "Resume start_epoch %d is already at or beyond configured n_epochs %d; nothing to do.",
                self.start_epoch,
                self.cfg.n_epochs,
            )
            return last_return

        with utils.GlobalContext({'phase': 'train', 'policy': 'sampling'}):
            self.logger.info('Obtaining samples...')
            for epoch in tqdm(range(self.start_epoch, self.cfg.n_epochs)):
                if self.cfg.cascade.use_cascade:
                    if not self.auto_branch.enabled:
                        self._try_grow_policy_stage(epoch)
                    self._try_grow_skill_stage(epoch)

                self.logger.info('epoch #%d | ' % epoch)
                self._itr_start_time = time.time()
                self.total_epoch = epoch

                self._set_models_mode('eval')
                if self.cfg.stage == 'pre_training' and self.agent.traj_encoder is not None:
                    self.agent.traj_encoder.eval()

                if self.cfg.n_epochs_per_eval != 0 and (self.step_itr + 1) % self.cfg.n_epochs_per_eval == 0:
                    self.task_adapter.evaluate(self.step_itr, self.total_epoch, self.writer)
                    self.log_diagnostics()

                self._set_models_mode('train')
                if self.cfg.stage == 'pre_training' and self.agent.traj_encoder is not None:
                    self.agent.traj_encoder.train()

                time_sampling = [0.0]
                with MeasureAndAccTime(time_sampling):
                    step_paths = self.task_adapter.get_train_trajectories(self.cfg.traj_batch_size)
                sampling_breakdown = {}
                consume_sampling_metrics = getattr(self.task_adapter, "consume_train_sampling_metrics", None)
                if callable(consume_sampling_metrics):
                    sampling_breakdown = consume_sampling_metrics() or {}
                
                self.total_env_steps += sum([len(path['dones']) for path in step_paths])
                
                last_return = self.train_once(
                    step_paths,
                    extra_scalar_metrics={
                        'TimeSampling': time_sampling[0],
                        **sampling_breakdown,
                    },
                )

                self.step_itr += 1
                if self.cfg.cascade.use_cascade:
                    self.auto_branch.maybe_handle_epoch_end(epoch, step_paths, self.step_itr)

                # Saving
                new_save = (self.cfg.n_epochs_per_save != 0 and self.step_itr % self.cfg.n_epochs_per_save == 0)
                pt_save = (self.cfg.n_epochs_per_pt_save != 0 and self.step_itr % self.cfg.n_epochs_per_pt_save == 0)
                if new_save or pt_save:
                    self.save(epoch, new_save, pt_save)

                # Logging
                if self.step_itr % self.cfg.n_epochs_per_log == 0:
                    self.log_diagnostics()

        return last_return

    def _try_grow_policy_stage(self, epoch):
        policy = self.agent.sac_trainer.skill_policy._module
        if isinstance(policy, WithEncoder):
            policy = policy.module
            
        if not isinstance(policy, CascadeActor):
            return

        current_stages = len(policy.stages)
        if current_stages < self.cfg.cascade.num_policy_levels:
            if epoch > 0 and epoch % self.cfg.cascade.epochs_per_policy_stage == 0:
                self.logger.info(f"[Cascade] Adding Stage {current_stages} at epoch {epoch}")
                self.agent.add_policy_stage()
                self.logger.info(f"[Cascade] Optimizer updated. Total stages: {len(policy.stages)}")

    def _try_grow_skill_stage(self, epoch):
        if not is_pretraining_stage(self.cfg):
            return
        if not self.cfg.algo.use_hierarchical_skill:
            return
        if self.agent.current_skill_level >= self.cfg.algo.num_skill_levels:
            return
        if epoch > 0 and epoch % self.cfg.algo.epochs_per_skill_stage == 0:
            self.agent.current_skill_level += 1
            self.logger.info(
                f"[Skill] Activating skill level {self.agent.current_skill_level} at epoch {epoch}"
            )

    def train_once(self, paths, extra_scalar_metrics={}):
        self.coverage_tracker.update_train_paths(paths)
        data = agent_utils.process_samples(paths, self.cfg.sac_discount)
        metrics = aggregate_safety_metrics(paths, getattr(self.cfg, "safety", None), self.cfg.env.task)
        trained_this_epoch = True
        
        time_training = [0.0]
        # KME Refresh (Delegated to Agent)
        if should_use_kme(self.cfg):
            self.agent._maybe_refresh_idk_from_replay(metrics)
            
        with MeasureAndAccTime(time_training):
            self.agent._update_replay_buffer(data)
            
            # KME buffer check (logic specific to pre-training setup for KME)
            # This logic is tightly coupled with replay buffer and agent state.
            # Ideally should be in agent or adapter, but trainer manages loop.
            # I'll keep it here but note it's slightly coupled.
            if self.replay_buffer.n_transitions_stored < self.cfg.sac_min_buffer_size:
                 if should_use_kme(self.cfg) and self.cfg.kernel_map:
                     self.agent.path_datas.append(data)
                 trained_this_epoch = False
                 metrics['ReplayWarmupOnly'] = 1.0
                 metrics['ReplayNumTransitions'] = float(self.replay_buffer.n_transitions_stored)
            else:
                # Initial KME build
                if should_use_kme(self.cfg) and not self.agent.init_kme and is_pretraining_stage(self.cfg):
                    self.agent._build_idk_initial(metrics)
                    # If kernel map enabled, we need to re-populate buffer with KME features
                    if self.cfg.kernel_map:
                        self.replay_buffer.clear()
                        for traj_data in self.agent.path_datas:
                            self.agent._update_replay_buffer(traj_data)
                        self.agent.path_datas = []
                
                epoch_data = agent_utils.flatten_data(data, self.agent.device)
                metrics.update(self.agent.update(epoch_data))
            
        if trained_this_epoch:
            metrics.setdefault('ReplayNumTransitions', float(self.replay_buffer.n_transitions_stored))
        
        # Logging performance
        performance = utils.log_performance_ex(
            self.step_itr,
            batch=TrajectoryBatch.from_trajectory_list(self.env.spec, paths),
            discount=self.cfg.sac_discount,
        )
        self._log_metrics(performance, metrics, extra_scalar_metrics, time_training[0])
        
        return np.mean(performance['undiscounted_returns'])

    def save(self, epoch, new_save, pt_save):
        model_dir = os.path.join(self.work_dir, f'models/epoch-{epoch}')
        os.makedirs(model_dir, exist_ok=True)
        self.agent.save(os.path.join(model_dir, 'skill_policy.pt'))
        if is_pretraining_stage(self.cfg) and self.agent.traj_encoder is not None:
             # Save traj encoder
             torch.save({
                 'discrete': self.cfg.discrete,
                 'dim_skill': self.cfg.dim_skill,
                 'traj_encoder': self.agent.traj_encoder,
                 'traj_latent_normalizer': self.agent.traj_latent_normalizer,
             }, os.path.join(model_dir, 'traj_encoder.pt'))
        self._save_resume_state(epoch, model_dir)
        self.logger.info('Saved snapshot')

    def _resume_config_signature(self):
        return {
            'task': self.cfg.env.task,
            'algo': self.cfg.algo.algo,
            'stage': self.cfg.log.stage,
            'dim_skill': self.cfg.algo.dim_skill,
            'discrete': self.cfg.algo.discrete,
            'encoder': self.cfg.net.encoder,
            'ac_backbone': self.cfg.net.ac_backbone,
            'use_cascade': self.cfg.cascade.use_cascade,
            'num_policy_levels': self.cfg.cascade.num_policy_levels,
            'use_hierarchical_skill': self.cfg.algo.use_hierarchical_skill,
            'use_hierarchical_policy': self.cfg.algo.use_hierarchical_policy,
            'num_skill_levels': self.cfg.algo.num_skill_levels,
            'use_hierarchical_phi': self.cfg.algo.use_hierarchical_phi,
            'hierarchical_phi_depth': self.cfg.algo.hierarchical_phi_depth,
            'use_kme': self.cfg.algo.use_kme,
            'idk_subsample_size': self.cfg.algo.idk_subsample_size,
            'traj_latent_norm': self.cfg.algo.traj_latent_norm,
            'traj_latent_norm_eps': self.cfg.algo.traj_latent_norm_eps,
        }

    def _build_resume_state(self, epoch):
        return {
            'format': 'metra_resume_v1',
            'work_dir': self.work_dir,
            'config_signature': self._resume_config_signature(),
            'trainer_state': {
                'step_itr': int(self.step_itr),
                'total_epoch': int(self.total_epoch),
                'total_env_steps': int(self.total_env_steps),
                'next_epoch': int(epoch) + 1,
            },
            'agent_state': self.agent.get_resume_state(),
            'extra_state': self._extra_resume_state(),
        }

    def _save_resume_state(self, epoch, model_dir):
        checkpoint = self._build_resume_state(epoch)
        latest_path = os.path.join(self.work_dir, 'models', 'latest_resume.pt')
        epoch_path = os.path.join(model_dir, 'resume_state.pt')
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, epoch_path)

    def _extra_resume_state(self):
        return {
            'auto_branch': self.auto_branch.state_dict(),
            'coverage_tracker': self.coverage_tracker.state_dict(),
        }

    def _load_extra_resume_state(self, state):
        self.auto_branch.load_state_dict(state.get('auto_branch', {}))
        self.coverage_tracker.load_state_dict(state.get('coverage_tracker', {}))

    def load_resume_checkpoint(self, checkpoint_path):
        checkpoint = safe_torch_load(checkpoint_path, map_location=self.agent.device)
        self._validate_resume_checkpoint(checkpoint)

        agent_state = checkpoint.get('agent_state')
        if agent_state is None:
            raise KeyError("Resume checkpoint is missing agent_state")
        self.agent.load_resume_state(agent_state)

        trainer_state = checkpoint.get('trainer_state', {})
        self.step_itr = int(trainer_state.get('step_itr', 0))
        self.total_epoch = int(trainer_state.get('total_epoch', 0))
        self.total_env_steps = int(trainer_state.get('total_env_steps', 0))
        self.start_epoch = int(trainer_state.get('next_epoch', self.step_itr))
        self._load_extra_resume_state(checkpoint.get('extra_state', {}))
        self.logger.info("Resumed training state from %s (next_epoch=%d)", checkpoint_path, self.start_epoch)

    def _validate_resume_checkpoint(self, checkpoint):
        if not isinstance(checkpoint, dict):
            raise ValueError("Resume checkpoint must be a dict")

        signature = checkpoint.get('config_signature', {})
        expected = self._resume_config_signature()
        required_keys = ['task', 'algo', 'stage', 'dim_skill', 'encoder', 'use_cascade']
        for key in required_keys:
            if key in signature and signature[key] != expected[key]:
                raise ValueError(
                    f"Resume checkpoint mismatch for {key}: checkpoint={signature[key]!r}, current={expected[key]!r}"
                )

        optional_strict_keys = [
            'ac_backbone',
            'num_policy_levels',
            'use_hierarchical_skill',
            'use_hierarchical_policy',
            'num_skill_levels',
            'use_hierarchical_phi',
            'hierarchical_phi_depth',
            'use_kme',
            'idk_subsample_size',
            'traj_latent_norm',
            'traj_latent_norm_eps',
        ]
        for key in optional_strict_keys:
            if key in signature and signature[key] != expected[key]:
                raise ValueError(
                    f"Resume checkpoint mismatch for {key}: checkpoint={signature[key]!r}, current={expected[key]!r}"
                )

    def _set_models_mode(self, mode):
        policy = self.agent.sac_trainer.skill_policy
        if mode == 'train': policy.train()
        else: policy.eval()

    def log_diagnostics(self, pause_for_plot=False):
        total_time = (time.time() - self._start_time)
        epoch_time = (time.time() - self._itr_start_time)
        self.logger.info(f'Time {total_time:.2f} s | EpochTime {epoch_time:.2f} s')
        self.writer.add_scalar('TotalEnvSteps', self.total_env_steps, self.total_epoch)
        self.writer.add_scalar('TimeTotal', total_time, self.total_epoch)
        self.writer.flush()

    def _log_metrics(self, performance, metrics, extra, time_training):
        prefix = utils.get_metric_prefix() + 'METRA/'
        logged_values = {}
        
        # Log performance
        self.writer.add_scalar(prefix + 'AverageExternalDiscountedReturn', np.mean(performance['discounted_returns']), self.step_itr)
        self.writer.add_scalar(prefix + 'AverageExternalReturn', np.mean(performance['undiscounted_returns']), self.step_itr)
        
        # Log metrics
        for k, v in metrics.items():
            val = v.item() if torch.is_tensor(v) else v
            logged_values[k] = val
            self.writer.add_scalar(prefix + k, val, self.step_itr)
        for k, v in (extra or {}).items():
            val = v.item() if torch.is_tensor(v) else v
            logged_values[k] = val
            self.writer.add_scalar(prefix + k, val, self.step_itr)
            
        # Log grad norms
        with torch.no_grad():
            from utils.utils import compute_total_norm
            total_norm = compute_total_norm(self.agent.all_parameters())
            self.writer.add_scalar(prefix + 'TotalGradNormAll', total_norm.item(), self.step_itr)
            
        self.writer.add_scalar(prefix + 'TimeTraining', time_training, self.step_itr)
        logged_values["TimeTraining"] = time_training
        if logged_values:
            self.logger.info(
                "Metrics %s",
                " ".join(
                    f"{k}={float(v):.6g}" for k, v in sorted(logged_values.items())
                    if isinstance(v, (int, float, np.floating, np.integer))
                ),
            )

class MeasureAndAccTime:
    def __init__(self, target):
        self._target = target
    def __enter__(self):
        self._time_enter = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._target[0] += (time.time() - self._time_enter)
