import math
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from core.experimental.idk_csd_config import IdkCsdConfig
from core.metra_agent import MetraAgent
from core.metra_trainer import MeasureAndAccTime, MetraTrainer
from core.stage_contract import is_pretraining_stage
from data_structs.trajectory_batch import TrajectoryBatch
from utils import agent_utils
from utils import utils


class IdkCsdTrainer(MetraTrainer):
    def __init__(self, config: IdkCsdConfig, agent: MetraAgent, env, replay_buffer, work_dir):
        super().__init__(config, agent, env, replay_buffer, work_dir)
        self.idk_update_interval = int(config.algo.idk_update_interval)
        self.last_idk_update = 0
        self._logged_joint_schedule_warning = False

    def train(self):
        if not is_pretraining_stage(self.cfg):
            return super().train()

        last_return = None
        self.task_adapter.on_train_start()
        self._maybe_log_joint_schedule_warning()

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

                self.total_env_steps += sum(len(path['dones']) for path in step_paths)
                last_return = self.train_once_custom(
                    step_paths,
                    extra_scalar_metrics={'TimeSampling': time_sampling[0]},
                )

                self.step_itr += 1
                if self.cfg.cascade.use_cascade:
                    self.auto_branch.maybe_handle_epoch_end(epoch, step_paths, self.step_itr)

                new_save = self.cfg.n_epochs_per_save != 0 and self.step_itr % self.cfg.n_epochs_per_save == 0
                pt_save = self.cfg.n_epochs_per_pt_save != 0 and self.step_itr % self.cfg.n_epochs_per_pt_save == 0
                if new_save or pt_save:
                    self.save(epoch, new_save, pt_save)

                if self.step_itr % self.cfg.n_epochs_per_log == 0:
                    self.log_diagnostics()

        return last_return

    def train_once_custom(self, paths, extra_scalar_metrics=None):
        extra_scalar_metrics = extra_scalar_metrics or {}
        data = self._process_samples_with_time(paths)
        metrics = {}
        time_training = [0.0]

        with MeasureAndAccTime(time_training):
            self.agent._update_replay_buffer(data)

            if not self.agent.shared_kernel_initialized:
                self.agent._build_idk_initial(metrics)
                self.last_idk_update = self.step_itr
            elif self.idk_update_interval > 0 and (self.step_itr - self.last_idk_update) >= self.idk_update_interval:
                self.logger.info("Refreshing IDK and contrastive IK maps from replay...")
                self.agent._build_idk_initial(metrics)
                self.last_idk_update = self.step_itr

            if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.cfg.sac_min_buffer_size:
                return 0.0

            epoch_data = agent_utils.flatten_data(data, self.agent.device)
            for _ in range(self.cfg.train.trans_optimization_epochs):
                mix_lambda = self._compute_mix_lambda()
                step_metrics = self.agent.update_joint(epoch_data, live_paths=paths, mix_lambda=mix_lambda)
                metrics.update(step_metrics)

        performance = utils.log_performance_ex(
            self.step_itr,
            batch=TrajectoryBatch.from_trajectory_list(self.env.spec, paths),
            discount=self.cfg.sac_discount,
        )
        self._log_metrics(performance, metrics, extra_scalar_metrics, time_training[0])
        return np.mean(performance['undiscounted_returns'])

    def _process_samples_with_time(self, paths):
        data = agent_utils.process_samples(paths, self.cfg.sac_discount)
        data = defaultdict(list, data)

        for path in paths:
            traj_len = int(len(path['dones']))
            data['time_idxs'].append(np.arange(traj_len, dtype=np.float32))
            data['next_time_idxs'].append(np.arange(1, traj_len + 1, dtype=np.float32))

        return data

    def _compute_mix_lambda(self) -> float:
        warmup_steps = int(self.cfg.algo.contrastive_warmup_epochs) * int(self.cfg.train.trans_optimization_epochs)
        if warmup_steps <= 0:
            return 0.0

        progress = min(max(float(self.agent.total_train_steps) / float(warmup_steps), 0.0), 1.0)
        schedule = getattr(self.cfg.algo, 'contrastive_mix_schedule', 'cosine')

        if schedule == 'cosine':
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if schedule == 'linear':
            return 1.0 - progress
        if schedule == 'exp':
            exp_k = float(getattr(self.cfg.algo, 'contrastive_exp_k', 5.0))
            if abs(exp_k) < 1e-8:
                return 1.0 - progress
            numerator = math.exp(-exp_k * progress) - math.exp(-exp_k)
            denominator = 1.0 - math.exp(-exp_k)
            return numerator / denominator
        raise ValueError(f"Unsupported contrastive_mix_schedule: {schedule}")

    def _maybe_log_joint_schedule_warning(self):
        if self._logged_joint_schedule_warning:
            return
        self.logger.info(
            "IDK-CSD now uses joint phi/policy updates every optimization step; "
            "contrastive_n_epochs and contrastive_m_epochs remain CLI-compatible but no longer control scheduling."
        )
        self._logged_joint_schedule_warning = True

    def _extra_resume_state(self):
        state = super()._extra_resume_state()
        state.update({
            'last_idk_update': int(self.last_idk_update),
        })
        return state

    def _load_extra_resume_state(self, state):
        super()._load_extra_resume_state(state)
        self.last_idk_update = int(state.get('last_idk_update', self.last_idk_update))
