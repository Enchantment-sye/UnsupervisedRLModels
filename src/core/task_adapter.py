from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
import numpy as np
import torch
import os
from envs import should_use_isaaclab_backend
from envs.ogbench_scene_kitchen_like_eval import is_ogbench_scene_metric_key
from utils import utils, video_motion
from data_structs.trajectory_batch import TrajectoryBatch
from workers.rollout import SkillRolloutWorker
from core.skill_selector import DiscreteSkillSelector, CEMSkillSelector
from core.stage_contract import is_finetune_stage, is_zero_training_stage, requires_best_skill_search, uses_skill_inputs
from core.metrics.trajectory_structure import (
    SkipReason,
    build_structure_eval_options,
    compute_training_eval_structure_metrics,
    exception_skip_metrics,
    interval_skip_metrics,
    video_trajectories_are_suitable,
)
from core.metra_viz import plot_skill_xy_trajectories


def _task_requests_galaxea_sim(config) -> bool:
    task = getattr(config, 'task', '')
    return isinstance(task, str) and task.startswith('galaxea_')


def _task_requests_ogbench_visual(config) -> bool:
    task = getattr(config, 'task', '')
    return isinstance(task, str) and task.startswith('ogbench_') and bool(getattr(config, 'encoder', 0))


def _task_requests_ogbench_scene(config) -> bool:
    task = getattr(config, 'task', '')
    return isinstance(task, str) and task.startswith('ogbench_') and 'scene' in task


@contextmanager
def _temporary_video_capture_mode(env, enabled):
    setter = getattr(env, 'set_video_capture_active', None)
    if not enabled or not callable(setter):
        yield
        return

    setter(True)
    try:
        yield
    finally:
        setter(False)


class TaskAdapter(ABC):
    def __init__(self, config, env, agent, work_dir, logger):
        self.cfg = config
        self.env = env
        self.agent = agent
        self.work_dir = work_dir
        self.logger = logger
        
    @abstractmethod
    def get_train_trajectories(self, batch_size):
        pass
    
    @abstractmethod
    def evaluate(self, step_itr, total_epoch, writer, *, log_policy_coverage_to_writer=True):
        pass
        
    @abstractmethod
    def on_train_start(self):
        pass

class SkillDiscoveryTaskAdapter(TaskAdapter):
    _OFFICIAL_KITCHEN_TASKS = (
        'd4rl_kitchen',
        'metra_kitchen',
        'kitchen',
    )
    _D4RL_KITCHEN_EVAL_TASKS = (
        'microwave',
        'kettle',
        'light switch',
        'top left burner',
    )
    _XY_BIN_LEGACY_KEYS = (
        'MjNumUniqueCoords',
        'MjNumTrajs',
        'MjAvgTrajLen',
        'MjNumCoords',
    )
    _LOCOMOTION_POLICY_COVERAGE_KEYS = (
        'MjNumTrajs',
        'MjAvgTrajLen',
        'MjNumCoords',
        'MjNumUniqueCoords',
        'PolicyStateCoverageXYBins',
        'PolicyFinalXYDispMean',
        'PolicyFinalXYDispMax',
        'PolicyXRange',
        'PolicyYRange',
        'PolicyMeanSpeed',
    )

    def __init__(self, config, env, agent, work_dir, logger):
        super().__init__(config, env, agent, work_dir, logger)
        self.best_skill = None
        self.rollout_worker = None
        self._parallel_train_collector = None
        self._generic_parallel_collector = None
        self._kitchen_parallel_collector = None
        self._last_train_sampling_metrics = {}
        self.coverage_tracker = None
        self._structure_metrics_eval_count = 0
        
    def on_train_start(self):
        if not requires_best_skill_search(self.cfg):
            return
        if self.best_skill is None:
            restored_skill = getattr(self.agent, 'best_skill', None)
            if restored_skill is not None:
                self.best_skill = np.asarray(restored_skill, dtype=np.float32)
            else:
                self.best_skill = self.find_best_skill()
        self.agent.best_skill = np.asarray(self.best_skill, dtype=np.float32)

    def _active_skill_level(self):
        if not self.cfg.use_hierarchical_skill:
            return 1
        return max(1, min(self.cfg.num_skill_levels, getattr(self.agent, 'current_skill_level', 1)))

    def _skill_shape(self):
        if self.cfg.use_hierarchical_skill:
            return (self.cfg.num_skill_levels, self.cfg.dim_skill)
        return (self.cfg.dim_skill,)

    def _reshape_skill(self, skill):
        skill = np.asarray(skill, dtype=np.float32)
        if not self.cfg.use_hierarchical_skill:
            return skill.reshape(self.cfg.dim_skill)
        return skill.reshape(self.cfg.num_skill_levels, self.cfg.dim_skill)

    def _sample_skill_batch(self, batch_size, rng=None):
        rng = rng or np.random
        if self.cfg.use_hierarchical_skill:
            active_levels = self._active_skill_level()
            skills = np.zeros((batch_size, self.cfg.num_skill_levels, self.cfg.dim_skill), dtype=np.float32)
            if self.cfg.discrete:
                idxs = rng.randint(0, self.cfg.dim_skill, size=(batch_size, active_levels))
                eye = np.eye(self.cfg.dim_skill, dtype=np.float32)
                for level in range(active_levels):
                    skills[:, level, :] = eye[idxs[:, level]]
            else:
                active = rng.randn(batch_size, active_levels, self.cfg.dim_skill).astype(np.float32)
                if self.cfg.unit_length:
                    norms = np.linalg.norm(active, axis=-1, keepdims=True) + 1e-8
                    active = active / norms
                skills[:, :active_levels, :] = active
            return skills

        if self.cfg.discrete:
            idxs = rng.randint(0, self.cfg.dim_skill, batch_size)
            return np.eye(self.cfg.dim_skill, dtype=np.float32)[idxs]

        skills = rng.randn(batch_size, self.cfg.dim_skill).astype(np.float32)
        if self.cfg.unit_length:
            n = np.linalg.norm(skills, axis=1, keepdims=True) + 1e-8
            skills = skills / n
        return skills

    def sample_skills(self, batch_size, rng=None):
        if not uses_skill_inputs(self.cfg):
            return np.zeros((batch_size, 0), dtype=np.float32)
        return self._sample_skill_batch(batch_size, rng=rng)

    def build_skill_extras(self, skills):
        skills = list(skills)
        if not uses_skill_inputs(self.cfg):
            return [None] * len(skills)
        return [{'skill': self._reshape_skill(skill)} for skill in skills]

    def collect_policy_trajectories(
            self,
            extras,
            *,
            deterministic_policy,
            rollout_seed,
            state_record_pixeled=False,
            video_frame_source=None,
            reset_perturbations=None):
        extras = list(extras)
        reset_perturbations = self._normalize_reset_perturbations(reset_perturbations, len(extras))
        if (
                _task_requests_galaxea_sim(self.cfg)
                and int(getattr(self.cfg, 'n_parallel', 1) or 1) > 1
        ):
            try:
                collector = self._get_galaxea_parallel_train_collector()
                trajectories = collector.collect_fixed(
                    self.agent.sac_trainer.skill_policy,
                    extras=extras,
                    deterministic_policy=deterministic_policy,
                    state_record_pixeled=state_record_pixeled,
                    video_frame_source=video_frame_source,
                    reset_perturbations=reset_perturbations,
                )
                collector.consume_timing_metrics()
                if len(trajectories) == len(extras):
                    return trajectories
                self.logger.warning(
                    "Galaxea parallel eval returned %d/%d trajectories; falling back to serial rollout.",
                    len(trajectories),
                    len(extras),
                )
            except Exception as exc:
                self.logger.warning("Galaxea parallel eval/video rollout failed; falling back to serial: %s", exc)

        if self._should_use_generic_parallel_sampler(for_eval=True, state_record_pixeled=state_record_pixeled):
            try:
                collector = self._get_generic_parallel_collector()
                trajectories = collector.collect_fixed(
                    self.agent.sac_trainer.skill_policy,
                    extras=extras,
                    deterministic_policy=deterministic_policy,
                    state_record_pixeled=state_record_pixeled,
                    video_frame_source=video_frame_source,
                    reset_perturbations=reset_perturbations,
                )
                if len(trajectories) == len(extras):
                    return trajectories
                self.logger.warning(
                    "Generic parallel eval/video returned %d/%d trajectories; falling back to serial rollout.",
                    len(trajectories),
                    len(extras),
                )
                self._discard_generic_parallel_collector()
            except Exception as exc:
                self._discard_generic_parallel_collector()
                if not bool(getattr(self.cfg, 'parallel_sampler_fail_open', True)):
                    raise
                self.logger.warning("Generic parallel eval/video rollout failed; falling back to serial: %s", exc)

        if self._should_use_kitchen_parallel_sampler(for_eval=True, state_record_pixeled=state_record_pixeled):
            collector = self._get_kitchen_parallel_collector()
            trajectories = collector.collect_fixed(
                self.agent.sac_trainer.skill_policy,
                extras=extras,
                deterministic_policy=deterministic_policy,
                state_record_pixeled=state_record_pixeled,
                video_frame_source=video_frame_source,
                reset_perturbations=reset_perturbations,
            )
            if len(trajectories) != len(extras):
                raise RuntimeError(
                    f"Kitchen parallel eval sampler returned {len(trajectories)}/{len(extras)} trajectories."
                )
            return trajectories

        return self._collect_policy_trajectories_serial(
            extras,
            deterministic_policy=deterministic_policy,
            rollout_seed=rollout_seed,
            state_record_pixeled=state_record_pixeled,
            video_frame_source=video_frame_source,
            reset_perturbations=reset_perturbations,
        )

    def _collect_policy_trajectories_serial(
            self,
            extras,
            *,
            deterministic_policy,
            rollout_seed,
            state_record_pixeled=False,
            video_frame_source=None,
            reset_perturbations=None):
        reset_perturbations = self._normalize_reset_perturbations(reset_perturbations, len(extras))
        rollout_worker = SkillRolloutWorker(
            rollout_seed,
            self.cfg.time_limit,
            cur_extra_keys=['skill'] if uses_skill_inputs(self.cfg) else [],
            pixeled=self.cfg.encoder,
            config=self.cfg,
        )
        batches = []
        for idx, extra in enumerate(extras):
            batch = rollout_worker.rollout(
                self.env,
                self.agent.sac_trainer.skill_policy,
                extra,
                deterministic_policy=deterministic_policy,
                state_record_pixeled=state_record_pixeled,
                video_frame_source=video_frame_source,
                reset_perturbation=reset_perturbations[idx],
            )
            batches.append(batch)
        return TrajectoryBatch.concatenate(*batches).to_trajectory_list()

    @staticmethod
    def _normalize_reset_perturbations(reset_perturbations, expected_length):
        if reset_perturbations is None:
            return [None] * int(expected_length)
        reset_perturbations = list(reset_perturbations)
        if len(reset_perturbations) != int(expected_length):
            raise ValueError(
                f"Expected {expected_length} reset perturbations, got {len(reset_perturbations)}."
            )
        return reset_perturbations

    def build_auto_branch_probe_extras(self, num_episodes, branch_id):
        if not uses_skill_inputs(self.cfg):
            return [None] * num_episodes

        if self.cfg.discrete and not self.cfg.use_hierarchical_skill:
            if self.cfg.dim_skill <= num_episodes:
                return [{'skill': np.eye(self.cfg.dim_skill, dtype=np.float32)[idx]} for idx in range(self.cfg.dim_skill)]
            skill_ids = np.linspace(0, self.cfg.dim_skill - 1, num=num_episodes, dtype=int)
            skill_ids = np.unique(skill_ids)
            return [{'skill': np.eye(self.cfg.dim_skill, dtype=np.float32)[idx]} for idx in skill_ids]

        seed = int(self.cfg.seed + branch_id * 9973)
        if not self.cfg.auto_branch.seeded_probe_skills:
            seed += int(getattr(self.agent, 'total_train_steps', 0))
        rng = np.random.RandomState(seed)
        skills = self._sample_skill_batch(num_episodes, rng=rng)
        return [{'skill': np.asarray(skill, dtype=np.float32)} for skill in skills]

    def get_train_trajectories(self, batch_size):
        if should_use_isaaclab_backend(self.cfg) and int(getattr(self.cfg, 'isaaclab_num_envs', 1) or 1) > 1:
            collector = self._get_parallel_train_collector()
            trajectories = collector.collect(
                self.agent.sac_trainer.skill_policy,
                target_num_trajectories=batch_size,
                sample_extra_fn=self._sample_single_train_extra,
            )
            self._last_train_sampling_metrics = collector.consume_timing_metrics()
            return trajectories

        if _task_requests_galaxea_sim(self.cfg) and int(getattr(self.cfg, 'n_parallel', 1) or 1) > 1:
            collector = self._get_galaxea_parallel_train_collector()
            trajectories = collector.collect(
                self.agent.sac_trainer.skill_policy,
                target_num_trajectories=batch_size,
                sample_extra_fn=self._sample_single_train_extra,
            )
            self._last_train_sampling_metrics = collector.consume_timing_metrics()
            return trajectories

        if self._should_use_kitchen_parallel_sampler(for_eval=False, state_record_pixeled=False):
            collector = self._get_kitchen_parallel_collector()
            trajectories = collector.collect(
                self.agent.sac_trainer.skill_policy,
                target_num_trajectories=batch_size,
                sample_extra_fn=self._sample_single_train_extra,
            )
            self._last_train_sampling_metrics = collector.consume_timing_metrics()
            if len(trajectories) != batch_size:
                raise RuntimeError(
                    f"Kitchen parallel train sampler returned {len(trajectories)}/{batch_size} trajectories."
                )
            return trajectories

        if self._should_use_generic_parallel_sampler(for_eval=False, state_record_pixeled=False):
            try:
                collector = self._get_generic_parallel_collector()
                trajectories = collector.collect(
                    self.agent.sac_trainer.skill_policy,
                    target_num_trajectories=batch_size,
                    sample_extra_fn=self._sample_single_train_extra,
                )
                self._last_train_sampling_metrics = collector.consume_timing_metrics()
                if len(trajectories) == batch_size:
                    return trajectories
                self.logger.warning(
                    "Generic parallel train sampler returned %d/%d trajectories; falling back to serial rollout.",
                    len(trajectories),
                    batch_size,
                )
                self._discard_generic_parallel_collector()
            except Exception as exc:
                self._discard_generic_parallel_collector()
                if not bool(getattr(self.cfg, 'parallel_sampler_fail_open', True)):
                    raise
                self.logger.warning("Generic parallel train sampler failed; falling back to serial: %s", exc)

        if self.rollout_worker is None:
            self.rollout_worker = SkillRolloutWorker(
                self.cfg.seed,
                self.cfg.time_limit,
                cur_extra_keys=['skill'] if uses_skill_inputs(self.cfg) else [],
                pixeled=self.cfg.encoder,
                config=self.cfg,
            )
            
        kwargs = self._get_train_trajectories_kwargs(batch_size)
        
        batches = []
        extras = kwargs.get('extras', [None]*batch_size)
        timing_totals = {}
        
        policy = self.agent.sac_trainer.skill_policy
        
        for i in range(batch_size):
            extra = extras[i]
            batch = self.rollout_worker.rollout(
                self.env, 
                policy, 
                extra, 
                deterministic_policy=False, 
                state_record_pixeled=False
            )
            batches.append(batch)
            rollout_metrics = self.rollout_worker.consume_timing_metrics()
            for key, value in rollout_metrics.items():
                timing_totals[key] = timing_totals.get(key, 0.0) + float(value)
            
        trajectories = TrajectoryBatch.concatenate(*batches)
        self._last_train_sampling_metrics = timing_totals
        return trajectories.to_trajectory_list()

    def consume_train_sampling_metrics(self):
        metrics = dict(self._last_train_sampling_metrics)
        self._last_train_sampling_metrics = {}
        return metrics

    def _get_train_trajectories_kwargs(self, batch_size):
        if not uses_skill_inputs(self.cfg):
            if self.rollout_worker is not None:
                self.rollout_worker._cur_extra_keys = []
            return {}

        if is_finetune_stage(self.cfg):
            if self.best_skill is None:
                self.on_train_start()
            extras = [{'skill': np.asarray(self.best_skill, dtype=np.float32)} for _ in range(batch_size)]
            return dict(extras=extras)

        if self.rollout_worker is not None:
            self.rollout_worker._cur_extra_keys = ['skill']

        skills = self.sample_skills(batch_size)
        extras = self.build_skill_extras(skills)
        return dict(extras=extras)

    def _sample_single_train_extra(self):
        if not uses_skill_inputs(self.cfg):
            return None

        if is_finetune_stage(self.cfg):
            if self.best_skill is None:
                self.on_train_start()
            return {'skill': np.asarray(self.best_skill, dtype=np.float32)}

        skill = self.sample_skills(1)[0]
        return {'skill': self._reshape_skill(skill)}

    def _get_parallel_train_collector(self):
        if self._parallel_train_collector is not None:
            return self._parallel_train_collector

        from envs.isaaclab.parallel_train import IsaacLabParallelTrajectoryCollector

        handles_getter = getattr(self.env, "get_parallel_train_handles", None)
        if not callable(handles_getter):
            raise RuntimeError(
                "IsaacLab parallel training requested, but the current env does not expose parallel train handles."
            )
        self._parallel_train_collector = IsaacLabParallelTrajectoryCollector(self.cfg, handles_getter())
        return self._parallel_train_collector

    def _get_galaxea_parallel_train_collector(self):
        if self._parallel_train_collector is not None:
            return self._parallel_train_collector

        from envs.galaxea_sim_parallel import GalaxeaSimProcessTrajectoryCollector

        self._parallel_train_collector = GalaxeaSimProcessTrajectoryCollector(
            self.cfg,
            num_envs=int(getattr(self.cfg, 'n_parallel', 1) or 1),
        )
        return self._parallel_train_collector

    def _get_generic_parallel_collector(self):
        if self._generic_parallel_collector is not None:
            return self._generic_parallel_collector

        from envs.generic_parallel import GenericProcessTrajectoryCollector

        self._generic_parallel_collector = GenericProcessTrajectoryCollector(
            self.cfg,
            num_workers=self._generic_parallel_num_workers(),
        )
        return self._generic_parallel_collector

    def _get_kitchen_parallel_collector(self):
        if self._kitchen_parallel_collector is not None:
            return self._kitchen_parallel_collector

        from envs.kitchen_parallel import KitchenProcessTrajectoryCollector

        self._kitchen_parallel_collector = KitchenProcessTrajectoryCollector(
            self.cfg,
            num_workers=self._generic_parallel_num_workers(),
        )
        return self._kitchen_parallel_collector

    def _discard_generic_parallel_collector(self):
        collector = self._generic_parallel_collector
        self._generic_parallel_collector = None
        close = getattr(collector, 'close', None)
        if callable(close):
            close()

    def _generic_parallel_num_workers(self):
        configured = int(getattr(self.cfg, 'parallel_sampler_num_workers', 0) or 0)
        if configured > 0:
            return configured
        return int(getattr(self.cfg, 'n_parallel', 1) or 1)

    def _should_use_generic_parallel_sampler(self, *, for_eval: bool, state_record_pixeled: bool):
        if should_use_isaaclab_backend(self.cfg) or _task_requests_galaxea_sim(self.cfg):
            return False
        if getattr(self.cfg, 'task', '') in self._OFFICIAL_KITCHEN_TASKS:
            return False
        if bool(getattr(getattr(self.cfg, 'safety', None), 'enabled', 0)):
            return False
        if self._generic_parallel_num_workers() <= 1:
            return False
        if for_eval:
            if not bool(getattr(self.cfg, 'eval_parallel_sampler_enabled', False)):
                return False
            if state_record_pixeled and not bool(getattr(self.cfg, 'eval_video_parallel_sampler_enabled', True)):
                return False
            return True
        return bool(getattr(self.cfg, 'parallel_sampler_enabled', False))

    def _should_use_kitchen_parallel_sampler(self, *, for_eval: bool, state_record_pixeled: bool):
        if getattr(self.cfg, 'task', '') not in self._OFFICIAL_KITCHEN_TASKS:
            return False
        if self._generic_parallel_num_workers() <= 1:
            return False
        if for_eval:
            if not bool(getattr(self.cfg, 'eval_parallel_sampler_enabled', False)):
                return False
            if state_record_pixeled and not bool(getattr(self.cfg, 'eval_video_parallel_sampler_enabled', True)):
                return False
            return True
        return bool(getattr(self.cfg, 'parallel_sampler_enabled', False))

    def close(self):
        for collector in (
                self._parallel_train_collector,
                self._generic_parallel_collector,
                self._kitchen_parallel_collector):
            close = getattr(collector, 'close', None)
            if callable(close):
                close()

    def find_best_skill(self):
        if not uses_skill_inputs(self.cfg):
            return np.zeros((0,), dtype=np.float32)

        if self.rollout_worker is None:
            self.rollout_worker = SkillRolloutWorker(
                self.cfg.seed,
                self.cfg.time_limit,
                cur_extra_keys=['skill'] if uses_skill_inputs(self.cfg) else [],
                pixeled=self.cfg.encoder,
                config=self.cfg,
            )

        if self.cfg.use_hierarchical_skill:
            candidates = self._sample_skill_batch(32)
            best_return = -float('inf')
            best_skill = candidates[0]
            self.logger.info("Selecting best hierarchical skill by random search...")
            for skill in candidates:
                avg_ret = 0.0
                for _ in range(2):
                    batch = self.rollout_worker.rollout(
                        self.env,
                        self.agent.sac_trainer.skill_policy,
                        {'skill': skill},
                        deterministic_policy=False,
                        state_record_pixeled=False,
                    )
                    avg_ret += float(batch.to_trajectory_list()[0]['rewards'].sum())
                avg_ret /= 2
                if avg_ret > best_return:
                    best_return = avg_ret
                    best_skill = skill
            return self._reshape_skill(best_skill)

        if self.cfg.discrete:
            selector = DiscreteSkillSelector(
                env=self.env,
                actor=self.agent.sac_trainer.skill_policy,
                worker=self.rollout_worker,
                device=self.agent.device,
                dim_skill=self.cfg.dim_skill,
                logger=self.logger
            )
        else:
            selector = CEMSkillSelector(
                env=self.env,
                actor=self.agent.sac_trainer.skill_policy,
                worker=self.rollout_worker,
                device=self.agent.device,
                dim_skill=self.cfg.dim_skill,
                logger=self.logger
            )
        return self._reshape_skill(selector.select())

    def evaluate(self, step_itr, total_epoch, writer, *, log_policy_coverage_to_writer=True):
        return self._evaluate_impl(
            step_itr,
            total_epoch,
            writer,
            log_policy_coverage_to_writer=log_policy_coverage_to_writer,
        )

    def _evaluate_impl(self, step_itr, total_epoch, writer, *, log_policy_coverage_to_writer: bool):
        trajectories = self.collect_policy_coverage_trajectories(total_epoch)
        video_trajectories = None
        video_policy_mode = None
        
        # Calculate Metrics
        eval_metrics = {}
        sum_returns = 0
        for traj in trajectories:
            sum_returns += traj['rewards'].sum()
        num_eval_trajs = max(len(trajectories), 1)
        eval_metrics['ReturnOverall'] = sum_returns / num_eval_trajs

        # Log Evaluation
        with utils.GlobalContext({'phase': 'eval', 'policy': 'skill'}):
             performance = utils.log_performance_ex(
                step_itr,
                TrajectoryBatch.from_trajectory_list(self.env.spec, trajectories),
                discount=self.cfg.sac_discount,
                additional_records=eval_metrics,
            )
             for k, v in performance['scalars'].items():
                writer.add_scalar('eval/' + k, v, step_itr)
        self._log_d4rl_kitchen_eval_metrics(trajectories, step_itr, writer)
        ogbench_scene_metrics = self._log_ogbench_scene_kitchen_like_eval_metrics(
            trajectories,
            step_itr,
            writer,
        )
        policy_coverage_metrics = self.compute_policy_coverage_metrics(trajectories)
        tracker = getattr(self, 'coverage_tracker', None)
        if tracker is not None:
            tracker_metrics = {}
            tracker_metrics.update(tracker.compute_policy_metrics(trajectories))
            tracker_metrics.update(tracker.compute_queue_metrics())
            tracker_metrics.update(tracker.compute_total_metrics())
            policy_coverage_metrics.update(tracker_metrics)
        if log_policy_coverage_to_writer:
            self.log_policy_coverage_metrics_to_writer(policy_coverage_metrics, step_itr, writer)
        self._maybe_log_skill_xy_trajectories(step_itr, total_epoch, writer)

        # 2. Video Recording / Motion Analysis
        motion_analysis_enabled = bool(self.cfg.motion_analysis.enabled)
        should_collect_video = bool(self.cfg.eval_record_video or motion_analysis_enabled)
        if should_collect_video:
            if self.cfg.eval_record_video:
                self.logger.info("Recording video...")
            else:
                self.logger.info("Collecting video rollouts for motion analysis...")

            if not uses_skill_inputs(self.cfg):
                video_extras = [None] * self.cfg.num_video_repeats
            elif is_finetune_stage(self.cfg):
                if self.best_skill is None:
                    self.on_train_start()
                fixed_skill = np.asarray(self.best_skill, dtype=np.float32)
                video_extras = [{'skill': fixed_skill} for _ in range(self.cfg.num_video_repeats)]
            else:
                video_skills = self._get_video_skills()
                video_extras = self.build_skill_extras(video_skills)

            video_frame_source = None
            video_view_context = nullcontext()
            warmup_render_capture_fn = None
            if should_use_isaaclab_backend(self.cfg):
                video_frame_source = getattr(self.cfg, 'isaaclab_video_source', 'observation')
                if video_frame_source == 'render':
                    try:
                        from envs.isaaclab.viewer_runtime import (
                            temporary_video_viewer_preset,
                            warmup_render_capture as _warmup_render_capture,
                        )
                    except ImportError:
                        video_view_context = nullcontext()
                    else:
                        warmup_render_capture_fn = _warmup_render_capture
                        video_view_context = temporary_video_viewer_preset(
                            self.env,
                            getattr(self.cfg, 'isaaclab_video_viewer_preset', 'inherit'),
                        )
            elif _task_requests_galaxea_sim(self.cfg):
                video_frame_source = getattr(self.cfg, 'galaxea_sim_video_source', 'observation')
            elif _task_requests_ogbench_scene(self.cfg):
                video_frame_source = getattr(self.cfg, 'ogbench_video_source', 'blog')

            video_shape = self._get_video_record_shape()

            video_reset_perturbations = None
            if _task_requests_ogbench_scene(self.cfg):
                video_reset_perturbations = self._build_ogbench_video_reset_perturbations(
                    len(video_extras),
                )

            video_capture_context = _temporary_video_capture_mode(
                self.env,
                _task_requests_ogbench_scene(self.cfg),
            )
            with video_view_context, video_capture_context:
                if (
                    video_frame_source == 'render'
                    and hasattr(self.env, 'capture_video_frame')
                    and warmup_render_capture_fn is not None
                ):
                    warmup_render_capture_fn(
                        lambda: self.env.capture_video_frame(source=video_frame_source),
                    )
                video_trajectories = self.collect_policy_trajectories(
                    video_extras,
                    deterministic_policy=True,
                    rollout_seed=self.cfg.seed + 100 + total_epoch,
                    state_record_pixeled=True,
                    video_frame_source=video_frame_source,
                    reset_perturbations=video_reset_perturbations,
                )
                video_policy_mode = 'deterministic'

            if self.cfg.eval_record_video:
                utils.record_video(
                    self.work_dir,
                    step_itr,
                    'Video_RandomZ',
                    video_trajectories,
                    n_cols=self._get_video_n_cols(),
                    skip_frames=self.cfg.video_skip_frames,
                    shape=video_shape,
                    async_encode=bool(getattr(self.cfg, 'async_video_encoding', False)),
                    logger=self.logger,
                )

            video_tensor = None
            video_entries = None
            video_pixel_metrics = None
            try:
                video_tensor = utils.trajectories_to_video_tensor(
                    video_trajectories,
                    skip_frames=self.cfg.video_skip_frames,
                    shape=video_shape,
                )
                video_entries = [
                    {
                        'video_id': video_motion.format_video_id(idx, self.cfg.num_video_repeats),
                        'frames': np.transpose(video_tensor[idx], (0, 2, 3, 1)),
                    }
                    for idx in range(video_tensor.shape[0])
                ]
                video_pixel_metrics = video_motion.compute_video_pixel_motion_metrics(
                    video_entries,
                    self.cfg.motion_analysis,
                    num_video_repeats=self.cfg.num_video_repeats,
                )
                for metric_name, metric_value in video_pixel_metrics.items():
                    if np.isscalar(metric_value) and np.isfinite(float(metric_value)):
                        writer.add_scalar(f'eval/{metric_name}', float(metric_value), step_itr)
            except Exception as exc:
                self.logger.warning("[VideoPixelMetrics] failed during eval: %s", exc)

            if motion_analysis_enabled:
                try:
                    if video_tensor is None:
                        video_tensor = utils.trajectories_to_video_tensor(
                            video_trajectories,
                            skip_frames=self.cfg.video_skip_frames,
                            shape=video_shape,
                        )
                    if video_entries is None:
                        video_entries = [
                            {
                                'video_id': video_motion.format_video_id(idx, self.cfg.num_video_repeats),
                                'frames': np.transpose(video_tensor[idx], (0, 2, 3, 1)),
                            }
                            for idx in range(video_tensor.shape[0])
                        ]
                    motion_result = video_motion.analyze_video_collection(
                        video_entries,
                        self.cfg.motion_analysis,
                    )
                    if video_pixel_metrics is not None:
                        motion_result['video_pixel_metrics'] = video_pixel_metrics
                        motion_result.update(video_pixel_metrics)
                    video_motion.log_motion_analysis(motion_result, logger=self.logger)
                    writer.add_scalar(
                        'eval/MotionLargeMotionFrameRatioMean',
                        motion_result['mean_large_motion_ratio'],
                        step_itr,
                    )
                except Exception as exc:
                    self.logger.warning("[MotionAnalysis] failed during eval: %s", exc)
        self._maybe_log_structure_metrics(
            step_itr,
            total_epoch,
            writer,
            video_trajectories=video_trajectories,
            video_policy_mode=video_policy_mode,
        )
        return {
            'trajectories': trajectories,
            'policy_coverage_metrics': policy_coverage_metrics,
            'ogbench_scene_kitchen_like_metrics': ogbench_scene_metrics,
        }

    def _maybe_log_structure_metrics(
            self,
            step_itr,
            total_epoch,
            writer,
            *,
            video_trajectories=None,
            video_policy_mode=None):
        if not bool(getattr(self.cfg, 'eval_structure_metrics', False)):
            return
        if writer is None:
            return

        self._structure_metrics_eval_count += 1
        backends = getattr(self.cfg, 'eval_structure_metrics_backends', 'temporal,ikse')
        interval = max(1, int(getattr(self.cfg, 'eval_structure_metrics_interval', 1)))
        if self._structure_metrics_eval_count % interval != 0:
            self._write_structure_metrics(writer, step_itr, interval_skip_metrics(backends))
            return

        try:
            trajectories, options, skill_ids, source_metrics = self._collect_structure_metric_trajectories(
                total_epoch,
                video_trajectories=video_trajectories,
                video_policy_mode=video_policy_mode,
            )
            metrics = compute_training_eval_structure_metrics(
                trajectories,
                options=options,
                skill_ids=skill_ids,
                cfg=self.cfg,
                env_name=getattr(self.cfg, 'task', ''),
                device=getattr(self.cfg, 'device', 'cpu'),
                backends=backends,
                used_video_trajectories=bool(source_metrics.get('StructureMetricsUsedVideoTrajectories', 0.0)),
                used_extra_rollouts=bool(source_metrics.get('StructureMetricsUsedExtraRollouts', 0.0)),
                options_subsampled=bool(source_metrics.get('StructureMetricsSubsampled', 0.0)),
            )
            metrics.update(source_metrics)
        except Exception as exc:
            if not bool(getattr(self.cfg, 'eval_structure_metrics_fail_open', True)):
                raise
            self.logger.warning("[StructureMetrics] failed during eval; training will continue: %s", exc)
            metrics = exception_skip_metrics(backends, reason=SkipReason.BACKEND_EXCEPTION)
        self._write_structure_metrics(writer, step_itr, metrics)

    def _maybe_log_skill_xy_trajectories(self, step_itr, total_epoch, writer):
        if not bool(getattr(self.cfg, 'eval_skill_xy_plot', True)):
            return
        if not self._task_supports_skill_xy_plot():
            return

        try:
            rollouts_per_skill = max(1, int(getattr(self.cfg, 'eval_skill_xy_plot_rollouts_per_skill', 3)))
            base_skills = self._get_base_video_skills()
            if len(base_skills) == 0:
                return

            repeated_skills = np.repeat(base_skills, rollouts_per_skill, axis=0)
            extras = self.build_skill_extras(repeated_skills)
            trajectories = self.collect_policy_trajectories(
                extras,
                deterministic_policy=True,
                rollout_seed=int(getattr(self.cfg, 'seed', 0)) + 700000 + int(total_epoch),
                state_record_pixeled=False,
            )
            plot_skill_xy_trajectories(
                trajectories,
                n_trajs_per_skill=rollouts_per_skill,
                snapshot_dir=self.work_dir,
                writer=writer,
                step_itr=step_itr,
                plot_axis=getattr(self.cfg, 'eval_plot_axis', None),
                logger=self.logger,
            )
        except Exception as exc:
            self.logger.warning("[SkillXYTrajPlot] failed during eval; training will continue: %s", exc)

    def _task_supports_skill_xy_plot(self):
        task = str(getattr(self.cfg, 'task', '') or '').lower()
        return task.startswith('dmc_') or 'ant' in task

    def _collect_structure_metric_trajectories(self, total_epoch, *, video_trajectories=None, video_policy_mode=None):
        source_metrics = {
            'StructureMetricsUsedVideoTrajectories': 0.0,
            'StructureMetricsUsedExtraRollouts': 0.0,
            'StructureMetricsVideoSkipReasonCode': float(SkipReason.OK),
        }
        policy_mode = getattr(self.cfg, 'eval_structure_metrics_policy_mode', 'deterministic')
        rollouts_per_skill = int(getattr(self.cfg, 'eval_structure_metrics_rollouts_per_skill', 3))
        reset_perturb_scale = float(getattr(self.cfg, 'eval_structure_metrics_reset_perturb_scale', 0.0))
        source_metrics['StructureMetricsResetPerturbScale'] = reset_perturb_scale
        source_metrics['StructureMetricsResetPerturbedRollouts'] = 0.0
        if reset_perturb_scale > 0.0:
            source_metrics['StructureMetricsVideoSkipReasonCode'] = float(SkipReason.VIDEO_TRAJECTORIES_NOT_SUITABLE)
        if (
                reset_perturb_scale <= 0.0
                and bool(getattr(self.cfg, 'eval_structure_metrics_use_video_trajectories', True))
        ):
            suitable, reason, inferred_skill_ids = video_trajectories_are_suitable(
                video_trajectories,
                min_trajs_per_cluster=rollouts_per_skill,
                policy_mode=policy_mode,
                video_policy_mode=video_policy_mode,
            )
            if suitable:
                source_metrics['StructureMetricsUsedVideoTrajectories'] = 1.0
                return video_trajectories, None, inferred_skill_ids, source_metrics
            source_metrics['StructureMetricsVideoSkipReasonCode'] = float(reason)

        option_bundle = build_structure_eval_options(
            discrete=bool(getattr(self.cfg, 'discrete', False)),
            dim_skill=int(getattr(self.cfg, 'dim_skill', 0)),
            unit_length=bool(getattr(self.cfg, 'unit_length', True)),
            rollouts_per_skill=rollouts_per_skill,
            num_skills=int(getattr(self.cfg, 'eval_structure_metrics_num_skills', -1)),
            max_trajs=int(getattr(self.cfg, 'eval_structure_metrics_max_trajs', 96)),
            anchor_seed=int(getattr(self.cfg, 'eval_structure_metrics_anchor_seed', 0)),
            num_random_trajectories=int(getattr(self.cfg, 'num_random_trajectories', 16)),
            use_hierarchical_skill=bool(getattr(self.cfg, 'use_hierarchical_skill', False)),
            num_skill_levels=int(getattr(self.cfg, 'num_skill_levels', 1)),
        )
        if option_bundle.options.shape[0] == 0:
            return [], option_bundle.options, option_bundle.skill_ids, source_metrics

        extras = self.build_skill_extras(option_bundle.options)
        reset_perturbations = self._build_structure_reset_perturbations(
            len(extras),
            total_epoch=total_epoch,
            scale=reset_perturb_scale,
        )
        structure_trajectories = self.collect_policy_trajectories(
            extras,
            deterministic_policy=(policy_mode == 'deterministic'),
            rollout_seed=int(getattr(self.cfg, 'seed', 0)) + 5000 + int(total_epoch),
            state_record_pixeled=False,
            reset_perturbations=reset_perturbations,
        )
        source_metrics['StructureMetricsUsedExtraRollouts'] = 1.0
        source_metrics['StructureMetricsResetPerturbedRollouts'] = float(
            any(perturbation is not None for perturbation in reset_perturbations)
        )
        source_metrics['StructureMetricsSubsampled'] = float(option_bundle.subsampled)
        return structure_trajectories, option_bundle.options, option_bundle.skill_ids, source_metrics

    def _build_structure_reset_perturbations(self, num_rollouts, *, total_epoch, scale):
        scale = float(scale)
        if scale <= 0.0:
            return [None] * int(num_rollouts)
        base_seed = (
            int(getattr(self.cfg, 'seed', 0))
            + 9000003
            + int(total_epoch) * 1009
            + int(getattr(self.cfg, 'eval_structure_metrics_anchor_seed', 0)) * 9176
        )
        return [(base_seed + idx, scale) for idx in range(int(num_rollouts))]

    def _build_ogbench_video_reset_perturbations(self, num_rollouts):
        base_seed = int(getattr(self.cfg, 'eval_video_reset_seed', 1000003))
        seed = int(getattr(self.cfg, 'seed', 0)) + base_seed
        return [(seed, 1.0) for _ in range(int(num_rollouts))]

    def _get_video_record_shape(self):
        if _task_requests_ogbench_scene(self.cfg):
            size = int(getattr(self.cfg, 'ogbench_video_render_size', 0) or 0)
            if size > 0:
                return size, size
        size = int(getattr(self.cfg, 'render_size', 64))
        return size, size

    def _write_structure_metrics(self, writer, step_itr, metrics):
        # This hook only covers the task_adapter training eval path. The legacy
        # src/core/metra.py and metra_evaluation.py paths are intentionally left
        # unchanged in this phase.
        write_legacy = bool(getattr(self.cfg, 'eval_structure_metrics_write_legacy_tags', False))
        for key, value in sorted((metrics or {}).items()):
            if not write_legacy and key in {'Entropy_Raw', 'Entropy_Enc', 'DBI_Raw', 'DBI_Enc'}:
                continue
            writer.add_scalar(f'eval/{key}', float(value), step_itr)

    def collect_policy_coverage_trajectories(self, total_epoch):
        extras = self._build_quantitative_eval_extras(self.cfg.num_random_trajectories)
        return self.collect_policy_trajectories(
            extras,
            deterministic_policy=True,
            rollout_seed=self.cfg.seed + 100 + total_epoch,
            state_record_pixeled=False,
        )

    def _build_quantitative_eval_extras(self, num_eval_trajs):
        if not uses_skill_inputs(self.cfg):
            return [None] * num_eval_trajs

        if is_finetune_stage(self.cfg):
            if self.best_skill is None:
                self.on_train_start()
            repeated_skill = np.repeat(np.asarray(self.best_skill, dtype=np.float32)[None, ...], num_eval_trajs, axis=0)
            return [{'skill': s} for s in repeated_skill]

        eval_skills = self.sample_skills(num_eval_trajs)
        return self.build_skill_extras(eval_skills)

    def compute_policy_coverage_metrics(self, trajectories):
        calc_eval_metrics = getattr(self.env, 'calc_eval_metrics', None)
        if not callable(calc_eval_metrics):
            return {}

        raw_metrics = calc_eval_metrics(trajectories, is_option_trajectories=True)
        if not raw_metrics:
            return {}

        raw_metrics = dict(raw_metrics)
        if any(is_ogbench_scene_metric_key(key) for key in raw_metrics):
            return {}
        if (
                'MjNumUniqueCoords' not in raw_metrics
                and 'PolicyStateCoverageXYBins' not in raw_metrics
        ):
            return {}

        coverage_metrics = {}
        for key in self._LOCOMOTION_POLICY_COVERAGE_KEYS:
            if key in raw_metrics:
                coverage_metrics[key] = float(raw_metrics[key])
        if (
                'PolicyStateCoverageXYBins' not in coverage_metrics
                and 'MjNumUniqueCoords' in raw_metrics
        ):
            coverage_metrics['PolicyStateCoverageXYBins'] = float(raw_metrics['MjNumUniqueCoords'])
        return coverage_metrics

    def log_policy_coverage_metrics_to_writer(self, metrics, step_itr, writer):
        if writer is None:
            return
        for key, value in metrics.items():
            writer.add_scalar(f'eval/{key}', value, step_itr)

    def log_policy_coverage_metrics_to_logger(self, metrics, step_itr, *, print_to_stdout: bool = False):
        if not metrics:
            return
        ordered_keys = self._LOCOMOTION_POLICY_COVERAGE_KEYS
        parts = []
        for key in ordered_keys:
            if key not in metrics:
                continue
            value = metrics[key]
            if float(value).is_integer():
                value_str = str(int(value))
            else:
                value_str = f'{value:.4f}'
            parts.append(f'{key} = {value_str}')
        if not parts:
            return
        message = f"Step {step_itr}: " + ' | '.join(parts)
        self.logger.info(message)
        if print_to_stdout:
            print(message)

    def _log_ogbench_scene_kitchen_like_eval_metrics(self, trajectories, step_itr, writer):
        calc_eval_metrics = getattr(self.env, 'calc_eval_metrics', None)
        if not callable(calc_eval_metrics):
            return {}

        raw_metrics = calc_eval_metrics(trajectories, is_option_trajectories=True)
        if not raw_metrics:
            return {}

        metrics = {
            key: value
            for key, value in dict(raw_metrics).items()
            if is_ogbench_scene_metric_key(key)
        }
        if not metrics:
            return {}

        for key, value in sorted(metrics.items()):
            if isinstance(value, (bool, np.bool_)):
                scalar = float(value)
            elif np.isscalar(value) and np.isfinite(float(value)):
                scalar = float(value)
            else:
                continue
            if writer is not None:
                writer.add_scalar(f'eval/{key}', scalar, step_itr)
            self.logger.info(f"Step {step_itr}: Eval {key} = {scalar}")
        return metrics

    def _log_d4rl_kitchen_eval_metrics(self, trajectories, step_itr, writer):
        if self.cfg.task not in self._OFFICIAL_KITCHEN_TASKS:
            return

        calc_eval_metrics = getattr(self.env, 'calc_eval_metrics', None)
        if not callable(calc_eval_metrics):
            raise AttributeError(
                "Official METRA Kitchen eval requires env.calc_eval_metrics()."
            )

        official_metrics = calc_eval_metrics(trajectories, is_option_trajectories=True)
        for key in (
                'KitchenTaskBottomBurner',
                'KitchenTaskLightSwitch',
                'KitchenTaskSlideCabinet',
                'KitchenTaskHingeCabinet',
                'KitchenTaskMicrowave',
                'KitchenTaskKettle',
                'KitchenOverall',
                'KitchenPolicyTaskCoverage',
                'KitchenBottomBurnerSuccessRate',
                'KitchenLightSwitchSuccessRate',
                'KitchenSlideCabinetSuccessRate',
                'KitchenHingeCabinetSuccessRate',
                'KitchenMicrowaveSuccessRate',
                'KitchenKettleSuccessRate',
                'KitchenAvgCompletedTasksPerTraj',
                'KitchenBestCompletedTasksPerTraj',
                'KitchenMissingSuccessKeys'):
            if key not in official_metrics:
                raise KeyError(f"Missing official Kitchen metric {key}")
            value = official_metrics[key]
            writer.add_scalar(f'eval/{key}', value, step_itr)
            self.logger.info(f"Step {step_itr}: Eval {key} = {value}")

        kitchen_aliases = {
            'KitchenTaskBottomBurner': 'eval/kitchen/bottom_burner_success',
            'KitchenTaskLightSwitch': 'eval/kitchen/light_switch_success',
            'KitchenTaskSlideCabinet': 'eval/kitchen/slide_cabinet_success',
            'KitchenTaskHingeCabinet': 'eval/kitchen/hinge_cabinet_success',
            'KitchenTaskMicrowave': 'eval/kitchen/microwave_success',
            'KitchenTaskKettle': 'eval/kitchen/kettle_success',
            'KitchenOverall': 'eval/kitchen/overall_6task_coverage',
            'KitchenPolicyTaskCoverage': 'eval/kitchen/policy_task_coverage',
            'KitchenAvgCompletedTasksPerTraj': 'eval/avg_completed_tasks',
            'KitchenBestCompletedTasksPerTraj': 'eval/kitchen/best_completed_tasks',
        }
        for key, tag in kitchen_aliases.items():
            writer.add_scalar(tag, official_metrics[key], step_itr)

        self._log_d4rl_kitchen_legacy_eval_metrics(trajectories, step_itr, writer)

    def _log_d4rl_kitchen_legacy_eval_metrics(self, trajectories, step_itr, writer):
        kitchen_successes = {task: [] for task in self._D4RL_KITCHEN_EVAL_TASKS}
        completed_tasks_counts = []

        for path in trajectories:
            ep_completed_count = 0
            env_infos = path.get('env_infos', {})
            for task in self._D4RL_KITCHEN_EVAL_TASKS:
                success_key = f'{task} success'
                success_values = env_infos.get(success_key)
                if success_values is None or len(success_values) == 0:
                    continue
                is_completed = success_values[-1] > 0.5
                kitchen_successes[task].append(float(is_completed))
                if is_completed:
                    ep_completed_count += 1
            completed_tasks_counts.append(ep_completed_count)

        for task, values in kitchen_successes.items():
            if not values:
                continue
            avg_success = np.mean(values)
            writer.add_scalar(f'eval_legacy/{task} success_rate', avg_success, step_itr)
            self.logger.info(f"Step {step_itr}: Eval legacy {task} success_rate = {avg_success:.2f}")

        if completed_tasks_counts:
            avg_completed = np.mean(completed_tasks_counts)
            writer.add_scalar('eval_legacy/avg_completed_tasks', avg_completed, step_itr)
            self.logger.info(f"Step {step_itr}: Eval legacy Avg Completed Tasks = {avg_completed:.10f}")

    def _get_base_video_skills(self):
        if not uses_skill_inputs(self.cfg):
            return np.zeros((1, 0), dtype=np.float32)

        if is_finetune_stage(self.cfg) and self.best_skill is not None:
            fixed_skill = np.asarray(self.best_skill, dtype=np.float32)
            return fixed_skill[None, ...]

        if getattr(self.cfg, 'use_hierarchical_skill', False):
            return self.sample_skills(16)

        if self.cfg.discrete:
            return np.eye(self.cfg.dim_skill, dtype=np.float32)
        else:
            if self.cfg.dim_skill == 2:
                radius = 1. if self.cfg.unit_length else 1.5
                video_skills = []
                for angle in [3, 2, 1, 4]:
                    video_skills.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                video_skills.append([0, 0])
                for angle in [0, 5, 6, 7]:
                    video_skills.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                video_skills = np.asarray(video_skills, dtype=np.float32)
            else:
                video_skills = np.random.randn(9, self.cfg.dim_skill)
                if self.cfg.unit_length:
                    video_skills = video_skills / np.linalg.norm(video_skills, axis=1, keepdims=True)
                video_skills = video_skills.astype(np.float32)

        return video_skills

    def _get_video_skills(self):
        return np.repeat(self._get_base_video_skills(), self.cfg.num_video_repeats, axis=0)

    def _get_video_n_cols(self):
        if (
            uses_skill_inputs(self.cfg)
            and not self.cfg.discrete
            and not self.cfg.use_hierarchical_skill
            and not is_finetune_stage(self.cfg)
        ):
            return 3 * max(1, int(self.cfg.num_video_repeats))
        return None
