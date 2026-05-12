from __future__ import annotations

import logging
import os
import copy
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from core.cov_encoder.coverage_encoder import (
    DirectCoverageEncoder,
    load_coverage_encoder_checkpoint,
)
from core.metra_agent import MetraAgent
from core.metra_config import MetraConfig
from envs import make_env
from iod.coverage_tracker import CoverageTracker
from memory.replay_buffer import PathBufferEx
from utils import utils as metra_utils

from .config import MassPixelConfig
from .nn_mass import StreamingNNMass
from .reward_adapter import MassRewardAdapter
from .utils import (
    ScalarAccumulator,
    env_step,
    infer_pixel_shape,
    obs_image,
    sample_action,
    save_config_json,
    stack_path,
    timestamped_work_dir,
)


def obs_to_video_frame(obs, pixel_shape) -> np.ndarray:
    frame = np.asarray(obs)
    if frame.ndim == 1:
        frame = frame.reshape(tuple(pixel_shape))
    if frame.ndim != 3:
        raise ValueError(f"Expected image observation rank 3 or flat image, got shape={frame.shape}")
    if frame.shape[0] in (1, 3, 6, 9, 12) and frame.shape[-1] not in (1, 3, 6, 9, 12):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] < 3:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] != 3:
        if frame.shape[-1] % 3 == 0:
            frame = frame[..., -3:]
        else:
            frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if frame.size and float(np.nanmax(frame)) <= 1.5:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def make_video_grid(
    trajectories: List[List[np.ndarray]],
    *,
    rows: int = 3,
    cols: int = 3,
    skip_frames: int = 1,
) -> List[np.ndarray]:
    if not trajectories:
        raise ValueError("make_video_grid needs at least one trajectory")
    skip = max(1, int(skip_frames))
    cells = max(1, int(rows) * int(cols))
    clipped = [list(traj[::skip]) for traj in trajectories[:cells]]
    first = next((traj[0] for traj in clipped if traj), None)
    if first is None:
        raise ValueError("make_video_grid received only empty trajectories")
    h, w, c = first.shape
    blank = np.zeros((h, w, c), dtype=np.uint8)
    while len(clipped) < cells:
        clipped.append([blank])
    max_len = max(len(traj) for traj in clipped)
    grid_frames = []
    for t in range(max_len):
        row_frames = []
        for r in range(rows):
            col_frames = []
            for col in range(cols):
                traj = clipped[r * cols + col]
                frame = traj[min(t, len(traj) - 1)] if traj else blank
                if frame.shape != blank.shape:
                    raise ValueError(f"All video frames must share shape {blank.shape}, got {frame.shape}")
                col_frames.append(frame)
            row_frames.append(np.concatenate(col_frames, axis=1))
        grid_frames.append(np.concatenate(row_frames, axis=0))
    return grid_frames


def _extract_env_info(step_output) -> Dict:
    if not isinstance(step_output, dict):
        return {}
    info = step_output.get("info", {}) or {}
    return dict(info) if isinstance(info, dict) else {}


def _append_env_info(bucket, info: Dict) -> None:
    for key, value in (info or {}).items():
        bucket[key].append(value)


def _finalize_env_infos(bucket) -> Dict[str, np.ndarray]:
    return {key: np.asarray(values) for key, values in bucket.items()}


class MassPixelTrainer:
    def __init__(self, cfg: MassPixelConfig):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self.mass_device = self._resolve_mass_device(cfg.mass_device)
        metra_utils.set_seed_everywhere(cfg.seed)

        self.work_dir = timestamped_work_dir(cfg.workspace_root, cfg.task, cfg.seed)
        save_config_json(os.path.join(self.work_dir, "args.json"), cfg)
        os.makedirs(os.path.join(self.work_dir, "models"), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.work_dir, "tb"))
        self.logger = self._build_logger()

        self.metra_cfg = self._build_metra_config()
        self.env = make_env(mode="train", config=self.metra_cfg)
        self.pixel_shape = infer_pixel_shape(self.env)
        self.action_dim = int(self.env.spec.action_space.flat_dim)
        self.replay_buffer = PathBufferEx(
            capacity_in_transitions=int(cfg.sac_max_buffer_size),
            pixel_shape=self.pixel_shape,
        )
        self.agent = MetraAgent(self.metra_cfg, self.env, self.replay_buffer)

        self.coverage_encoder, self.coverage_checkpoint = self._build_coverage_encoder()
        self.cov_z_dim = int(getattr(self.coverage_encoder, "latent_dim", cfg.cov_latent_dim))
        self.mass = StreamingNNMass(
            z_dim=self.cov_z_dim,
            c=cfg.mass_c,
            psi=cfg.mass_psi,
            alpha=cfg.mass_alpha,
            short_size=cfg.mass_short_size,
            long_size=cfg.mass_long_size,
            w_short=cfg.mass_w_short,
            w_long=cfg.mass_w_long,
            reward_clip=cfg.mass_reward_clip,
            device=self.mass_device,
        )
        self.reward_adapter = MassRewardAdapter(
            coverage_encoder=self.coverage_encoder,
            mass_model=self.mass,
            lambda_action=cfg.lambda_action,
            lambda_delta_action=cfg.lambda_delta_action,
            lambda_done=cfg.lambda_done,
            encode_batch_size=cfg.mass_encode_batch_size,
            device=self.device,
        )
        self.coverage_tracker = CoverageTracker(cfg.task)
        self.metrics = ScalarAccumulator()
        self.global_step = 0
        self.epoch = 0
        self.episode_length = 0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._seed_obs: List[np.ndarray] = []
        self._seed_next_obs: List[np.ndarray] = []
        self._seed_actions: List[np.ndarray] = []
        self._parallel_collector = None
        self._eval_parallel_collector = None

    def _build_logger(self):
        logger = logging.getLogger("MassPixelTrainer")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(os.path.join(self.work_dir, "debug.log"), mode="a")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def _resolve_device(self, requested: str) -> torch.device:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _resolve_mass_device(self, requested: str) -> torch.device:
        requested = str(requested).strip().lower()
        if requested == "auto":
            return self.device
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            return self._resolve_device("cuda")
        raise ValueError(f"Unsupported mass_device: {requested!r}")

    def _build_metra_config(self) -> MetraConfig:
        cfg = MetraConfig()
        cfg.seed = self.cfg.seed
        cfg.device = str(self.device)
        cfg.use_gpu = int(self.device.type == "cuda")
        cfg.sample_cpu = 1

        cfg.env.task = self.cfg.task
        cfg.env.time_limit = self.cfg.time_limit
        cfg.env.framestack = self.cfg.framestack
        cfg.env.action_repeat = self.cfg.action_repeat
        cfg.env.render_size = self.cfg.render_size
        cfg.env.flatten_obs = self.cfg.flatten_obs
        cfg.env.camera = self.cfg.camera
        cfg.env.dmc_camera = self.cfg.dmc_camera

        cfg.log.workspace_root = self.work_dir
        cfg.log.stage = "zero_training"
        cfg.log.run_group = self.cfg.algo
        cfg.log.n_epochs = self.cfg.n_epochs
        cfg.log.n_epochs_per_eval = self.cfg.n_epochs_per_eval
        cfg.log.n_epochs_per_log = max(1, self.cfg.n_epochs_per_log)
        cfg.log.n_epochs_per_save = max(1, self.cfg.n_epochs_per_save)
        cfg.log.n_epochs_per_pt_save = max(1, self.cfg.n_epochs_per_pt_save)
        cfg.log.num_random_trajectories = self.cfg.eval_state_coverage_trajectories
        cfg.log.eval_plot_axis = self.cfg.eval_plot_axis

        cfg.net.encoder = self.cfg.encoder
        cfg.net.encoder_type = self.cfg.rl_encoder_type
        cfg.net.finetune_encoder = bool(self.cfg.finetune_rl_encoder)

        cfg.algo.algo = "metra"
        cfg.algo.dim_skill = 0
        cfg.algo.discrete = 0
        cfg.algo.inner = 1
        cfg.algo.alpha = self.cfg.alpha

        cfg.train.traj_batch_size = self.cfg.traj_batch_size
        cfg.train.trans_minibatch_size = self.cfg.trans_minibatch_size
        cfg.train.trans_optimization_epochs = self.cfg.trans_optimization_epochs
        cfg.train.sac_max_buffer_size = self.cfg.sac_max_buffer_size
        cfg.train.sac_min_buffer_size = self.cfg.sac_min_buffer_size
        cfg.train.common_lr = self.cfg.common_lr
        cfg.train.lr_op = self.cfg.lr_op
        cfg.train.sac_lr_q = self.cfg.sac_lr_q
        cfg.train.sac_lr_a = self.cfg.sac_lr_a
        cfg.train.sac_discount = self.cfg.sac_discount
        cfg.train.sac_tau = self.cfg.sac_tau
        cfg.train.sac_scale_reward = self.cfg.sac_scale_reward
        cfg.train.sac_target_coef = self.cfg.sac_target_coef
        cfg.train.policy_delay = self.cfg.policy_delay
        cfg.train.actor_start_steps = self.cfg.actor_start_steps
        cfg.train.n_parallel = self.cfg.n_parallel
        cfg.train.n_thread = self.cfg.n_thread
        cfg.train.parallel_sampler_enabled = self.cfg.parallel_sampler_enabled
        cfg.train.parallel_sampler_num_workers = self.cfg.parallel_sampler_num_workers
        cfg.train.parallel_sampler_fail_open = self.cfg.parallel_sampler_fail_open
        cfg.train.eval_parallel_sampler_enabled = self.cfg.eval_parallel_sampler_enabled
        cfg.train.eval_video_parallel_sampler_enabled = self.cfg.eval_video_parallel_sampler_enabled
        cfg.train.replay_staging_enabled = self.cfg.replay_staging_enabled
        cfg.train.replay_staging_pin_memory = self.cfg.replay_staging_pin_memory
        cfg.replay_staging_enabled = self.cfg.replay_staging_enabled
        cfg.replay_staging_pin_memory = self.cfg.replay_staging_pin_memory
        return cfg

    def _build_coverage_encoder(self):
        if self.cfg.cov_encoder_type == "checkpoint":
            encoder, checkpoint = load_coverage_encoder_checkpoint(
                self.cfg.cov_encoder_path,
                pixel_shape=self.pixel_shape,
                action_dim=self.action_dim,
                latent_dim=self.cfg.cov_latent_dim,
                device=self.device,
                strict_metadata=True,
                freeze=True,
            )
            return encoder, checkpoint

        if self.cfg.cov_encoder_type == "resnet-101":
            model_dir = self.cfg.cov_resnet_path
        elif self.cfg.cov_encoder_type == "dinov3":
            model_dir = self.cfg.cov_dino_path
        else:
            raise ValueError(f"Unsupported coverage encoder type: {self.cfg.cov_encoder_type}")

        encoder = DirectCoverageEncoder(
            encoder_type=self.cfg.cov_encoder_type,
            model_dir=model_dir,
            pixel_shape=self.pixel_shape,
            action_dim=self.action_dim,
            device=self.device,
        )
        encoder.freeze()
        checkpoint = {
            "encoder_type": self.cfg.cov_encoder_type,
            "model_dir": os.path.abspath(os.path.expanduser(model_dir)),
            "pixel_shape": self.pixel_shape,
            "action_dim": self.action_dim,
            "latent_dim": int(encoder.latent_dim),
            "task": self.cfg.task,
            "config": {},
            "direct_backbone": True,
        }
        return encoder, checkpoint

    def run(self):
        self.logger.info("Workspace directory: %s", self.work_dir)
        self.logger.info("Using device: %s", self.device)
        self.logger.info(
            "Using MASS device: %s; coverage encode batch size=%d",
            self.mass_device,
            self.cfg.mass_encode_batch_size,
        )
        if self.cfg.cov_encoder_type == "checkpoint":
            self.logger.info("Loaded coverage encoder checkpoint: %s", self.cfg.cov_encoder_path)
        else:
            self.logger.info(
                "Using direct coverage encoder: type=%s latent_dim=%d model_dir=%s",
                self.cfg.cov_encoder_type,
                self.cov_z_dim,
                self.coverage_checkpoint.get("model_dir"),
            )
        try:
            self._collect_seed_data()
            self._build_initial_mass()
            self._train_rl()
            self._save("final")
        finally:
            self.writer.flush()
            self.writer.close()
            try:
                self.env.close()
            except Exception:
                pass
            self._close_parallel_collectors()

    def _collect_seed_data(self):
        self.logger.info("[Phase 1] collecting %d seed steps", self.cfg.seed_steps)
        timestep = self.env.reset()
        obs = obs_image(timestep)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.episode_length = 0

        for _ in range(self.cfg.seed_steps):
            action = sample_action(self.env)
            out = env_step(self.env, action)
            next_obs = obs_image(out)
            done = bool(out.get("is_terminal", False)) or self.episode_length + 1 >= self.cfg.time_limit
            self._add_transition(obs, action, next_obs, done, self.prev_action, reward=0.0)
            self._seed_obs.append(np.asarray(obs).reshape(-1).copy())
            self._seed_next_obs.append(np.asarray(next_obs).reshape(-1).copy())
            self._seed_actions.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())

            obs = next_obs
            self.prev_action = np.asarray(action, dtype=np.float32).reshape(-1)
            self.episode_length += 1
            self.global_step += 1
            if done:
                self.metrics.add(**{"rl/episode_length": self.episode_length, "rl/terminal_rate": 1.0})
                obs = obs_image(self.env.reset())
                self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
                self.episode_length = 0
            else:
                self.metrics.add(**{"rl/terminal_rate": 0.0})

        if not self._seed_next_obs:
            raise RuntimeError("seed_steps produced no observations; MASS needs seed data to build initial partitions")
        self.logger.info("[Phase 1] replay size=%d", self.replay_buffer.n_transitions_stored)
        return obs

    def _add_transition(self, obs, action, next_obs, done, prev_action, *, reward: float):
        path = stack_path(
            {
                "obs": [obs],
                "next_obs": [next_obs],
                "actions": [np.asarray(action, dtype=np.float32)],
                "prev_actions": [np.asarray(prev_action, dtype=np.float32)],
                "rewards": [float(reward)],
                "dones": [float(done)],
            }
        )
        self.replay_buffer.add_path(path)

    def _add_transition_array(self, obs, action, next_obs, done, prev_action, *, reward: float):
        self._add_transition(obs, action, next_obs, done, prev_action, reward=reward)

    @torch.no_grad()
    def _build_initial_mass(self):
        self.logger.info("[Phase 2] building initial MASS partitions")
        next_arr = np.asarray(self._seed_next_obs, dtype=np.float32)
        self.coverage_encoder.eval()
        z = self.reward_adapter.encode_observations(next_arr)
        stats = self.mass.build_initial_partitions(z)
        self._write_mass_stats(stats, self.global_step)
        self.logger.info("[Phase 2] initial MASS stats: %s", stats)

    def _train_rl(self):
        self.logger.info(
            "[Phase 3] training reward-free RL for n_epochs=%d, traj_batch_size=%d",
            self.cfg.n_epochs,
            self.cfg.traj_batch_size,
        )
        for epoch in range(self.cfg.n_epochs):
            self.epoch = epoch + 1
            self._collect_policy_trajectories()
            if self._can_update_sac():
                for _ in range(self.cfg.trans_optimization_epochs):
                    self._update_agent()
            self._maybe_refresh_mass_for_epoch()

            if self.cfg.n_epochs_per_log > 0 and self.epoch % self.cfg.n_epochs_per_log == 0:
                self._write_mass_stats(self.mass.stats(), self.global_step)
                self._flush_metrics()

            should_save = (
                self.cfg.n_epochs_per_save > 0 and self.epoch % self.cfg.n_epochs_per_save == 0
            ) or (
                self.cfg.n_epochs_per_pt_save > 0 and self.epoch % self.cfg.n_epochs_per_pt_save == 0
            )
            if should_save:
                self._save(f"epoch-{self.epoch}")

            if self.cfg.n_epochs_per_eval > 0 and self.epoch % self.cfg.n_epochs_per_eval == 0:
                self._evaluate()
        self._flush_metrics()

    def _collect_policy_trajectories(self):
        if self._should_use_parallel_sampler(for_eval=False):
            try:
                paths = self._get_parallel_collector(for_eval=False).collect(
                    self.agent.sac_trainer.skill_policy,
                    target_num_trajectories=self.cfg.traj_batch_size,
                    sample_extra_fn=lambda: None,
                )
                self._record_parallel_timing(self._parallel_collector)
                for path in paths:
                    self._ingest_collected_path(path)
                self.coverage_tracker.update_train_paths(paths)
                return
            except Exception as exc:  # noqa: BLE001 - optional infrastructure should fail open when configured.
                if not self.cfg.parallel_sampler_fail_open:
                    raise
                self.logger.warning("Parallel MASS rollout failed; falling back to serial rollout: %s", exc)
                self._discard_parallel_collector(for_eval=False)

        train_paths = []
        for _ in range(self.cfg.traj_batch_size):
            obs = obs_image(self.env.reset())
            prev_action = np.zeros(self.action_dim, dtype=np.float32)
            length = 0
            done = False
            env_infos = defaultdict(list)
            while not done and length < self.cfg.time_limit:
                action = self._policy_action(obs)
                out = env_step(self.env, action)
                next_obs = obs_image(out)
                done = bool(out.get("is_terminal", False)) or length + 1 >= self.cfg.time_limit
                _append_env_info(env_infos, _extract_env_info(out))

                reward_out = self.reward_adapter.compute_step_reward(
                    next_obs,
                    action=action,
                    prev_action=prev_action,
                    done=done,
                    update_rms=True,
                )
                r_int = float(reward_out["r_int"].reshape(-1)[0].item())
                self._record_reward_metrics(reward_out)
                self._add_transition(obs, action, next_obs, done, prev_action, reward=r_int)
                self.mass.add_z(reward_out["z_next"])

                self.global_step += 1
                length += 1
                self.metrics.add(**{"rl/terminal_rate": float(done)})

                obs = next_obs
                prev_action = np.asarray(action, dtype=np.float32).reshape(-1)

            self.metrics.add(**{"rl/episode_length": length})
            train_paths.append({"env_infos": _finalize_env_infos(env_infos)})
        if train_paths:
            self.coverage_tracker.update_train_paths(train_paths)

    def _ingest_collected_path(self, path):
        observations = np.asarray(path["observations"])
        next_observations = np.asarray(path["next_observations"])
        actions = np.asarray(path["actions"], dtype=np.float32)
        dones = np.asarray(path.get("dones", np.zeros(actions.shape[0], dtype=bool))).reshape(-1)
        prev_action = np.zeros(self.action_dim, dtype=np.float32)
        length = int(actions.shape[0])
        for idx in range(length):
            obs = observations[idx]
            next_obs = next_observations[idx]
            action = actions[idx].reshape(-1)
            done = bool(dones[idx])
            reward_out = self.reward_adapter.compute_step_reward(
                next_obs,
                action=action,
                prev_action=prev_action,
                done=done,
                update_rms=True,
            )
            r_int = float(reward_out["r_int"].reshape(-1)[0].item())
            self._record_reward_metrics(reward_out)
            self._add_transition_array(obs, action, next_obs, done, prev_action, reward=r_int)
            self.mass.add_z(reward_out["z_next"])
            self.global_step += 1
            self.metrics.add(**{"rl/terminal_rate": float(done)})
            prev_action = action
        self.metrics.add(**{"rl/episode_length": length})

    def _should_use_parallel_sampler(self, *, for_eval: bool) -> bool:
        if self._parallel_num_workers() <= 1:
            return False
        if self._task_needs_serial_sampler():
            return False
        if for_eval:
            return bool(self.cfg.eval_parallel_sampler_enabled and self.cfg.eval_video_parallel_sampler_enabled)
        return bool(self.cfg.parallel_sampler_enabled)

    def _parallel_num_workers(self) -> int:
        configured = int(self.cfg.parallel_sampler_num_workers or 0)
        if configured > 0:
            return configured
        return int(self.cfg.n_parallel or 1)

    def _task_needs_serial_sampler(self) -> bool:
        task = str(self.cfg.task)
        if task in ("d4rl_kitchen", "kitchen", "metra_kitchen"):
            return True
        if task.startswith("isaaclab:") or task.startswith("isaaclab_") or task.startswith("galaxea_"):
            return True
        return False

    def _collector_cfg(self):
        cfg = copy.deepcopy(self.metra_cfg)
        cfg.device = "cpu"
        cfg.use_gpu = 0
        cfg.sample_cpu = 1
        cfg.net.finetune_encoder = False
        cfg.train.n_parallel = self.cfg.n_parallel
        cfg.train.parallel_sampler_num_workers = self.cfg.parallel_sampler_num_workers
        cfg.train.parallel_sampler_enabled = self.cfg.parallel_sampler_enabled
        cfg.train.eval_parallel_sampler_enabled = self.cfg.eval_parallel_sampler_enabled
        cfg.train.eval_video_parallel_sampler_enabled = self.cfg.eval_video_parallel_sampler_enabled
        return cfg

    def _get_parallel_collector(self, *, for_eval: bool):
        attr = "_eval_parallel_collector" if for_eval else "_parallel_collector"
        collector = getattr(self, attr)
        if collector is not None:
            return collector
        from envs.generic_parallel import GenericProcessTrajectoryCollector

        collector = GenericProcessTrajectoryCollector(
            self._collector_cfg(),
            num_workers=self._parallel_num_workers(),
        )
        setattr(self, attr, collector)
        return collector

    def _discard_parallel_collector(self, *, for_eval: bool):
        attr = "_eval_parallel_collector" if for_eval else "_parallel_collector"
        collector = getattr(self, attr)
        setattr(self, attr, None)
        close = getattr(collector, "close", None)
        if callable(close):
            close()

    def _close_parallel_collectors(self):
        self._discard_parallel_collector(for_eval=False)
        self._discard_parallel_collector(for_eval=True)

    def _record_parallel_timing(self, collector):
        if collector is None:
            return
        consume = getattr(collector, "consume_timing_metrics", None)
        if not callable(consume):
            return
        for key, value in consume().items():
            self.metrics.add(**{f"rl/{key}": value})

    def _can_update_sac(self) -> bool:
        stored = int(self.replay_buffer.n_transitions_stored)
        return stored >= self.cfg.sac_min_buffer_size and stored >= self.cfg.trans_minibatch_size

    def _maybe_refresh_mass_for_epoch(self):
        if self.cfg.mass_refresh_interval > 0 and self.epoch % self.cfg.mass_refresh_interval == 0:
            stats = self.mass.rolling_refresh(self.cfg.mass_refresh_num)
            self._write_mass_stats(stats, self.global_step)

    @torch.no_grad()
    def _policy_action(self, obs):
        policy = self.agent.sac_trainer.skill_policy
        policy.eval()
        action, _ = policy.get_action(obs)
        policy.train()
        return np.asarray(action, dtype=np.float32)

    def _update_agent(self):
        batch = self.agent._sample_replay_buffer()
        reward_out = self.reward_adapter.compute_batch_reward(batch, update_rms=False)
        batch["rewards"] = reward_out["rewards"]
        self.agent._normalize_sac_scalars(batch)
        metrics: Dict[str, torch.Tensor] = {}
        self.agent._optimize_op(metrics, batch, self.agent.total_train_steps)
        self.agent.total_train_steps += 1
        self._record_reward_metrics(reward_out)
        self._record_rl_metrics(metrics, reward_out)

    def _record_reward_metrics(self, reward_out):
        r_cov = reward_out["r_cov"].detach().float().reshape(-1)
        r_int = reward_out["r_int"].detach().float().reshape(-1)
        self.metrics.add(
            **{
                "train/r_cov_mean": r_cov.mean(),
                "train/r_cov_std": r_cov.std(unbiased=False) if r_cov.numel() > 1 else 0.0,
                "train/r_cov_min": r_cov.min(),
                "train/r_cov_max": r_cov.max(),
                "train/r_int_mean": r_int.mean(),
                "train/r_int_std": r_int.std(unbiased=False) if r_int.numel() > 1 else 0.0,
                "rl/action_norm": reward_out["action_norm"].mean(),
                "rl/delta_action_norm": reward_out["delta_action_norm"].mean(),
            }
        )

    def _record_rl_metrics(self, metrics, reward_out):
        if "LossQf1" in metrics and "LossQf2" in metrics:
            critic_loss = metrics["LossQf1"].detach() + metrics["LossQf2"].detach()
            self.metrics.add(**{"rl/critic_loss": critic_loss})
        if "LossSacp" in metrics:
            self.metrics.add(**{"rl/actor_loss": metrics["LossSacp"].detach()})
        if "Q1Mean" in metrics and "Q2Mean" in metrics:
            q_mean = 0.5 * (metrics["Q1Mean"].detach() + metrics["Q2Mean"].detach())
            self.metrics.add(**{"rl/q_mean": q_mean})

    def _write_mass_stats(self, stats, step):
        tag_map = {
            "short_size": "mass/short_size",
            "long_size": "mass/long_size",
            "empty_cell_ratio_short": "mass/empty_cell_ratio_short",
            "empty_cell_ratio_long": "mass/empty_cell_ratio_long",
            "cell_entropy_short": "mass/cell_entropy_short",
            "cell_entropy_long": "mass/cell_entropy_long",
            "mean_anchor_distance_short": "mass/mean_anchor_distance_short",
            "mean_anchor_distance_long": "mass/mean_anchor_distance_long",
            "repartition_count": "mass/repartition_count",
            "refresh_count": "mass/refresh_count",
        }
        for key, tag in tag_map.items():
            if key in stats:
                self.writer.add_scalar(tag, float(stats[key]), step)

    def _flush_metrics(self):
        values = self.metrics.pop_means()
        for tag, value in values.items():
            self.writer.add_scalar(tag, value, self.global_step)
        self.writer.flush()
        if values:
            self.logger.info("epoch=%d step=%d metrics=%s", self.epoch, self.global_step, values)

    @torch.no_grad()
    def _evaluate(self):
        num_rollouts = int(self.cfg.num_random_trajectories)
        deterministic = self.cfg.eval_video_policy_mode == "deterministic"
        if self._should_use_parallel_sampler(for_eval=True):
            try:
                paths = self._get_parallel_collector(for_eval=True).collect_fixed(
                    self.agent.sac_trainer.skill_policy,
                    extras=[None] * num_rollouts,
                    deterministic_policy=deterministic,
                    state_record_pixeled=True,
                )
                self._record_parallel_timing(self._eval_parallel_collector)
                self._evaluate_paths(paths, plot_env=self.env)
                self._log_eval_state_coverage()
                return
            except Exception as exc:  # noqa: BLE001 - eval sampler should also fail open.
                if not self.cfg.parallel_sampler_fail_open:
                    raise
                self.logger.warning("Parallel MASS eval rollout failed; falling back to serial eval: %s", exc)
                self._discard_parallel_collector(for_eval=True)

        eval_env = make_env(mode="eval", config=self.metra_cfg)
        prev_force = self.agent.sac_trainer.skill_policy._force_use_mode_actions
        self.agent.sac_trainer.skill_policy._force_use_mode_actions = bool(deterministic)
        returns = []
        lengths = []
        video_rollouts: List[List[np.ndarray]] = []
        eval_paths = []
        try:
            for _ in range(num_rollouts):
                obs = obs_image(eval_env.reset())
                total = 0.0
                length = 0
                frames = [obs_to_video_frame(obs, self.pixel_shape)]
                observations = []
                next_observations = []
                rewards = []
                dones = []
                env_infos = defaultdict(list)
                done = False
                while not done and length < self.cfg.time_limit:
                    observations.append(np.asarray(obs).copy())
                    action = self._policy_action(obs)
                    out = env_step(eval_env, action)
                    step_reward = float(out.get("reward", 0.0))
                    total += step_reward
                    length += 1
                    done = bool(out.get("is_terminal", False)) or length >= self.cfg.time_limit
                    _append_env_info(env_infos, _extract_env_info(out))
                    rewards.append(step_reward)
                    dones.append(done)
                    obs = obs_image(out)
                    next_observations.append(np.asarray(obs).copy())
                    frames.append(obs_to_video_frame(obs, self.pixel_shape))
                returns.append(total)
                lengths.append(length)
                video_rollouts.append(frames)
                eval_paths.append(
                    {
                        "observations": np.asarray(observations),
                        "next_observations": np.asarray(next_observations),
                        "rewards": np.asarray(rewards, dtype=np.float32),
                        "dones": np.asarray(dones, dtype=bool),
                        "env_infos": _finalize_env_infos(env_infos),
                    }
                )
            if self.cfg.eval_record_video:
                self._write_eval_video(video_rollouts)
            self._write_eval_traj_plot(eval_env, eval_paths)
            if returns:
                self.writer.add_scalar("eval/episode_reward", float(np.mean(returns)), self.global_step)
                self.writer.add_scalar("eval/episode_length", float(np.mean(lengths)), self.global_step)
                self.writer.add_scalar("eval/num_random_trajectories", float(num_rollouts), self.global_step)
                self.logger.info(
                    "eval epoch=%d step=%d reward_mean=%.4f length_mean=%.2f video=%s",
                    self.epoch,
                    self.global_step,
                    float(np.mean(returns)),
                    float(np.mean(lengths)),
                    bool(self.cfg.eval_record_video),
                )
        finally:
            self.agent.sac_trainer.skill_policy._force_use_mode_actions = prev_force
            try:
                eval_env.close()
            except Exception:
                pass
        self._log_eval_state_coverage()

    def _evaluate_paths(self, paths, *, plot_env=None):
        returns = []
        lengths = []
        video_rollouts: List[List[np.ndarray]] = []
        for path in paths:
            rewards = np.asarray(path.get("rewards", []), dtype=np.float32).reshape(-1)
            observations = np.asarray(path["observations"])
            next_observations = np.asarray(path["next_observations"])
            frames = [obs_to_video_frame(obs, self.pixel_shape) for obs in observations]
            if len(next_observations):
                frames.append(obs_to_video_frame(next_observations[-1], self.pixel_shape))
            returns.append(float(rewards.sum()) if rewards.size else 0.0)
            lengths.append(int(observations.shape[0]))
            video_rollouts.append(frames)
        if self.cfg.eval_record_video:
            self._write_eval_video(video_rollouts)
        self._write_eval_traj_plot(plot_env or self.env, paths)
        if returns:
            self.writer.add_scalar("eval/episode_reward", float(np.mean(returns)), self.global_step)
            self.writer.add_scalar("eval/episode_length", float(np.mean(lengths)), self.global_step)
            self.writer.add_scalar("eval/num_random_trajectories", float(len(paths)), self.global_step)
            self.logger.info(
                "eval epoch=%d step=%d reward_mean=%.4f length_mean=%.2f video=%s parallel=True",
                self.epoch,
                self.global_step,
                float(np.mean(returns)),
                float(np.mean(lengths)),
                bool(self.cfg.eval_record_video),
            )

    def _write_eval_traj_plot(self, env, paths) -> None:
        if not paths:
            return
        render_trajectories = getattr(env, "render_trajectories", None)
        if not callable(render_trajectories):
            self.logger.warning("MASS eval trajectory plot skipped: env does not expose render_trajectories")
            return
        try:
            rng = np.random.RandomState(int(self.cfg.seed) + int(self.epoch))
            colors = metra_utils.get_skill_colors(rng.randn(len(paths), 2) * 4.0)
            with metra_utils.FigManager(
                self.work_dir,
                self.global_step,
                "TrajPlot_RandomZ",
                writer=self.writer,
                global_step=self.global_step,
            ) as fm:
                render_trajectories(paths, colors, self.cfg.eval_plot_axis, fm.ax)
        except Exception as exc:  # noqa: BLE001 - trajectory plots are diagnostic only.
            self.logger.warning("MASS eval trajectory plot skipped: %s", exc)

    @torch.no_grad()
    def _log_eval_state_coverage(self):
        try:
            paths = self._collect_eval_state_coverage_paths()
            self._write_eval_coverage_metrics(paths)
        except Exception as exc:  # noqa: BLE001 - coverage eval should not break training when fail-open is enabled.
            if not self.cfg.parallel_sampler_fail_open:
                raise
            self.logger.warning("MASS eval state coverage failed; continuing without coverage metrics: %s", exc)

    @torch.no_grad()
    def _collect_eval_state_coverage_paths(self):
        num_rollouts = int(self.cfg.eval_state_coverage_trajectories)
        if self._should_use_parallel_eval_coverage_sampler():
            try:
                paths = self._get_parallel_collector(for_eval=True).collect_fixed(
                    self.agent.sac_trainer.skill_policy,
                    extras=[None] * num_rollouts,
                    deterministic_policy=True,
                    state_record_pixeled=False,
                )
                self._record_parallel_timing(self._eval_parallel_collector)
                if len(paths) == num_rollouts:
                    return paths
                self.logger.warning(
                    "Parallel MASS coverage eval returned %d/%d trajectories; falling back to serial eval.",
                    len(paths),
                    num_rollouts,
                )
                self._discard_parallel_collector(for_eval=True)
            except Exception as exc:  # noqa: BLE001
                if not self.cfg.parallel_sampler_fail_open:
                    raise
                self.logger.warning("Parallel MASS coverage eval failed; falling back to serial eval: %s", exc)
                self._discard_parallel_collector(for_eval=True)

        eval_env = make_env(mode="eval", config=self.metra_cfg)
        policy = self.agent.sac_trainer.skill_policy
        prev_force = getattr(policy, "_force_use_mode_actions", False)
        policy._force_use_mode_actions = True
        paths = []
        try:
            for _ in range(num_rollouts):
                paths.append(self._collect_single_eval_coverage_path(eval_env))
        finally:
            policy._force_use_mode_actions = prev_force
            try:
                eval_env.close()
            except Exception:
                pass
        return paths

    def _should_use_parallel_eval_coverage_sampler(self) -> bool:
        return (
            self._parallel_num_workers() > 1
            and not self._task_needs_serial_sampler()
            and bool(self.cfg.eval_parallel_sampler_enabled)
        )

    @torch.no_grad()
    def _collect_single_eval_coverage_path(self, eval_env):
        obs = obs_image(eval_env.reset())
        rewards = []
        dones = []
        env_infos = defaultdict(list)
        done = False
        length = 0
        while not done and length < self.cfg.time_limit:
            action = self._policy_action(obs)
            out = env_step(eval_env, action)
            _append_env_info(env_infos, _extract_env_info(out))
            rewards.append(float(out.get("reward", 0.0)))
            done = bool(out.get("is_terminal", False)) or length + 1 >= self.cfg.time_limit
            dones.append(done)
            obs = obs_image(out)
            length += 1
        return {
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "env_infos": _finalize_env_infos(env_infos),
        }

    def _write_eval_coverage_metrics(self, paths):
        metrics = {}
        metrics.update(self.coverage_tracker.compute_policy_metrics(paths))
        metrics.update(self.coverage_tracker.compute_queue_metrics())
        metrics.update(self.coverage_tracker.compute_total_metrics())
        metrics["NumStateCoverageTrajectories"] = float(len(paths or []))
        for key, value in sorted(metrics.items()):
            self.writer.add_scalar(f"eval/{key}", float(value), self.global_step)
        self.writer.flush()
        if metrics:
            self.logger.info("eval state coverage step=%d metrics=%s", self.global_step, metrics)

    def _write_eval_video(self, video_rollouts: List[List[np.ndarray]]) -> None:
        grid_frames = make_video_grid(
            video_rollouts,
            rows=self.cfg.eval_video_grid_rows,
            cols=self.cfg.eval_video_grid_cols,
            skip_frames=self.cfg.video_skip_frames,
        )
        video_np = np.stack(grid_frames, axis=0)
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0).float().div(255.0)
        self.writer.add_video("eval/random_rollouts_3x3", video_tensor, self.global_step, fps=self.cfg.eval_video_fps)
        self.writer.flush()

        video_dir = os.path.join(self.work_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"eval_epoch-{self.epoch}_step-{self.global_step}_3x3.mp4")
        try:
            import imageio.v2 as imageio

            imageio.mimsave(video_path, list(video_np), fps=self.cfg.eval_video_fps)
            self.logger.info("Saved eval video grid: %s", video_path)
        except Exception as exc:  # noqa: BLE001 - video file is best-effort; TensorBoard already has it.
            self.logger.warning("Failed to save eval video grid to %s: %s", video_path, exc)

    def _save(self, name: str):
        model_dir = os.path.join(self.work_dir, "models", name)
        os.makedirs(model_dir, exist_ok=True)
        self.agent.save(os.path.join(model_dir, "skill_policy.pt"))
        torch.save(
            {
                "state_dict": self.coverage_encoder.state_dict()
                if self.cfg.cov_encoder_type == "checkpoint"
                else None,
                "encoder_type": self.cfg.cov_encoder_type,
                "source_cov_encoder_path": self.cfg.cov_encoder_path,
                "model_dir": self.coverage_checkpoint.get("model_dir"),
                "pixel_shape": self.pixel_shape,
                "action_dim": self.action_dim,
                "latent_dim": self.cov_z_dim,
                "task": self.cfg.task,
                "global_step": self.global_step,
                "source_config": self.coverage_checkpoint.get("config", {}),
            },
            os.path.join(model_dir, "coverage_encoder.pt"),
        )
