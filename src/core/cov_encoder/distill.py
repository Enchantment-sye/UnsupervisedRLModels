from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs import make_env
from utils import utils as metra_utils

from .config import CovEncoderDistillConfig
from .coverage_encoder import CoverageEncoder, ResNet101Teacher, save_coverage_encoder_checkpoint
from .utils import (
    env_step,
    infer_pixel_shape,
    obs_image,
    sample_action,
    save_config_json,
    timestamped_distill_work_dir,
    to_torch,
)


class CoverageEncoderDistillTrainer:
    """Algorithm-agnostic coverage encoder distillation runner."""

    def __init__(self, cfg: CovEncoderDistillConfig):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        metra_utils.set_seed_everywhere(cfg.seed)

        self.work_dir = timestamped_distill_work_dir(cfg.workspace_root, cfg.task, cfg.seed)
        os.makedirs(os.path.join(self.work_dir, "models"), exist_ok=True)
        save_config_json(os.path.join(self.work_dir, "args.json"), cfg)
        self.writer = SummaryWriter(os.path.join(self.work_dir, "tb"))
        self.logger = self._build_logger()

        self.env = make_env(mode="train", config=cfg)
        self.pixel_shape = infer_pixel_shape(self.env)
        self.action_dim = int(self.env.spec.action_space.flat_dim)

        self.coverage_encoder = CoverageEncoder(
            pixel_shape=self.pixel_shape,
            action_dim=self.action_dim,
            latent_dim=cfg.cov_latent_dim,
        ).to(self.device)
        self.teacher = ResNet101Teacher(cfg.teacher_path, device=self.device, pixel_shape=self.pixel_shape)
        self.optimizer = torch.optim.Adam(self.coverage_encoder.parameters(), lr=cfg.distill_lr)

        self.save_path = self._resolve_save_path()
        self._obs: List[np.ndarray] = []
        self._next_obs: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []

    def _resolve_device(self, requested: str) -> torch.device:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _resolve_save_path(self) -> str:
        if self.cfg.cov_encoder_save_path:
            return os.path.abspath(os.path.expanduser(self.cfg.cov_encoder_save_path))
        return os.path.join(self.work_dir, "models", "coverage_encoder.pt")

    def _build_logger(self):
        logger = logging.getLogger("CoverageEncoderDistillTrainer")
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

    def run(self) -> str:
        self.logger.info("Workspace directory: %s", self.work_dir)
        self.logger.info("Using device: %s", self.device)
        self.logger.info("Coverage encoder checkpoint will be saved to: %s", self.save_path)
        try:
            self._collect_random_data()
            self._distill()
            self._save()
            return self.save_path
        finally:
            self.writer.flush()
            self.writer.close()
            try:
                self.env.close()
            except Exception:
                pass

    def _collect_random_data(self):
        self.logger.info("[Distill] collecting %d random environment steps", self.cfg.distill_sample_steps)
        timestep = self.env.reset()
        obs = obs_image(timestep)
        episode_length = 0

        for step in range(self.cfg.distill_sample_steps):
            action = sample_action(self.env)
            out = env_step(self.env, action)
            next_obs = obs_image(out)
            done = bool(out.get("is_terminal", False)) or episode_length + 1 >= self.cfg.time_limit

            self._obs.append(np.asarray(obs).reshape(-1).copy())
            self._next_obs.append(np.asarray(next_obs).reshape(-1).copy())
            self._actions.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())

            obs = next_obs
            episode_length += 1
            if done:
                self.writer.add_scalar("distill/episode_length", episode_length, step + 1)
                obs = obs_image(self.env.reset())
                episode_length = 0

        if not self._obs:
            raise RuntimeError("distill_sample_steps produced no observations")
        self.logger.info("[Distill] collected transitions=%d", len(self._obs))

    def _distill(self):
        self.logger.info("[Distill] training coverage encoder for %d steps", self.cfg.distill_train_steps)
        obs_arr = np.asarray(self._obs, dtype=np.float32)
        next_arr = np.asarray(self._next_obs, dtype=np.float32)
        action_arr = np.asarray(self._actions, dtype=np.float32)
        n = obs_arr.shape[0]
        teacher_features = self._precompute_teacher_features(obs_arr)
        batch_size = min(self.cfg.distill_batch_size, n)
        if self.cfg.smoke_test:
            batch_size = min(batch_size, 8)

        self.coverage_encoder.train()
        for step in range(self.cfg.distill_train_steps):
            idx = np.random.choice(n, batch_size, replace=True)
            batch = {
                "obs": to_torch(obs_arr[idx], self.device),
                "next_obs": to_torch(next_arr[idx], self.device),
                "actions": to_torch(action_arr[idx], self.device),
                "teacher_features": teacher_features[idx].to(self.device),
            }
            losses = self.coverage_encoder.compute_cov_loss(
                batch,
                teacher=None,
                lambda_dist=self.cfg.lambda_dist,
                lambda_aug=self.cfg.lambda_aug,
                lambda_var=self.cfg.lambda_var,
                lambda_cov=self.cfg.lambda_cov,
                lambda_inv=self.cfg.lambda_inv,
                var_gamma=self.cfg.cov_var_gamma,
            )
            self.optimizer.zero_grad()
            losses["loss_total"].backward()
            self.optimizer.step()

            if step % max(1, self.cfg.log_interval) == 0 or step + 1 == self.cfg.distill_train_steps:
                scalars = {}
                for key, value in losses.items():
                    scalars[f"cov/{key}"] = float(value.detach().mean().item())
                    self.writer.add_scalar(f"cov/{key}", scalars[f"cov/{key}"], step)
                self.writer.flush()
                self.logger.info("distill_step=%d metrics=%s", step + 1, scalars)
        self.coverage_encoder.freeze()
        self.logger.info("[Distill] coverage encoder distillation complete")

    @torch.no_grad()
    def _precompute_teacher_features(self, obs_arr: np.ndarray) -> torch.Tensor:
        self.logger.info("[Distill] precomputing teacher features for %d observations", obs_arr.shape[0])
        feats = []
        batch_size = 64 if not self.cfg.smoke_test else 16
        for start in range(0, obs_arr.shape[0], batch_size):
            obs = to_torch(obs_arr[start : start + batch_size], self.device)
            feats.append(self.teacher(obs).detach().cpu())
        return torch.cat(feats, dim=0)

    def _save(self) -> None:
        save_coverage_encoder_checkpoint(
            self.save_path,
            self.coverage_encoder,
            pixel_shape=self.pixel_shape,
            action_dim=self.action_dim,
            latent_dim=self.cfg.cov_latent_dim,
            task=self.cfg.task,
            teacher_path=self.cfg.teacher_path,
            config=asdict(self.cfg),
            global_step=self.cfg.distill_sample_steps,
            distill_steps=self.cfg.distill_train_steps,
        )
        self.logger.info("[Distill] saved coverage encoder to %s", self.save_path)
