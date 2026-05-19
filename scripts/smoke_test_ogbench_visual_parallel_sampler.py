#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from envs.generic_parallel import GenericProcessTrajectoryCollector  # noqa: E402


class RandomPolicy:
    def __init__(self, action_dim, seed=0):
        self._rng = np.random.RandomState(seed)
        self._force_use_mode_actions = False
        self._action_dim = int(action_dim)

    def reset(self):
        pass

    def get_actions(self, obs):
        obs = np.asarray(obs)
        batch = int(obs.shape[0])
        actions = self._rng.uniform(-1.0, 1.0, size=(batch, self._action_dim)).astype(np.float32)
        return actions, {"mean": actions.copy()}


def main():
    parser = argparse.ArgumentParser(description="Smoke test OGBench visual generic parallel rollout.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--render-size", type=int, default=128)
    parser.add_argument("--video-render-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = _cfg(args)
    policy = RandomPolicy(action_dim=5, seed=args.seed)
    collector = GenericProcessTrajectoryCollector(cfg, num_workers=args.num_workers)
    try:
        train_paths = collector.collect(
            policy,
            target_num_trajectories=2,
            sample_extra_fn=lambda: {"skill": np.asarray([1.0, 0.0], dtype=np.float32)},
        )
        _validate_train_paths(train_paths, args)
        train_metrics = collector.consume_timing_metrics()

        video_paths = collector.collect_fixed(
            policy,
            extras=[
                {"skill": np.asarray([1.0, 0.0], dtype=np.float32)},
                {"skill": np.asarray([0.0, 1.0], dtype=np.float32)},
            ],
            deterministic_policy=True,
            state_record_pixeled=True,
            video_frame_source="blog",
        )
        _validate_video_paths(video_paths, args)
        video_metrics = collector.consume_timing_metrics()
    finally:
        collector.close()

    print(f"Collected {len(train_paths)} OGBench visual trajectories with {args.num_workers} workers.")
    print(f"Observation shape: {train_paths[0]['observations'].shape}")
    print(f"ParallelSamplerNumWorkers: {train_metrics['ParallelSamplerNumWorkers']}")
    print(f"Video observation shape: {video_paths[0]['observations'].shape}")
    print(f"VideoParallelSamplerNumWorkers: {video_metrics['ParallelSamplerNumWorkers']}")
    print("Video eval generic parallel enabled: True")


def _cfg(args):
    return SimpleNamespace(
        task="ogbench_scene",
        env_backend="url",
        stage="pre_training",
        algo=SimpleNamespace(algo="metra", dim_skill=2),
        dim_skill=2,
        encoder=1,
        seed=int(args.seed),
        action_repeat=1,
        time_limit=int(args.max_steps),
        render_size=int(args.render_size),
        framestack=3,
        flatten_obs=0,
        parallel_sampler_enabled=True,
        parallel_sampler_num_workers=int(args.num_workers),
        parallel_sampler_start_method="auto",
        n_parallel=int(args.num_workers),
        ogbench_video_source="blog",
        ogbench_video_render_size=int(args.video_render_size),
        ogbench_video_opaque_arm=1,
    )


def _validate_train_paths(paths, args):
    if len(paths) != 2:
        raise AssertionError(f"Expected 2 train paths, got {len(paths)}.")
    expected_dim = int(args.render_size) * int(args.render_size) * 3 * 3
    for path in paths:
        if path["observations"].shape != (int(args.max_steps), expected_dim):
            raise AssertionError(f"Unexpected train observation shape {path['observations'].shape}.")
        env_infos = path.get("env_infos", {})
        if "privileged/block_0_pos" not in env_infos or "button_states" not in env_infos:
            raise AssertionError("Privileged OGBench Scene env_infos did not survive parallel rollout.")


def _validate_video_paths(paths, args):
    if len(paths) != 2:
        raise AssertionError(f"Expected 2 video paths, got {len(paths)}.")
    expected_shape = (int(args.max_steps), int(args.video_render_size), int(args.video_render_size), 3)
    for path in paths:
        if path["observations"].shape != expected_shape:
            raise AssertionError(f"Unexpected video observation shape {path['observations'].shape}.")
        env_infos = path.get("env_infos", {})
        if "privileged/block_0_pos" not in env_infos or "button_states" not in env_infos:
            raise AssertionError("Privileged OGBench Scene env_infos did not survive video rollout.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
