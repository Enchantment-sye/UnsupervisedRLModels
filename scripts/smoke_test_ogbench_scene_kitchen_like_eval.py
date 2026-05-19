#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from envs.ogbench_scene_kitchen_like_eval import OGBenchSceneKitchenLikeEvaluator  # noqa: E402


STATE_SCENE_ID = "scene-v0"
PIXEL_SCENE_ID = "visual-scene-v0"
SCENE_ALIASES = {
    "ogbench_scene",
    "ogbench_scene-v0",
    "ogbench_scene-play-v0",
    "ogbench_visual-scene-v0",
    "ogbench_visual-scene-play-v0",
    "scene",
    "scene-v0",
    "scene-play-v0",
    "visual-scene",
    "visual-scene-v0",
    "visual-scene-play-v0",
}


def main():
    parser = argparse.ArgumentParser(description="Smoke test OGBench Scene Kitchen-like eval metrics.")
    parser.add_argument("--encoder", type=int, choices=(0, 1), default=0)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--num-trajs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env_id = _resolve_env_id(args.env_id, args.encoder)

    import gymnasium as gym
    import ogbench  # noqa: F401

    env = gym.make(env_id)
    rng = np.random.RandomState(args.seed)
    results = []
    try:
        for traj_idx in range(args.num_trajs):
            obs, reset_info = _reset(env, args.seed + traj_idx)
            _ = obs
            evaluator = OGBenchSceneKitchenLikeEvaluator()
            evaluator.reset(reset_info)
            for _step in range(args.max_steps):
                action = _sample_action(env.action_space, rng)
                obs, reward, terminated, truncated, info = env.step(action)
                _ = obs, reward
                evaluator.update(info)
                if terminated or truncated:
                    break
            results.append(evaluator.result())
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    metrics = OGBenchSceneKitchenLikeEvaluator.aggregate(results)
    _validate_metrics(metrics)
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")


def _resolve_env_id(env_id, encoder):
    if env_id is None or env_id in SCENE_ALIASES:
        return PIXEL_SCENE_ID if bool(int(encoder)) else STATE_SCENE_ID
    return env_id


def _reset(env, seed):
    try:
        return env.reset(seed=int(seed))
    except TypeError:
        return env.reset()


def _sample_action(space, rng):
    sample = getattr(space, "sample", None)
    if callable(sample):
        return sample()
    low = np.asarray(space.low, dtype=np.float32)
    high = np.asarray(space.high, dtype=np.float32)
    return rng.uniform(low, high).astype(np.float32)


def _validate_metrics(metrics):
    for key, value in metrics.items():
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            continue
        raise TypeError(f"Metric {key!r} has non-loggable type {type(value)!r}.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
