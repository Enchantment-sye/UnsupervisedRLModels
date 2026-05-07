#!/usr/bin/env python3

import json
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, "src")
TRAIN_DIR = os.path.join(ROOT_DIR, "scripts", "train")
for path in (ROOT_DIR, SRC_DIR, TRAIN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from envs import make_env
from workspace_common import WorkspaceContext


def build_env_config(task_name):
    camera = "head" if task_name.startswith("bigym_") else "corner"
    return SimpleNamespace(
        task=task_name,
        env_backend="isaaclab" if task_name.startswith("isaaclab_") else "url",
        isaaclab_task="",
        isaaclab_num_envs=1,
        isaaclab_headless=1,
        isaaclab_enable_cameras=1,
        isaaclab_render_mode="rgb_array",
        isaaclab_image_source="auto",
        isaaclab_camera_key=None,
        encoder=0,
        render_size=32,
        flatten_obs=1,
        seed=0,
        device="cuda:0",
        time_limit=200,
        framestack=1,
        frame_stack=1,
        action_repeat=1,
        camera=camera,
        dmc_camera=-1,
    )


def summarise_obs(obs):
    summary = {}
    for key, value in obs.items():
        if key == "info":
            summary["info_keys"] = sorted(value.keys())
            continue
        array = np.asarray(value)
        summary[f"{key}.shape"] = list(array.shape)
    return summary


def run_env_smoke(task_name):
    cfg = build_env_config(task_name)
    env = make_env(mode="train", config=cfg)
    try:
        reset_obs = env.reset()
        action = np.zeros(env.act_space["action"].shape, dtype=np.float32)
        step_obs = env.step({"action": action})
        return {
            "status": "ok",
            "reset": summarise_obs(reset_obs),
            "step": summarise_obs(step_obs),
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def validate_workspace_default():
    cfg = SimpleNamespace(
        use_gpu=False,
        seed=0,
        env=SimpleNamespace(task="isaaclab_reach_franka"),
        algo=SimpleNamespace(algo="metra_cascade", idk_subsample_size=256),
        net=SimpleNamespace(finetune_encoder=False),
        log=SimpleNamespace(workspace_root="/share/shangyy", stage="pre_training"),
    )
    ctx = WorkspaceContext.create(cfg)
    try:
        work_dir = ctx.work_dir
        if not work_dir.startswith("/share/shangyy/isaaclab_reach_franka/"):
            raise RuntimeError(f"workspace default mismatch: {work_dir}")
        return work_dir
    finally:
        shutil.rmtree(ctx.work_dir, ignore_errors=True)


def main():
    tasks = [
        "dmc_walker_walk",
        "metaworld_reach",
        "d4rl_kitchen",
        "bigym_reach_target",
        "debug_dummy",
    ]
    results = {}
    for task_name in tasks:
        results[task_name] = run_env_smoke(task_name)
    workspace_default = validate_workspace_default()
    payload = {
        "status": "ok",
        "workspace_default": workspace_default,
        "tasks": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
