#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("MUJOCO_GL", "egl")
MUJOCO_PATH = Path("/home/shangyy/.mujoco/mujoco210")
if "MUJOCO_PY_MUJOCO_PATH" not in os.environ and MUJOCO_PATH.is_dir():
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = str(MUJOCO_PATH)

ld_paths = [str(MUJOCO_PATH / "bin"), "/usr/lib/nvidia"]
current_ld_paths = [
    path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path
]
for path in ld_paths:
    if Path(path).is_dir() and path not in current_ld_paths:
        current_ld_paths.append(path)
if current_ld_paths:
    os.environ["LD_LIBRARY_PATH"] = ":".join(current_ld_paths)


def _make_config():
    return SimpleNamespace(
        task="d4rl_kitchen",
        env_backend="url",
        action_repeat=2,
        render_size=64,
        time_limit=50,
        framestack=1,
    )


def main():
    from envs import make_env

    env = make_env(mode="eval", config=_make_config())
    try:
        env.reset()
        env_infos = {}
        required_task_keys = [
            f"metric_success_task_relevant/goal_{goal_idx}"
            for goal_idx in range(6)
        ]
        required_all_object_keys = [
            f"metric_success_all_objects/goal_{goal_idx}"
            for goal_idx in range(6)
        ]
        required_info_keys = required_task_keys + required_all_object_keys

        for _ in range(3):
            action = env.act_space["action"].sample()
            timestep = env.step({"action": action})
            info = timestep.get("info", {})
            for key in required_info_keys:
                assert key in info, f"Missing {key} in step info"
                env_infos.setdefault(key, []).append(info[key])

        trajectory = {
            "env_infos": {
                key: np.asarray(values)
                for key, values in env_infos.items()
            }
        }
        metrics = env.calc_eval_metrics([trajectory], is_option_trajectories=True)
        required_metric_keys = [
            "KitchenTaskBottomBurner",
            "KitchenTaskLightSwitch",
            "KitchenTaskSlideCabinet",
            "KitchenTaskHingeCabinet",
            "KitchenTaskMicrowave",
            "KitchenTaskKettle",
            "KitchenOverall",
        ]
        for key in required_metric_keys:
            assert key in metrics, f"Missing {key} in calc_eval_metrics output"
        assert 0 <= metrics["KitchenOverall"] <= 6, metrics
        print("METRA official Kitchen 6-task eval sanity check passed.")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
