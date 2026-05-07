from __future__ import annotations

import warnings

import numpy as np


KITCHEN_TASKS = (
    ("BottomBurner", ("metric_success_task_relevant/goal_0", "metric_success/goal_0", "bottom left burner success", "bottom_burner success")),
    ("LightSwitch", ("metric_success_task_relevant/goal_1", "metric_success/goal_1", "light switch success", "light_switch success")),
    ("SlideCabinet", ("metric_success_task_relevant/goal_2", "metric_success/goal_2", "slide cabinet success", "slide_cabinet success")),
    ("HingeCabinet", ("metric_success_task_relevant/goal_3", "metric_success/goal_3", "hinge cabinet success", "hinge_cabinet success")),
    ("Microwave", ("metric_success_task_relevant/goal_4", "metric_success/goal_4", "microwave success")),
    ("Kettle", ("metric_success_task_relevant/goal_5", "metric_success/goal_5", "kettle success")),
)


def _trajectory_env_infos(traj):
    if isinstance(traj, dict):
        return traj.get("env_infos", {})
    return getattr(traj, "env_infos", {})


def _success_from_values(values):
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    return bool(np.any(arr.astype(float) > 0.5))


def _extract_kitchen_task_success_matrix(trajectories, *, warn_missing):
    successes = np.zeros((len(trajectories), len(KITCHEN_TASKS)), dtype=bool)
    missing = []

    for traj_idx, traj in enumerate(trajectories):
        env_infos = _trajectory_env_infos(traj)
        for task_idx, (task_name, candidate_keys) in enumerate(KITCHEN_TASKS):
            found = False
            for key in candidate_keys:
                if key not in env_infos:
                    continue
                success = _success_from_values(env_infos[key])
                if success is None:
                    continue
                successes[traj_idx, task_idx] = success
                found = True
                break
            if not found:
                missing.append(f"traj_{traj_idx}:{task_name}")

    if missing and warn_missing:
        warnings.warn(
            "Missing Kitchen task success keys for "
            f"{len(missing)} trajectory/task entries: {', '.join(missing[:12])}",
            RuntimeWarning,
            stacklevel=2,
        )

    return successes, missing


def extract_kitchen_task_success_matrix(trajectories) -> np.ndarray:
    successes, _ = _extract_kitchen_task_success_matrix(trajectories, warn_missing=True)
    return successes


def calc_kitchen_eval_metrics(trajectories) -> dict:
    success_matrix, missing = _extract_kitchen_task_success_matrix(trajectories, warn_missing=True)
    num_traj = len(trajectories)
    completed_per_task = success_matrix.any(axis=0) if num_traj else np.zeros(len(KITCHEN_TASKS), dtype=bool)
    completed_per_traj = success_matrix.sum(axis=1) if num_traj else np.array([], dtype=int)

    metrics = {}
    for task_idx, (task_name, _) in enumerate(KITCHEN_TASKS):
        metrics[f"KitchenTask{task_name}"] = int(completed_per_task[task_idx])

    coverage = int(completed_per_task.sum())
    metrics["KitchenOverall"] = coverage
    metrics["KitchenPolicyTaskCoverage"] = coverage

    for task_idx, (task_name, _) in enumerate(KITCHEN_TASKS):
        rate = float(success_matrix[:, task_idx].mean()) if num_traj else 0.0
        metrics[f"Kitchen{task_name}SuccessRate"] = rate

    metrics["KitchenAvgCompletedTasksPerTraj"] = float(completed_per_traj.mean()) if num_traj else 0.0
    metrics["KitchenBestCompletedTasksPerTraj"] = int(completed_per_traj.max()) if num_traj else 0
    metrics["KitchenMissingSuccessKeys"] = int(len(missing))
    return metrics
