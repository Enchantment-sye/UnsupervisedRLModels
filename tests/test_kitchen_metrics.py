import warnings

import numpy as np
import pytest

from envs.kitchen.metrics import (
    calc_kitchen_eval_metrics,
    extract_kitchen_task_success_matrix,
)


TASK_RELEVANT_KEYS = [
    f"metric_success_task_relevant/goal_{idx}"
    for idx in range(6)
]


def _traj(successes, *, key_prefix="metric_success_task_relevant"):
    return {
        "rewards": np.zeros(3),
        "env_infos": {
            f"{key_prefix}/goal_{idx}": np.asarray([0, 0, int(success)])
            for idx, success in enumerate(successes)
        },
    }


def test_all_six_tasks_completed_has_overall_six():
    metrics = calc_kitchen_eval_metrics([_traj([1, 1, 1, 1, 1, 1])])

    assert metrics["KitchenOverall"] == 6
    assert metrics["KitchenPolicyTaskCoverage"] == 6
    assert metrics["KitchenAvgCompletedTasksPerTraj"] == 6.0
    assert metrics["KitchenBestCompletedTasksPerTraj"] == 6
    assert metrics["KitchenMissingSuccessKeys"] == 0


def test_only_microwave_and_kettle_completed_has_overall_two():
    metrics = calc_kitchen_eval_metrics([_traj([0, 0, 0, 0, 1, 1])])

    assert metrics["KitchenOverall"] == 2
    assert metrics["KitchenPolicyTaskCoverage"] == 2
    assert metrics["KitchenTaskMicrowave"] == 1
    assert metrics["KitchenTaskKettle"] == 1
    assert metrics["KitchenTaskBottomBurner"] == 0


def test_success_rates_and_completed_task_counts_are_per_trajectory():
    trajectories = [
        _traj([1, 0, 0, 0, 1, 0]),
        _traj([0, 1, 0, 0, 1, 1]),
        _traj([0, 0, 0, 0, 0, 1]),
    ]

    metrics = calc_kitchen_eval_metrics(trajectories)

    assert metrics["KitchenOverall"] == 4
    assert metrics["KitchenBottomBurnerSuccessRate"] == pytest.approx(1 / 3)
    assert metrics["KitchenLightSwitchSuccessRate"] == pytest.approx(1 / 3)
    assert metrics["KitchenMicrowaveSuccessRate"] == pytest.approx(2 / 3)
    assert metrics["KitchenKettleSuccessRate"] == pytest.approx(2 / 3)
    assert metrics["KitchenAvgCompletedTasksPerTraj"] == pytest.approx(2.0)
    assert metrics["KitchenBestCompletedTasksPerTraj"] == 3


def test_metric_success_alias_is_accepted():
    matrix = extract_kitchen_task_success_matrix([
        _traj([0, 0, 1, 0, 0, 0], key_prefix="metric_success")
    ])

    assert matrix.tolist() == [[False, False, True, False, False, False]]


def test_current_task_specific_keys_are_accepted():
    trajectories = [
        {
            "env_infos": {
                "bottom left burner success": np.asarray([0, 1]),
                "light switch success": np.asarray([0, 0]),
                "slide cabinet success": np.asarray([0, 1]),
                "hinge cabinet success": np.asarray([0, 0]),
                "microwave success": np.asarray([0, 1]),
                "kettle success": np.asarray([0, 0]),
            },
        }
    ]

    metrics = calc_kitchen_eval_metrics(trajectories)

    assert metrics["KitchenOverall"] == 3
    assert metrics["KitchenBottomBurnerSuccessRate"] == 1.0
    assert metrics["KitchenSlideCabinetSuccessRate"] == 1.0
    assert metrics["KitchenMicrowaveSuccessRate"] == 1.0
    assert metrics["KitchenMissingSuccessKeys"] == 0


def test_missing_success_keys_warn_and_are_counted():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = calc_kitchen_eval_metrics([
            {"env_infos": {TASK_RELEVANT_KEYS[4]: np.asarray([1])}},
        ])

    assert any("Missing Kitchen task success keys" in str(item.message) for item in caught)
    assert metrics["KitchenMissingSuccessKeys"] == 5
    assert metrics["KitchenOverall"] == 1
