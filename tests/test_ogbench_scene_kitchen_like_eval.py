import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("src"))

from envs.ogbench_scene_kitchen_like_eval import (
    DRAWER_CUBE_TARGET,
    METRIC_PREFIX,
    OGBenchSceneKitchenLikeEvaluator,
    calc_ogbench_scene_kitchen_like_metrics,
    reset_info_key,
)


def _info(
        cube=(0.35, 0.05, 0.02),
        buttons=(0, 0),
        drawer=0.0,
        window=0.0,
        *,
        fallback_buttons=False):
    info = {
        "privileged/block_0_pos": np.asarray(cube, dtype=np.float64),
        "privileged/drawer_pos": np.asarray([drawer], dtype=np.float64),
        "privileged/window_pos": np.asarray(window, dtype=np.float64),
    }
    if fallback_buttons:
        info["privileged/button_0_state"] = np.asarray([buttons[0]], dtype=np.int64)
        info["privileged/button_1_state"] = np.asarray(buttons[1], dtype=np.int64)
    else:
        info["button_states"] = np.asarray(buttons, dtype=np.int64)
    return info


def _run(*infos):
    evaluator = OGBenchSceneKitchenLikeEvaluator()
    evaluator.reset(_info())
    for info in infos:
        evaluator.update(info)
    return evaluator.result()


def test_button_predicates_and_fallback_fields():
    evaluator = OGBenchSceneKitchenLikeEvaluator()
    evaluator.reset(_info(fallback_buttons=True))

    evaluator.update(_info(buttons=(1, 0), fallback_buttons=True))
    evaluator.update(_info(buttons=(1, 1), fallback_buttons=True))
    result = evaluator.result()

    assert result["predicates"]["button0_toggled"] is True
    assert result["predicates"]["button1_toggled"] is True
    assert result["events"]["button0_toggled"] == 0
    assert result["events"]["button1_toggled"] == 1


def test_drawer_opened_and_closed_requires_prior_open():
    result = _run(
        _info(drawer=0.02),
        _info(drawer=-0.13),
        _info(drawer=0.02),
    )

    assert result["predicates"]["drawer_opened"] is True
    assert result["predicates"]["drawer_closed"] is True
    assert result["events"]["drawer_opened"] == 1
    assert result["events"]["drawer_closed"] == 2

    no_open = _run(_info(drawer=0.02))
    assert no_open["predicates"]["drawer_closed"] is False


def test_window_opened_and_closed_requires_prior_open():
    result = _run(
        _info(window=0.02),
        _info(window=0.16),
        _info(window=0.02),
    )

    assert result["predicates"]["window_opened"] is True
    assert result["predicates"]["window_closed"] is True
    assert result["events"]["window_opened"] == 1
    assert result["events"]["window_closed"] == 2

    no_open = _run(_info(window=0.02))
    assert no_open["predicates"]["window_closed"] is False


def test_cube_predicates():
    result = _run(
        _info(cube=(0.45, 0.05, 0.02)),
        _info(cube=(0.35, 0.13, 0.07)),
        _info(cube=DRAWER_CUBE_TARGET),
    )

    assert result["predicates"]["cube_moved"] is True
    assert result["predicates"]["cube_lifted"] is True
    assert result["predicates"]["cube_to_table_region"] is True
    assert result["predicates"]["cube_in_drawer"] is True


def test_shape_robustness_for_arrays_scalars_and_lists():
    evaluator = OGBenchSceneKitchenLikeEvaluator()
    reset_info = {
        "privileged/block_0_pos": np.asarray([[0.35, 0.05, 0.02]], dtype=np.float64),
        "button_states": [[0, 0]],
        "privileged/drawer_pos": np.asarray([0.0], dtype=np.float64),
        "privileged/window_pos": 0.0,
    }
    evaluator.reset(reset_info)
    evaluator.update({
        "privileged/block_0_pos": [0.45, 0.05, 0.07],
        "button_states": np.asarray([[1, 0]], dtype=np.int64),
        "privileged/drawer_pos": -0.13,
        "privileged/window_pos": np.asarray([0.16], dtype=np.float64),
    })
    result = evaluator.result()

    assert result["predicates"]["button0_toggled"] is True
    assert result["predicates"]["drawer_opened"] is True
    assert result["predicates"]["window_opened"] is True
    assert result["predicates"]["cube_moved"] is True
    assert result["predicates"]["cube_lifted"] is True


def test_missing_scene_field_error_is_clear():
    evaluator = OGBenchSceneKitchenLikeEvaluator()
    evaluator.reset(_info())
    bad_info = _info()
    bad_info.pop("privileged/drawer_pos")

    with pytest.raises(KeyError, match="Missing OGBench Scene info field"):
        evaluator.update(bad_info)


def test_aggregate_predicate_and_object_coverage():
    traj1 = _run(
        _info(buttons=(1, 0), drawer=-0.13, cube=(0.45, 0.05, 0.02)),
        _info(buttons=(1, 0), drawer=0.02, cube=(0.45, 0.05, 0.02)),
    )
    traj2 = _run(
        _info(buttons=(0, 1), window=0.16, cube=(0.35, 0.13, 0.07)),
        _info(buttons=(0, 1), window=0.02, cube=(0.35, 0.13, 0.07)),
        _info(buttons=(0, 1), window=0.02, cube=DRAWER_CUBE_TARGET),
    )

    metrics = OGBenchSceneKitchenLikeEvaluator.aggregate([traj1, traj2])

    assert metrics[f"{METRIC_PREFIX}/AtomicCoverage"] == 10
    assert metrics[f"{METRIC_PREFIX}/AtomicCoverageRatio"] == 1.0
    assert metrics[f"{METRIC_PREFIX}/MeanAtomicCompletion"] == 5.5
    assert metrics[f"{METRIC_PREFIX}/MaxAtomicCompletion"] == 7
    assert metrics[f"{METRIC_PREFIX}/Predicate/button0_toggled"] == 0.5
    assert metrics[f"{METRIC_PREFIX}/PredicateUnion/cube_in_drawer"] == 1
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/CubeXYBinCoverage"] == 4
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/DrawerPositionCoverage"] == 2
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/WindowPositionCoverage"] == 2
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/ButtonStateCoverage"] == 3
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/ButtonStateCoverageRatio"] == 0.75
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/CubeInDrawerRate"] == 0.5
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/DrawerOpenRate"] == 0.5
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/WindowOpenRate"] == 0.5
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/MeanMaxCubeXYDisplacement"] > 0.0
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/MaxCubeXYDisplacement"] > 0.0
    assert np.isclose(metrics[f"{METRIC_PREFIX}/ObjectCoverage/MeanMaxCubeZ"], 0.045)
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/MeanDrawerMinPos"] < 0.0
    assert metrics[f"{METRIC_PREFIX}/ObjectCoverage/MeanWindowMaxPos"] > 0.0


def test_calc_metrics_skips_non_scene_trajectories():
    assert calc_ogbench_scene_kitchen_like_metrics([
        {"env_infos": {"coordinates": np.zeros((2, 2), dtype=np.float32)}}
    ]) == {}


def test_calc_metrics_reconstructs_reset_and_step_info_from_trajectory():
    reset_info = _info()
    step_info = _info(buttons=(1, 0), cube=(0.45, 0.05, 0.02))
    env_infos = {
        key: np.asarray([value])
        for key, value in step_info.items()
    }
    for key, value in reset_info.items():
        env_infos[reset_info_key(key)] = np.asarray([value])

    metrics = calc_ogbench_scene_kitchen_like_metrics([{"env_infos": env_infos}])

    assert metrics[f"{METRIC_PREFIX}/AtomicCoverage"] == 2
    assert metrics[f"{METRIC_PREFIX}/Predicate/button0_toggled"] == 1.0
    assert metrics[f"{METRIC_PREFIX}/Predicate/cube_moved"] == 1.0
