import warnings

import numpy as np

from iod.coverage_tracker import CoverageTracker


def _locomotion_path(x):
    return {
        "env_infos": {
            "coordinates": np.asarray([[x, 0.0]], dtype=np.float32),
            "next_coordinates": np.asarray([[x, 0.0]], dtype=np.float32),
        }
    }


def _kitchen_path(successes):
    return {
        "env_infos": {
            f"metric_success_task_relevant/goal_{idx}": np.asarray([float(success)])
            for idx, success in enumerate(successes)
        }
    }


def test_queue_keeps_only_recent_trajectories():
    tracker = CoverageTracker("ant", queue_size=3)

    tracker.update_train_paths([_locomotion_path(x) for x in range(5)])

    assert tracker.compute_queue_metrics()["QueueStateCoverageXYBins"] == 3
    assert list(tracker.queue) == [
        {(2, 0)},
        {(3, 0)},
        {(4, 0)},
    ]


def test_default_queue_size_is_100000_train_trajectories():
    tracker = CoverageTracker("ant")

    assert tracker.queue_size == 100000


def test_total_coverage_does_not_decrease_when_queue_evicts():
    tracker = CoverageTracker("ant", queue_size=3)

    tracker.update_train_paths([_locomotion_path(x) for x in range(5)])

    assert tracker.compute_queue_metrics()["QueueStateCoverageXYBins"] == 3
    assert tracker.compute_total_metrics()["TotalStateCoverageXYBins"] == 5


def test_kitchen_mask_union_is_correct():
    tracker = CoverageTracker("kitchen", queue_size=3)

    tracker.update_train_paths([
        _kitchen_path([1, 0, 0, 0, 0, 0]),
        _kitchen_path([0, 0, 0, 0, 1, 0]),
        _kitchen_path([0, 1, 0, 0, 0, 1]),
    ])

    assert tracker.compute_queue_metrics()["KitchenQueueTaskCoverage"] == 4
    assert tracker.compute_total_metrics()["KitchenTotalTaskCoverage"] == 4
    assert tracker.compute_policy_metrics([
        _kitchen_path([0, 0, 1, 0, 0, 0]),
        _kitchen_path([0, 0, 0, 1, 0, 0]),
    ])["KitchenPolicyTaskCoverage"] == 2


def test_state_dict_load_state_dict_preserves_metrics():
    tracker = CoverageTracker("ant", queue_size=3)
    tracker.update_train_paths([_locomotion_path(x) for x in range(5)])

    restored = CoverageTracker("ant", queue_size=1)
    restored.load_state_dict(tracker.state_dict())

    assert restored.compute_queue_metrics() == tracker.compute_queue_metrics()
    assert restored.compute_total_metrics() == tracker.compute_total_metrics()


def test_missing_coordinates_do_not_crash_and_report_missing_info_once():
    tracker = CoverageTracker("ant", queue_size=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = tracker.compute_policy_metrics([{"env_infos": {}}])
        repeat_metrics = tracker.compute_policy_metrics([{"env_infos": {}}])

    assert metrics["PolicyStateCoverageXYBins"] == 0
    assert metrics["MissingCoverageInfo"] == 1
    assert repeat_metrics["MissingCoverageInfo"] == 1
    assert len(caught) == 1


def test_halfcheetah_coverage_uses_x_bins_only():
    tracker = CoverageTracker("halfcheetah", queue_size=3)

    path = {
        "env_infos": {
            "coordinates": np.asarray([[0.2, 0.2], [0.2, 7.7]], dtype=np.float32),
            "next_coordinates": np.asarray([[0.2, 7.7], [1.2, 9.2]], dtype=np.float32),
        }
    }
    metrics = tracker.compute_policy_metrics([path])

    assert metrics["PolicyStateCoverageXYBins"] == 2


def test_missing_train_coordinates_mark_queue_and_total_metrics():
    tracker = CoverageTracker("ant", queue_size=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracker.update_train_paths([{"env_infos": {}}])

    queue_metrics = tracker.compute_queue_metrics()
    total_metrics = tracker.compute_total_metrics()

    assert queue_metrics["QueueStateCoverageXYBins"] == 0
    assert queue_metrics["MissingCoverageInfo"] == 1
    assert total_metrics["TotalStateCoverageXYBins"] == 0
    assert total_metrics["MissingCoverageInfo"] == 1
    assert len(caught) == 1


def test_ogbench_scene_coverage_tracker_is_noop():
    tracker = CoverageTracker("ogbench_scene", queue_size=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracker.update_train_paths([{"env_infos": {}}])
        policy_metrics = tracker.compute_policy_metrics([{"env_infos": {}}])

    assert policy_metrics == {}
    assert tracker.compute_queue_metrics() == {}
    assert tracker.compute_total_metrics() == {}
    assert list(tracker.queue) == []
    assert len(caught) == 0
