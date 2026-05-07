import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from core.metrics.trajectory_structure import (
    SkipReason,
    compute_training_eval_structure_metrics,
    video_trajectories_are_suitable,
)


def _cfg():
    return SimpleNamespace(
        eval_structure_metrics_states_per_traj=3,
        eval_structure_metrics_max_points=100,
        eval_structure_metrics_anchor_seed=1,
        eval_structure_metrics_rollouts_per_skill=3,
        eval_structure_metrics_policy_mode="stochastic",
        temporal_graph_knn_k=2,
        soft_dtw_gamma=1.0,
    )


def _traj(skill_id=0, offset=0.0, *, nan=False, include_skill=True, include_state=True):
    coords = np.asarray([[offset, 0.0], [offset + 0.2, 0.1], [offset + 0.4, 0.3]], dtype=np.float32)
    if nan:
        coords[1, 0] = np.nan
    trajectory = {
        "observations": coords.copy() if include_state else np.zeros((3, 2), dtype=np.float32),
        "env_infos": {},
        "agent_infos": {},
    }
    if include_state:
        trajectory["env_infos"] = {
            "coordinates": coords,
            "next_coordinates": coords + 0.1,
        }
    if include_skill:
        trajectory["agent_infos"]["skill"] = np.eye(2, dtype=np.float32)[skill_id].repeat(3, axis=0)
    return trajectory


def _temporal_metrics(trajectories, skill_ids=None):
    return compute_training_eval_structure_metrics(
        trajectories,
        skill_ids=skill_ids,
        cfg=_cfg(),
        env_name="ant",
        backends="temporal",
    )


def test_too_few_clusters_skip_code():
    metrics = _temporal_metrics([_traj(0, idx * 0.1) for idx in range(3)], np.asarray([0, 0, 0]))

    assert metrics["StructureMetricsTemporalSkipped"] == 1.0
    assert metrics["StructureMetricsTemporalSkipReasonCode"] == float(SkipReason.TOO_FEW_CLUSTERS)


def test_too_few_trajs_per_cluster_skip_code():
    trajectories = [_traj(0, 0.0), _traj(0, 0.1), _traj(1, 2.0), _traj(1, 2.1)]
    metrics = _temporal_metrics(trajectories, np.asarray([0, 0, 1, 1]))

    assert metrics["StructureMetricsTemporalSkipped"] == 1.0
    assert metrics["StructureMetricsTemporalSkipReasonCode"] == float(SkipReason.TOO_FEW_TRAJS_PER_CLUSTER)


def test_missing_skill_labels_skip_code():
    trajectories = [_traj(include_skill=False, offset=idx * 0.1) for idx in range(6)]
    metrics = _temporal_metrics(trajectories, None)

    assert metrics["StructureMetricsTemporalSkipped"] == 1.0
    assert metrics["StructureMetricsTemporalSkipReasonCode"] == float(SkipReason.MISSING_SKILL_LABELS)


def test_missing_coordinates_skip_code():
    metrics = _temporal_metrics([_traj(include_state=False) for _ in range(6)], np.asarray([0, 0, 0, 1, 1, 1]))

    assert metrics["StructureMetricsTemporalSkipped"] == 1.0
    assert metrics["StructureMetricsTemporalSkipReasonCode"] == float(SkipReason.MISSING_STATE_POINTS_OR_COORDINATES)


def test_nonfinite_distance_matrix_skip_code():
    trajectories = [_traj(0, idx * 0.1, nan=(idx == 0)) for idx in range(3)]
    trajectories += [_traj(1, 2.0 + idx * 0.1) for idx in range(3)]
    metrics = _temporal_metrics(trajectories, np.asarray([0, 0, 0, 1, 1, 1]))

    assert metrics["StructureMetricsTemporalSkipped"] == 1.0
    assert metrics["StructureMetricsTemporalSkipReasonCode"] == float(SkipReason.NONFINITE_DISTANCE_MATRIX)


def test_video_trajectories_not_suitable_skip_code():
    trajectories = [_traj(0, 0.0), _traj(0, 0.1)]

    suitable, reason, labels = video_trajectories_are_suitable(
        trajectories,
        min_trajs_per_cluster=3,
        policy_mode="deterministic",
        video_policy_mode="stochastic",
    )

    assert suitable is False
    assert reason == int(SkipReason.VIDEO_TRAJECTORIES_NOT_SUITABLE)
    assert labels is None
