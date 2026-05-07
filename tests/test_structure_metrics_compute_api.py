import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from core.metrics.trajectory_structure import compute_training_eval_structure_metrics


def _traj(skill_id, offset):
    coords = np.asarray(
        [
            [offset, 0.0],
            [offset + 0.1, 0.2],
            [offset + 0.2, 0.4],
        ],
        dtype=np.float32,
    )
    return {
        "observations": coords.copy(),
        "env_infos": {
            "coordinates": coords,
            "next_coordinates": coords + np.asarray([0.05, 0.05], dtype=np.float32),
        },
        "agent_infos": {
            "skill": np.eye(2, dtype=np.float32)[skill_id].repeat(coords.shape[0], axis=0),
        },
        "rewards": np.ones(coords.shape[0], dtype=np.float32),
    }


def _cfg(backends="temporal,ikse"):
    return SimpleNamespace(
        eval_structure_metrics_backends=backends,
        eval_structure_metrics_states_per_traj=3,
        eval_structure_metrics_max_points=100,
        eval_structure_metrics_anchor_seed=123,
        eval_structure_metrics_rollouts_per_skill=3,
        eval_structure_metrics_policy_mode="stochastic",
        temporal_graph_knn_k=2,
        soft_dtw_gamma=1.0,
    )


def test_compute_api_returns_temporal_and_ikse_keys_without_writer_or_logger():
    trajectories = [_traj(0, idx * 0.01) for idx in range(3)]
    trajectories += [_traj(1, 2.0 + idx * 0.01) for idx in range(3)]
    skill_ids = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

    metrics = compute_training_eval_structure_metrics(
        trajectories,
        skill_ids=skill_ids,
        cfg=_cfg(),
        env_name="ant",
        backends=["temporal", "ikse"],
    )

    assert metrics["StructureMetricsBackendMask"] == 3.0
    assert metrics["StructureMetricsNumBackends"] == 2.0
    assert metrics["StructureMetricsSkipped"] == 0.0
    assert "Entropy_TemporalParticle_XY" in metrics
    assert "DBI_TemporalMedoid_XY" in metrics
    assert "Score_TemporalParticle_DBI" in metrics
    assert "Entropy_IKDE_XY" in metrics
    assert "DBI_IKMeanRatio_Legacy_XY" in metrics
    assert "IKSE_LegacyDBI" in metrics


def test_single_backend_masks_are_stable():
    trajectories = [_traj(0, idx * 0.01) for idx in range(3)]
    trajectories += [_traj(1, 2.0 + idx * 0.01) for idx in range(3)]
    skill_ids = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

    metrics = compute_training_eval_structure_metrics(
        trajectories,
        skill_ids=skill_ids,
        cfg=_cfg("temporal"),
        env_name="ant",
        backends="temporal",
    )

    assert metrics["StructureMetricsBackendMask"] == 1.0
    assert metrics["StructureMetricsNumBackends"] == 1.0
    assert metrics["MetricBackend_UsesTemporalDistance"] == 1.0
    assert metrics["MetricBackend_UsesIsolationKernel"] == 0.0
