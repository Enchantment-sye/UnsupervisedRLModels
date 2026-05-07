import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from core import task_adapter as task_adapter_module
from core.task_adapter import SkillDiscoveryTaskAdapter


class _Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = float(value)


def _cfg(enabled):
    return SimpleNamespace(
        eval_structure_metrics=enabled,
        eval_structure_metrics_backends="temporal,ikse",
        eval_structure_metrics_interval=1,
        eval_structure_metrics_fail_open=True,
        eval_structure_metrics_write_legacy_tags=False,
        eval_structure_metrics_policy_mode="deterministic",
        eval_structure_metrics_use_video_trajectories=False,
        eval_structure_metrics_rollouts_per_skill=3,
        eval_structure_metrics_num_skills=2,
        eval_structure_metrics_max_trajs=6,
        eval_structure_metrics_anchor_seed=0,
        eval_structure_metrics_max_points=60,
        eval_structure_metrics_states_per_traj=3,
        discrete=True,
        dim_skill=2,
        unit_length=True,
        num_random_trajectories=2,
        use_hierarchical_skill=False,
        num_skill_levels=1,
        seed=1,
        task="ant",
    )


def _adapter(enabled):
    return SkillDiscoveryTaskAdapter(
        _cfg(enabled),
        env=SimpleNamespace(spec=object()),
        agent=None,
        work_dir="/tmp",
        logger=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
    )


def test_disabled_structure_metrics_does_not_call_compute(monkeypatch):
    adapter = _adapter(False)
    called = {"value": False}
    monkeypatch.setattr(
        task_adapter_module,
        "compute_training_eval_structure_metrics",
        lambda *args, **kwargs: called.__setitem__("value", True),
    )

    writer = _Writer()
    adapter._maybe_log_structure_metrics(1, 1, writer)

    assert called["value"] is False
    assert writer.scalars == {}


def test_enabled_structure_metrics_writes_eval_tags(monkeypatch):
    adapter = _adapter(True)
    monkeypatch.setattr(
        adapter,
        "_collect_structure_metric_trajectories",
        lambda *args, **kwargs: (
            [{"observations": np.zeros((3, 2), dtype=np.float32)}],
            None,
            np.asarray([0], dtype=np.int64),
            {"StructureMetricsUsedExtraRollouts": 1.0},
        ),
    )
    monkeypatch.setattr(
        task_adapter_module,
        "compute_training_eval_structure_metrics",
        lambda *args, **kwargs: {
            "Entropy_TemporalParticle_XY": 1.5,
            "DBI_TemporalMedoid_XY": 2.5,
            "StructureMetricsSkipped": 0.0,
        },
    )

    writer = _Writer()
    adapter._maybe_log_structure_metrics(4, 2, writer)

    assert writer.scalars["eval/Entropy_TemporalParticle_XY"] == 1.5
    assert writer.scalars["eval/DBI_TemporalMedoid_XY"] == 2.5
    assert writer.scalars["eval/StructureMetricsUsedExtraRollouts"] == 1.0


def test_compute_exception_fail_open_writes_skipped_tags(monkeypatch):
    adapter = _adapter(True)
    monkeypatch.setattr(
        adapter,
        "_collect_structure_metric_trajectories",
        lambda *args, **kwargs: ([{}], None, np.asarray([0]), {}),
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(task_adapter_module, "compute_training_eval_structure_metrics", _raise)

    writer = _Writer()
    adapter._maybe_log_structure_metrics(5, 3, writer)

    assert writer.scalars["eval/StructureMetricsSkipped"] == 1.0
    assert writer.scalars["eval/StructureMetricsSkipReasonCode"] == 11.0
    assert writer.scalars["eval/StructureMetricsTemporalSkipped"] == 1.0
    assert writer.scalars["eval/StructureMetricsIKSkipped"] == 1.0
