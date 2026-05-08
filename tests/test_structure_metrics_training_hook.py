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
        eval_structure_metrics_reset_perturb_scale=0.0,
        eval_structure_metrics_degenerate_policy="skip",
        eval_record_video=0,
        num_video_repeats=2,
        render_size=4,
        video_skip_frames=1,
        motion_analysis=SimpleNamespace(enabled=False),
        encoder=0,
        sac_discount=0.99,
        stage="pre_training",
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


def test_video_eval_uses_deterministic_policy(monkeypatch):
    adapter = _adapter(False)
    adapter.cfg.eval_record_video = 1
    adapter.cfg.dim_skill = 2
    calls = []

    coverage_trajectory = {
        "observations": np.zeros((1, 2), dtype=np.float32),
        "next_observations": np.zeros((1, 2), dtype=np.float32),
        "actions": np.zeros((1, 1), dtype=np.float32),
        "rewards": np.zeros(1, dtype=np.float32),
        "dones": np.ones(1, dtype=bool),
        "agent_infos": {},
        "env_infos": {},
    }
    monkeypatch.setattr(adapter, "collect_policy_coverage_trajectories", lambda total_epoch: [coverage_trajectory])
    monkeypatch.setattr(adapter, "_log_d4rl_kitchen_eval_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "compute_policy_coverage_metrics", lambda trajectories: {})
    monkeypatch.setattr(adapter, "log_policy_coverage_metrics_to_writer", lambda *args, **kwargs: None)
    monkeypatch.setattr(task_adapter_module.utils, "log_performance_ex", lambda *args, **kwargs: {"scalars": {}})
    monkeypatch.setattr(task_adapter_module.utils, "record_video", lambda *args, **kwargs: None)

    def _collect(extras, **kwargs):
        calls.append(kwargs)
        return [
            {
                "observations": np.zeros((1, 2), dtype=np.float32),
                "next_observations": np.zeros((1, 2), dtype=np.float32),
                "actions": np.zeros((1, 1), dtype=np.float32),
                "rewards": np.zeros(1, dtype=np.float32),
                "dones": np.ones(1, dtype=bool),
                "agent_infos": {},
                "env_infos": {},
            }
            for _ in extras
        ]

    monkeypatch.setattr(adapter, "collect_policy_trajectories", _collect)

    adapter._evaluate_impl(1, 7, _Writer(), log_policy_coverage_to_writer=True)

    assert calls
    assert calls[-1]["deterministic_policy"] is True
    assert calls[-1]["state_record_pixeled"] is True


def test_structure_metrics_perturbation_forces_extra_deterministic_rollouts(monkeypatch):
    adapter = _adapter(True)
    adapter.cfg.task = "d4rl_kitchen"
    adapter.cfg.eval_structure_metrics_reset_perturb_scale = 1e-4
    calls = []

    def _collect(extras, **kwargs):
        calls.append((list(extras), kwargs))
        return [
            {
                "observations": np.zeros((3, 2), dtype=np.float32),
                "next_observations": np.zeros((3, 2), dtype=np.float32),
                "actions": np.zeros((3, 1), dtype=np.float32),
                "rewards": np.zeros(3, dtype=np.float32),
                "dones": np.zeros(3, dtype=bool),
                "agent_infos": {"skill": np.repeat(extra["skill"][None, :], 3, axis=0)},
                "env_infos": {},
            }
            for extra in extras
        ]

    monkeypatch.setattr(adapter, "collect_policy_trajectories", _collect)
    video_trajectories = [{"agent_infos": {"skill": np.eye(2, dtype=np.float32)[0][None, :]}} for _ in range(6)]

    trajectories, options, skill_ids, source_metrics = adapter._collect_structure_metric_trajectories(
        3,
        video_trajectories=video_trajectories,
        video_policy_mode="deterministic",
    )

    assert len(trajectories) == 6
    assert options.shape[0] == 6
    assert skill_ids.tolist() == [0, 0, 0, 1, 1, 1]
    assert source_metrics["StructureMetricsUsedVideoTrajectories"] == 0.0
    assert source_metrics["StructureMetricsUsedExtraRollouts"] == 1.0
    assert source_metrics["StructureMetricsResetPerturbedRollouts"] == 1.0
    assert calls[0][1]["deterministic_policy"] is True
    assert len(calls[0][1]["reset_perturbations"]) == 6
    assert calls[0][1]["reset_perturbations"][0][1] == 1e-4
