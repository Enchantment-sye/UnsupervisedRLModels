import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from core.task_adapter import SkillDiscoveryTaskAdapter


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        self.warnings.append(args[0] if args else "")


class _Policy:
    pass


def _cfg(**overrides):
    cfg = SimpleNamespace(
        task="dmc_walker_walk",
        env_backend="url",
        stage="pre_training",
        algo=SimpleNamespace(algo="metra", dim_skill=2),
        dim_skill=2,
        discrete=False,
        unit_length=True,
        use_hierarchical_skill=False,
        num_skill_levels=1,
        n_parallel=4,
        parallel_sampler_enabled=False,
        parallel_sampler_num_workers=2,
        parallel_sampler_fail_open=True,
        eval_parallel_sampler_enabled=False,
        eval_video_parallel_sampler_enabled=True,
        safety=SimpleNamespace(enabled=0),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _adapter(cfg):
    return SkillDiscoveryTaskAdapter(
        cfg,
        env=SimpleNamespace(spec=object()),
        agent=SimpleNamespace(sac_trainer=SimpleNamespace(skill_policy=_Policy())),
        work_dir="/tmp",
        logger=_Logger(),
    )


class _Collector:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.collect_calls = 0
        self.collect_fixed_calls = 0
        self.fixed_state_record_pixeled = None

    def collect(self, policy, *, target_num_trajectories, sample_extra_fn):
        self.collect_calls += 1
        if self.fail:
            raise RuntimeError("collector boom")
        for _ in range(target_num_trajectories):
            extra = sample_extra_fn()
            assert "skill" in extra
        return [_path() for _ in range(target_num_trajectories)]

    def collect_fixed(
            self,
            policy,
            *,
            extras,
            deterministic_policy,
            state_record_pixeled=False,
            video_frame_source=None,
            reset_perturbations=None):
        self.collect_fixed_calls += 1
        self.fixed_state_record_pixeled = state_record_pixeled
        if self.fail:
            raise RuntimeError("collector boom")
        assert deterministic_policy in (True, False)
        return [_path() for _ in extras]

    def consume_timing_metrics(self):
        return {"TimeParallelSampler": 0.25, "ParallelSamplerNumWorkers": 2.0}


def _path():
    return {
        "observations": np.zeros((2, 2), dtype=np.float32),
        "next_observations": np.zeros((2, 2), dtype=np.float32),
        "actions": np.zeros((2, 1), dtype=np.float32),
        "rewards": np.zeros(2, dtype=np.float32),
        "dones": np.asarray([False, True]),
        "agent_infos": {"skill": np.zeros((2, 2), dtype=np.float32)},
        "env_infos": {},
    }


def test_disabled_eval_parallel_sampler_uses_serial(monkeypatch):
    adapter = _adapter(_cfg(eval_parallel_sampler_enabled=False))
    monkeypatch.setattr(adapter, "_get_generic_parallel_collector", lambda: (_ for _ in ()).throw(AssertionError("should not call collector")))
    monkeypatch.setattr(adapter, "_collect_policy_trajectories_serial", lambda *args, **kwargs: ["serial"])

    result = adapter.collect_policy_trajectories(
        [{"skill": np.asarray([1.0, 0.0], dtype=np.float32)}],
        deterministic_policy=True,
        rollout_seed=1,
    )

    assert result == ["serial"]


def test_enabled_train_parallel_sampler_calls_generic_collector(monkeypatch):
    collector = _Collector()
    adapter = _adapter(_cfg(parallel_sampler_enabled=True))
    monkeypatch.setattr(adapter, "_get_generic_parallel_collector", lambda: collector)

    paths = adapter.get_train_trajectories(3)

    assert len(paths) == 3
    assert collector.collect_calls == 1
    metrics = adapter.consume_train_sampling_metrics()
    assert metrics["TimeParallelSampler"] == 0.25
    assert metrics["ParallelSamplerNumWorkers"] == 2.0


def test_kitchen_disables_generic_process_parallel_sampler():
    adapter = _adapter(
        _cfg(
            task="d4rl_kitchen",
            parallel_sampler_enabled=True,
            eval_parallel_sampler_enabled=True,
        )
    )

    assert adapter._should_use_generic_parallel_sampler(
        for_eval=False,
        state_record_pixeled=False,
    ) is False
    assert adapter._should_use_generic_parallel_sampler(
        for_eval=True,
        state_record_pixeled=False,
    ) is False
    assert adapter._should_use_kitchen_parallel_sampler(
        for_eval=False,
        state_record_pixeled=False,
    ) is True
    assert adapter._should_use_kitchen_parallel_sampler(
        for_eval=True,
        state_record_pixeled=False,
    ) is True


def test_kitchen_train_parallel_sampler_calls_kitchen_collector(monkeypatch):
    collector = _Collector()
    adapter = _adapter(_cfg(task="d4rl_kitchen", parallel_sampler_enabled=True))
    monkeypatch.setattr(adapter, "_get_kitchen_parallel_collector", lambda: collector)
    monkeypatch.setattr(
        adapter,
        "_get_generic_parallel_collector",
        lambda: (_ for _ in ()).throw(AssertionError("should not call generic collector")),
    )

    paths = adapter.get_train_trajectories(3)

    assert len(paths) == 3
    assert collector.collect_calls == 1
    metrics = adapter.consume_train_sampling_metrics()
    assert metrics["ParallelSamplerNumWorkers"] == 2.0


def test_kitchen_eval_parallel_sampler_calls_kitchen_collector(monkeypatch):
    collector = _Collector()
    adapter = _adapter(_cfg(task="kitchen", eval_parallel_sampler_enabled=True))
    monkeypatch.setattr(adapter, "_get_kitchen_parallel_collector", lambda: collector)
    monkeypatch.setattr(
        adapter,
        "_get_generic_parallel_collector",
        lambda: (_ for _ in ()).throw(AssertionError("should not call generic collector")),
    )

    paths = adapter.collect_policy_trajectories(
        [{"skill": np.asarray([1.0, 0.0], dtype=np.float32)}],
        deterministic_policy=False,
        rollout_seed=1,
        state_record_pixeled=True,
        video_frame_source="render",
    )

    assert len(paths) == 1
    assert collector.collect_fixed_calls == 1
    assert collector.fixed_state_record_pixeled is True


def test_kitchen_parallel_startup_failure_raises_without_serial_fallback(monkeypatch):
    adapter = _adapter(_cfg(task="metra_kitchen", parallel_sampler_enabled=True))
    monkeypatch.setattr(
        adapter,
        "_get_kitchen_parallel_collector",
        lambda: (_ for _ in ()).throw(RuntimeError("startup boom")),
    )
    monkeypatch.setattr(
        adapter,
        "_collect_policy_trajectories_serial",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not fall back")),
    )

    try:
        adapter.get_train_trajectories(1)
    except RuntimeError as exc:
        assert "startup boom" in str(exc)
    else:
        raise AssertionError("Kitchen parallel startup failure should raise")


def test_kitchen_parallel_failure_raises_without_serial_fallback(monkeypatch):
    adapter = _adapter(_cfg(task="d4rl_kitchen", parallel_sampler_enabled=True))
    monkeypatch.setattr(adapter, "_get_kitchen_parallel_collector", lambda: _Collector(fail=True))
    monkeypatch.setattr(
        adapter,
        "_collect_policy_trajectories_serial",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not fall back")),
    )

    try:
        adapter.get_train_trajectories(1)
    except RuntimeError as exc:
        assert "collector boom" in str(exc)
    else:
        raise AssertionError("Kitchen parallel failure should raise")


def test_eval_parallel_sampler_fail_open_falls_back_to_serial(monkeypatch):
    adapter = _adapter(_cfg(eval_parallel_sampler_enabled=True, parallel_sampler_fail_open=True))
    monkeypatch.setattr(adapter, "_get_generic_parallel_collector", lambda: _Collector(fail=True))
    monkeypatch.setattr(adapter, "_collect_policy_trajectories_serial", lambda *args, **kwargs: ["serial"])

    result = adapter.collect_policy_trajectories(
        [{"skill": np.asarray([1.0, 0.0], dtype=np.float32)}],
        deterministic_policy=True,
        rollout_seed=1,
    )

    assert result == ["serial"]
    assert adapter.logger.warnings


def test_eval_video_parallel_sampler_uses_fixed_collector(monkeypatch):
    collector = _Collector()
    adapter = _adapter(
        _cfg(
            eval_parallel_sampler_enabled=True,
            eval_video_parallel_sampler_enabled=True,
        )
    )
    monkeypatch.setattr(adapter, "_get_generic_parallel_collector", lambda: collector)

    paths = adapter.collect_policy_trajectories(
        [{"skill": np.asarray([1.0, 0.0], dtype=np.float32)}],
        deterministic_policy=False,
        rollout_seed=1,
        state_record_pixeled=True,
        video_frame_source="render",
    )

    assert len(paths) == 1
    assert collector.collect_fixed_calls == 1
    assert collector.fixed_state_record_pixeled is True
