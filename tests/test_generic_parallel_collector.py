import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from envs.generic_parallel import (
    GenericProcessTrajectoryCollector,
    _GenericEnvConstructor,
    resolve_parallel_sampler_start_method,
)


class _FakeWorker:
    def __init__(self, slot, horizon=3):
        self.slot = slot
        self.horizon = horizon
        self.t = 0
        self.closed = False
        self.video_capture_modes = []

    def reset(self, blocking=False):
        def _result():
            self.t = 0
            return self._timestep(is_first=True)

        return _result() if blocking else _result

    def step(self, action, blocking=False):
        assert "action" in action

        def _result():
            self.t += 1
            return self._timestep(is_first=False)

        return _result() if blocking else _result

    def call(self, name, *args, **kwargs):
        if name == "set_video_capture_active":
            active = bool(args[0])

            def _set_result():
                self.video_capture_modes.append(active)

            return _set_result

        assert name == "capture_video_frame"

        def _result():
            return np.full((4, 4, 3), self.slot + 10, dtype=np.uint8)

        return _result

    def close(self):
        self.closed = True

    def _timestep(self, *, is_first):
        return {
            "image": np.full((4, 4, 3), self.slot, dtype=np.uint8),
            "state": np.asarray([self.slot, self.t], dtype=np.float32),
            "reward": float(self.t),
            "is_first": is_first,
            "is_last": self.t >= self.horizon,
            "is_terminal": self.t >= self.horizon,
            "info": {
                "coordinates": np.asarray([self.slot, self.t], dtype=np.float32),
            },
        }


class _FakePolicy:
    def __init__(self):
        self._force_use_mode_actions = False
        self.reset_count = 0
        self.obs_batches = []

    def reset(self):
        self.reset_count += 1

    def get_actions(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        self.obs_batches.append(obs.copy())
        actions = np.ones((obs.shape[0], 1), dtype=np.float32) * (2.0 if self._force_use_mode_actions else 1.0)
        return actions, {
            "log_prob": np.zeros((obs.shape[0], 1), dtype=np.float32),
            "pre_tanh_value": actions + 0.5,
        }


def _cfg():
    return SimpleNamespace(
        task="dmc_walker_walk",
        stage="pre_training",
        algo=SimpleNamespace(algo="metra", dim_skill=2),
        encoder=0,
        time_limit=3,
        parallel_sampler_start_method="auto",
    )


def _collector(num_workers=2):
    collector = GenericProcessTrajectoryCollector.__new__(GenericProcessTrajectoryCollector)
    collector.cfg = _cfg()
    collector._num_workers = num_workers
    collector._workers = [_FakeWorker(slot) for slot in range(num_workers)]
    collector._timing_totals = collector._new_timing_totals()
    return collector


def test_dmc_pixel_auto_parallel_sampler_start_method_prefers_forkserver(monkeypatch):
    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork", "spawn", "forkserver"])
    cfg = SimpleNamespace(
        task="dmc_quadruped_run_forward_color",
        encoder=1,
        parallel_sampler_start_method="auto",
    )

    assert resolve_parallel_sampler_start_method(cfg) == "forkserver"


def test_dmc_pixel_auto_parallel_sampler_start_method_falls_back_to_spawn(monkeypatch):
    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork", "spawn"])
    cfg = SimpleNamespace(
        task="dmc_quadruped_run_forward_color",
        encoder=1,
        parallel_sampler_start_method="auto",
    )

    assert resolve_parallel_sampler_start_method(cfg) == "spawn"


def test_ogbench_visual_auto_parallel_sampler_start_method_prefers_forkserver(monkeypatch):
    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork", "spawn", "forkserver"])
    cfg = SimpleNamespace(
        task="ogbench_scene",
        encoder=1,
        parallel_sampler_start_method="auto",
    )

    assert resolve_parallel_sampler_start_method(cfg) == "forkserver"


def test_ogbench_visual_auto_parallel_sampler_start_method_falls_back_to_spawn(monkeypatch):
    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork", "spawn"])
    cfg = SimpleNamespace(
        task="ogbench_scene",
        encoder=1,
        parallel_sampler_start_method="auto",
    )

    assert resolve_parallel_sampler_start_method(cfg) == "spawn"


def test_ogbench_worker_constructor_bootstraps_render_env(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    fake_env_module = SimpleNamespace(
        make_env=lambda **kwargs: (os.environ.get("MUJOCO_GL"), os.environ.get("PYOPENGL_PLATFORM"))
    )
    monkeypatch.setitem(sys.modules, "envs", fake_env_module)

    cfg = SimpleNamespace(task="ogbench_scene", encoder=1, seed=11)
    result = _GenericEnvConstructor(cfg, worker_id=0)()

    assert result == ("egl", "egl")


def test_non_dmc_or_state_auto_parallel_sampler_start_method_keeps_legacy_behavior(monkeypatch):
    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork", "spawn", "forkserver"])

    assert resolve_parallel_sampler_start_method(SimpleNamespace(
        task="dmc_quadruped_run_forward_color",
        encoder=0,
        parallel_sampler_start_method="auto",
    )) is None
    assert resolve_parallel_sampler_start_method(SimpleNamespace(
        task="maze_ant",
        encoder=1,
        parallel_sampler_start_method="auto",
    )) is None


def test_explicit_forkserver_start_method_falls_back_to_spawn_when_needed(monkeypatch):
    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork", "spawn"])
    cfg = SimpleNamespace(
        task="maze_ant",
        encoder=0,
        parallel_sampler_start_method="forkserver",
    )

    assert resolve_parallel_sampler_start_method(cfg) == "spawn"


def test_collect_train_paths_match_rollout_schema_and_target_count():
    collector = _collector(num_workers=3)
    policy = _FakePolicy()
    skill = np.asarray([1.0, 0.0], dtype=np.float32)

    paths = collector.collect(
        policy,
        target_num_trajectories=2,
        sample_extra_fn=lambda: {"skill": skill},
    )

    assert len(paths) == 2
    for path in paths:
        assert set(path) == {
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "dones",
            "agent_infos",
            "env_infos",
        }
        assert path["observations"].shape == (3, 2)
        assert path["next_observations"].shape == (3, 2)
        assert path["actions"].shape == (3, 1)
        assert path["agent_infos"]["skill"].shape == (3, 2)
        assert path["agent_infos"]["pre_tanh_value"].shape == (3, 1)
        assert path["env_infos"]["coordinates"].shape == (3, 2)
        assert path["dones"][-1]
    assert policy.reset_count == 1
    assert collector.consume_timing_metrics()["ParallelSamplerNumWorkers"] == 3.0


def test_collect_fixed_preserves_extra_order_and_deterministic_mode():
    collector = _collector(num_workers=2)
    policy = _FakePolicy()
    extras = [
        {"skill": np.asarray([1.0, 0.0], dtype=np.float32)},
        {"skill": np.asarray([0.0, 1.0], dtype=np.float32)},
        {"skill": np.asarray([0.5, 0.5], dtype=np.float32)},
    ]

    paths = collector.collect_fixed(policy, extras=extras, deterministic_policy=True)

    assert len(paths) == len(extras)
    for path, extra in zip(paths, extras):
        expected = np.repeat(extra["skill"][None, :], path["actions"].shape[0], axis=0)
        assert np.allclose(path["agent_infos"]["skill"], expected)
        assert np.allclose(path["actions"], 2.0)
    assert policy._force_use_mode_actions is False


def test_collect_fixed_video_records_worker_frames():
    collector = _collector(num_workers=1)
    policy = _FakePolicy()

    paths = collector.collect_fixed(
        policy,
        extras=[{"skill": np.asarray([1.0, 0.0], dtype=np.float32)}],
        deterministic_policy=False,
        state_record_pixeled=True,
        video_frame_source="render",
    )

    assert paths[0]["observations"].shape == (3, 4, 4, 3)
    assert paths[0]["next_observations"].shape == (3, 4, 4, 3)
    assert np.all(paths[0]["observations"][0] == 10)
    assert np.all(paths[0]["next_observations"][-1] == 10)


def test_ogbench_fixed_video_toggles_worker_capture_modes():
    collector = _collector(num_workers=2)
    collector.cfg.task = "ogbench_scene"
    collector.cfg.encoder = 1
    policy = _FakePolicy()

    collector.collect_fixed(
        policy,
        extras=[
            {"skill": np.asarray([1.0, 0.0], dtype=np.float32)},
            {"skill": np.asarray([0.0, 1.0], dtype=np.float32)},
        ],
        deterministic_policy=True,
        state_record_pixeled=True,
        video_frame_source="blog",
    )

    assert collector._workers[0].video_capture_modes == [True, False]
    assert collector._workers[1].video_capture_modes == [True, False]


def test_ogbench_fixed_video_restores_worker_capture_modes_on_failure(monkeypatch):
    collector = _collector(num_workers=2)
    collector.cfg.task = "ogbench_scene"
    collector.cfg.encoder = 1
    policy = _FakePolicy()

    def _boom(*args, **kwargs):
        raise RuntimeError("fixed collection boom")

    monkeypatch.setattr(collector, "_collect_fixed_impl", _boom)

    try:
        collector.collect_fixed(
            policy,
            extras=[{"skill": np.asarray([1.0, 0.0], dtype=np.float32)}],
            deterministic_policy=True,
            state_record_pixeled=True,
            video_frame_source="blog",
        )
    except RuntimeError as exc:
        assert "fixed collection boom" in str(exc)
    else:
        raise AssertionError("Expected fixed collection failure")

    assert collector._workers[0].video_capture_modes == [True, False]
    assert collector._workers[1].video_capture_modes == [True, False]
