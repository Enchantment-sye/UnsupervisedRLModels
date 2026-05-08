import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from envs.generic_parallel import GenericProcessTrajectoryCollector


class _FakeWorker:
    def __init__(self, slot, horizon=3):
        self.slot = slot
        self.horizon = horizon
        self.t = 0
        self.closed = False

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
        assert name == "capture_video_frame"

        def _result():
            return np.full((4, 4, 3), self.slot + 10, dtype=np.uint8)

        return _result

    def close(self):
        self.closed = True

    def _timestep(self, *, is_first):
        return {
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
        stage="pre_training",
        algo=SimpleNamespace(algo="metra", dim_skill=2),
        encoder=0,
        time_limit=3,
    )


def _collector(num_workers=2):
    collector = GenericProcessTrajectoryCollector.__new__(GenericProcessTrajectoryCollector)
    collector.cfg = _cfg()
    collector._num_workers = num_workers
    collector._workers = [_FakeWorker(slot) for slot in range(num_workers)]
    collector._timing_totals = collector._new_timing_totals()
    return collector


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
