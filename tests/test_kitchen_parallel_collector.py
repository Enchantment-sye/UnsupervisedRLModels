import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from envs.kitchen_parallel import KitchenProcessTrajectoryCollector


class _FakeKitchenWorker:
    def __init__(self, slot, horizon=2):
        self.slot = slot
        self.horizon = horizon
        self.t = 0
        self.closed = False
        self.reset_perturbations = []

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
        if name == "set_next_reset_perturbation":
            seed, scale = args

            def _result():
                self.reset_perturbations.append((int(seed), float(scale)))
                return True

            return _result

        assert name == "capture_video_frame"

        def _result():
            return np.full((4, 4, 3), self.slot, dtype=np.uint8)

        return _result

    def close(self):
        self.closed = True

    def _timestep(self, *, is_first):
        return {
            "image": np.asarray([self.slot, self.t, self.t + 1], dtype=np.uint8),
            "reward": float(self.t),
            "is_first": is_first,
            "is_last": self.t >= self.horizon,
            "is_terminal": self.t >= self.horizon,
            "info": {
                "coordinates": np.asarray([self.slot, self.t], dtype=np.float32),
                "next_coordinates": np.asarray([self.slot, self.t + 1], dtype=np.float32),
                "metric_success_task_relevant/goal_0": float(self.t >= self.horizon),
            },
        }


class _FakePolicy:
    def reset(self):
        pass

    def get_actions(self, obs):
        obs = np.asarray(obs)
        actions = np.ones((obs.shape[0], 1), dtype=np.float32)
        return actions, {
            "log_prob": np.zeros((obs.shape[0], 1), dtype=np.float32),
        }


def _cfg():
    return SimpleNamespace(
        stage="pre_training",
        algo=SimpleNamespace(algo="metra", dim_skill=2),
        encoder=1,
        time_limit=2,
    )


def test_kitchen_collector_matches_rollout_schema_with_fake_workers():
    collector = KitchenProcessTrajectoryCollector(
        _cfg(),
        num_workers=2,
        worker_factory=lambda slot: _FakeKitchenWorker(slot),
    )
    skill = np.asarray([1.0, 0.0], dtype=np.float32)

    paths = collector.collect(
        _FakePolicy(),
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
        assert path["observations"].shape == (2, 3)
        assert path["next_observations"].shape == (2, 3)
        assert path["actions"].shape == (2, 1)
        assert path["agent_infos"]["skill"].shape == (2, 2)
        assert "metric_success_task_relevant/goal_0" in path["env_infos"]
        assert "coordinates" in path["env_infos"]
        assert "next_coordinates" in path["env_infos"]

    metrics = collector.consume_timing_metrics()
    assert metrics["ParallelSamplerNumWorkers"] == 2.0


def test_kitchen_collect_fixed_applies_reset_perturbations_before_reset():
    workers = {}

    def _worker_factory(slot):
        worker = _FakeKitchenWorker(slot)
        workers[slot] = worker
        return worker

    collector = KitchenProcessTrajectoryCollector(
        _cfg(),
        num_workers=2,
        worker_factory=_worker_factory,
    )
    skill = np.asarray([1.0, 0.0], dtype=np.float32)

    paths = collector.collect_fixed(
        _FakePolicy(),
        extras=[{"skill": skill}, {"skill": skill}],
        deterministic_policy=True,
        reset_perturbations=[(101, 1e-4), (102, 1e-4)],
    )

    assert len(paths) == 2
    observed = sorted(
        perturbation
        for worker in workers.values()
        for perturbation in worker.reset_perturbations
    )
    assert observed == [(101, 1e-4), (102, 1e-4)]
