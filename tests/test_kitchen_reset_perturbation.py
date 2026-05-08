import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("src"))

try:
    from envs.kitchen.kitchen import KitchenEnv
except Exception as exc:
    pytest.skip(f"Kitchen D4RL/MuJoCo stack is unavailable: {exc}", allow_module_level=True)


class _FakeSim:
    def __init__(self):
        self.data = SimpleNamespace(
            qpos=np.zeros(12, dtype=np.float64),
            qvel=np.arange(12, dtype=np.float64),
        )
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class _FakeKitchenBackend:
    def __init__(self):
        self.sim = _FakeSim()


def _fake_env():
    env = KitchenEnv.__new__(KitchenEnv)
    env._env = _FakeKitchenBackend()
    env._next_reset_perturbation = None
    return env


def test_next_reset_perturbation_is_one_shot_and_reproducible():
    first = _fake_env()
    second = _fake_env()

    assert first.set_next_reset_perturbation(seed=123, scale=1e-4) is True
    assert first._apply_next_reset_perturbation() is True
    assert first._apply_next_reset_perturbation() is False

    assert second.set_next_reset_perturbation(seed=123, scale=1e-4) is True
    assert second._apply_next_reset_perturbation() is True

    np.testing.assert_allclose(first._env.sim.data.qpos, second._env.sim.data.qpos)
    np.testing.assert_allclose(first._env.sim.data.qvel, np.arange(12, dtype=np.float64))
    np.testing.assert_allclose(second._env.sim.data.qvel, np.arange(12, dtype=np.float64))
    assert np.any(np.abs(first._env.sim.data.qpos[:9]) > 0.0)
    np.testing.assert_allclose(first._env.sim.data.qpos[9:], 0.0)
    assert first._env.sim.forward_calls == 1
