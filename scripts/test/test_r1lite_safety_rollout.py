import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

if "gym" not in sys.modules:
    gym_mod = types.ModuleType("gym")
    spaces_mod = types.ModuleType("gym.spaces")

    class _Space:
        def __init__(self, *args, **kwargs):
            self.shape = kwargs.get("shape", ())

    spaces_mod.Discrete = _Space
    spaces_mod.Dict = _Space
    spaces_mod.Box = _Space
    gym_mod.spaces = spaces_mod
    sys.modules["gym"] = gym_mod
    sys.modules["gym.spaces"] = spaces_mod

from config.base import SafetyConfig
from envs.wrappers import NormalizeAction, TimeLimit
from workers.rollout import SkillRolloutWorker


def _safety_state():
    return {
        "left_arm_joint_position": np.zeros(6, dtype=np.float32),
        "right_arm_joint_position": np.zeros(6, dtype=np.float32),
        "left_arm_gripper_position": np.asarray([0.2], dtype=np.float32),
        "right_arm_gripper_position": np.asarray([0.2], dtype=np.float32),
    }


class _DummyEnv:
    def __init__(self):
        self.received_actions = []

    def reset(self):
        return {
            "state": np.zeros(14, dtype=np.float32),
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "info": {"safety_state": _safety_state()},
        }

    def step(self, action):
        if isinstance(action, dict):
            action = action["action"]
        self.received_actions.append(np.asarray(action, dtype=np.float32).copy())
        return {
            "state": np.zeros(14, dtype=np.float32),
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "reward": 0.0,
            "is_terminal": True,
            "info": {"safety_state": _safety_state()},
        }

    def safety_physical_action_bounds(self):
        return -np.ones(14, dtype=np.float32) * 3.0, np.ones(14, dtype=np.float32) * 3.0

    def safety_denormalize_action(self, action):
        return action

    def safety_normalize_action(self, action):
        return action


class _DummyPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self._force_use_mode_actions = False

    def reset(self):
        pass

    def get_action(self, _obs):
        return self.action.copy(), {}


class _NoTimingWrappedEnv:
    @property
    def act_space(self):
        return {
            "action": SimpleNamespace(
                low=-np.ones(2, dtype=np.float32),
                high=np.ones(2, dtype=np.float32),
            )
        }

    @property
    def obs_space(self):
        return {}


def _cfg(enabled):
    return SimpleNamespace(
        env=SimpleNamespace(task="galaxea_r1lite_blocks_stack_easy"),
        safety=SafetyConfig(
            enabled=int(enabled),
            mode="sim",
            safety_yaml=str(ROOT / "configs/safety/r1lite_redlines.yaml"),
            lbsgd_steps=2,
        ),
    )


def test_rollout_env_step_receives_safe_action_and_records_raw_safe():
    env = _DummyEnv()
    policy = _DummyPolicy(np.full(14, 3.0, dtype=np.float32))
    worker = SkillRolloutWorker(seed=0, time_limit=2, cur_extra_keys=[], pixeled=False, config=_cfg(True))

    worker.start_rollout(env, policy, deterministic_policy=True)
    worker.step_rollout(env, policy)

    assert len(env.received_actions) == 1
    assert not np.allclose(env.received_actions[0], policy.action)
    assert np.allclose(worker._actions[0], env.received_actions[0])
    assert "raw_action" in worker._agent_infos
    assert "safe_action" in worker._agent_infos
    assert "safety_correction_norm" in worker._agent_infos
    assert "safety_qp_active" in worker._env_infos


def test_rollout_safety_disabled_preserves_raw_action_path():
    env = _DummyEnv()
    policy = _DummyPolicy(np.full(14, 3.0, dtype=np.float32))
    worker = SkillRolloutWorker(seed=0, time_limit=2, cur_extra_keys=[], pixeled=False, config=_cfg(False))

    worker.start_rollout(env, policy)
    worker.step_rollout(env, policy)

    assert np.allclose(env.received_actions[0], policy.action)
    assert "raw_action" not in worker._agent_infos


def test_optional_timing_metrics_getattr_default_survives_wrappers():
    env = TimeLimit(NormalizeAction(_NoTimingWrappedEnv()), duration=2)

    assert getattr(env, "consume_timing_metrics", None) is None


def test_accumulate_env_timing_metrics_skips_wrapped_env_without_metrics():
    env = TimeLimit(NormalizeAction(_NoTimingWrappedEnv()), duration=2)
    worker = SkillRolloutWorker(seed=0, time_limit=2, cur_extra_keys=[], pixeled=False, config=_cfg(False))
    before = worker.consume_timing_metrics()

    worker._accumulate_env_timing_metrics(env)

    assert worker.consume_timing_metrics() == before
