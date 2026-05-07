import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from envs.galaxea_sim_parallel import GalaxeaSimProcessTrajectoryCollector
from safety.types import SafetyReport


class _FakeWorker:
    def __init__(self, slot):
        self.slot = int(slot)
        self.step_count = 0
        self.capture_sources = []
        self.received_actions = []

    def reset(self, blocking=False):
        self.step_count = 0
        timestep = self._timestep(done=False)
        return timestep if blocking else (lambda: timestep)

    def step(self, action, blocking=False):
        action_value = action.get("action", action) if isinstance(action, dict) else action
        self.received_actions.append(np.asarray(action_value, dtype=np.float32).copy())
        self.step_count += 1
        timestep = self._timestep(done=self.step_count >= 2)
        return timestep if blocking else (lambda: timestep)

    def call(self, name, *args, **_kwargs):
        if name == "get_train_image_tensor":
            return lambda: np.full((4,), self.slot * 10 + self.step_count, dtype=np.uint8)
        if name == "capture_video_frame":
            source = args[0] if args else None

            def _capture():
                self.capture_sources.append(source)
                value = self.slot * 50 + self.step_count
                return np.full((2, 2, 3), value, dtype=np.uint8)

            return _capture
        if name in ("safety_denormalize_action", "safety_normalize_action"):
            return lambda action: action
        raise AttributeError(name)

    def close(self):
        pass

    def _timestep(self, *, done):
        image_value = self.slot * 20 + self.step_count
        return {
            "state": np.full((4,), image_value, dtype=np.float32),
            "image": np.full((2, 2, 3), image_value, dtype=np.uint8),
            "reward": float(self.slot + self.step_count),
            "is_last": bool(done),
            "is_terminal": bool(done),
            "info": {
                "safety_state": {
                    "left_arm_joint_position": np.zeros(6, dtype=np.float32),
                    "right_arm_joint_position": np.zeros(6, dtype=np.float32),
                    "left_arm_gripper_position": np.zeros(1, dtype=np.float32),
                    "right_arm_gripper_position": np.zeros(1, dtype=np.float32),
                }
            },
        }


class _FakePolicy:
    def __init__(self):
        self._force_use_mode_actions = False
        self.reset_calls = 0
        self.inputs = []

    def reset(self):
        self.reset_calls += 1

    def get_actions(self, obs):
        obs = np.asarray(obs)
        self.inputs.append(obs.copy())
        batch = obs.shape[0]
        actions = np.stack(
            [
                np.linspace(0.0, 1.0, batch, dtype=np.float32),
                np.linspace(1.0, 2.0, batch, dtype=np.float32),
            ],
            axis=1,
        )
        return actions, {"policy_row": np.arange(batch, dtype=np.float32)}


class _FakeSafetyController:
    def filter_action(self, raw_action, **_kwargs):
        report = SafetyReport(
            safety_enabled=True,
            safety_mode="sim",
            safety_triggered=True,
            safety_correction_norm=1.0,
        )
        return np.asarray(raw_action, dtype=np.float32) + 1.0, report


def _collector(num_envs=2):
    cfg = SimpleNamespace(
        encoder=1,
        stage="pre_training",
        dim_skill=2,
        task="galaxea_r1lite_blocks_stack_easy",
    )
    collector = GalaxeaSimProcessTrajectoryCollector.__new__(GalaxeaSimProcessTrajectoryCollector)
    collector.cfg = cfg
    collector._num_envs = int(num_envs)
    collector._workers = [_FakeWorker(slot) for slot in range(num_envs)]
    collector._timing_totals = {
        "TimeSamplingEnv": 0.0,
        "TimeImagePostprocess": 0.0,
    }
    collector._safety_controllers = [_FakeSafetyController() for _ in range(num_envs)]
    collector._prev_safe_actions = [None for _ in range(num_envs)]
    return collector


def _extras(count):
    return [
        {"skill": np.asarray([idx, idx + 0.5], dtype=np.float32)}
        for idx in range(count)
    ]


def test_collect_fixed_preserves_extra_order_and_policy_observations():
    collector = _collector(num_envs=2)
    policy = _FakePolicy()

    paths = collector.collect_fixed(
        policy,
        extras=_extras(3),
        deterministic_policy=True,
        state_record_pixeled=False,
    )

    assert len(paths) == 3
    assert policy.reset_calls == 1
    assert policy._force_use_mode_actions is False
    assert paths[0]["observations"].shape == (2, 4)
    assert np.allclose(paths[0]["agent_infos"]["skill"][0], np.asarray([0.0, 0.5], dtype=np.float32))
    assert np.allclose(paths[1]["agent_infos"]["skill"][0], np.asarray([1.0, 1.5], dtype=np.float32))
    assert np.allclose(paths[2]["agent_infos"]["skill"][0], np.asarray([2.0, 2.5], dtype=np.float32))


def test_collect_fixed_records_video_frames_and_passes_source():
    collector = _collector(num_envs=2)
    policy = _FakePolicy()

    paths = collector.collect_fixed(
        policy,
        extras=_extras(2),
        deterministic_policy=False,
        state_record_pixeled=True,
        video_frame_source="third_person",
    )

    assert len(paths) == 2
    assert paths[0]["observations"].shape == (2, 2, 2, 3)
    assert collector._workers[0].capture_sources
    assert set(collector._workers[0].capture_sources) == {"third_person"}
    assert set(collector._workers[1].capture_sources) == {"third_person"}


def test_collect_fixed_keeps_safety_metrics_and_safe_actions():
    collector = _collector(num_envs=2)
    policy = _FakePolicy()

    paths = collector.collect_fixed(
        policy,
        extras=_extras(1),
        deterministic_policy=False,
        state_record_pixeled=False,
    )

    path = paths[0]
    assert "raw_action" in path["agent_infos"]
    assert "safe_action" in path["agent_infos"]
    assert "safety_correction_norm" in path["agent_infos"]
    assert "safety_enabled" in path["env_infos"]
    assert np.allclose(path["agent_infos"]["safe_action"], path["agent_infos"]["raw_action"] + 1.0)
    assert np.allclose(collector._workers[0].received_actions[0], path["agent_infos"]["safe_action"][0])
