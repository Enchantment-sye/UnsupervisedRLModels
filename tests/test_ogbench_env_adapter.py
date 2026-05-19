import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from envs.ogbench_env import OGBenchEnv, resolve_ogbench_scene_env_id
from envs.ogbench_scene_kitchen_like_eval import reset_info_key
from envs import make_env
from workers.rollout import SkillRolloutWorker


class _Space:
    def __init__(self, shape, low=-1.0, high=1.0, dtype=np.float32):
        self.shape = tuple(shape)
        self.low = np.full(self.shape, low, dtype=dtype)
        self.high = np.full(self.shape, high, dtype=dtype)
        self.dtype = dtype


class _FakeRawSceneEnv:
    def __init__(self, *, pixels=False, width=64, height=64):
        self.pixels = bool(pixels)
        self.width = int(width)
        self.height = int(height)
        self._render_width = int(width)
        self._render_height = int(height)
        self._renderer = None
        self.render_cameras = []
        self.render_arm_alphas = []
        self._model = _FakeModel() if pixels else None
        self.observation_space = _Space((self.height, self.width, 3), 0, 255, np.uint8) if pixels else _Space((40,))
        self.action_space = _Space((5,))
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        cube_x = 0.35 + float(np.random.random()) * 0.01
        return self._obs(), _scene_info(buttons=(0, 0), cube=(cube_x, 0.05, 0.02))

    def step(self, action):
        assert np.asarray(action).shape == (5,)
        self.steps += 1
        info = _scene_info(
            buttons=(1, 0),
            cube=(0.45, 0.05, 0.02),
            drawer=-0.13,
            window=0.0,
        )
        return self._obs(), 0.0, False, self.steps >= 2, info

    def _obs(self):
        if self.pixels:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return np.zeros(40, dtype=np.float32)

    def render(self, camera=None):
        self.render_cameras.append(camera)
        self.render_arm_alphas.append(self._model.material("ur5e/linkgray").rgba[3] if self._model is not None else 1.0)
        value = 7 if camera == "front" else 5
        return np.full((self._render_height, self._render_width, 3), value, dtype=np.uint8)


class _FakeMaterial:
    def __init__(self, alpha):
        self.rgba = np.asarray([0.1, 0.2, 0.3, alpha], dtype=np.float64)


class _FakeModel:
    def __init__(self):
        self._materials = {
            name: _FakeMaterial(0.1)
            for name in (
                "ur5e/robotiq/metal",
                "ur5e/robotiq/silicone",
                "ur5e/robotiq/gray",
                "ur5e/robotiq/black",
                "ur5e/robotiq/pad_gray",
                "ur5e/black",
                "ur5e/jointgray",
                "ur5e/linkgray",
                "ur5e/lightblue",
            )
        }

    def material(self, name):
        return self._materials[name]


class _Policy:
    def reset(self):
        self._force_use_mode_actions = False

    def get_action(self, obs):
        return np.zeros(5, dtype=np.float32), {}


def _scene_info(cube, buttons=(0, 0), drawer=0.0, window=0.0):
    return {
        "privileged/block_0_pos": np.asarray(cube, dtype=np.float64),
        "button_states": np.asarray(buttons, dtype=np.int64),
        "privileged/drawer_pos": np.asarray([drawer], dtype=np.float64),
        "privileged/window_pos": np.asarray([window], dtype=np.float64),
        "goal": np.zeros(3, dtype=np.float32),
        "goal_rendered": np.zeros((64, 64, 3), dtype=np.uint8),
    }


def _install_fake_ogbench(monkeypatch):
    made = []
    make_kwargs = []

    def make(env_id, **kwargs):
        made.append(env_id)
        make_kwargs.append(dict(kwargs))
        return _FakeRawSceneEnv(
            pixels=env_id == "visual-scene-v0",
            width=kwargs.get("width", 64),
            height=kwargs.get("height", 64),
        )

    monkeypatch.setitem(sys.modules, "ogbench", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(make=make))
    return made, make_kwargs


def test_encoder_resolves_state_and_pixel_scene_ids():
    assert resolve_ogbench_scene_env_id("ogbench_scene", encoder=0) == "scene-v0"
    assert resolve_ogbench_scene_env_id("ogbench_scene", encoder=1) == "visual-scene-v0"
    assert resolve_ogbench_scene_env_id("ogbench_scene-play-v0", encoder=0) == "scene-v0"
    assert resolve_ogbench_scene_env_id("ogbench_visual-scene-play-v0", encoder=1) == "visual-scene-v0"


def test_adapter_preserves_last_reset_info_and_step_info(monkeypatch):
    made, make_kwargs = _install_fake_ogbench(monkeypatch)
    env = OGBenchEnv(task_name="ogbench_scene", encoder=0, seed=123)

    reset_timestep = env.reset()
    step_timestep = env.step({"action": np.zeros(5, dtype=np.float32)})

    assert made == ["scene-v0"]
    assert make_kwargs == [{"disable_env_checker": True}]
    assert reset_timestep["state"].shape == (40,)
    assert reset_timestep["image"].shape == (64, 64, 3)
    assert "privileged/block_0_pos" in env.last_reset_info
    assert step_timestep["info"]["button_states"][0] == 1
    assert reset_info_key("privileged/block_0_pos") in step_timestep["info"]


def test_pixel_adapter_uses_encoder_one(monkeypatch):
    made, _ = _install_fake_ogbench(monkeypatch)
    env = OGBenchEnv(task_name="ogbench_scene-play-v0", encoder=1, size=(128, 128))

    timestep = env.reset()

    assert made == ["visual-scene-v0"]
    assert timestep["image"].shape == (128, 128, 3)
    assert timestep["state"].shape == (128 * 128 * 3,)
    assert env.obs_space["image"].shape == (128, 128, 3)


def test_adapter_captures_render_video_frame(monkeypatch):
    _install_fake_ogbench(monkeypatch)
    env = OGBenchEnv(task_name="ogbench_scene", encoder=1, size=(128, 128))
    env.reset()

    frame = env.capture_video_frame(source="render")

    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8
    assert int(frame[0, 0, 0]) == 7
    assert env._env.render_cameras[-1] == "front"
    assert env._env.render_arm_alphas[-1] == 1.0
    assert env._env._model.material("ur5e/linkgray").rgba[3] == 0.1


def test_adapter_observation_video_source_keeps_pixel_camera_and_transparent_arm(monkeypatch):
    _install_fake_ogbench(monkeypatch)
    env = OGBenchEnv(task_name="ogbench_scene", encoder=1, size=(128, 128))
    env.reset()

    frame = env.capture_video_frame(source="observation")

    assert frame.shape == (128, 128, 3)
    assert int(frame[0, 0, 0]) == 5
    assert env._env.render_cameras[-1] == "front_pixels"
    assert env._env.render_arm_alphas[-1] == 0.1


def test_adapter_video_capture_mode_uses_independent_video_size(monkeypatch):
    _install_fake_ogbench(monkeypatch)
    env = OGBenchEnv(task_name="ogbench_scene", encoder=1, size=(128, 128), video_size=256)
    env.reset()

    env.set_video_capture_active(True)
    frame = env.capture_video_frame(source="blog")
    env.set_video_capture_active(False)

    assert frame.shape == (256, 256, 3)
    assert env._env._render_height == 128
    assert env._env._render_width == 128


def test_adapter_next_reset_seed_hook_is_used(monkeypatch):
    _install_fake_ogbench(monkeypatch)
    env = OGBenchEnv(task_name="ogbench_scene", encoder=0)
    calls = []

    original_reset = env._env.reset

    def reset(**kwargs):
        calls.append(dict(kwargs))
        return original_reset(**kwargs)

    env._env.reset = reset
    env.set_next_reset_perturbation(1234, 1.0)
    env.reset()

    assert calls == [{"seed": 1234}]


def test_adapter_next_reset_seed_controls_global_numpy_temporarily(monkeypatch):
    _install_fake_ogbench(monkeypatch)
    env = OGBenchEnv(task_name="ogbench_scene", encoder=0)
    before_state = np.random.get_state()

    env.set_next_reset_perturbation(1234, 1.0)
    env.reset()
    cube1 = env.last_reset_info["privileged/block_0_pos"].copy()
    after_first_state = np.random.get_state()
    env.set_next_reset_perturbation(1234, 1.0)
    env.reset()
    cube2 = env.last_reset_info["privileged/block_0_pos"].copy()
    after_second_state = np.random.get_state()

    assert np.allclose(cube1, cube2)
    assert all(np.array_equal(a, b) for a, b in zip(before_state, after_first_state))
    assert all(np.array_equal(a, b) for a, b in zip(after_first_state, after_second_state))


def test_make_env_uses_encoder_as_scene_source_of_truth(monkeypatch):
    made, make_kwargs = _install_fake_ogbench(monkeypatch)
    cfg = SimpleNamespace(
        task="ogbench_scene-play-v0",
        seed=0,
        action_repeat=1,
        render_size=128,
        encoder=1,
        flatten_obs=1,
        time_limit=5,
        framestack=3,
    )

    env = make_env("train", cfg)
    timestep = env.reset()

    assert made == ["visual-scene-v0"]
    assert make_kwargs == [{"disable_env_checker": True, "width": 128, "height": 128}]
    assert env.env_id == "visual-scene-v0"
    assert env.obs_space["image"].shape == (128, 128, 9)
    assert timestep["image"].shape == (128, 128, 9)


def test_rollout_worker_carries_adapter_info_into_env_infos(monkeypatch):
    _install_fake_ogbench(monkeypatch)
    env = OGBenchEnv(task_name="ogbench_scene", encoder=0)
    worker = SkillRolloutWorker(
        seed=0,
        time_limit=2,
        cur_extra_keys=[],
        pixeled=False,
        config=None,
    )

    batch = worker.rollout(env, _Policy(), deterministic_policy=True)
    path = batch.to_trajectory_list()[0]

    assert "privileged/block_0_pos" in path["env_infos"]
    assert "button_states" in path["env_infos"]
    assert reset_info_key("privileged/block_0_pos") in path["env_infos"]
    assert path["env_infos"]["privileged/block_0_pos"].shape[0] == 2
