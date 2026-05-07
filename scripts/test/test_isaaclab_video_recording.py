import os
import sys
from types import SimpleNamespace

import gym
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.metra_config import MetraConfig, get_parser, make_config_from_args
from envs import normalize_env_backend_for_task, should_use_isaaclab_backend
from src.envs.isaaclab.adapters.single_agent_box import IsaacLabSingleAgentBoxEnv
from src.envs.isaaclab.base_spec import IsaacLabTaskSpec
from src.envs.isaaclab.cfg_builders import (
    GALAXEA_PANORAMA_FIXED_VIEWER_SETTINGS,
    GALAXEA_WORKSTATION_VIEWER_SETTINGS,
    galaxea_workstation_cfg_builder,
    resolve_viewer_preset_settings,
)
from src.envs.isaaclab.factory import resolve_isaaclab_request
from src.envs.isaaclab.viewer_runtime import (
    reapply_active_viewer_preset,
    temporary_video_viewer_preset,
    warmup_render_capture,
)
from utils import utils
from workers.rollout import SkillRolloutWorker


class _FakeViewportCameraController:
    def __init__(self):
        self.calls = []

    def update_view_to_world(self):
        self.calls.append(("origin", "world"))

    def update_view_to_asset_root(self, asset_name):
        self.calls.append(("origin", "asset_root", asset_name))

    def update_view_location(self, eye=None, lookat=None):
        self.calls.append(("location", tuple(float(x) for x in eye), tuple(float(x) for x in lookat)))


class _FakeSim:
    def __init__(self):
        self.calls = []

    def set_camera_view(self, *, eye, target):
        self.calls.append((tuple(float(x) for x in eye), tuple(float(x) for x in target)))


class _FakeVideoEnv:
    def __init__(self, *, render_frames=None, done_after=1):
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Dict(
                    {
                        "joint_pos": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                        "joint_vel": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                        "left_ee_pose": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                        "right_ee_pose": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                        "object_pose": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                        "goal_pose": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                        "front_rgb": gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
                    }
                )
            }
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.unwrapped = self
        self.viewport_camera_controller = _FakeViewportCameraController()
        self.sim = _FakeSim()
        self._step_count = 0
        self._done_after = int(done_after)
        if render_frames is None:
            render_frames = [np.full((8, 8, 3), (120, 130, 140), dtype=np.uint8)]
        self._render_frames = [None if frame is None else np.asarray(frame).copy() for frame in render_frames]
        self._render_idx = 0

    def _policy_obs(self):
        return {
            "policy": {
                "joint_pos": np.full((1, 2), 1.0 + self._step_count, dtype=np.float32),
                "joint_vel": np.full((1, 2), 2.0 + self._step_count, dtype=np.float32),
                "left_ee_pose": np.full((1, 1), 3.0, dtype=np.float32),
                "right_ee_pose": np.full((1, 1), 4.0, dtype=np.float32),
                "object_pose": np.full((1, 1), 5.0, dtype=np.float32),
                "goal_pose": np.full((1, 1), 6.0, dtype=np.float32),
                "front_rgb": np.full((1, 8, 8, 3), (10, 20, 30), dtype=np.uint8),
            }
        }

    def reset(self, **kwargs):
        self._step_count = 0
        self._render_idx = 0
        return self._policy_obs(), {}

    def step(self, action):
        self._step_count += 1
        return self._policy_obs(), 1.0, self._step_count >= self._done_after, False, {}

    def render(self, mode="rgb_array"):
        idx = min(self._render_idx, len(self._render_frames) - 1)
        frame = self._render_frames[idx]
        self._render_idx += 1
        return None if frame is None else frame.copy()


class _RecordingPolicy:
    def __init__(self):
        self.observations = []
        self._force_use_mode_actions = False

    def reset(self):
        self.observations.clear()

    def get_action(self, obs):
        self.observations.append(np.asarray(obs).copy())
        return np.zeros(2, dtype=np.float32), {}


def _make_wrapper(*, video_source="observation", render_frames=None, done_after=1):
    request = SimpleNamespace(
        flatten_obs=0,
        render_size=8,
        seed=0,
        encoder=1,
        image_source="camera",
        camera_key="front_rgb",
        render_mode="rgb_array",
        video_source=video_source,
        video_viewer_preset="inherit",
    )
    spec = IsaacLabTaskSpec(
        task_name="isaaclab_r1_lift_bin",
        env_id="Isaac-R1-Lift-Bin-IK-Rel-Direct-v0",
        workflow_type="direct",
        obs_type="box",
        action_type="box",
        requires_cameras=False,
        supports_render_rgb=True,
        supports_camera_obs=True,
        camera_obs_key="front_rgb",
        adapter_cls=IsaacLabSingleAgentBoxEnv,
        cfg_builder=galaxea_workstation_cfg_builder,
    )
    return IsaacLabSingleAgentBoxEnv(
        env=_FakeVideoEnv(render_frames=render_frames, done_after=done_after),
        task_spec=spec,
        request=request,
    )


def test_isaaclab_video_args_round_trip_to_request():
    parser = get_parser()
    args = parser.parse_args(
        [
            "--env-backend",
            "isaaclab",
            "--task",
            "isaaclab_r1_lift_bin",
            "--encoder",
            "1",
            "--isaaclab-video-source",
            "render",
            "--isaaclab-video-viewer-preset",
            "panorama_fixed",
        ]
    )
    cfg = make_config_from_args(args, cls=MetraConfig)

    assert cfg.env.isaaclab_video_source == "render"
    assert cfg.env.isaaclab_video_viewer_preset == "panorama_fixed"

    request = resolve_isaaclab_request(cfg, mode="train")
    assert request.video_source == "render"
    assert request.video_viewer_preset == "panorama_fixed"


def test_isaaclab_task_normalizes_backend_without_explicit_flag():
    parser = get_parser()
    args = parser.parse_args(
        [
            "--task",
            "isaaclab_r1_lift_bin",
            "--encoder",
            "1",
        ]
    )

    assert args.env_backend == "url"
    cfg = make_config_from_args(args, cls=MetraConfig)

    assert args.env_backend == "isaaclab"
    assert cfg.env_backend == "isaaclab"
    assert should_use_isaaclab_backend(args)
    assert should_use_isaaclab_backend(cfg)


def test_isaaclab_backend_helper_matches_task_prefix_and_explicit_override():
    assert normalize_env_backend_for_task("isaaclab_r1_lift_bin", "url") == "isaaclab"
    assert normalize_env_backend_for_task("isaaclab:Isaac-R1-Lift-Bin-IK-Rel-Direct-v0", "url") == "isaaclab"
    assert normalize_env_backend_for_task("dmc_walker_walk", "isaaclab") == "isaaclab"
    assert normalize_env_backend_for_task("dmc_walker_walk", "url") == "url"
    assert should_use_isaaclab_backend(task_name="isaaclab_r1_lift_bin", env_backend="url")
    assert should_use_isaaclab_backend(task_name="dmc_walker_walk", env_backend="isaaclab")
    assert not should_use_isaaclab_backend(task_name="dmc_walker_walk", env_backend="url")


def test_capture_video_frame_uses_render_source_independently_from_policy_image():
    wrapper = _make_wrapper(video_source="render")
    timestep = wrapper.reset()

    np.testing.assert_array_equal(timestep["image"], np.full((8, 8, 3), (10, 20, 30), dtype=np.uint8))

    video_frame = wrapper.capture_video_frame(source="render")
    np.testing.assert_array_equal(video_frame, np.full((8, 8, 3), (120, 130, 140), dtype=np.uint8))


def test_capture_video_frame_returns_none_for_all_zero_render_warmup_frame():
    wrapper = _make_wrapper(
        video_source="render",
        render_frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
    )
    wrapper.reset()

    assert wrapper.capture_video_frame(source="render") is None


def test_warmup_render_capture_waits_for_stable_render_frames():
    pink_box = np.full((8, 8, 3), (180, 20, 180), dtype=np.uint8)
    third_person = np.full((8, 8, 3), (120, 130, 140), dtype=np.uint8)
    snow = np.random.default_rng(0).integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    blur = np.full((8, 8, 3), (121, 131, 141), dtype=np.uint8)
    stable_1 = np.full((8, 8, 3), (122, 132, 142), dtype=np.uint8)
    stable_2 = np.full((8, 8, 3), (123, 133, 143), dtype=np.uint8)
    frames = iter([pink_box, third_person, snow, blur, stable_1, stable_2])

    frame, stabilized = warmup_render_capture(lambda: next(frames, stable_2))

    assert stabilized is True
    np.testing.assert_array_equal(frame, stable_2)


def test_panorama_fixed_viewer_preset_resolves_to_world_camera_for_r1_tasks():
    wrapper = _make_wrapper(video_source="render")
    preset_name, settings = resolve_viewer_preset_settings(wrapper._task_spec, "panorama_fixed")

    assert preset_name == "panorama_fixed"
    assert settings["origin_type"] == "world"
    assert settings["eye"] == (1.95, 1.45, 2.35)
    assert settings["lookat"] == (0.55, 0.0, 1.02)


def test_temporary_video_viewer_preset_switches_to_panorama_and_restores_follow_view():
    wrapper = _make_wrapper(video_source="render")
    controller = wrapper._env.viewport_camera_controller

    with temporary_video_viewer_preset(wrapper, "panorama_fixed"):
        pass

    assert controller.calls[0] == ("origin", "world")
    assert controller.calls[1] == (
        "location",
        (1.95, 1.45, 2.35),
        (0.55, 0.0, 1.02),
    )
    assert controller.calls[2] == ("origin", "asset_root", "robot")
    assert controller.calls[3] == (
        "location",
        GALAXEA_WORKSTATION_VIEWER_SETTINGS["eye"],
        GALAXEA_WORKSTATION_VIEWER_SETTINGS["lookat"],
    )


def test_reapply_active_viewer_preset_uses_current_panorama_setting():
    wrapper = _make_wrapper(video_source="render")
    wrapper._active_viewer_preset = "panorama_fixed"
    controller = wrapper._env.viewport_camera_controller

    applied = reapply_active_viewer_preset(wrapper)

    assert applied is True
    assert controller.calls[0] == ("origin", "world")
    assert controller.calls[1] == (
        "location",
        (1.95, 1.45, 2.35),
        (0.55, 0.0, 1.02),
    )


def test_rollout_records_render_video_frames_without_changing_policy_inputs():
    wrapper = _make_wrapper(video_source="render")
    policy = _RecordingPolicy()
    worker = SkillRolloutWorker(seed=0, time_limit=4, cur_extra_keys=[], pixeled=True)

    batch = worker.rollout(
        wrapper,
        policy,
        deterministic_policy=False,
        state_record_pixeled=True,
        video_frame_source="render",
    )
    trajectory = batch.to_trajectory_list()[0]

    assert len(policy.observations) == 1
    np.testing.assert_array_equal(policy.observations[0], np.full((8, 8, 3), (10, 20, 30), dtype=np.uint8))
    np.testing.assert_array_equal(
        trajectory["observations"][0],
        np.full((8, 8, 3), (120, 130, 140), dtype=np.uint8),
    )


def test_rollout_warms_up_initial_black_render_frame_before_recording():
    valid = np.full((8, 8, 3), (120, 130, 140), dtype=np.uint8)
    wrapper = _make_wrapper(
        video_source="render",
        render_frames=[np.zeros((8, 8, 3), dtype=np.uint8), valid],
    )
    policy = _RecordingPolicy()
    worker = SkillRolloutWorker(seed=0, time_limit=4, cur_extra_keys=[], pixeled=True)

    batch = worker.rollout(
        wrapper,
        policy,
        deterministic_policy=False,
        state_record_pixeled=True,
        video_frame_source="render",
    )
    trajectory = batch.to_trajectory_list()[0]

    np.testing.assert_array_equal(trajectory["observations"][0], valid)


def test_rollout_waits_out_non_black_transitional_render_frames_before_recording():
    pink_box = np.full((8, 8, 3), (180, 20, 180), dtype=np.uint8)
    third_person = np.full((8, 8, 3), (120, 130, 140), dtype=np.uint8)
    snow = np.random.default_rng(0).integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    blur = np.full((8, 8, 3), (121, 131, 141), dtype=np.uint8)
    stable_1 = np.full((8, 8, 3), (122, 132, 142), dtype=np.uint8)
    stable_2 = np.full((8, 8, 3), (123, 133, 143), dtype=np.uint8)
    wrapper = _make_wrapper(
        video_source="render",
        render_frames=[pink_box, third_person, snow, blur, stable_1, stable_2],
    )
    policy = _RecordingPolicy()
    worker = SkillRolloutWorker(seed=0, time_limit=4, cur_extra_keys=[], pixeled=True)

    batch = worker.rollout(
        wrapper,
        policy,
        deterministic_policy=False,
        state_record_pixeled=True,
        video_frame_source="render",
    )
    trajectory = batch.to_trajectory_list()[0]

    np.testing.assert_array_equal(trajectory["observations"][0], stable_2)


def test_rollout_reapplies_active_panorama_preset_before_render_bootstrap():
    pink_box = np.full((8, 8, 3), (180, 20, 180), dtype=np.uint8)
    stable_1 = np.full((8, 8, 3), (122, 132, 142), dtype=np.uint8)
    stable_2 = np.full((8, 8, 3), (123, 133, 143), dtype=np.uint8)
    wrapper = _make_wrapper(
        video_source="render",
        render_frames=[pink_box, stable_1, stable_2],
    )
    wrapper._active_viewer_preset = "panorama_fixed"
    policy = _RecordingPolicy()
    worker = SkillRolloutWorker(seed=0, time_limit=4, cur_extra_keys=[], pixeled=True)

    batch = worker.rollout(
        wrapper,
        policy,
        deterministic_policy=False,
        state_record_pixeled=True,
        video_frame_source="render",
    )
    trajectory = batch.to_trajectory_list()[0]

    np.testing.assert_array_equal(trajectory["observations"][0], stable_2)
    assert wrapper._env.viewport_camera_controller.calls[0] == ("origin", "world")


def test_rollout_defers_unstable_reset_bootstrap_until_after_first_step():
    pink_box = np.full((8, 8, 3), (180, 20, 180), dtype=np.uint8)
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    third_person = np.full((8, 8, 3), (120, 130, 140), dtype=np.uint8)
    snow = np.random.default_rng(1).integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    blur = np.full((8, 8, 3), (121, 131, 141), dtype=np.uint8)
    stable_1 = np.full((8, 8, 3), (122, 132, 142), dtype=np.uint8)
    stable_2 = np.full((8, 8, 3), (123, 133, 143), dtype=np.uint8)
    wrapper = _make_wrapper(
        video_source="render",
        render_frames=[pink_box] + [black] * 9 + [third_person, snow, blur, stable_1, stable_2],
    )
    wrapper._active_viewer_preset = "panorama_fixed"
    policy = _RecordingPolicy()
    worker = SkillRolloutWorker(seed=0, time_limit=4, cur_extra_keys=[], pixeled=True)

    batch = worker.rollout(
        wrapper,
        policy,
        deterministic_policy=False,
        state_record_pixeled=True,
        video_frame_source="render",
    )
    trajectory = batch.to_trajectory_list()[0]

    np.testing.assert_array_equal(trajectory["observations"][0], stable_2)


def test_rollout_reuses_last_valid_render_frame_when_render_temporarily_returns_black():
    valid = np.full((8, 8, 3), (120, 130, 140), dtype=np.uint8)
    wrapper = _make_wrapper(
        video_source="render",
        render_frames=[valid, np.zeros((8, 8, 3), dtype=np.uint8), valid],
        done_after=2,
    )
    policy = _RecordingPolicy()
    worker = SkillRolloutWorker(seed=0, time_limit=4, cur_extra_keys=[], pixeled=True)

    batch = worker.rollout(
        wrapper,
        policy,
        deterministic_policy=False,
        state_record_pixeled=True,
        video_frame_source="render",
    )
    trajectory = batch.to_trajectory_list()[0]

    assert trajectory["observations"].shape[0] == 2
    np.testing.assert_array_equal(trajectory["observations"][0], valid)
    np.testing.assert_array_equal(trajectory["observations"][1], valid)


def test_trajectories_to_video_tensor_repeats_last_valid_frame_for_shorter_rollouts():
    frame_a = np.full((8, 8, 3), 11, dtype=np.uint8)
    frame_b = np.full((8, 8, 3), 22, dtype=np.uint8)
    frame_c = np.full((8, 8, 3), 33, dtype=np.uint8)
    frame_d = np.full((8, 8, 3), 44, dtype=np.uint8)
    tensor = utils.trajectories_to_video_tensor(
        [
            {"observations": np.asarray([frame_a], dtype=np.uint8)},
            {"observations": np.asarray([frame_b, frame_c, frame_d], dtype=np.uint8)},
        ],
        shape=(8, 8),
    )

    assert tensor.shape == (2, 3, 3, 8, 8)
    np.testing.assert_array_equal(np.moveaxis(tensor[0, 1], 0, -1), frame_a)
    np.testing.assert_array_equal(np.moveaxis(tensor[0, 2], 0, -1), frame_a)


def test_prepare_video_duplicates_last_real_tile_instead_of_padding_with_black():
    videos = np.stack(
        [
            np.full((1, 3, 8, 8), fill_value=value, dtype=np.uint8)
            for value in (10, 20, 30, 40, 50)
        ],
        axis=0,
    )

    montage = utils.prepare_video(videos)

    assert montage.shape == (1, 16, 32, 3)
    bottom_row = montage[0, 8:16]
    np.testing.assert_array_equal(bottom_row[:, 8:16], np.full((8, 8, 3), 50 / 255.0, dtype=np.float32))
    np.testing.assert_array_equal(bottom_row[:, 16:24], np.full((8, 8, 3), 50 / 255.0, dtype=np.float32))
    np.testing.assert_array_equal(bottom_row[:, 24:32], np.full((8, 8, 3), 50 / 255.0, dtype=np.float32))
