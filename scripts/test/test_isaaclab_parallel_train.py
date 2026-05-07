import os
import sys
from types import SimpleNamespace

import gym
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.envs.isaaclab.adapters.single_agent_box import IsaacLabSingleAgentBoxEnv
from src.envs.isaaclab.base_spec import IsaacLabTaskSpec
from src.envs.isaaclab.parallel_train import IsaacLabParallelTrajectoryCollector


class _FakeBatchedIsaacEnv:
    def __init__(self, num_envs=4, done_after=2):
        self.num_envs = int(num_envs)
        self.device = "cpu"
        self._done_after = int(done_after)
        self._step_counts = np.zeros(self.num_envs, dtype=np.int32)
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
        self.single_action_space = gym.spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
        self.unwrapped = self

    def _policy_obs(self):
        steps = self._step_counts.astype(np.float32)
        rgb = np.stack(
            [
                np.full((8, 8, 3), fill_value=10 + idx + int(step), dtype=np.uint8)
                for idx, step in enumerate(steps)
            ],
            axis=0,
        )
        return {
            "policy": {
                "joint_pos": np.stack([np.array([1.0 + step, 2.0 + step], dtype=np.float32) for step in steps], axis=0),
                "joint_vel": np.stack([np.array([3.0 + step, 4.0 + step], dtype=np.float32) for step in steps], axis=0),
                "left_ee_pose": np.stack([np.array([5.0 + step], dtype=np.float32) for step in steps], axis=0),
                "right_ee_pose": np.stack([np.array([6.0 + step], dtype=np.float32) for step in steps], axis=0),
                "object_pose": np.stack([np.array([7.0 + step], dtype=np.float32) for step in steps], axis=0),
                "goal_pose": np.stack([np.array([8.0 + step], dtype=np.float32) for step in steps], axis=0),
                "front_rgb": torch.as_tensor(rgb),
            }
        }

    def reset(self, **kwargs):
        self._step_counts[:] = 0
        return self._policy_obs(), {}

    def step(self, action):
        self._step_counts += 1
        terminated = self._step_counts >= self._done_after
        rewards = self._step_counts.astype(np.float32)
        for idx in np.where(terminated)[0]:
            self._step_counts[idx] = 0
        return self._policy_obs(), rewards, terminated, np.zeros_like(terminated), {}

    def _reset_idx(self, env_ids):
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        self._step_counts[env_ids] = 0

    def _get_observations(self):
        return self._policy_obs()

    def close(self):
        pass


class _FakePolicy:
    def __init__(self):
        self._force_use_mode_actions = False

    def reset(self):
        pass

    def get_actions(self, observations):
        if torch.is_tensor(observations):
            batch_size = observations.shape[0]
        else:
            batch_size = np.asarray(observations).shape[0]
        return np.zeros((batch_size, 2), dtype=np.float32), {
            "log_prob": np.zeros((batch_size,), dtype=np.float32),
        }


def _make_request(num_envs=4):
    return SimpleNamespace(
        flatten_obs=1,
        render_size=8,
        seed=0,
        encoder=1,
        image_source="camera",
        camera_key="front_rgb",
        render_mode="rgb_array",
        video_source="observation",
        video_viewer_preset="inherit",
        num_envs=num_envs,
        device="cpu",
    )


def _make_task_spec():
    return IsaacLabTaskSpec(
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
    )


def test_single_agent_wrapper_caches_gpu_training_image_tensor():
    wrapper = IsaacLabSingleAgentBoxEnv(
        env=_FakeBatchedIsaacEnv(num_envs=4),
        task_spec=_make_task_spec(),
        request=_make_request(num_envs=4),
    )

    timestep = wrapper.reset()

    assert timestep["image"].shape == (8 * 8 * 3,)
    train_tensor = wrapper.get_train_image_tensor()
    assert torch.is_tensor(train_tensor)
    assert tuple(train_tensor.shape) == (8 * 8 * 3,)


def test_parallel_collector_collects_requested_number_of_trajectories():
    wrapper = IsaacLabSingleAgentBoxEnv(
        env=_FakeBatchedIsaacEnv(num_envs=4, done_after=2),
        task_spec=_make_task_spec(),
        request=_make_request(num_envs=4),
    )
    handles = wrapper.get_parallel_train_handles()
    cfg = SimpleNamespace(
        seed=0,
        time_limit=2,
        encoder=1,
        stage="pre_training",
        algo=SimpleNamespace(dim_skill=2),
    )
    collector = IsaacLabParallelTrajectoryCollector(cfg, handles)

    trajectories = collector.collect(
        _FakePolicy(),
        target_num_trajectories=3,
        sample_extra_fn=lambda: {"skill": np.array([1.0, 0.0], dtype=np.float32)},
    )

    assert len(trajectories) == 3
    for trajectory in trajectories:
        assert trajectory["observations"].shape[1] == 8 * 8 * 3
        assert trajectory["actions"].shape[1] == 2
        assert trajectory["rewards"].shape[0] == trajectory["dones"].shape[0]
        assert "skill" in trajectory["agent_infos"]

    metrics = collector.consume_timing_metrics()
    assert metrics["TimeSamplingEnv"] >= 0.0
    assert metrics["TimeImagePostprocess"] >= 0.0
