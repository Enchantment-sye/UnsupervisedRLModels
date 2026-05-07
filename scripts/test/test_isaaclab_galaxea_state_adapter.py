from types import SimpleNamespace

import gym
import numpy as np

from src.envs.isaaclab.adapters.single_agent_box import IsaacLabSingleAgentBoxEnv
from src.envs.isaaclab.base_spec import IsaacLabTaskSpec


class _FakeEnv:
    def __init__(self):
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Dict(
                    {
                        "joint_pos": gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32),
                        "joint_vel": gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32),
                        "left_ee_pose": gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
                        "right_ee_pose": gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
                        "object_pose": gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
                        "goal_pose": gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
                        "last_joints": gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32),
                        "front_rgb": gym.spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8),
                        "front_depth": gym.spaces.Box(0.0, 10.0, shape=(240, 320, 1), dtype=np.float32),
                    }
                )
            }
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)


def _make_wrapper():
    request = SimpleNamespace(
        flatten_obs=0,
        render_size=64,
        seed=0,
        encoder=0,
        image_source="render",
        camera_key=None,
        render_mode="rgb_array",
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
    )
    return IsaacLabSingleAgentBoxEnv(env=_FakeEnv(), task_spec=spec, request=request)


def test_galaxea_state_shape_excludes_images_depth_and_last_joints():
    wrapper = _make_wrapper()
    assert wrapper._infer_state_space().shape == (56,)


def test_galaxea_state_vector_keeps_expected_key_order():
    wrapper = _make_wrapper()
    obs = {
        "policy": {
            "goal_pose": np.full((1, 7), 6.0, dtype=np.float32),
            "joint_pos": np.full((1, 14), 1.0, dtype=np.float32),
            "joint_vel": np.full((1, 14), 2.0, dtype=np.float32),
            "left_ee_pose": np.full((1, 7), 3.0, dtype=np.float32),
            "right_ee_pose": np.full((1, 7), 4.0, dtype=np.float32),
            "object_pose": np.full((1, 7), 5.0, dtype=np.float32),
            "last_joints": np.full((1, 14), 9.0, dtype=np.float32),
            "front_rgb": np.zeros((1, 32, 32, 3), dtype=np.uint8),
            "front_depth": np.zeros((1, 32, 32, 1), dtype=np.float32),
        }
    }
    state = wrapper._extract_state(obs)
    expected = np.concatenate(
        [
            np.full(14, 1.0, dtype=np.float32),
            np.full(14, 2.0, dtype=np.float32),
            np.full(7, 3.0, dtype=np.float32),
            np.full(7, 4.0, dtype=np.float32),
            np.full(7, 5.0, dtype=np.float32),
            np.full(7, 6.0, dtype=np.float32),
        ]
    )
    np.testing.assert_allclose(state, expected)
