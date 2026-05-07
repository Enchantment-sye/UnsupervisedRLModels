import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from ros2_bridge import R1BridgeOutputs, R1WorkbenchPolicyBridge
from src.envs.isaaclab.registry import get_task_spec


class _DummyRuntime:
    def predict(self, *, state, image=None):
        assert state.ndim == 1
        if image is not None:
            assert image.shape == (16, 16, 3)
        return R1BridgeOutputs(
            left_arm_joint_target=np.ones(6, dtype=np.float32),
            right_arm_joint_target=np.ones(6, dtype=np.float32) * 2.0,
            torso_joint_target=np.ones(3, dtype=np.float32) * 3.0,
            left_gripper_target=0.25,
            right_gripper_target=0.75,
        )


def test_r1_registry_defaults():
    lift_spec = get_task_spec("isaaclab_r1_lift_bin")
    assert lift_spec.env_id == "Isaac-R1-Lift-Bin-IK-Rel-Direct-v0"
    assert lift_spec.camera_obs_key == "front_rgb"
    assert lift_spec.default_image_source_encoder1 == "camera"


def test_r1_bridge_state_and_pixel_modes():
    runtime = _DummyRuntime()

    state_bridge = R1WorkbenchPolicyBridge(mode="state")
    state_bridge.update_joint_positions(left_arm=np.zeros(6), right_arm=np.zeros(6), torso=np.zeros(3))
    state_bridge.update_grippers(left_gripper=[0.0], right_gripper=[0.0])
    state_bridge.update_end_effector_pose(left_pose=np.zeros(7), right_pose=np.zeros(7))
    state_bridge.update_task_poses(object_pose=np.zeros(7), goal_pose=np.zeros(7))
    outputs = state_bridge.step(runtime)
    assert outputs.left_gripper_target == 0.25

    pixel_bridge = R1WorkbenchPolicyBridge(mode="pixel")
    pixel_bridge.update_joint_positions(left_arm=np.zeros(6), right_arm=np.zeros(6), torso=np.zeros(3))
    pixel_bridge.update_grippers(left_gripper=[0.0], right_gripper=[0.0])
    pixel_bridge.update_end_effector_pose(left_pose=np.zeros(7), right_pose=np.zeros(7))
    pixel_bridge.update_task_poses(object_pose=np.zeros(7), goal_pose=np.zeros(7))
    pixel_bridge.update_front_rgb(np.zeros((16, 16, 3), dtype=np.uint8))
    pixel_outputs = pixel_bridge.step(runtime)
    assert pixel_outputs.right_gripper_target == 0.75
