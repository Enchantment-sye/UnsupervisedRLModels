from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, Optional, Protocol

import numpy as np

from safety import build_safety_controller


@dataclass
class R1ObservationBuffer:
    left_arm_joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    right_arm_joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    left_gripper: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    right_gripper: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    torso_joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    left_arm_joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    right_arm_joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    torso_joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    left_arm_effort: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    right_arm_effort: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    left_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    right_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    object_pose: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    goal_pose: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    front_rgb: Optional[np.ndarray] = None
    head_rgb: Optional[np.ndarray] = None
    left_wrist_rgb: Optional[np.ndarray] = None
    right_wrist_rgb: Optional[np.ndarray] = None
    status: Dict[str, int] = field(default_factory=dict)
    topic_age_s: Dict[str, float] = field(default_factory=dict)
    stamp_sec: float = 0.0

    def as_state_vector(self) -> np.ndarray:
        parts = (
            self.left_arm_joint_pos,
            self.right_arm_joint_pos,
            self.left_gripper,
            self.right_gripper,
            self.torso_joint_pos,
            self.left_ee_pose,
            self.right_ee_pose,
            self.object_pose,
            self.goal_pose,
        )
        return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts], axis=0)

    def as_safety_state(self) -> Dict[str, object]:
        return {
            "left_arm_joint_position": self.left_arm_joint_pos.copy(),
            "right_arm_joint_position": self.right_arm_joint_pos.copy(),
            "left_arm_gripper_position": self.left_gripper.copy(),
            "right_arm_gripper_position": self.right_gripper.copy(),
            "left_arm_joint_velocity": self.left_arm_joint_vel.copy(),
            "right_arm_joint_velocity": self.right_arm_joint_vel.copy(),
            "torso_joint_position": self.torso_joint_pos.copy(),
            "torso_joint_velocity": self.torso_joint_vel.copy(),
            "left_arm_effort": self.left_arm_effort.copy(),
            "right_arm_effort": self.right_arm_effort.copy(),
            "left_arm_ee_pose": self.left_ee_pose.copy(),
            "right_arm_ee_pose": self.right_ee_pose.copy(),
            "status": dict(self.status),
            "topic_age_s": dict(self.topic_age_s),
            "camera_available": {
                "rgb_head": self.head_rgb is not None or self.front_rgb is not None,
                "rgb_left_hand": self.left_wrist_rgb is not None,
                "rgb_right_hand": self.right_wrist_rgb is not None,
            },
            "timestamp": float(self.stamp_sec),
        }


@dataclass
class R1BridgeOutputs:
    left_arm_joint_target: np.ndarray
    right_arm_joint_target: np.ndarray
    torso_joint_target: np.ndarray
    left_gripper_target: float
    right_gripper_target: float

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "left_arm_joint_target": self.left_arm_joint_target,
            "right_arm_joint_target": self.right_arm_joint_target,
            "torso_joint_target": self.torso_joint_target,
            "left_gripper_target": np.asarray([self.left_gripper_target], dtype=np.float32),
            "right_gripper_target": np.asarray([self.right_gripper_target], dtype=np.float32),
        }


class PolicyRuntime(Protocol):
    def predict(self, *, state: np.ndarray, image: Optional[np.ndarray] = None) -> R1BridgeOutputs:
        """Return fixed-workcell arm/gripper/torso targets from the latest observation."""


class R1WorkbenchPolicyBridge:
    """Bridge-side state assembler for R1 fixed-workstation inference.

    This layer is intentionally separate from the training loop so the ROS2 topic
    plumbing can evolve without affecting METRA rollout/training code.
    """

    def __init__(self, mode: str = "state", *, tri_view: bool = False, safety_config=None, dry_run: bool = True):
        if mode not in ("state", "pixel"):
            raise ValueError(f"Unsupported bridge mode: {mode!r}")
        self.mode = mode
        self.tri_view = bool(tri_view)
        self.dry_run = bool(dry_run)
        self.buffer = R1ObservationBuffer()
        self._safety_controller = build_safety_controller(safety_config, env=None) if safety_config is not None else None
        self._prev_safe_action = None
        self._has_joint_state = False
        self._has_gripper_state = False
        self._has_image = False

    def update_joint_positions(self, *, left_arm, right_arm, torso, left_vel=None, right_vel=None, torso_vel=None,
                               left_effort=None, right_effort=None):
        self.buffer.left_arm_joint_pos = np.asarray(left_arm, dtype=np.float32).reshape(-1)
        self.buffer.right_arm_joint_pos = np.asarray(right_arm, dtype=np.float32).reshape(-1)
        self.buffer.torso_joint_pos = np.asarray(torso, dtype=np.float32).reshape(-1)
        if left_vel is not None:
            self.buffer.left_arm_joint_vel = np.asarray(left_vel, dtype=np.float32).reshape(-1)
        if right_vel is not None:
            self.buffer.right_arm_joint_vel = np.asarray(right_vel, dtype=np.float32).reshape(-1)
        if torso_vel is not None:
            self.buffer.torso_joint_vel = np.asarray(torso_vel, dtype=np.float32).reshape(-1)
        if left_effort is not None:
            self.buffer.left_arm_effort = np.asarray(left_effort, dtype=np.float32).reshape(-1)
        if right_effort is not None:
            self.buffer.right_arm_effort = np.asarray(right_effort, dtype=np.float32).reshape(-1)
        self._has_joint_state = True

    def update_grippers(self, *, left_gripper, right_gripper):
        self.buffer.left_gripper = np.asarray(left_gripper, dtype=np.float32).reshape(-1)
        self.buffer.right_gripper = np.asarray(right_gripper, dtype=np.float32).reshape(-1)
        self._has_gripper_state = True

    def update_end_effector_pose(self, *, left_pose, right_pose):
        self.buffer.left_ee_pose = np.asarray(left_pose, dtype=np.float32).reshape(-1)
        self.buffer.right_ee_pose = np.asarray(right_pose, dtype=np.float32).reshape(-1)

    def update_task_poses(self, *, object_pose, goal_pose):
        self.buffer.object_pose = np.asarray(object_pose, dtype=np.float32).reshape(-1)
        self.buffer.goal_pose = np.asarray(goal_pose, dtype=np.float32).reshape(-1)

    def update_front_rgb(self, image, *, stamp_sec: float = 0.0):
        self.buffer.front_rgb = np.asarray(image, dtype=np.uint8)
        self.buffer.head_rgb = self.buffer.front_rgb
        self.buffer.stamp_sec = float(stamp_sec)
        self._has_image = True

    def update_triview_rgb(self, *, left_wrist=None, right_wrist=None, head=None, stamp_sec: float = 0.0):
        if left_wrist is not None:
            self.buffer.left_wrist_rgb = np.asarray(left_wrist, dtype=np.uint8)
        if right_wrist is not None:
            self.buffer.right_wrist_rgb = np.asarray(right_wrist, dtype=np.uint8)
        if head is not None:
            self.buffer.head_rgb = np.asarray(head, dtype=np.uint8)
            self.buffer.front_rgb = self.buffer.head_rgb
        self.buffer.stamp_sec = float(stamp_sec)
        self._has_image = True

    def update_status(self, **status):
        for key, value in status.items():
            try:
                self.buffer.status[key] = int(value)
            except Exception:
                self.buffer.status[key] = 1

    def update_topic_age(self, **topic_age_s):
        self.buffer.topic_age_s.update({key: float(value) for key, value in topic_age_s.items()})

    def ready(self) -> bool:
        if not (self._has_joint_state and self._has_gripper_state):
            return False
        if self.mode == "pixel":
            if self.tri_view:
                return (
                    self.buffer.left_wrist_rgb is not None and
                    self.buffer.right_wrist_rgb is not None and
                    (self.buffer.head_rgb is not None or self.buffer.front_rgb is not None)
                )
            return self._has_image and self.buffer.front_rgb is not None
        return True

    def build_policy_inputs(self):
        image = None
        if self.mode == "pixel":
            if self.tri_view:
                head = self.buffer.head_rgb if self.buffer.head_rgb is not None else self.buffer.front_rgb
                image = np.concatenate(
                    [self.buffer.left_wrist_rgb, self.buffer.right_wrist_rgb, head],
                    axis=-1,
                )
            else:
                image = self.buffer.front_rgb
        return {
            "state": self.buffer.as_state_vector(),
            "image": image,
        }

    def step(self, runtime: PolicyRuntime) -> R1BridgeOutputs:
        if not self.ready():
            raise RuntimeError("Bridge does not have enough sensor data for the configured mode.")
        inputs = self.build_policy_inputs()
        raw_outputs = runtime.predict(state=inputs["state"], image=inputs["image"])
        return self._filter_outputs(raw_outputs)

    def _filter_outputs(self, outputs: R1BridgeOutputs) -> R1BridgeOutputs:
        if self._safety_controller is None:
            return outputs
        raw_action = outputs_to_r1lite_action(outputs)
        safety_state = self.buffer.as_safety_state()
        if not (self.mode == "pixel" and self.tri_view):
            safety_state.pop("camera_available", None)
        safe_action, _report = self._safety_controller.filter_action(
            raw_action=raw_action,
            safety_state=safety_state,
            policy_obs=self.build_policy_inputs(),
            prev_action=self._prev_safe_action,
        )
        self._prev_safe_action = np.asarray(safe_action, dtype=np.float32).copy()
        return outputs_from_r1lite_action(safe_action, torso_joint_target=outputs.torso_joint_target)


def outputs_to_r1lite_action(outputs: R1BridgeOutputs) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(outputs.left_arm_joint_target, dtype=np.float32).reshape(-1)[:6],
            np.asarray(outputs.right_arm_joint_target, dtype=np.float32).reshape(-1)[:6],
            np.asarray([outputs.left_gripper_target], dtype=np.float32),
            np.asarray([outputs.right_gripper_target], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def outputs_from_r1lite_action(action, *, torso_joint_target) -> R1BridgeOutputs:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    left = np.zeros(6, dtype=np.float32)
    right = np.zeros(6, dtype=np.float32)
    left[:min(6, action.size)] = action[:min(6, action.size)]
    if action.size > 6:
        right[:min(6, action.size - 6)] = action[6:12]
    left_gripper = float(action[12]) if action.size > 12 else 0.0
    right_gripper = float(action[13]) if action.size > 13 else 0.0
    return R1BridgeOutputs(
        left_arm_joint_target=left,
        right_arm_joint_target=right,
        torso_joint_target=np.asarray(torso_joint_target, dtype=np.float32).reshape(-1),
        left_gripper_target=left_gripper,
        right_gripper_target=right_gripper,
    )


try:  # pragma: no cover - depends on ROS2 runtime availability
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray
except ImportError:  # pragma: no cover
    rclpy = None
    Node = object
    Image = JointState = Float32 = Float32MultiArray = Int32MultiArray = None


class Ros2R1WorkbenchBridgeNode(Node):  # pragma: no cover - integration-only
    """Reference ROS2 node for fixed-workcell inference.

    The node publishes bridge-level relay topics rather than directly binding to
    vendor-specific SDK command messages. This keeps the policy bridge safe to run
    in dry-run mode and makes it easier to add the final driver adapters later.
    """

    def __init__(self, runtime: PolicyRuntime, mode: str = "state", *, tri_view: bool = False,
                 safety_config=None, dry_run: bool = True):
        if rclpy is None:
            raise ImportError("ROS2 runtime is unavailable. Source the ROS2 Humble environment first.")
        super().__init__("metra_r1_workbench_bridge")
        self._runtime = runtime
        self._dry_run = bool(dry_run)
        self._bridge = R1WorkbenchPolicyBridge(
            mode=mode,
            tri_view=tri_view,
            safety_config=safety_config,
            dry_run=dry_run,
        )

        self._arm_sub = self.create_subscription(JointState, "/metra_bridge/input/arm_joint_state", self._on_arm_state, 10)
        self._torso_sub = self.create_subscription(JointState, "/metra_bridge/input/torso_joint_state", self._on_torso_state, 10)
        self._gripper_sub = self.create_subscription(JointState, "/metra_bridge/input/gripper_state", self._on_gripper_state, 10)
        self._image_sub = self.create_subscription(Image, "/metra_bridge/input/front_rgb", self._on_front_rgb, 10)
        self._head_image_sub = self.create_subscription(Image, "/metra_bridge/input/head_rgb", self._on_head_rgb, 10)
        self._left_wrist_image_sub = self.create_subscription(Image, "/metra_bridge/input/left_wrist_rgb", self._on_left_wrist_rgb, 10)
        self._right_wrist_image_sub = self.create_subscription(Image, "/metra_bridge/input/right_wrist_rgb", self._on_right_wrist_rgb, 10)
        self._status_sub = self.create_subscription(Int32MultiArray, "/metra_bridge/input/status", self._on_status, 10)

        self._left_arm_pub = self.create_publisher(Float32MultiArray, "/metra_bridge/output/left_arm_joint_target", 10)
        self._right_arm_pub = self.create_publisher(Float32MultiArray, "/metra_bridge/output/right_arm_joint_target", 10)
        self._torso_pub = self.create_publisher(Float32MultiArray, "/metra_bridge/output/torso_joint_target", 10)
        self._left_gripper_pub = self.create_publisher(Float32, "/metra_bridge/output/left_gripper_target", 10)
        self._right_gripper_pub = self.create_publisher(Float32, "/metra_bridge/output/right_gripper_target", 10)

    def _on_arm_state(self, msg: JointState):
        positions = np.asarray(msg.position, dtype=np.float32)
        if positions.size < 12:
            return
        velocities = np.asarray(msg.velocity, dtype=np.float32) if len(msg.velocity) >= 12 else None
        efforts = np.asarray(msg.effort, dtype=np.float32) if len(msg.effort) >= 12 else None
        self._bridge.update_joint_positions(
            left_arm=positions[:6],
            right_arm=positions[6:12],
            torso=self._bridge.buffer.torso_joint_pos,
            left_vel=velocities[:6] if velocities is not None else None,
            right_vel=velocities[6:12] if velocities is not None else None,
            left_effort=efforts[:6] if efforts is not None else None,
            right_effort=efforts[6:12] if efforts is not None else None,
        )
        self._maybe_publish()

    def _on_torso_state(self, msg: JointState):
        self._bridge.update_joint_positions(
            left_arm=self._bridge.buffer.left_arm_joint_pos,
            right_arm=self._bridge.buffer.right_arm_joint_pos,
            torso=np.asarray(msg.position, dtype=np.float32),
            torso_vel=np.asarray(msg.velocity, dtype=np.float32) if len(msg.velocity) else None,
        )
        self._maybe_publish()

    def _on_gripper_state(self, msg: JointState):
        positions = np.asarray(msg.position, dtype=np.float32)
        if positions.size < 2:
            return
        self._bridge.update_grippers(left_gripper=positions[:1], right_gripper=positions[1:2])
        self._maybe_publish()

    def _on_front_rgb(self, msg: Image):
        image = self._decode_image(msg)
        self._bridge.update_front_rgb(image, stamp_sec=float(msg.header.stamp.sec))
        self._maybe_publish()

    def _on_head_rgb(self, msg: Image):
        self._bridge.update_triview_rgb(head=self._decode_image(msg), stamp_sec=float(msg.header.stamp.sec))
        self._maybe_publish()

    def _on_left_wrist_rgb(self, msg: Image):
        self._bridge.update_triview_rgb(left_wrist=self._decode_image(msg), stamp_sec=float(msg.header.stamp.sec))
        self._maybe_publish()

    def _on_right_wrist_rgb(self, msg: Image):
        self._bridge.update_triview_rgb(right_wrist=self._decode_image(msg), stamp_sec=float(msg.header.stamp.sec))
        self._maybe_publish()

    def _on_status(self, msg: Int32MultiArray):
        values = list(msg.data)
        keys = (
            "feedback_status_arm_left",
            "feedback_status_arm_right",
            "feedback_status_torso",
            "feedback_status_chassis",
            "feedback_status_gripper",
        )
        self._bridge.update_status(**{key: values[idx] for idx, key in enumerate(keys) if idx < len(values)})
        self._maybe_publish()

    @staticmethod
    def _decode_image(msg: Image):
        image = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.height > 0 and msg.width > 0:
            image = image.reshape(msg.height, msg.width, -1)[..., :3]
        return image

    def _publish_array(self, publisher, values):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in np.asarray(values, dtype=np.float32).reshape(-1)]
        publisher.publish(msg)

    def _maybe_publish(self):
        if not self._bridge.ready():
            return
        outputs = self._bridge.step(self._runtime)
        if self._dry_run:
            return
        self._publish_array(self._left_arm_pub, outputs.left_arm_joint_target)
        self._publish_array(self._right_arm_pub, outputs.right_arm_joint_target)
        self._publish_array(self._torso_pub, outputs.torso_joint_target)

        left_gripper = Float32()
        left_gripper.data = float(outputs.left_gripper_target)
        self._left_gripper_pub.publish(left_gripper)

        right_gripper = Float32()
        right_gripper.data = float(outputs.right_gripper_target)
        self._right_gripper_pub.publish(right_gripper)
