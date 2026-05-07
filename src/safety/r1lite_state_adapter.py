from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .types import flatten_numeric


R1LITE_DEFAULT_LAYOUT = {
    "left_arm": [0, 6],
    "right_arm": [6, 12],
    "left_gripper": [12, 13],
    "right_gripper": [13, 14],
    "torso": [],
    "chassis": [],
}


@dataclass
class ActionLayout:
    left_arm: Tuple[int, int] = (0, 6)
    right_arm: Tuple[int, int] = (6, 12)
    left_gripper: Tuple[int, int] = (12, 13)
    right_gripper: Tuple[int, int] = (13, 14)
    torso: Tuple[int, int] = (0, 0)
    chassis: Tuple[int, int] = (0, 0)

    @property
    def controlled_q_dim(self) -> int:
        return max(self.right_gripper[1], self.left_gripper[1], self.right_arm[1], self.left_arm[1])


@dataclass
class SafetyLimits:
    q_min: Optional[np.ndarray] = None
    q_max: Optional[np.ndarray] = None
    dq_max: Optional[np.ndarray] = None
    ddq_max: Optional[np.ndarray] = None
    jerk_max: Optional[np.ndarray] = None
    tau_max_safe: Optional[np.ndarray] = None
    tau_trip: Optional[np.ndarray] = None
    gripper_min: Optional[np.ndarray] = None
    gripper_max: Optional[np.ndarray] = None
    action_low: Optional[np.ndarray] = None
    action_high: Optional[np.ndarray] = None
    human_zone_radius: float = 1.5
    topic_timeout_s: float = 0.5
    effort_trip_frames: int = 3
    real_mode_calibrated: bool = False
    collision_backend: str = "proxy"
    proxy_only_collision: bool = True
    layout: ActionLayout = field(default_factory=ActionLayout)
    raw_yaml: Dict[str, Any] = field(default_factory=dict)

    def ensure_dim(self, dim: int) -> "SafetyLimits":
        def _fill(value, default):
            if value is None or np.asarray(value).size == 0:
                return np.full((dim,), default, dtype=np.float32)
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size < dim:
                pad = np.full((dim - arr.size,), default, dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            return arr[:dim].astype(np.float32)

        low_default = -np.inf
        high_default = np.inf
        if self.action_low is not None and np.asarray(self.action_low).size >= dim:
            low_default = np.asarray(self.action_low, dtype=np.float32).reshape(-1)[:dim]
        if self.action_high is not None and np.asarray(self.action_high).size >= dim:
            high_default = np.asarray(self.action_high, dtype=np.float32).reshape(-1)[:dim]

        self.q_min = _fill(self.q_min, low_default if np.isscalar(low_default) else 0.0)
        self.q_max = _fill(self.q_max, high_default if np.isscalar(high_default) else 0.0)
        if not np.isscalar(low_default):
            self.q_min = np.maximum(self.q_min, low_default[:dim])
        if not np.isscalar(high_default):
            self.q_max = np.minimum(self.q_max, high_default[:dim])
        self.dq_max = _fill(self.dq_max, 1.0)
        self.ddq_max = _fill(self.ddq_max, 2.0)
        self.jerk_max = _fill(self.jerk_max, np.inf)
        self.tau_max_safe = _fill(self.tau_max_safe, np.inf)
        self.tau_trip = _fill(self.tau_trip, np.inf)
        self.action_low = _fill(self.action_low, -np.inf)
        self.action_high = _fill(self.action_high, np.inf)
        if self.gripper_min is None:
            self.gripper_min = np.zeros((2,), dtype=np.float32)
        if self.gripper_max is None:
            self.gripper_max = np.ones((2,), dtype=np.float32)
        return self


@dataclass
class R1LiteSafetyState:
    raw: Dict[str, Any]
    q: Optional[np.ndarray] = None
    dq: Optional[np.ndarray] = None
    effort: Optional[np.ndarray] = None
    left_ee_pose: Optional[np.ndarray] = None
    right_ee_pose: Optional[np.ndarray] = None
    timestamp: float = 0.0


def is_r1lite_safety_task(config_or_task: Any) -> bool:
    task = getattr(config_or_task, "task", config_or_task)
    if not isinstance(task, str):
        env_cfg = getattr(config_or_task, "env", None)
        task = getattr(env_cfg, "task", task)
    if not isinstance(task, str):
        return False
    lowered = task.lower()
    return lowered.startswith("galaxea_r1lite") or lowered.startswith("r1lite_")


def _as_tuple_range(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None or value == []:
        return (0, 0)
    if isinstance(value, dict):
        return (int(value.get("start", default[0])), int(value.get("stop", default[1])))
    arr = list(value)
    if len(arr) != 2:
        return default
    return (int(arr[0]), int(arr[1]))


def load_safety_limits(path: str = "", *, action_low=None, action_high=None) -> SafetyLimits:
    data: Dict[str, Any] = {}
    if path:
        expanded = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"Safety YAML not found: {expanded}")
        if yaml is None:
            raise ImportError("pyyaml is required to read safety_yaml")
        with open(expanded, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    layout_data = data.get("action_layout", {}) if isinstance(data, dict) else {}
    layout = ActionLayout(
        left_arm=_as_tuple_range(layout_data.get("left_arm", R1LITE_DEFAULT_LAYOUT["left_arm"]), (0, 6)),
        right_arm=_as_tuple_range(layout_data.get("right_arm", R1LITE_DEFAULT_LAYOUT["right_arm"]), (6, 12)),
        left_gripper=_as_tuple_range(layout_data.get("left_gripper", R1LITE_DEFAULT_LAYOUT["left_gripper"]), (12, 13)),
        right_gripper=_as_tuple_range(layout_data.get("right_gripper", R1LITE_DEFAULT_LAYOUT["right_gripper"]), (13, 14)),
        torso=_as_tuple_range(layout_data.get("torso", []), (0, 0)),
        chassis=_as_tuple_range(layout_data.get("chassis", []), (0, 0)),
    )

    limits = data.get("limits", {}) if isinstance(data, dict) else {}
    gripper = limits.get("gripper", {}) if isinstance(limits, dict) else {}
    watchdog = data.get("watchdog", {}) if isinstance(data, dict) else {}
    collision = data.get("collision", {}) if isinstance(data, dict) else {}
    mode_restrictions = data.get("mode_restrictions", {}) if isinstance(data, dict) else {}

    ret = SafetyLimits(
        q_min=_array_or_none(limits.get("q_min")),
        q_max=_array_or_none(limits.get("q_max")),
        dq_max=_array_or_none(limits.get("dq_max")),
        ddq_max=_array_or_none(limits.get("ddq_max")),
        jerk_max=_array_or_none(limits.get("jerk_max")),
        tau_max_safe=_array_or_none(limits.get("tau_max_safe")),
        tau_trip=_array_or_none(limits.get("tau_trip")),
        gripper_min=_array_or_none(gripper.get("min")),
        gripper_max=_array_or_none(gripper.get("max")),
        action_low=_array_or_none(action_low),
        action_high=_array_or_none(action_high),
        human_zone_radius=float(watchdog.get("human_zone_radius_m", 1.5)),
        topic_timeout_s=float(watchdog.get("topic_timeout_s", 0.5)),
        effort_trip_frames=int(watchdog.get("effort_trip_frames", 3)),
        real_mode_calibrated=bool(mode_restrictions.get("real_mode_calibrated", False)),
        collision_backend=str(collision.get("backend", "proxy")),
        proxy_only_collision=str(collision.get("backend", "proxy")).lower() != "full",
        layout=layout,
        raw_yaml=data,
    )
    return ret


def _array_or_none(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = flatten_numeric(value)
    if arr is None:
        return None
    return arr.astype(np.float32)


def adapt_safety_state(raw_state: Any, *, layout: Optional[ActionLayout] = None) -> R1LiteSafetyState:
    raw = raw_state if isinstance(raw_state, dict) else {}
    layout = layout or ActionLayout()
    q_parts = [
        _first_numeric(raw, ("left_arm_joint_position", "left_arm_joint_pos", "left_arm_qpos", "left_arm_joints")),
        _first_numeric(raw, ("right_arm_joint_position", "right_arm_joint_pos", "right_arm_qpos", "right_arm_joints")),
        _first_numeric(raw, ("left_arm_gripper_position", "left_gripper_position", "left_gripper")),
        _first_numeric(raw, ("right_arm_gripper_position", "right_gripper_position", "right_gripper")),
    ]
    dq_parts = [
        _first_numeric(raw, ("left_arm_joint_velocity", "left_arm_joint_vel", "left_arm_qvel")),
        _first_numeric(raw, ("right_arm_joint_velocity", "right_arm_joint_vel", "right_arm_qvel")),
        _first_numeric(raw, ("left_arm_gripper_velocity", "left_gripper_velocity")),
        _first_numeric(raw, ("right_arm_gripper_velocity", "right_gripper_velocity")),
    ]
    effort_parts = [
        _first_numeric(raw, ("left_arm_effort", "left_arm_joint_effort", "left_arm_torque")),
        _first_numeric(raw, ("right_arm_effort", "right_arm_joint_effort", "right_arm_torque")),
        _first_numeric(raw, ("left_gripper_effort", "left_arm_gripper_effort")),
        _first_numeric(raw, ("right_gripper_effort", "right_arm_gripper_effort")),
    ]
    q = _concat_available(q_parts)
    dq = _concat_available(dq_parts)
    effort = _concat_available(effort_parts)
    return R1LiteSafetyState(
        raw=raw,
        q=q,
        dq=dq,
        effort=effort,
        left_ee_pose=_first_numeric(raw, ("left_arm_ee_pose", "left_ee_pose")),
        right_ee_pose=_first_numeric(raw, ("right_arm_ee_pose", "right_ee_pose")),
        timestamp=float(np.asarray(raw.get("timestamp", raw.get("stamp_sec", 0.0))).reshape(-1)[0])
        if raw.get("timestamp", raw.get("stamp_sec", None)) is not None
        else 0.0,
    )


def _first_numeric(mapping: Dict[str, Any], names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        arr = flatten_numeric(mapping.get(name))
        if arr is not None:
            return arr
    return None


def _concat_available(parts: Iterable[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    arrays = [np.asarray(part, dtype=np.float32).reshape(-1) for part in parts if part is not None]
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0).astype(np.float32)
