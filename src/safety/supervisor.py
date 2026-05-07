from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .r1lite_state_adapter import SafetyLimits, adapt_safety_state
from .types import RedlineResult, coerce_bool, flatten_numeric


class R1LiteSupervisor:
    def __init__(self, cfg, limits: SafetyLimits, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.limits = limits
        self.logger = logger or logging.getLogger(__name__)
        self._effort_over_count = 0

    @property
    def mode(self) -> str:
        return str(getattr(self.cfg, "mode", "sim"))

    def check(self, safety_state: Any) -> RedlineResult:
        if safety_state is None or not isinstance(safety_state, dict):
            if self.mode == "real" or int(getattr(self.cfg, "stop_on_missing_safety_state", 1)):
                if self.mode == "real":
                    return RedlineResult(True, "missing_safety_state", count=1)
            return RedlineResult(False)

        if self.mode == "real" and not self.limits.real_mode_calibrated:
            return RedlineResult(True, "real_mode_requires_calibrated_safety_yaml", count=1)
        if self.mode == "real":
            real_error = self._real_mode_readiness_error(safety_state)
            if real_error:
                return RedlineResult(True, real_error, count=1)

        for key in ("emergency_stop", "e_stop", "estop", "hardware_estop"):
            if key in safety_state and coerce_bool(safety_state[key]):
                return RedlineResult(True, f"{key}_triggered", count=1)

        for key in ("human_zone_violation", "person_in_safety_zone"):
            if key in safety_state and coerce_bool(safety_state[key]):
                return RedlineResult(True, key, action="retreat", count=1)
        human_distance = flatten_numeric(safety_state.get("human_distance_m"))
        if human_distance is not None and human_distance.size and float(np.min(human_distance)) < self.limits.human_zone_radius:
            return RedlineResult(True, "human_zone_violation", action="retreat", count=1)

        for key in ("camera_missing", "head_rgb_missing", "left_wrist_rgb_missing", "right_wrist_rgb_missing", "depth_missing"):
            if key in safety_state and coerce_bool(safety_state[key]):
                return RedlineResult(True, key, count=1)
        camera_available = safety_state.get("camera_available")
        if isinstance(camera_available, dict):
            for key in ("rgb_head", "rgb_left_hand", "rgb_right_hand"):
                if key in camera_available and not coerce_bool(camera_available[key]):
                    return RedlineResult(True, f"camera_missing:{key}", count=1)

        for key in ("torso_command_conflict", "torso_velocity_and_position_command"):
            if key in safety_state and coerce_bool(safety_state[key]):
                return RedlineResult(True, "torso_command_conflict", count=1)

        for key in ("geofence_violation", "geofence_trip", "workspace_proxy_violation"):
            if key in safety_state and coerce_bool(safety_state[key]):
                return RedlineResult(True, key, action="retreat", count=1)
        for key in ("cable_proxy_violation", "external_cable_violation"):
            if key in safety_state and coerce_bool(safety_state[key]):
                return RedlineResult(True, key, action="retreat", count=1)

        status_error = self._first_status_error(safety_state)
        if status_error:
            return RedlineResult(True, status_error, count=1)

        timeout_error = self._first_timeout_error(safety_state)
        if timeout_error:
            return RedlineResult(True, timeout_error, count=1)

        effort_trip = self._check_effort_trip(safety_state)
        if effort_trip:
            return RedlineResult(True, effort_trip, count=1)

        return RedlineResult(False)

    def check_post_action(self, safety_state: Any, action, *, lbsgd_infeasible=False, qp_infeasible=False) -> RedlineResult:
        if lbsgd_infeasible:
            return RedlineResult(True, "lbsgd_infeasible", count=1)
        if qp_infeasible:
            return RedlineResult(True, "qp_infeasible", count=1)
        return self.check(safety_state)

    def _first_status_error(self, safety_state: Dict[str, Any]) -> str:
        status_keys = [
            key for key in safety_state
            if "status" in str(key).lower() or "diagnostic" in str(key).lower() or str(key).lower() in ("can_error", "hdas_error")
        ]
        for key in status_keys:
            value = safety_state[key]
            if isinstance(value, str):
                if value.strip().lower() not in ("", "ok", "normal", "0", "none"):
                    return f"status_error:{key}"
                continue
            arr = flatten_numeric(value)
            if arr is not None and arr.size and np.any(arr != 0):
                return f"status_error:{key}"
            if arr is None and coerce_bool(value):
                return f"status_error:{key}"
        return ""

    def _real_mode_readiness_error(self, safety_state: Dict[str, Any]) -> str:
        dim = max(1, int(self.limits.layout.controlled_q_dim))
        required_limits = {
            "joint_limits": (self.limits.q_min, self.limits.q_max),
            "velocity_limits": (self.limits.dq_max,),
            "acceleration_limits": (self.limits.ddq_max,),
        }
        for name, arrays in required_limits.items():
            for arr in arrays:
                numeric = flatten_numeric(arr)
                if numeric is None or numeric.size < dim or not np.all(np.isfinite(numeric[:dim])):
                    return f"real_mode_missing_{name}"
        for key in ("left_arm_joint_position", "right_arm_joint_position"):
            if key not in safety_state:
                return f"real_mode_missing_{key}"
        has_status = any("status" in str(key).lower() or "diagnostic" in str(key).lower() for key in safety_state)
        if not has_status:
            return "real_mode_missing_status_check"
        has_watchdog = (
            "topic_age_s" in safety_state or
            "topic_ages_s" in safety_state or
            any("timeout" in str(key).lower() or "hz_error" in str(key).lower() for key in safety_state)
        )
        if not has_watchdog:
            return "real_mode_missing_topic_watchdog"
        return ""

    def _first_timeout_error(self, safety_state: Dict[str, Any]) -> str:
        timeout_keys = [key for key in safety_state if "timeout" in str(key).lower() or "hz_error" in str(key).lower()]
        for key in timeout_keys:
            if coerce_bool(safety_state[key]):
                return f"topic_timeout:{key}"
        topic_age = safety_state.get("topic_age_s", safety_state.get("topic_ages_s"))
        if isinstance(topic_age, dict):
            for key, value in topic_age.items():
                arr = flatten_numeric(value)
                if arr is not None and arr.size and float(np.max(arr)) > self.limits.topic_timeout_s:
                    return f"topic_timeout:{key}"
        return ""

    def _check_effort_trip(self, safety_state: Dict[str, Any]) -> str:
        adapted = adapt_safety_state(safety_state, layout=self.limits.layout)
        if adapted.effort is None or self.limits.tau_trip is None:
            self._effort_over_count = 0
            return ""
        effort = np.asarray(adapted.effort, dtype=np.float32).reshape(-1)
        dim = min(effort.size, self.limits.tau_trip.size)
        if dim == 0:
            self._effort_over_count = 0
            return ""
        over = bool(np.any(np.abs(effort[:dim]) > self.limits.tau_trip[:dim]))
        self._effort_over_count = self._effort_over_count + 1 if over else 0
        if self._effort_over_count >= max(1, int(self.limits.effort_trip_frames)):
            return "effort_trip"
        return ""
