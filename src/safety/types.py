from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


SAFETY_MODE_TO_CODE = {
    "off": 0.0,
    "sim": 1.0,
    "shadow": 2.0,
    "real": 3.0,
}


@dataclass
class RedlineResult:
    triggered: bool = False
    reason: str = ""
    action: str = "stop"
    count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintEvaluation:
    names: List[str] = field(default_factory=list)
    margins: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    raw_violation_count: int = 0
    proxy_only: bool = False

    @property
    def min_margin(self) -> float:
        if self.margins.size == 0:
            return float("inf")
        return float(np.min(self.margins))

    @property
    def violation_count(self) -> int:
        if self.margins.size == 0:
            return 0
        return int(np.sum(self.margins < 0.0))


@dataclass
class SafetyReport:
    safety_enabled: bool = False
    safety_mode: str = "off"
    safety_triggered: bool = False
    safety_supervisor_preempted: bool = False
    safety_lbsgd_steps: int = 0
    safety_qp_active: bool = False
    safety_min_margin: float = float("inf")
    safety_redline_count: int = 0
    safety_infeasible: bool = False
    safety_stop_reason: str = ""
    safety_qp_infeasible: bool = False
    safety_lbsgd_infeasible: bool = False
    safety_stop_action: bool = False
    safety_raw_action_violation: bool = False
    safety_safe_action_violation: bool = False
    safety_correction_norm: float = 0.0
    safety_warmup_phase: int = 0
    safety_qp_runtime_enabled: bool = False
    safety_lbsgd_runtime_enabled: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "SafetyReport") -> "SafetyReport":
        self.safety_enabled = self.safety_enabled or other.safety_enabled
        if other.safety_mode != "off":
            self.safety_mode = other.safety_mode
        self.safety_triggered = self.safety_triggered or other.safety_triggered
        self.safety_supervisor_preempted = self.safety_supervisor_preempted or other.safety_supervisor_preempted
        self.safety_lbsgd_steps += int(other.safety_lbsgd_steps)
        self.safety_qp_active = self.safety_qp_active or other.safety_qp_active
        self.safety_min_margin = _safe_min(self.safety_min_margin, other.safety_min_margin)
        self.safety_redline_count += int(other.safety_redline_count)
        self.safety_infeasible = self.safety_infeasible or other.safety_infeasible
        self.safety_qp_infeasible = self.safety_qp_infeasible or other.safety_qp_infeasible
        self.safety_lbsgd_infeasible = self.safety_lbsgd_infeasible or other.safety_lbsgd_infeasible
        self.safety_stop_action = self.safety_stop_action or other.safety_stop_action
        self.safety_raw_action_violation = self.safety_raw_action_violation or other.safety_raw_action_violation
        self.safety_safe_action_violation = self.safety_safe_action_violation or other.safety_safe_action_violation
        self.safety_correction_norm = max(float(self.safety_correction_norm), float(other.safety_correction_norm))
        self.safety_warmup_phase = max(int(self.safety_warmup_phase), int(other.safety_warmup_phase))
        self.safety_qp_runtime_enabled = self.safety_qp_runtime_enabled or other.safety_qp_runtime_enabled
        self.safety_lbsgd_runtime_enabled = self.safety_lbsgd_runtime_enabled or other.safety_lbsgd_runtime_enabled
        if other.safety_stop_reason:
            self.safety_stop_reason = other.safety_stop_reason
        self.details.update(other.details)
        return self

    def with_preempt(self, redline: RedlineResult) -> "SafetyReport":
        self.safety_triggered = True
        self.safety_supervisor_preempted = True
        self.safety_redline_count += max(1, int(redline.count or 1))
        self.safety_stop_reason = redline.reason
        self.safety_stop_action = redline.action == "stop"
        if redline.details:
            self.details.update(redline.details)
        return self

    def with_infeasible(self, reason: str, *, source: str) -> "SafetyReport":
        self.safety_triggered = True
        self.safety_infeasible = True
        self.safety_stop_reason = reason
        if source == "lbsgd":
            self.safety_lbsgd_infeasible = True
        if source == "qp":
            self.safety_qp_infeasible = True
        return self

    def to_env_info(self) -> Dict[str, Any]:
        min_margin = self.safety_min_margin
        if min_margin == float("inf"):
            min_margin = np.float32(1e6)
        return {
            "safety_enabled": np.float32(1.0 if self.safety_enabled else 0.0),
            "safety_mode": np.float32(SAFETY_MODE_TO_CODE.get(self.safety_mode, -1.0)),
            "safety_triggered": np.float32(1.0 if self.safety_triggered else 0.0),
            "safety_supervisor_preempted": np.float32(1.0 if self.safety_supervisor_preempted else 0.0),
            "safety_lbsgd_steps": np.float32(self.safety_lbsgd_steps),
            "safety_qp_active": np.float32(1.0 if self.safety_qp_active else 0.0),
            "safety_min_margin": np.float32(min_margin),
            "safety_redline_count": np.float32(self.safety_redline_count),
            "safety_infeasible": np.float32(1.0 if self.safety_infeasible else 0.0),
            "safety_qp_infeasible": np.float32(1.0 if self.safety_qp_infeasible else 0.0),
            "safety_lbsgd_infeasible": np.float32(1.0 if self.safety_lbsgd_infeasible else 0.0),
            "safety_stop_action": np.float32(1.0 if self.safety_stop_action else 0.0),
            "safety_raw_action_violation": np.float32(1.0 if self.safety_raw_action_violation else 0.0),
            "safety_safe_action_violation": np.float32(1.0 if self.safety_safe_action_violation else 0.0),
            "safety_correction_norm": np.float32(self.safety_correction_norm),
            "safety_warmup_phase": np.float32(self.safety_warmup_phase),
            "safety_qp_runtime_enabled": np.float32(1.0 if self.safety_qp_runtime_enabled else 0.0),
            "safety_lbsgd_runtime_enabled": np.float32(1.0 if self.safety_lbsgd_runtime_enabled else 0.0),
            "safety_stop_reason": self.safety_stop_reason,
        }


def _safe_min(a: float, b: float) -> float:
    vals = [x for x in (float(a), float(b)) if np.isfinite(x)]
    if not vals:
        return float("inf")
    return float(min(vals))


def coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "error", "fault", "missing")
    array = np.asarray(value)
    if array.size == 0:
        return False
    if array.dtype == bool:
        return bool(array.reshape(-1).any())
    if np.issubdtype(array.dtype, np.number):
        return bool(np.any(array.reshape(-1) != 0))
    return bool(value)


def flatten_numeric(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if not np.issubdtype(array.dtype, np.number):
        return None
    if array.ndim >= 3:
        return None
    return array.astype(np.float32).reshape(-1)


def first_present(mapping: Dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None
