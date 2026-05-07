from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .types import SAFETY_MODE_TO_CODE
from .r1lite_state_adapter import is_r1lite_safety_task


def aggregate_safety_metrics(paths, safety_cfg, task: str = "") -> Dict[str, float]:
    cfg_like = type("_SafetyCfg", (), {})()
    cfg_like.safety = safety_cfg
    cfg_like.env = type("_EnvCfg", (), {"task": task})()
    if safety_cfg is not None and not is_r1lite_safety_task(cfg_like):
        safety_cfg = type("_DisabledSafetyCfg", (), {"enabled": 0, "mode": "off"})()
        cfg_like.safety = safety_cfg
    return compute_safety_metrics(paths, cfg_like)


def compute_safety_metrics(paths, cfg) -> Dict[str, float]:
    safety_cfg = getattr(cfg, "safety", cfg)
    enabled = float(bool(int(getattr(safety_cfg, "enabled", 0))))
    mode = str(getattr(safety_cfg, "mode", "off"))
    metrics = {
        "SafetyEnabled": enabled,
        "SafetyMode": float(SAFETY_MODE_TO_CODE.get(mode, -1.0)),
    }
    values = {
        "preempt": [],
        "redline": [],
        "lbsgd_steps": [],
        "lbsgd_infeasible": [],
        "qp_active": [],
        "qp_infeasible": [],
        "correction_norm": [],
        "min_margin": [],
        "stop_action": [],
        "raw_violation": [],
        "safe_violation": [],
    }
    for path in paths:
        env_infos = path.get("env_infos", {}) or {}
        agent_infos = path.get("agent_infos", {}) or {}
        _extend(values["preempt"], env_infos.get("safety_supervisor_preempted"))
        _extend(values["redline"], env_infos.get("safety_redline_count"))
        _extend(values["lbsgd_steps"], env_infos.get("safety_lbsgd_steps"))
        _extend(values["lbsgd_infeasible"], env_infos.get("safety_lbsgd_infeasible"))
        _extend(values["qp_active"], env_infos.get("safety_qp_active"))
        _extend(values["qp_infeasible"], env_infos.get("safety_qp_infeasible"))
        _extend(values["correction_norm"], agent_infos.get("safety_correction_norm", env_infos.get("safety_correction_norm")))
        _extend(values["min_margin"], env_infos.get("safety_min_margin"))
        _extend(values["stop_action"], env_infos.get("safety_stop_action"))
        _extend(values["raw_violation"], env_infos.get("safety_raw_action_violation"))
        _extend(values["safe_violation"], env_infos.get("safety_safe_action_violation"))

    metrics.update({
        "SafetySupervisorPreemptCount": _sum(values["preempt"]),
        "SafetyRedlineCount": _sum(values["redline"]),
        "SafetyLBSGDStepsMean": _mean(values["lbsgd_steps"]),
        "SafetyLBSGDInfeasibleCount": _sum(values["lbsgd_infeasible"]),
        "SafetyQPActiveCount": _sum(values["qp_active"]),
        "SafetyQPInfeasibleCount": _sum(values["qp_infeasible"]),
        "SafetyCorrectionNormMean": _mean(values["correction_norm"]),
        "SafetyCorrectionNormMax": _max(values["correction_norm"]),
        "SafetyMinMarginMean": _mean(values["min_margin"]),
        "SafetyMinMarginMin": _min(values["min_margin"]),
        "SafetyStopActionCount": _sum(values["stop_action"]),
        "SafetyRawActionViolationRate": _mean(values["raw_violation"]),
        "SafetySafeActionViolationRate": _mean(values["safe_violation"]),
    })
    return metrics


def _extend(dst, value):
    if value is None:
        return
    arr = np.asarray(value)
    if arr.dtype == object:
        for item in arr.reshape(-1):
            _extend(dst, item)
        return
    try:
        dst.extend(arr.astype(np.float32).reshape(-1).tolist())
    except Exception:
        return


def _sum(values):
    return float(np.sum(values)) if values else 0.0


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _max(values):
    return float(np.max(values)) if values else 0.0


def _min(values):
    return float(np.min(values)) if values else 0.0
