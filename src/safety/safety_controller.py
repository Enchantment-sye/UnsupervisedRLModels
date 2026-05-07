from __future__ import annotations

import logging
import os
import math
from typing import Callable, Optional, Tuple

import numpy as np

from .constraints import evaluate_numpy, make_constraint_context, stop_action_physical
from .lbsgd_governor import LBSGDGovernor
from .qp_shield import QPShield
from .r1lite_state_adapter import is_r1lite_safety_task, load_safety_limits
from .supervisor import R1LiteSupervisor
from .types import SafetyReport


def safety_enabled_for_config(config) -> bool:
    safety_cfg = getattr(config, "safety", config)
    if not bool(int(getattr(safety_cfg, "enabled", 0))):
        return False
    if str(getattr(safety_cfg, "mode", "sim")) == "off":
        return False
    return is_r1lite_safety_task(config)


def build_safety_controller(config, *, env=None, logger: Optional[logging.Logger] = None) -> Optional["SafetyController"]:
    if not safety_enabled_for_config(config):
        return None
    return SafetyController(config, env=env, logger=logger)


class SafetyController:
    def __init__(self, config, *, env=None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.cfg = getattr(config, "safety", config)
        self.logger = logger or logging.getLogger(__name__)
        action_low, action_high = self._resolve_physical_action_bounds(env)
        self.limits = load_safety_limits(getattr(self.cfg, "safety_yaml", ""), action_low=action_low, action_high=action_high)
        if action_low is not None:
            self.limits.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        if action_high is not None:
            self.limits.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.supervisor = R1LiteSupervisor(self.cfg, self.limits, logger=self.logger)
        self.lbsgd = LBSGDGovernor(self.cfg, logger=self.logger)
        self.qp = QPShield(self.cfg, logger=self.logger)
        self.prev_safe_physical_action = None
        self.prev2_safe_physical_action = None
        self._filter_calls = 0
        self._last_qp_enabled = False
        self._last_lbsgd_enabled = False

    @property
    def enabled(self) -> bool:
        return bool(int(getattr(self.cfg, "enabled", 0))) and str(getattr(self.cfg, "mode", "sim")) != "off"

    def filter_action(
            self,
            *,
            raw_action,
            safety_state,
            policy_obs=None,
            prev_action=None,
            action_to_physical: Optional[Callable] = None,
            action_from_physical: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, SafetyReport]:
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        report = SafetyReport(safety_enabled=self.enabled, safety_mode=str(getattr(self.cfg, "mode", "sim")))
        if not self.enabled:
            return raw_action, report
        step_index = self._filter_calls
        self._filter_calls += 1
        schedule = self._runtime_schedule(step_index)
        report.safety_warmup_phase = schedule["phase"]
        report.safety_qp_runtime_enabled = schedule["qp_enabled"]
        report.safety_lbsgd_runtime_enabled = schedule["lbsgd_enabled"]

        if (
                (schedule["qp_enabled"] and not self._last_qp_enabled)
                or (schedule["lbsgd_enabled"] and not self._last_lbsgd_enabled)
        ):
            self._reset_action_history()
        self._last_qp_enabled = schedule["qp_enabled"]
        self._last_lbsgd_enabled = schedule["lbsgd_enabled"]

        raw_physical = self._to_physical(raw_action, action_to_physical)
        prev_physical = self.prev_safe_physical_action if schedule["accel_enabled"] else None
        prev2_physical = self.prev2_safe_physical_action if schedule["accel_enabled"] else None

        ctx = make_constraint_context(
            safety_state,
            self.limits,
            action_dim=raw_physical.size,
            semantics=str(getattr(self.cfg, "action_semantics", "auto")),
            dt=float(getattr(self.cfg, "dt", 0.05)),
            qpos_margin=float(getattr(self.cfg, "qpos_margin", 0.05)),
            dq_limit_scale=float(getattr(self.cfg, "dq_limit_scale", 0.25)),
            ddq_limit_scale=float(getattr(self.cfg, "ddq_limit_scale", 0.25)),
            tau_limit_scale=float(getattr(self.cfg, "tau_limit_scale", 0.25)),
            min_barrier_margin=float(getattr(self.cfg, "min_barrier_margin", 1e-4)),
            prev_action=prev_physical,
            prev2_action=prev2_physical,
            acceleration_enabled=schedule["accel_enabled"],
        )

        raw_eval = evaluate_numpy(raw_physical, ctx, include_proxy=True)
        report.safety_raw_action_violation = raw_eval.violation_count > 0
        report.safety_min_margin = raw_eval.min_margin
        if schedule["shadow_only"]:
            report.safety_safe_action_violation = report.safety_raw_action_violation
            report.safety_correction_norm = 0.0
            return raw_action.astype(np.float32), report

        if bool(int(getattr(self.cfg, "supervisor_enabled", 1))):
            redline = self.supervisor.check(safety_state)
            if redline.triggered:
                report.with_preempt(redline)
                stop = self._stop_action(raw_action, safety_state, action_to_physical, action_from_physical)
                report.safety_correction_norm = float(np.linalg.norm(stop - raw_action))
                return stop, report

        if not schedule["qp_enabled"] and not schedule["lbsgd_enabled"]:
            report.safety_safe_action_violation = report.safety_raw_action_violation
            report.safety_correction_norm = 0.0
            return raw_action.astype(np.float32), report

        action_physical = raw_physical
        if schedule["lbsgd_enabled"]:
            action_physical, lbsgd_report = self.lbsgd.project(
                action_physical,
                ctx,
                policy_obs=policy_obs,
                runtime_lbsgd_steps=schedule["lbsgd_steps"],
                runtime_barrier_eta=schedule["barrier_eta"],
            )
            report.merge(lbsgd_report)
            report.safety_warmup_phase = schedule["phase"]
            report.safety_qp_runtime_enabled = schedule["qp_enabled"]
            report.safety_lbsgd_runtime_enabled = schedule["lbsgd_enabled"]
            if lbsgd_report.safety_infeasible:
                redline = self.supervisor.check_post_action(safety_state, action_physical, lbsgd_infeasible=True)
                report.with_preempt(redline)
                stop = self._stop_action(raw_action, safety_state, action_to_physical, action_from_physical, ctx=ctx)
                report.safety_correction_norm = float(np.linalg.norm(stop - raw_action))
                return stop, report

        if schedule["qp_enabled"]:
            action_physical, qp_report = self.qp.project(action_physical, ctx)
            report.merge(qp_report)
            report.safety_warmup_phase = schedule["phase"]
            report.safety_qp_runtime_enabled = schedule["qp_enabled"]
            report.safety_lbsgd_runtime_enabled = schedule["lbsgd_enabled"]
            if qp_report.safety_infeasible:
                redline = self.supervisor.check_post_action(safety_state, action_physical, qp_infeasible=True)
                report.with_preempt(redline)
                stop = self._stop_action(raw_action, safety_state, action_to_physical, action_from_physical, ctx=ctx)
                report.safety_correction_norm = float(np.linalg.norm(stop - raw_action))
                return stop, report

        if bool(int(getattr(self.cfg, "supervisor_enabled", 1))):
            final_redline = self.supervisor.check_post_action(safety_state, action_physical)
            if final_redline.triggered:
                report.with_preempt(final_redline)
                stop = self._stop_action(raw_action, safety_state, action_to_physical, action_from_physical, ctx=ctx)
                report.safety_correction_norm = float(np.linalg.norm(stop - raw_action))
                return stop, report

        safe_action = self._from_physical(action_physical, action_from_physical, fallback_shape=raw_action.shape)
        safe_eval = evaluate_numpy(action_physical, ctx, include_proxy=True)
        report.safety_min_margin = min(report.safety_min_margin, safe_eval.min_margin)
        report.safety_safe_action_violation = safe_eval.violation_count > 0
        report.safety_triggered = report.safety_triggered or bool(np.linalg.norm(safe_action - raw_action) > 1e-7)
        report.safety_correction_norm = float(np.linalg.norm(safe_action - raw_action))
        self.prev2_safe_physical_action = None if self.prev_safe_physical_action is None else self.prev_safe_physical_action.copy()
        self.prev_safe_physical_action = action_physical.copy()
        return safe_action.astype(np.float32), report

    def _runtime_schedule(self, step_index: int) -> dict:
        base_qp = bool(int(getattr(self.cfg, "qp_enabled", 1)))
        base_lbsgd = bool(int(getattr(self.cfg, "lbsgd_enabled", 1)))
        if str(getattr(self.cfg, "mode", "sim")) == "real":
            return {
                "phase": 4,
                "qp_enabled": base_qp,
                "lbsgd_enabled": base_lbsgd,
                "accel_enabled": True,
                "shadow_only": False,
                "lbsgd_steps": int(getattr(self.cfg, "lbsgd_steps", 8)),
                "barrier_eta": float(getattr(self.cfg, "barrier_eta", 1e-2)),
            }

        qp_warmup = int(getattr(self.cfg, "qp_warmup_steps", 0) or 0)
        lbsgd_warmup = int(getattr(self.cfg, "lbsgd_warmup_steps", 0) or 0)
        lbsgd_ramp = int(getattr(self.cfg, "lbsgd_ramp_steps", 0) or 0)
        accel_warmup = int(getattr(self.cfg, "accel_warmup_steps", 0) or 0)
        shadow_until = int(getattr(self.cfg, "shadow_until_steps", 0) or 0)
        shadow_only = str(getattr(self.cfg, "mode", "sim")) == "sim" and step_index < shadow_until
        qp_enabled = base_qp and step_index >= qp_warmup
        lbsgd_enabled = base_lbsgd and step_index >= lbsgd_warmup
        if lbsgd_enabled and base_qp:
            qp_enabled = True
        if shadow_only:
            qp_enabled = False
            lbsgd_enabled = False

        target_steps = int(getattr(self.cfg, "lbsgd_steps", 8))
        target_eta = float(getattr(self.cfg, "barrier_eta", 1e-2))
        runtime_steps = target_steps
        runtime_eta = target_eta
        phase = 4
        if not qp_enabled and not lbsgd_enabled:
            phase = 1  # supervisor-only
        elif qp_enabled and not lbsgd_enabled:
            phase = 2  # QP-only
        elif lbsgd_enabled and lbsgd_ramp > 0 and step_index < lbsgd_warmup + lbsgd_ramp:
            phase = 3  # QP + ramped LBSGD
            progress = max(1.0 / max(target_steps, 1), (step_index - lbsgd_warmup + 1) / float(lbsgd_ramp))
            progress = min(1.0, max(0.0, progress))
            runtime_steps = max(1, int(math.ceil(target_steps * progress)))
            runtime_eta = target_eta * progress
        return {
            "phase": phase,
            "qp_enabled": qp_enabled,
            "lbsgd_enabled": lbsgd_enabled,
            "accel_enabled": step_index >= accel_warmup,
            "shadow_only": shadow_only,
            "lbsgd_steps": runtime_steps,
            "barrier_eta": runtime_eta,
        }

    def _reset_action_history(self) -> None:
        self.prev_safe_physical_action = None
        self.prev2_safe_physical_action = None

    def _stop_action(self, raw_action, safety_state, action_to_physical, action_from_physical, ctx=None) -> np.ndarray:
        raw_physical = self._to_physical(raw_action, action_to_physical)
        if ctx is None:
            try:
                ctx = make_constraint_context(
                    safety_state,
                    self.limits,
                    action_dim=raw_physical.size,
                    semantics=str(getattr(self.cfg, "action_semantics", "auto")),
                    dt=float(getattr(self.cfg, "dt", 0.05)),
                    qpos_margin=float(getattr(self.cfg, "qpos_margin", 0.05)),
                    dq_limit_scale=float(getattr(self.cfg, "dq_limit_scale", 0.25)),
                    ddq_limit_scale=float(getattr(self.cfg, "ddq_limit_scale", 0.25)),
                    tau_limit_scale=float(getattr(self.cfg, "tau_limit_scale", 0.25)),
                    min_barrier_margin=float(getattr(self.cfg, "min_barrier_margin", 1e-4)),
                    acceleration_enabled=False,
                )
            except Exception:
                return np.zeros_like(np.asarray(raw_action, dtype=np.float32).reshape(-1))
        stop_physical = stop_action_physical(raw_physical, ctx)
        return self._from_physical(stop_physical, action_from_physical, fallback_shape=np.asarray(raw_action).shape)

    def _to_physical(self, action, transform):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if transform is None:
            return action
        transformed = transform({"action": action})
        if isinstance(transformed, dict):
            transformed = transformed.get("action", transformed)
        return np.asarray(transformed, dtype=np.float32).reshape(-1)

    def _from_physical(self, action, transform, *, fallback_shape):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if transform is None:
            return action.reshape(fallback_shape)
        transformed = transform({"action": action})
        if isinstance(transformed, dict):
            transformed = transformed.get("action", transformed)
        return np.asarray(transformed, dtype=np.float32).reshape(fallback_shape)

    def _resolve_physical_action_bounds(self, env):
        if env is None:
            return None, None
        info_getter = getattr(env, "safety_physical_action_bounds", None)
        if callable(info_getter):
            low, high = info_getter()
            return low, high
        try:
            space = env.spec.action_space
            return np.asarray(space.low, dtype=np.float32), np.asarray(space.high, dtype=np.float32)
        except Exception:
            return None, None
