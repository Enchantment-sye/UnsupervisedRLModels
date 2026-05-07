from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .r1lite_state_adapter import ActionLayout, R1LiteSafetyState, SafetyLimits, adapt_safety_state
from .types import ConstraintEvaluation


@dataclass
class ConstraintContext:
    state: R1LiteSafetyState
    limits: SafetyLimits
    action_dim: int
    semantics: str
    dt: float
    qpos_margin: float
    dq_limit_scale: float
    ddq_limit_scale: float
    tau_limit_scale: float
    min_barrier_margin: float
    prev_action: Optional[np.ndarray] = None
    prev2_action: Optional[np.ndarray] = None
    prev_velocity: Optional[np.ndarray] = None
    acceleration_enabled: bool = True

    @property
    def layout(self) -> ActionLayout:
        return self.limits.layout


def make_constraint_context(
        safety_state,
        limits: SafetyLimits,
        *,
        action_dim: int,
        semantics: str,
        dt: float,
        qpos_margin: float,
        dq_limit_scale: float,
        ddq_limit_scale: float,
        tau_limit_scale: float,
        min_barrier_margin: float,
        prev_action: Optional[np.ndarray] = None,
        prev2_action: Optional[np.ndarray] = None,
        prev_velocity: Optional[np.ndarray] = None,
        acceleration_enabled: bool = True,
) -> ConstraintContext:
    state = safety_state if isinstance(safety_state, R1LiteSafetyState) else adapt_safety_state(safety_state, layout=limits.layout)
    limits.ensure_dim(action_dim)
    inferred_prev_velocity = None
    if acceleration_enabled:
        if prev_velocity is not None:
            inferred_prev_velocity = prev_velocity
        elif state.dq is not None:
            inferred_prev_velocity = state.dq
    return ConstraintContext(
        state=state,
        limits=limits,
        action_dim=int(action_dim),
        semantics=resolve_action_semantics(semantics),
        dt=max(float(dt), 1e-6),
        qpos_margin=float(qpos_margin),
        dq_limit_scale=float(dq_limit_scale),
        ddq_limit_scale=float(ddq_limit_scale),
        tau_limit_scale=float(tau_limit_scale),
        min_barrier_margin=float(min_barrier_margin),
        prev_action=None if prev_action is None else np.asarray(prev_action, dtype=np.float32).reshape(-1)[:action_dim],
        prev2_action=None if prev2_action is None else np.asarray(prev2_action, dtype=np.float32).reshape(-1)[:action_dim],
        prev_velocity=_coerce_vector(inferred_prev_velocity, action_dim),
        acceleration_enabled=bool(acceleration_enabled),
    )


def resolve_action_semantics(semantics: str) -> str:
    if semantics == "auto":
        return "absolute_joint_position"
    if semantics not in ("absolute_joint_position", "delta_joint_position", "joint_velocity"):
        raise ValueError(f"Unsupported safety action semantics: {semantics!r}")
    return semantics


def evaluate_numpy(action: np.ndarray, ctx: ConstraintContext, *, include_proxy: bool = True) -> ConstraintEvaluation:
    action = np.asarray(action, dtype=np.float32).reshape(-1)[:ctx.action_dim]
    margins: List[float] = []
    names: List[str] = []

    low = ctx.limits.action_low[:ctx.action_dim]
    high = ctx.limits.action_high[:ctx.action_dim]
    _append_box(margins, names, action, low, high, "action")

    q_next, u = predict_next_q_and_velocity(action, ctx)
    q_dim = min(len(q_next), ctx.action_dim, len(ctx.limits.q_min))
    if q_dim:
        q_margin = _qpos_margin_vector(ctx, q_dim)
        _append_box(
            margins,
            names,
            q_next[:q_dim],
            ctx.limits.q_min[:q_dim] + q_margin,
            ctx.limits.q_max[:q_dim] - q_margin,
            "q_next",
        )
        dq_max = np.maximum(ctx.limits.dq_max[:q_dim] * ctx.dq_limit_scale, 0.0)
        _append_abs_limit(margins, names, u[:q_dim], dq_max, "dq")

    prev_u = _previous_velocity(ctx, len(u))
    if ctx.acceleration_enabled and prev_u is not None:
        acc_dim = min(len(prev_u), len(u), len(ctx.limits.ddq_max))
        if acc_dim:
            ddq_max = np.maximum(ctx.limits.ddq_max[:acc_dim] * ctx.ddq_limit_scale * ctx.dt, 0.0)
            _append_abs_limit(margins, names, u[:acc_dim] - prev_u[:acc_dim], ddq_max, "ddq")

    if ctx.prev_action is not None and ctx.prev2_action is not None and ctx.limits.jerk_max is not None:
        _, prev_u = predict_next_q_and_velocity(ctx.prev_action, ctx)
        _, prev2_u = predict_next_q_and_velocity(ctx.prev2_action, ctx)
        jerk_dim = min(len(u), len(prev_u), len(prev2_u), len(ctx.limits.jerk_max))
        if jerk_dim:
            jerk_max = np.maximum(ctx.limits.jerk_max[:jerk_dim] * ctx.dt * ctx.dt, 0.0)
            jerk = u[:jerk_dim] - 2.0 * prev_u[:jerk_dim] + prev2_u[:jerk_dim]
            _append_abs_limit(margins, names, jerk, jerk_max, "jerk")

    _append_gripper_limits(margins, names, q_next, ctx)

    if ctx.state.effort is not None and ctx.limits.tau_trip is not None:
        effort = np.asarray(ctx.state.effort, dtype=np.float32).reshape(-1)
        tau_dim = min(effort.size, ctx.limits.tau_trip.size)
        if tau_dim:
            tau_limit = ctx.limits.tau_trip[:tau_dim] * max(ctx.tau_limit_scale, 1e-6)
            _append_abs_limit(margins, names, effort[:tau_dim], tau_limit, "effort_feedback")

    if include_proxy:
        proxy_names, proxy_margins = proxy_collision_margins(ctx.state.raw, ctx)
        names.extend(proxy_names)
        margins.extend(proxy_margins)

    margin_arr = np.asarray(margins, dtype=np.float32)
    return ConstraintEvaluation(names=names, margins=margin_arr, proxy_only=bool(include_proxy and ctx.limits.proxy_only_collision))


def torch_margins(action, ctx: ConstraintContext):
    if torch is None:
        raise ImportError("torch is required for LBSGD margins")
    if not torch.is_tensor(action):
        action = torch.as_tensor(action, dtype=torch.float32)
    device = action.device
    dtype = action.dtype
    action = action.reshape(-1)[:ctx.action_dim]
    margins = []

    low = torch.as_tensor(ctx.limits.action_low[:ctx.action_dim], dtype=dtype, device=device)
    high = torch.as_tensor(ctx.limits.action_high[:ctx.action_dim], dtype=dtype, device=device)
    finite_low = torch.isfinite(low)
    finite_high = torch.isfinite(high)
    if bool(torch.any(finite_low)):
        margins.append((action - low)[finite_low])
    if bool(torch.any(finite_high)):
        margins.append((high - action)[finite_high])

    q_next, u = predict_next_q_and_velocity_torch(action, ctx)
    q_dim = min(q_next.shape[-1], ctx.action_dim, len(ctx.limits.q_min))
    if q_dim:
        q_margin = torch.as_tensor(_qpos_margin_vector(ctx, q_dim), dtype=dtype, device=device)
        q_min = torch.as_tensor(ctx.limits.q_min[:q_dim], dtype=dtype, device=device) + q_margin
        q_max = torch.as_tensor(ctx.limits.q_max[:q_dim], dtype=dtype, device=device) - q_margin
        finite_q_min = torch.isfinite(q_min)
        finite_q_max = torch.isfinite(q_max)
        if bool(torch.any(finite_q_min)):
            margins.append((q_next[:q_dim] - q_min)[finite_q_min])
        if bool(torch.any(finite_q_max)):
            margins.append((q_max - q_next[:q_dim])[finite_q_max])
        dq_max = torch.as_tensor(ctx.limits.dq_max[:q_dim] * ctx.dq_limit_scale, dtype=dtype, device=device)
        margins.append(dq_max - torch.abs(u[:q_dim]))

    prev_u = _previous_velocity_torch(ctx, u.shape[-1], dtype=dtype, device=device)
    if ctx.acceleration_enabled and prev_u is not None:
        acc_dim = min(prev_u.shape[-1], u.shape[-1], len(ctx.limits.ddq_max))
        if acc_dim:
            ddq_max = torch.as_tensor(
                ctx.limits.ddq_max[:acc_dim] * ctx.ddq_limit_scale * ctx.dt,
                dtype=dtype,
                device=device,
            )
            margins.append(ddq_max - torch.abs(u[:acc_dim] - prev_u[:acc_dim]))

    grip_idx, grip_low, grip_high = _gripper_indices_and_bounds(ctx)
    if grip_idx.size:
        idx = torch.as_tensor(grip_idx, dtype=torch.long, device=device)
        low_g = torch.as_tensor(grip_low, dtype=dtype, device=device)
        high_g = torch.as_tensor(grip_high, dtype=dtype, device=device)
        qg = q_next.index_select(0, idx)
        finite_low_g = torch.isfinite(low_g)
        finite_high_g = torch.isfinite(high_g)
        if bool(torch.any(finite_low_g)):
            margins.append((qg - low_g)[finite_low_g])
        if bool(torch.any(finite_high_g)):
            margins.append((high_g - qg)[finite_high_g])

    if not margins:
        return torch.ones((1,), dtype=dtype, device=device) * 1e6
    return torch.cat([m.reshape(-1) for m in margins], dim=0)


def predict_next_q_and_velocity(action: np.ndarray, ctx: ConstraintContext) -> Tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)[:ctx.action_dim]
    q = _current_q(ctx, len(action))
    if ctx.semantics == "absolute_joint_position":
        q_next = action.copy()
        u = (q_next - q) / ctx.dt
    elif ctx.semantics == "delta_joint_position":
        u = action / ctx.dt
        q_next = q + action
    else:
        u = action.copy()
        q_next = q + action * ctx.dt
    return q_next.astype(np.float32), u.astype(np.float32)


def predict_next_q_and_velocity_torch(action, ctx: ConstraintContext):
    device = action.device
    dtype = action.dtype
    q = torch.as_tensor(_current_q(ctx, action.shape[-1]), dtype=dtype, device=device)
    if ctx.semantics == "absolute_joint_position":
        q_next = action
        u = (q_next - q) / ctx.dt
    elif ctx.semantics == "delta_joint_position":
        u = action / ctx.dt
        q_next = q + action
    else:
        u = action
        q_next = q + action * ctx.dt
    return q_next, u


def project_to_box(action: np.ndarray, ctx: ConstraintContext) -> Tuple[np.ndarray, int, bool]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)[:ctx.action_dim].copy()
    lower = np.asarray(ctx.limits.action_low[:ctx.action_dim], dtype=np.float32).copy()
    upper = np.asarray(ctx.limits.action_high[:ctx.action_dim], dtype=np.float32).copy()
    q = _current_q(ctx, ctx.action_dim)

    q_dim = min(ctx.action_dim, len(ctx.limits.q_min), len(q))
    if q_dim:
        q_margin = _qpos_margin_vector(ctx, q_dim)
        q_low = ctx.limits.q_min[:q_dim] + q_margin
        q_high = ctx.limits.q_max[:q_dim] - q_margin
        if ctx.semantics == "absolute_joint_position":
            lower[:q_dim] = np.maximum(lower[:q_dim], q_low)
            upper[:q_dim] = np.minimum(upper[:q_dim], q_high)
        elif ctx.semantics == "delta_joint_position":
            lower[:q_dim] = np.maximum(lower[:q_dim], q_low - q[:q_dim])
            upper[:q_dim] = np.minimum(upper[:q_dim], q_high - q[:q_dim])
        else:
            lower[:q_dim] = np.maximum(lower[:q_dim], (q_low - q[:q_dim]) / ctx.dt)
            upper[:q_dim] = np.minimum(upper[:q_dim], (q_high - q[:q_dim]) / ctx.dt)

        dq_max = ctx.limits.dq_max[:q_dim] * ctx.dq_limit_scale
        if ctx.semantics == "absolute_joint_position":
            lower[:q_dim] = np.maximum(lower[:q_dim], q[:q_dim] - dq_max * ctx.dt)
            upper[:q_dim] = np.minimum(upper[:q_dim], q[:q_dim] + dq_max * ctx.dt)
        elif ctx.semantics == "delta_joint_position":
            lower[:q_dim] = np.maximum(lower[:q_dim], -dq_max * ctx.dt)
            upper[:q_dim] = np.minimum(upper[:q_dim], dq_max * ctx.dt)
        else:
            lower[:q_dim] = np.maximum(lower[:q_dim], -dq_max)
            upper[:q_dim] = np.minimum(upper[:q_dim], dq_max)

    prev_u = _previous_velocity(ctx, q_dim or ctx.action_dim)
    if ctx.acceleration_enabled and prev_u is not None:
        acc_dim = min(q_dim or ctx.action_dim, prev_u.size, ctx.limits.ddq_max.size)
        if acc_dim:
            du = ctx.limits.ddq_max[:acc_dim] * ctx.ddq_limit_scale * ctx.dt
            if ctx.semantics == "absolute_joint_position":
                lower[:acc_dim] = np.maximum(lower[:acc_dim], q[:acc_dim] + (prev_u[:acc_dim] - du) * ctx.dt)
                upper[:acc_dim] = np.minimum(upper[:acc_dim], q[:acc_dim] + (prev_u[:acc_dim] + du) * ctx.dt)
            elif ctx.semantics == "delta_joint_position":
                lower[:acc_dim] = np.maximum(lower[:acc_dim], (prev_u[:acc_dim] - du) * ctx.dt)
                upper[:acc_dim] = np.minimum(upper[:acc_dim], (prev_u[:acc_dim] + du) * ctx.dt)
            else:
                lower[:acc_dim] = np.maximum(lower[:acc_dim], prev_u[:acc_dim] - du)
                upper[:acc_dim] = np.minimum(upper[:acc_dim], prev_u[:acc_dim] + du)

    grip_idx, grip_low, grip_high = _gripper_indices_and_bounds(ctx)
    if grip_idx.size and ctx.semantics == "absolute_joint_position":
        lower[grip_idx] = np.maximum(lower[grip_idx], grip_low)
        upper[grip_idx] = np.minimum(upper[grip_idx], grip_high)

    infeasible = bool(np.any(lower > upper))
    if infeasible:
        midpoint = np.where(np.isfinite(lower + upper), 0.5 * (lower + upper), 0.0)
        return midpoint.astype(np.float32), int(action.size), True
    clipped = np.clip(action, lower, upper)
    return clipped.astype(np.float32), int(np.sum(np.abs(clipped - action) > 1e-7)), False


def stop_action_physical(raw_action: np.ndarray, ctx: Optional[ConstraintContext]) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if ctx is None:
        return np.zeros_like(raw_action)
    q = _current_q(ctx, raw_action.size)
    if ctx.semantics == "absolute_joint_position":
        action = q[:raw_action.size].copy()
    else:
        action = np.zeros_like(raw_action)
    action, _, _ = project_to_box(action, ctx)
    return action.astype(np.float32)


def proxy_collision_margins(raw_state: Dict, ctx: ConstraintContext) -> Tuple[List[str], List[float]]:
    names: List[str] = []
    margins: List[float] = []
    left = _pose_position(raw_state.get("left_arm_ee_pose", raw_state.get("left_ee_pose")))
    right = _pose_position(raw_state.get("right_arm_ee_pose", raw_state.get("right_ee_pose")))
    if left is not None and right is not None:
        dist = float(np.linalg.norm(left - right))
        names.append("proxy_left_ee_right_ee")
        margins.append(dist - float(ctx.limits.raw_yaml.get("collision", {}).get("self_collision_min_dist", 0.08)))
    for key in ("left_arm_ee_pose", "right_arm_ee_pose", "left_ee_pose", "right_ee_pose"):
        pos = _pose_position(raw_state.get(key))
        if pos is None:
            continue
        names.append(f"proxy_floor_{key}")
        margins.append(float(pos[2]) - float(ctx.limits.raw_yaml.get("collision", {}).get("floor_min_z", 0.02)))
    if "workspace_proxy_violation" in raw_state:
        names.append("proxy_workspace_boundary")
        margins.append(-1.0 if bool(raw_state["workspace_proxy_violation"]) else 1.0)
    if "cable_proxy_violation" in raw_state:
        names.append("proxy_cable")
        margins.append(-1.0 if bool(raw_state["cable_proxy_violation"]) else 1.0)
    return names, margins


def _append_box(margins: List[float], names: List[str], value, low, high, prefix: str) -> None:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    low = np.asarray(low, dtype=np.float32).reshape(-1)
    high = np.asarray(high, dtype=np.float32).reshape(-1)
    dim = min(value.size, low.size, high.size)
    for idx in range(dim):
        if np.isfinite(low[idx]):
            margins.append(float(value[idx] - low[idx]))
            names.append(f"{prefix}_lo_{idx}")
        if np.isfinite(high[idx]):
            margins.append(float(high[idx] - value[idx]))
            names.append(f"{prefix}_hi_{idx}")


def _append_abs_limit(margins: List[float], names: List[str], value, limit, prefix: str) -> None:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    limit = np.asarray(limit, dtype=np.float32).reshape(-1)
    dim = min(value.size, limit.size)
    for idx in range(dim):
        if np.isfinite(limit[idx]):
            margins.append(float(limit[idx] - abs(value[idx])))
            names.append(f"{prefix}_abs_{idx}")


def _append_gripper_limits(margins: List[float], names: List[str], q_next: np.ndarray, ctx: ConstraintContext) -> None:
    idx, low, high = _gripper_indices_and_bounds(ctx)
    if idx.size == 0:
        return
    q_next = np.asarray(q_next, dtype=np.float32).reshape(-1)
    idx = idx[idx < q_next.size]
    if idx.size == 0:
        return
    _append_box(margins, names, q_next[idx], low[:idx.size], high[:idx.size], "gripper")


def _gripper_indices_and_bounds(ctx: ConstraintContext):
    idx = []
    layout = ctx.layout
    if layout.left_gripper[1] > layout.left_gripper[0]:
        idx.extend(range(layout.left_gripper[0], layout.left_gripper[1]))
    if layout.right_gripper[1] > layout.right_gripper[0]:
        idx.extend(range(layout.right_gripper[0], layout.right_gripper[1]))
    idx_arr = np.asarray(idx, dtype=np.int64)
    low = np.asarray(ctx.limits.gripper_min, dtype=np.float32).reshape(-1)
    high = np.asarray(ctx.limits.gripper_max, dtype=np.float32).reshape(-1)
    if low.size == 1 and idx_arr.size > 1:
        low = np.repeat(low, idx_arr.size)
    if high.size == 1 and idx_arr.size > 1:
        high = np.repeat(high, idx_arr.size)
    return idx_arr, low[:idx_arr.size], high[:idx_arr.size]


def _current_q(ctx: ConstraintContext, dim: int) -> np.ndarray:
    if ctx.state.q is None or np.asarray(ctx.state.q).size == 0:
        return np.zeros((dim,), dtype=np.float32)
    q = np.asarray(ctx.state.q, dtype=np.float32).reshape(-1)
    if q.size < dim:
        q = np.concatenate([q, np.zeros((dim - q.size,), dtype=np.float32)], axis=0)
    return q[:dim].astype(np.float32)


def _previous_velocity(ctx: ConstraintContext, dim: int) -> Optional[np.ndarray]:
    if not ctx.acceleration_enabled:
        return None
    if ctx.prev_velocity is not None:
        return _pad_vector(ctx.prev_velocity, dim)
    if ctx.prev_action is None:
        return None
    _, prev_u = predict_next_q_and_velocity(ctx.prev_action, ctx)
    return _pad_vector(prev_u, dim)


def _previous_velocity_torch(ctx: ConstraintContext, dim: int, *, dtype, device):
    if torch is None or not ctx.acceleration_enabled:
        return None
    if ctx.prev_velocity is not None:
        return torch.as_tensor(_pad_vector(ctx.prev_velocity, dim), dtype=dtype, device=device)
    if ctx.prev_action is None:
        return None
    _, prev_u = predict_next_q_and_velocity_torch(
        torch.as_tensor(ctx.prev_action, dtype=dtype, device=device),
        ctx,
    )
    if prev_u.shape[-1] >= dim:
        return prev_u[:dim]
    pad = torch.zeros((dim - prev_u.shape[-1],), dtype=dtype, device=device)
    return torch.cat([prev_u, pad], dim=0)


def _coerce_vector(value, dim: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    return _pad_vector(np.asarray(value, dtype=np.float32).reshape(-1), dim)


def _pad_vector(value: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < dim:
        arr = np.concatenate([arr, np.zeros((dim - arr.size,), dtype=np.float32)], axis=0)
    return arr[:dim].astype(np.float32)


def _qpos_margin_vector(ctx: ConstraintContext, dim: int) -> np.ndarray:
    margin = np.full((dim,), float(ctx.qpos_margin), dtype=np.float32)
    gripper_idx, _, _ = _gripper_indices_and_bounds(ctx)
    gripper_idx = gripper_idx[gripper_idx < dim]
    if gripper_idx.size:
        margin[gripper_idx] = 0.0
    return margin


def _pose_position(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]
