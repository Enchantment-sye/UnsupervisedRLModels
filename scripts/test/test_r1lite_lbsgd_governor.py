import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config.base import SafetyConfig
from safety.constraints import evaluate_numpy, make_constraint_context
from safety.lbsgd_governor import LBSGDGovernor
from safety.r1lite_state_adapter import load_safety_limits


def _ctx(*, q_min=None, q_max=None):
    limits = load_safety_limits(str(ROOT / "configs/safety/r1lite_redlines.yaml"))
    if q_min is not None:
        limits.q_min = np.asarray(q_min, dtype=np.float32)
    if q_max is not None:
        limits.q_max = np.asarray(q_max, dtype=np.float32)
    state = {
        "left_arm_joint_position": np.zeros(6, dtype=np.float32),
        "right_arm_joint_position": np.zeros(6, dtype=np.float32),
        "left_arm_gripper_position": np.asarray([0.2], dtype=np.float32),
        "right_arm_gripper_position": np.asarray([0.2], dtype=np.float32),
    }
    return make_constraint_context(
        state,
        limits,
        action_dim=14,
        semantics="absolute_joint_position",
        dt=0.05,
        qpos_margin=0.05,
        dq_limit_scale=20.0,
        ddq_limit_scale=20.0,
        tau_limit_scale=0.25,
        min_barrier_margin=1e-4,
    )


def _cfg():
    return SafetyConfig(
        enabled=1,
        mode="sim",
        lbsgd_steps=4,
        lbsgd_lr=1e-2,
        barrier_eta=1e-2,
        min_barrier_margin=1e-4,
    )


def test_lbsgd_corrects_out_of_bounds_raw_action_to_feasible_interior():
    ctx = _ctx()
    raw = np.full(14, 3.0, dtype=np.float32)

    action, report = LBSGDGovernor(_cfg()).project(raw, ctx)
    eval_after = evaluate_numpy(action, ctx, include_proxy=False)

    assert not report.safety_infeasible
    assert eval_after.violation_count == 0
    assert eval_after.min_margin > 0.0
    assert np.linalg.norm(action - raw) > 0.0


def test_lbsgd_reports_infeasible_when_no_interior_exists():
    ctx = _ctx(q_min=np.zeros(14, dtype=np.float32), q_max=np.zeros(14, dtype=np.float32))
    raw = np.zeros(14, dtype=np.float32)

    _action, report = LBSGDGovernor(_cfg()).project(raw, ctx)

    assert report.safety_infeasible
    assert report.safety_lbsgd_infeasible
    assert report.safety_stop_reason == "lbsgd_no_feasible_interior"
