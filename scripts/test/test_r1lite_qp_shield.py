import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config.base import SafetyConfig
from safety.constraints import evaluate_numpy, make_constraint_context
from safety.qp_shield import QPShield
from safety.r1lite_state_adapter import load_safety_limits


def _ctx():
    limits = load_safety_limits(str(ROOT / "configs/safety/r1lite_redlines.yaml"))
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
        dq_limit_scale=0.25,
        ddq_limit_scale=0.25,
        tau_limit_scale=0.25,
        min_barrier_margin=1e-4,
    )


def test_qp_shield_clips_after_lbsgd_to_box_constraints():
    ctx = _ctx()
    action_after_lbsgd = np.full(14, 3.0, dtype=np.float32)

    projected, report = QPShield(SafetyConfig(enabled=1, mode="sim")).project(action_after_lbsgd, ctx)
    eval_after = evaluate_numpy(projected, ctx, include_proxy=False)

    assert report.safety_qp_active
    assert not report.safety_qp_infeasible
    assert report.details["safety_qp_num_clipped"] > 0
    assert eval_after.violation_count == 0


def test_qp_shield_reports_infeasible_for_empty_box():
    ctx = _ctx()
    ctx.limits.q_min[:] = 1.0
    ctx.limits.q_max[:] = -1.0

    _projected, report = QPShield(SafetyConfig(enabled=1, mode="sim")).project(np.zeros(14, dtype=np.float32), ctx)

    assert report.safety_infeasible
    assert report.safety_qp_infeasible
    assert report.safety_stop_reason == "qp_infeasible"
