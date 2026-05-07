import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config.base import SafetyConfig
from safety import build_safety_controller
from safety.r1lite_state_adapter import load_safety_limits
from safety.supervisor import R1LiteSupervisor
from safety.types import SafetyReport


def test_supervisor_missing_safety_state_stops_in_real_mode():
    cfg = SafetyConfig(enabled=1, mode="real")
    limits = load_safety_limits(str(ROOT / "configs/safety/r1lite_redlines.yaml"))

    redline = R1LiteSupervisor(cfg, limits).check(None)

    assert redline.triggered
    assert redline.reason == "missing_safety_state"


def test_supervisor_status_error_preempts_and_skips_lbsgd():
    cfg = SimpleNamespace(
        env=SimpleNamespace(task="galaxea_r1lite_blocks_stack_easy"),
        safety=SafetyConfig(
            enabled=1,
            mode="sim",
            safety_yaml=str(ROOT / "configs/safety/r1lite_redlines.yaml"),
        ),
    )
    controller = build_safety_controller(cfg)
    called = {"lbsgd": False}

    def _must_not_call(*_args, **_kwargs):
        called["lbsgd"] = True
        raise AssertionError("LBSGD must not run after supervisor preemption")

    controller.lbsgd.project = _must_not_call

    safe_action, report = controller.filter_action(
        raw_action=np.ones(14, dtype=np.float32),
        safety_state={
            "feedback_status_arm_left": 2,
            "left_arm_joint_position": np.zeros(6, dtype=np.float32),
            "right_arm_joint_position": np.zeros(6, dtype=np.float32),
            "left_arm_gripper_position": np.asarray([0.2], dtype=np.float32),
            "right_arm_gripper_position": np.asarray([0.2], dtype=np.float32),
        },
    )

    assert report.safety_supervisor_preempted
    assert report.safety_stop_reason.startswith("status_error")
    assert not called["lbsgd"]
    assert safe_action.shape == (14,)


def test_safety_warmup_enables_qp_before_lbsgd():
    cfg = SimpleNamespace(
        env=SimpleNamespace(task="galaxea_r1lite_blocks_stack_easy"),
        safety=SafetyConfig(
            enabled=1,
            mode="sim",
            safety_yaml=str(ROOT / "configs/safety/r1lite_redlines.yaml"),
            qp_warmup_steps=2,
            lbsgd_warmup_steps=4,
            lbsgd_ramp_steps=4,
        ),
    )
    controller = build_safety_controller(cfg)
    calls = {"qp": 0, "lbsgd": 0}

    def _qp(action, ctx):
        calls["qp"] += 1
        return action, SafetyReport(safety_enabled=True, safety_mode="sim")

    def _lbsgd(action, ctx, policy_obs=None, **_kwargs):
        calls["lbsgd"] += 1
        return action, SafetyReport(safety_enabled=True, safety_mode="sim")

    controller.qp.project = _qp
    controller.lbsgd.project = _lbsgd
    state = {
        "left_arm_joint_position": np.zeros(6, dtype=np.float32),
        "right_arm_joint_position": np.zeros(6, dtype=np.float32),
        "left_arm_gripper_position": np.asarray([0.0], dtype=np.float32),
        "right_arm_gripper_position": np.asarray([0.0], dtype=np.float32),
    }

    reports = [
        controller.filter_action(raw_action=np.zeros(14, dtype=np.float32), safety_state=state)[1]
        for _ in range(5)
    ]

    assert [int(report.safety_warmup_phase) for report in reports] == [1, 1, 2, 2, 3]
    assert calls["qp"] == 3
    assert calls["lbsgd"] == 1


def test_supervisor_only_warmup_does_not_store_raw_action_history():
    cfg = SimpleNamespace(
        env=SimpleNamespace(task="galaxea_r1lite_blocks_stack_easy"),
        safety=SafetyConfig(
            enabled=1,
            mode="sim",
            safety_yaml=str(ROOT / "configs/safety/r1lite_redlines.yaml"),
            qp_warmup_steps=10,
            lbsgd_warmup_steps=10,
        ),
    )
    controller = build_safety_controller(cfg)
    state = {
        "left_arm_joint_position": np.zeros(6, dtype=np.float32),
        "right_arm_joint_position": np.zeros(6, dtype=np.float32),
        "left_arm_gripper_position": np.asarray([0.0], dtype=np.float32),
        "right_arm_gripper_position": np.asarray([0.0], dtype=np.float32),
    }

    safe, report = controller.filter_action(raw_action=np.ones(14, dtype=np.float32), safety_state=state)

    assert np.allclose(safe, np.ones(14, dtype=np.float32))
    assert int(report.safety_warmup_phase) == 1
    assert controller.prev_safe_physical_action is None
    assert controller.prev2_safe_physical_action is None


def test_qp_transition_resets_bad_warmup_history_and_disables_accel_until_warmup():
    cfg = SimpleNamespace(
        env=SimpleNamespace(task="galaxea_r1lite_blocks_stack_easy"),
        safety=SafetyConfig(
            enabled=1,
            mode="sim",
            safety_yaml=str(ROOT / "configs/safety/r1lite_redlines.yaml"),
            qp_warmup_steps=1,
            lbsgd_enabled=0,
            accel_warmup_steps=10,
        ),
    )
    controller = build_safety_controller(cfg)
    seen = {}

    def _qp(action, ctx):
        seen["prev_action"] = ctx.prev_action
        seen["prev_velocity"] = ctx.prev_velocity
        seen["acceleration_enabled"] = ctx.acceleration_enabled
        return action, SafetyReport(safety_enabled=True, safety_mode="sim")

    controller.qp.project = _qp
    state = {
        "left_arm_joint_position": np.zeros(6, dtype=np.float32),
        "right_arm_joint_position": np.zeros(6, dtype=np.float32),
        "left_arm_gripper_position": np.asarray([0.0], dtype=np.float32),
        "right_arm_gripper_position": np.asarray([0.0], dtype=np.float32),
    }
    controller.prev_safe_physical_action = np.full(14, 3.0, dtype=np.float32)

    controller.filter_action(raw_action=np.zeros(14, dtype=np.float32), safety_state=state)
    controller.filter_action(raw_action=np.zeros(14, dtype=np.float32), safety_state=state)

    assert seen["prev_action"] is None
    assert seen["prev_velocity"] is None
    assert seen["acceleration_enabled"] is False


def test_sim_shadow_until_records_violations_without_preempt_or_filters():
    cfg = SimpleNamespace(
        env=SimpleNamespace(task="galaxea_r1lite_blocks_stack_easy"),
        safety=SafetyConfig(
            enabled=1,
            mode="sim",
            safety_yaml=str(ROOT / "configs/safety/r1lite_redlines.yaml"),
            qp_warmup_steps=0,
            lbsgd_warmup_steps=0,
            shadow_until_steps=2,
        ),
    )
    controller = build_safety_controller(cfg)
    calls = {"qp": 0, "lbsgd": 0}
    controller.qp.project = lambda action, ctx: (calls.__setitem__("qp", calls["qp"] + 1) or action, SafetyReport())
    controller.lbsgd.project = lambda action, ctx, **kwargs: (
        calls.__setitem__("lbsgd", calls["lbsgd"] + 1) or action,
        SafetyReport(),
    )
    raw = np.full(14, 3.0, dtype=np.float32)
    state = {
        "feedback_status_arm_left": 2,
        "left_arm_joint_position": np.zeros(6, dtype=np.float32),
        "right_arm_joint_position": np.zeros(6, dtype=np.float32),
        "left_arm_gripper_position": np.asarray([0.0], dtype=np.float32),
        "right_arm_gripper_position": np.asarray([0.0], dtype=np.float32),
    }

    safe, report = controller.filter_action(raw_action=raw, safety_state=state)

    assert np.allclose(safe, raw)
    assert report.safety_raw_action_violation
    assert not report.safety_supervisor_preempted
    assert calls == {"qp": 0, "lbsgd": 0}
