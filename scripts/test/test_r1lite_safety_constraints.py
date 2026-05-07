import sys
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from envs.galaxea_sim import GalaxeaSimEnv, TRIVIEW_CAMERA_KEYS, TRIVIEW_IMAGE_KEY
from config import get_parser, make_config_from_args
from safety.constraints import evaluate_numpy, make_constraint_context, project_to_box
from safety.r1lite_state_adapter import is_r1lite_safety_task, load_safety_limits


def _state():
    return {
        "left_arm_joint_position": np.zeros(6, dtype=np.float32),
        "right_arm_joint_position": np.zeros(6, dtype=np.float32),
        "left_arm_gripper_position": np.asarray([0.2], dtype=np.float32),
        "right_arm_gripper_position": np.asarray([0.2], dtype=np.float32),
    }


def _ctx():
    limits = load_safety_limits(str(ROOT / "configs/safety/r1lite_redlines.yaml"))
    return make_constraint_context(
        _state(),
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


def test_r1lite_task_gate_only_matches_r1lite():
    assert is_r1lite_safety_task(SimpleNamespace(env=SimpleNamespace(task="galaxea_r1lite_blocks_stack_easy")))
    assert not is_r1lite_safety_task(SimpleNamespace(env=SimpleNamespace(task="galaxea_blocks_stack_easy")))


def test_metra_config_deepcopy_supports_galaxea_parallel_collector():
    parser = get_parser()
    args = parser.parse_args([
        "--task",
        "galaxea_r1lite_blocks_stack_easy",
        "--encoder",
        "1",
        "--galaxea-sim-image-key",
        TRIVIEW_IMAGE_KEY,
        "--n_parallel",
        "2",
    ])
    cfg = make_config_from_args(args)

    cloned = copy.deepcopy(cfg)

    assert cloned.env.task == "galaxea_r1lite_blocks_stack_easy"
    assert cloned.n_parallel == 2
    assert cloned.net.encoder_type == "galaxea-r1lite-triview"


def test_constraints_detect_and_project_joint_limit_violation():
    ctx = _ctx()
    raw = np.full(14, 3.0, dtype=np.float32)

    before = evaluate_numpy(raw, ctx)
    projected, num_clipped, infeasible = project_to_box(raw, ctx)
    after = evaluate_numpy(projected, ctx)

    assert before.violation_count > 0
    assert num_clipped > 0
    assert not infeasible
    assert after.violation_count == 0
    assert after.min_margin >= -1e-6


def test_qpos_margin_does_not_make_closed_gripper_stop_infeasible():
    ctx = _ctx()
    stop = np.zeros(14, dtype=np.float32)

    projected, _num_clipped, infeasible = project_to_box(stop, ctx)
    after = evaluate_numpy(projected, ctx)

    assert not infeasible
    assert after.violation_count == 0


def test_acceleration_uses_feedback_velocity_before_prev_action_history():
    limits = load_safety_limits(str(ROOT / "configs/safety/r1lite_redlines.yaml"))
    state = _state()
    state["left_arm_joint_velocity"] = np.zeros(6, dtype=np.float32)
    state["right_arm_joint_velocity"] = np.zeros(6, dtype=np.float32)
    state["left_arm_gripper_velocity"] = np.zeros(1, dtype=np.float32)
    state["right_arm_gripper_velocity"] = np.zeros(1, dtype=np.float32)
    action = np.zeros(14, dtype=np.float32)
    action[12:] = 0.2

    ctx = make_constraint_context(
        state,
        limits,
        action_dim=14,
        semantics="absolute_joint_position",
        dt=0.05,
        qpos_margin=0.01,
        dq_limit_scale=1.0,
        ddq_limit_scale=1.0,
        tau_limit_scale=0.75,
        min_barrier_margin=1e-5,
        prev_action=np.full(14, 3.0, dtype=np.float32),
    )
    eval_after = evaluate_numpy(action, ctx, include_proxy=False)
    ddq_margins = [
        margin
        for name, margin in zip(eval_after.names, eval_after.margins)
        if name.startswith("ddq_abs_")
    ]

    assert ddq_margins
    assert min(ddq_margins) >= -1e-6


def test_acceleration_warmup_can_disable_prev_action_constraints():
    limits = load_safety_limits(str(ROOT / "configs/safety/r1lite_redlines.yaml"))
    action = np.zeros(14, dtype=np.float32)
    action[12:] = 0.2

    ctx = make_constraint_context(
        _state(),
        limits,
        action_dim=14,
        semantics="absolute_joint_position",
        dt=0.05,
        qpos_margin=0.01,
        dq_limit_scale=1.0,
        ddq_limit_scale=1.0,
        tau_limit_scale=0.75,
        min_barrier_margin=1e-5,
        prev_action=np.full(14, 3.0, dtype=np.float32),
        acceleration_enabled=False,
    )
    projected, _num_clipped, infeasible = project_to_box(action, ctx)
    eval_after = evaluate_numpy(projected, ctx, include_proxy=False)

    assert not infeasible
    assert not any(name.startswith("ddq_abs_") for name in eval_after.names)


def test_galaxea_safety_state_is_info_not_image_obs():
    env = GalaxeaSimEnv.__new__(GalaxeaSimEnv)
    env._size = (2, 2)
    env.flatten_obs = False
    env.encoder = True
    env._image_key = TRIVIEW_IMAGE_KEY
    env._uses_triview = True
    env._env_id = 0
    env._last_image = np.zeros((2, 2, 9), dtype=np.uint8)

    obs = {
        "upper_body_observations": {
            **{key: np.zeros((2, 2, 3), dtype=np.uint8) for key in TRIVIEW_CAMERA_KEYS},
            "left_arm_joint_position": np.arange(6, dtype=np.float32),
            "right_arm_joint_position": np.arange(6, dtype=np.float32),
        },
        "lower_body_observations": {
            "torso_joint_position": np.zeros(3, dtype=np.float32),
        },
    }

    timestep = env._build_timestep(obs, reward=0.0, is_first=True, is_last=False, is_terminal=False, info={})

    assert timestep["image"].shape == (2, 2, 9)
    assert "safety_state" in timestep["info"]
    assert "left_arm_joint_position" in timestep["info"]["safety_state"]
    assert not isinstance(timestep["image"], dict)
