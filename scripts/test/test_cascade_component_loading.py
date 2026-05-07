import os
import sys
import tempfile

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
TRAIN_DIR = os.path.join(REPO_ROOT, "scripts", "train")
for path in (REPO_ROOT, SRC_DIR, TRAIN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("METRA_STARTUP_QUIET", "1")

from core.metra_agent import MetraAgent
from core.metra_config import MetraConfig, get_parser, make_config_from_args
from workspace_common import build_env_and_replay_buffer


def build_agent(algo, extra_args=None):
    parser = get_parser()
    argv = [
        "--task", "debug_component_loading",
        "--stage", "pre_training",
        "--algo", algo,
        "--dim_skill", "3",
        "--encoder", "1",
        "--render_size", "64",
        "--seed", "7",
        "--use_gpu", "0",
        "--sample_cpu", "1",
        "--workspace_root", "/tmp",
        "--traj_batch_size", "2",
        "--trans_minibatch_size", "2",
        "--sac_min_buffer_size", "10",
        "--sac_max_buffer_size", "100",
    ]
    if extra_args:
        argv.extend(extra_args)
    args = parser.parse_args(argv)
    cfg = make_config_from_args(args, cls=MetraConfig)
    env, replay_buffer = build_env_and_replay_buffer(cfg, args)
    agent = MetraAgent(cfg, env, replay_buffer)
    return env, agent


def save_policy_checkpoint(path, agent):
    torch.save(
        {
            "discrete": agent.cfg.algo.discrete,
            "dim_skill": agent.cfg.algo.dim_skill,
            "policy": agent.sac_trainer.skill_policy,
        },
        path,
    )


def save_traj_encoder_checkpoint(path, agent):
    torch.save(
        {
            "discrete": agent.cfg.algo.discrete,
            "dim_skill": agent.cfg.algo.dim_skill,
            "traj_encoder": agent.traj_encoder,
        },
        path,
    )


def assert_state_dict_equal(lhs, rhs):
    assert set(lhs.keys()) == set(rhs.keys())
    for key in lhs.keys():
        assert torch.equal(lhs[key].detach().cpu(), rhs[key].detach().cpu()), key


def close_env(env):
    try:
        env.close()
    except Exception:
        pass


def test_cascade_component_checkpoint_load_grows_stage_count():
    src_env = None
    dst_env = None
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            src_env, src_agent = build_agent("metra_cascade")
            src_agent.ensure_policy_stage_count(4)

            policy_path = os.path.join(tmpdir, "cascade_skill_policy.pt")
            traj_encoder_path = os.path.join(tmpdir, "cascade_traj_encoder.pt")
            save_policy_checkpoint(policy_path, src_agent)
            save_traj_encoder_checkpoint(traj_encoder_path, src_agent)

            dst_env, dst_agent = build_agent("metra_cascade")
            assert dst_agent._get_cascade_stage_count() == 1

            dst_agent.load_component_checkpoints(policy_path, traj_encoder_path=traj_encoder_path)

            assert dst_agent._get_cascade_stage_count() == 4
            assert_state_dict_equal(
                dst_agent.sac_trainer.skill_policy.state_dict(),
                src_agent.sac_trainer.skill_policy.state_dict(),
            )
            assert_state_dict_equal(
                dst_agent.traj_encoder.state_dict(),
                src_agent.traj_encoder.state_dict(),
            )
        finally:
            close_env(src_env)
            close_env(dst_env)


def test_non_cascade_component_checkpoint_load_preserves_existing_behavior():
    src_env = None
    dst_env = None
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            src_env, src_agent = build_agent("metra")
            policy_path = os.path.join(tmpdir, "plain_skill_policy.pt")
            save_policy_checkpoint(policy_path, src_agent)

            dst_env, dst_agent = build_agent("metra")
            dst_agent.load_component_checkpoints(policy_path)

            assert dst_agent._get_cascade_stage_count() == 1
            assert_state_dict_equal(
                dst_agent.sac_trainer.skill_policy.state_dict(),
                src_agent.sac_trainer.skill_policy.state_dict(),
            )
        finally:
            close_env(src_env)
            close_env(dst_env)


def test_non_cascade_agent_rejects_cascade_component_checkpoint():
    src_env = None
    dst_env = None
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            src_env, src_agent = build_agent("metra_cascade")
            src_agent.ensure_policy_stage_count(3)
            policy_path = os.path.join(tmpdir, "cascade_skill_policy.pt")
            save_policy_checkpoint(policy_path, src_agent)

            dst_env, dst_agent = build_agent("metra")
            try:
                dst_agent.load_component_checkpoints(policy_path)
            except ValueError as exc:
                message = str(exc)
                assert "CascadeActor" in message
                assert "--use_cascade" in message
            else:
                raise AssertionError("Expected non-cascade agent to reject cascade checkpoint")
        finally:
            close_env(src_env)
            close_env(dst_env)


def test_cascade_agent_rejects_non_cascade_component_checkpoint():
    src_env = None
    dst_env = None
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            src_env, src_agent = build_agent("metra")
            policy_path = os.path.join(tmpdir, "plain_skill_policy.pt")
            save_policy_checkpoint(policy_path, src_agent)

            dst_env, dst_agent = build_agent("metra_cascade")
            try:
                dst_agent.load_component_checkpoints(policy_path)
            except ValueError as exc:
                message = str(exc)
                assert "non-cascade policy" in message
                assert "CascadeActor" in message
            else:
                raise AssertionError("Expected cascade agent to reject non-cascade checkpoint")
        finally:
            close_env(src_env)
            close_env(dst_env)


def main():
    test_cascade_component_checkpoint_load_grows_stage_count()
    test_non_cascade_component_checkpoint_load_preserves_existing_behavior()
    test_non_cascade_agent_rejects_cascade_component_checkpoint()
    test_cascade_agent_rejects_non_cascade_component_checkpoint()
    print("cascade component loading tests passed")


if __name__ == "__main__":
    main()
