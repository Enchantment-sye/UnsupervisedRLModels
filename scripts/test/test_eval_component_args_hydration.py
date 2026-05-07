import json
import logging
import os
import sys
import tempfile


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
TRAIN_DIR = os.path.join(REPO_ROOT, "scripts", "train")
EVAL_DIR = os.path.join(REPO_ROOT, "scripts", "eval")
for path in (REPO_ROOT, SRC_DIR, TRAIN_DIR, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_metra import EVAL_ARG_LOGGER, get_eval_parser, normalize_eval_args


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_component_eval_uses_source_run_args_for_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = os.path.join(tmpdir, "source_run")
        model_dir = os.path.join(run_dir, "models", "epoch-1")
        os.makedirs(model_dir, exist_ok=True)

        policy_path = os.path.join(model_dir, "skill_policy.pt")
        traj_path = os.path.join(model_dir, "traj_encoder.pt")
        with open(policy_path, "wb"):
            pass
        with open(traj_path, "wb"):
            pass

        source_args = {
            "task": "metaworld_dial_turn",
            "algo": "iksd",
            "stage": "pre_training",
            "dim_skill": 64,
            "discrete": 1,
            "encoder": 0,
            "use_cascade": False,
            "num_policy_levels": 2,
            "use_kme": True,
            "idk_subsample_size": 128,
            "seed": 42,
            "metric_num_sampled_points": 10,
            "skill_policy_path": None,
            "traj_encoder_path": None,
        }
        with open(os.path.join(run_dir, "args.json"), "w") as fh:
            json.dump(source_args, fh)

        parser = get_eval_parser()
        argv = [
            "--ckpt-path", policy_path,
            "--traj-encoder-path", traj_path,
            "--task", "dmc_quadruped_run_forward_color",
            "--algo", "metra_cascade",
            "--use_cascade",
            "--dim_skill", "16",
            "--encoder", "1",
            "--seed", "0",
            "--metric_num_sampled_points", "20",
            "--eval-mode", "trajectories",
        ]
        args = parser.parse_args(argv)

        handler = _ListHandler()
        EVAL_ARG_LOGGER.addHandler(handler)
        try:
            normalized = normalize_eval_args(args, parser=parser, argv=argv)
        finally:
            EVAL_ARG_LOGGER.removeHandler(handler)

        assert normalized.task == "metaworld_dial_turn"
        assert normalized.algo == "iksd"
        assert normalized.use_cascade is False
        assert normalized.dim_skill == 64
        assert normalized.encoder == 0
        assert normalized.seed == 0
        assert normalized.metric_num_sampled_points == 20
        assert normalized.eval_mode == "trajectories"
        assert normalized.skill_policy_path == policy_path
        assert normalized.traj_encoder_path == traj_path
        assert any("overrode conflicting CLI values" in message for message in handler.messages)


def main():
    test_component_eval_uses_source_run_args_for_structure()
    print("eval component args hydration tests passed")


if __name__ == "__main__":
    main()
