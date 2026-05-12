from __future__ import annotations

import argparse
from dataclasses import dataclass


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _mass_algo(value):
    if value in ("mass", "mass_pixel"):
        return value
    raise argparse.ArgumentTypeError("invalid choice: %r (choose from 'mass')" % (value,))


@dataclass
class MassPixelConfig:
    algo: str = "mass"
    seed: int = 0
    device: str = "cuda"
    task: str = "debug_dummy"
    n_epochs: int = 1000000
    traj_batch_size: int = 8
    trans_minibatch_size: int = 256
    trans_optimization_epochs: int = 200
    n_epochs_per_log: int = 25
    n_epochs_per_eval: int = 0
    n_epochs_per_save: int = 1000
    n_epochs_per_pt_save: int = 1000
    n_parallel: int = 4
    n_thread: int = 1
    parallel_sampler_enabled: bool = True
    parallel_sampler_num_workers: int = 0
    parallel_sampler_fail_open: bool = True
    eval_parallel_sampler_enabled: bool = True
    eval_video_parallel_sampler_enabled: bool = True
    num_random_trajectories: int = 9
    eval_record_video: bool = True
    eval_plot_axis: list[float] | None = None
    video_skip_frames: int = 1
    eval_video_grid_rows: int = 3
    eval_video_grid_cols: int = 3
    eval_video_fps: int = 15
    eval_video_policy_mode: str = "stochastic"
    eval_state_coverage_trajectories: int = 48
    workspace_root: str = "/share/shangyy"

    # Env / RL plumbing.
    render_size: int = 64
    framestack: int = 1
    action_repeat: int = 1
    time_limit: int = 200
    flatten_obs: int = 1
    camera: str = "corner"
    dmc_camera: int = -1
    encoder: int = 1
    common_lr: float = 1e-4
    lr_op: float | None = None
    sac_lr_q: float | None = None
    sac_lr_a: float | None = None
    alpha: float = 0.01
    sac_discount: float = 0.99
    sac_tau: float = 5e-3
    sac_scale_reward: float = 1.0
    sac_target_coef: float = 1.0
    sac_min_buffer_size: int = 10000
    sac_max_buffer_size: int = 300000
    policy_delay: int = 1
    actor_start_steps: int = 0
    rl_encoder_type: str = "original"
    finetune_rl_encoder: bool = False
    replay_staging_enabled: bool = True
    replay_staging_pin_memory: bool = True

    # Seed collection.
    seed_steps: int = 5000

    # Coverage encoder.
    cov_encoder_type: str = "checkpoint"
    cov_encoder_path: str = ""
    cov_resnet_path: str = "/home/shangyy/models/resnet-101/"
    cov_dino_path: str = "/home/shangyy/models/dinov3-vits16-pretrain-lvd1689m/"
    cov_latent_dim: int = 32
    # Deprecated MASS-side distillation knobs. The generic
    # train_cov_encoder_distill.py entrypoint owns distillation.
    teacher_path: str = "/home/shangyy/models/resnet-101/"
    cov_lr: float = 1e-4
    cov_warmup_steps: int = 2000
    freeze_cov_after_warmup: bool = True
    online_cov_update: bool = False
    lambda_dist: float = 1.0
    lambda_aug: float = 1.0
    lambda_var: float = 1.0
    lambda_cov: float = 0.04
    lambda_inv: float = 0.1
    cov_var_gamma: float = 1.0

    # MASS.
    mass_c: int = 64
    mass_psi: int = 128
    mass_alpha: float = 1.0
    mass_short_size: int = 50000
    mass_long_size: int = 300000
    mass_w_short: float = 0.7
    mass_w_long: float = 0.3
    mass_refresh_interval: int = 5
    mass_refresh_num: int = 8
    mass_reward_clip: float = 5.0
    mass_encode_batch_size: int = 128
    mass_device: str = "auto"

    # Intrinsic reward shaping.
    lambda_action: float = 1e-3
    lambda_delta_action: float = 1e-3
    lambda_done: float = 5.0

    # Debug.
    smoke_test: bool = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    hidden = argparse.SUPPRESS

    parser.add_argument("--algo", type=_mass_algo, default=MassPixelConfig.algo, metavar="{mass}")
    parser.add_argument("--algorithm", dest="algo", type=_mass_algo, help=hidden)
    parser.add_argument("--seed", type=int, default=MassPixelConfig.seed)
    parser.add_argument("--device", type=str, default=MassPixelConfig.device)
    parser.add_argument("--env", dest="task", type=str, default=MassPixelConfig.task)
    parser.add_argument("--task", dest="task", type=str)
    parser.add_argument("--n_epochs", type=int, default=MassPixelConfig.n_epochs)
    parser.add_argument("--traj_batch_size", type=int, default=MassPixelConfig.traj_batch_size)
    parser.add_argument("--trans_minibatch_size", type=int, default=MassPixelConfig.trans_minibatch_size)
    parser.add_argument("--trans_optimization_epochs", type=int, default=MassPixelConfig.trans_optimization_epochs)
    parser.add_argument("--n_epochs_per_log", type=int, default=MassPixelConfig.n_epochs_per_log)
    parser.add_argument("--n_epochs_per_eval", type=int, default=MassPixelConfig.n_epochs_per_eval)
    parser.add_argument("--n_epochs_per_save", type=int, default=MassPixelConfig.n_epochs_per_save)
    parser.add_argument("--n_epochs_per_pt_save", type=int, default=MassPixelConfig.n_epochs_per_pt_save)
    parser.add_argument("--n_parallel", type=int, default=MassPixelConfig.n_parallel)
    parser.add_argument("--n_thread", type=int, default=MassPixelConfig.n_thread)
    parser.add_argument("--parallel_sampler_enabled", action=argparse.BooleanOptionalAction, default=MassPixelConfig.parallel_sampler_enabled)
    parser.add_argument("--parallel_sampler_num_workers", type=int, default=MassPixelConfig.parallel_sampler_num_workers)
    parser.add_argument("--parallel_sampler_fail_open", type=_str_to_bool, default=MassPixelConfig.parallel_sampler_fail_open)
    parser.add_argument("--eval_parallel_sampler_enabled", action=argparse.BooleanOptionalAction, default=MassPixelConfig.eval_parallel_sampler_enabled)
    parser.add_argument("--eval_video_parallel_sampler_enabled", type=_str_to_bool, default=MassPixelConfig.eval_video_parallel_sampler_enabled)
    parser.add_argument("--num_random_trajectories", type=int, default=MassPixelConfig.num_random_trajectories)
    parser.add_argument("--eval_record_video", type=_str_to_bool, default=MassPixelConfig.eval_record_video)
    parser.add_argument("--eval_plot_axis", type=float, default=MassPixelConfig.eval_plot_axis, nargs="*")
    parser.add_argument("--video_skip_frames", type=int, default=MassPixelConfig.video_skip_frames)
    parser.add_argument("--eval_video_grid_rows", type=int, default=MassPixelConfig.eval_video_grid_rows)
    parser.add_argument("--eval_video_grid_cols", type=int, default=MassPixelConfig.eval_video_grid_cols)
    parser.add_argument("--eval_video_fps", type=int, default=MassPixelConfig.eval_video_fps)
    parser.add_argument(
        "--eval_video_policy_mode",
        type=str,
        default=MassPixelConfig.eval_video_policy_mode,
        choices=["stochastic", "deterministic"],
    )
    parser.add_argument(
        "--eval_state_coverage_trajectories",
        type=int,
        default=MassPixelConfig.eval_state_coverage_trajectories,
        help="Number of deterministic eval trajectories used only for state coverage metrics",
    )
    parser.add_argument("--workspace_root", type=str, default=MassPixelConfig.workspace_root)

    parser.add_argument("--render_size", type=int, default=MassPixelConfig.render_size)
    parser.add_argument("--framestack", type=int, default=MassPixelConfig.framestack)
    parser.add_argument("--action_repeat", type=int, default=MassPixelConfig.action_repeat)
    parser.add_argument("--time_limit", type=int, default=MassPixelConfig.time_limit)
    parser.add_argument("--flatten_obs", type=int, default=MassPixelConfig.flatten_obs, choices=[0, 1])
    parser.add_argument("--camera", type=str, default=MassPixelConfig.camera)
    parser.add_argument("--dmc_camera", type=int, default=MassPixelConfig.dmc_camera)
    parser.add_argument("--encoder", type=int, default=MassPixelConfig.encoder, choices=[0, 1])
    parser.add_argument("--common_lr", type=float, default=MassPixelConfig.common_lr)
    parser.add_argument("--lr_op", type=float, default=MassPixelConfig.lr_op)
    parser.add_argument("--sac_lr_q", type=float, default=MassPixelConfig.sac_lr_q)
    parser.add_argument("--sac_lr_a", type=float, default=MassPixelConfig.sac_lr_a)
    parser.add_argument("--alpha", type=float, default=MassPixelConfig.alpha)
    parser.add_argument("--sac_discount", type=float, default=MassPixelConfig.sac_discount)
    parser.add_argument("--sac_tau", type=float, default=MassPixelConfig.sac_tau)
    parser.add_argument("--sac_scale_reward", type=float, default=MassPixelConfig.sac_scale_reward)
    parser.add_argument("--sac_target_coef", type=float, default=MassPixelConfig.sac_target_coef)
    parser.add_argument("--sac_min_buffer_size", type=int, default=MassPixelConfig.sac_min_buffer_size)
    parser.add_argument("--sac_max_buffer_size", type=int, default=MassPixelConfig.sac_max_buffer_size)
    parser.add_argument("--policy_delay", type=int, default=MassPixelConfig.policy_delay)
    parser.add_argument("--actor_start_steps", type=int, default=MassPixelConfig.actor_start_steps)
    parser.add_argument(
        "--encoder_type",
        dest="rl_encoder_type",
        type=str,
        default=MassPixelConfig.rl_encoder_type,
        choices=["original", "resnet-101", "dinov3", "galaxea-r1lite-triview"],
    )
    parser.add_argument("--rl_encoder_type", dest="rl_encoder_type", type=str, choices=["original", "resnet-101", "dinov3", "galaxea-r1lite-triview"], help=hidden)
    parser.add_argument("--finetune_encoder", dest="finetune_rl_encoder", action="store_true", default=MassPixelConfig.finetune_rl_encoder)
    parser.add_argument("--finetune_rl_encoder", type=_str_to_bool, default=MassPixelConfig.finetune_rl_encoder, help=hidden)
    parser.add_argument("--replay_staging_enabled", action=argparse.BooleanOptionalAction, default=MassPixelConfig.replay_staging_enabled)
    parser.add_argument("--replay_staging_pin_memory", type=_str_to_bool, default=MassPixelConfig.replay_staging_pin_memory)

    parser.add_argument("--seed_steps", type=int, default=MassPixelConfig.seed_steps)

    parser.add_argument(
        "--cov_encoder_type",
        type=str,
        default=MassPixelConfig.cov_encoder_type,
        choices=["checkpoint", "resnet-101", "dinov3", "dino-v3"],
        help="Coverage encoder backend: distilled checkpoint, direct local ResNet-101, or direct local DINOv3",
    )
    parser.add_argument(
        "--cov_encoder_path",
        type=str,
        default=MassPixelConfig.cov_encoder_path,
        help="Path to a checkpoint produced by train_cov_encoder_distill.py when cov_encoder_type=checkpoint",
    )
    parser.add_argument("--cov_resnet_path", type=str, default=MassPixelConfig.cov_resnet_path)
    parser.add_argument("--cov_dino_path", type=str, default=MassPixelConfig.cov_dino_path)
    parser.add_argument("--cov_latent_dim", type=int, default=MassPixelConfig.cov_latent_dim)
    parser.add_argument("--teacher_path", type=str, default=MassPixelConfig.teacher_path, help=argparse.SUPPRESS)
    parser.add_argument("--cov_lr", type=float, default=MassPixelConfig.cov_lr, help=argparse.SUPPRESS)
    parser.add_argument("--cov_warmup_steps", type=int, default=MassPixelConfig.cov_warmup_steps, help=argparse.SUPPRESS)
    parser.add_argument(
        "--freeze_cov_after_warmup",
        type=_str_to_bool,
        default=MassPixelConfig.freeze_cov_after_warmup,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--online_cov_update", type=_str_to_bool, default=MassPixelConfig.online_cov_update)
    parser.add_argument("--lambda_dist", type=float, default=MassPixelConfig.lambda_dist, help=argparse.SUPPRESS)
    parser.add_argument("--lambda_aug", type=float, default=MassPixelConfig.lambda_aug, help=argparse.SUPPRESS)
    parser.add_argument("--lambda_var", type=float, default=MassPixelConfig.lambda_var, help=argparse.SUPPRESS)
    parser.add_argument("--lambda_cov", type=float, default=MassPixelConfig.lambda_cov, help=argparse.SUPPRESS)
    parser.add_argument("--lambda_inv", type=float, default=MassPixelConfig.lambda_inv, help=argparse.SUPPRESS)
    parser.add_argument("--cov_var_gamma", type=float, default=MassPixelConfig.cov_var_gamma, help=argparse.SUPPRESS)

    parser.add_argument("--mass_c", type=int, default=MassPixelConfig.mass_c)
    parser.add_argument("--mass_psi", type=int, default=MassPixelConfig.mass_psi)
    parser.add_argument("--mass_alpha", type=float, default=MassPixelConfig.mass_alpha)
    parser.add_argument("--mass_short_size", type=int, default=MassPixelConfig.mass_short_size)
    parser.add_argument("--mass_long_size", type=int, default=MassPixelConfig.mass_long_size)
    parser.add_argument("--mass_w_short", type=float, default=MassPixelConfig.mass_w_short)
    parser.add_argument("--mass_w_long", type=float, default=MassPixelConfig.mass_w_long)
    parser.add_argument(
        "--mass_refresh_interval",
        type=int,
        default=MassPixelConfig.mass_refresh_interval,
        help="MASS partition refresh interval in training epochs; <=0 disables refresh",
    )
    parser.add_argument("--mass_refresh_num", type=int, default=MassPixelConfig.mass_refresh_num)
    parser.add_argument("--mass_reward_clip", type=float, default=MassPixelConfig.mass_reward_clip)
    parser.add_argument(
        "--mass_encode_batch_size",
        type=int,
        default=MassPixelConfig.mass_encode_batch_size,
        help="Coverage encoder micro-batch size for MASS feature extraction",
    )
    parser.add_argument(
        "--mass_device",
        type=str,
        default=MassPixelConfig.mass_device,
        choices=["auto", "cuda", "cpu"],
        help="Device for MASS z buffers/counts; auto follows the RL device",
    )

    parser.add_argument("--lambda_action", type=float, default=MassPixelConfig.lambda_action)
    parser.add_argument("--lambda_delta_action", type=float, default=MassPixelConfig.lambda_delta_action)
    parser.add_argument("--lambda_done", type=float, default=MassPixelConfig.lambda_done)

    parser.add_argument("--smoke_test", type=_str_to_bool, default=MassPixelConfig.smoke_test)
    return parser


def parse_args(argv=None) -> MassPixelConfig:
    namespace = build_arg_parser().parse_args(argv)
    cfg = MassPixelConfig(**vars(namespace))
    if cfg.algo == "mass_pixel":
        cfg.algo = "mass"
    if cfg.cov_encoder_type == "dino-v3":
        cfg.cov_encoder_type = "dinov3"
    _validate(cfg)
    if cfg.smoke_test:
        cap_source = cfg.seed_steps if cfg.seed_steps > 0 else cfg.traj_batch_size * cfg.time_limit
        cfg.trans_minibatch_size = min(cfg.trans_minibatch_size, max(1, min(32, cap_source)))
        cfg.sac_min_buffer_size = min(cfg.sac_min_buffer_size, cfg.trans_minibatch_size)
    return cfg


def _validate(cfg: MassPixelConfig) -> None:
    if cfg.seed_steps < 0:
        raise ValueError("seed_steps must be >= 0")
    if cfg.n_epochs < 0:
        raise ValueError("n_epochs must be >= 0")
    if cfg.traj_batch_size < 1:
        raise ValueError("traj_batch_size must be >= 1")
    if cfg.trans_minibatch_size < 1:
        raise ValueError("trans_minibatch_size must be >= 1")
    if cfg.trans_optimization_epochs < 0:
        raise ValueError("trans_optimization_epochs must be >= 0")
    if cfg.n_parallel < 1:
        raise ValueError("n_parallel must be >= 1")
    if cfg.n_thread < 1:
        raise ValueError("n_thread must be >= 1")
    if cfg.parallel_sampler_num_workers < 0:
        raise ValueError("parallel_sampler_num_workers must be >= 0")
    if cfg.num_random_trajectories < 1:
        raise ValueError("num_random_trajectories must be >= 1")
    if cfg.eval_state_coverage_trajectories < 1:
        raise ValueError("eval_state_coverage_trajectories must be >= 1")
    if cfg.video_skip_frames < 1:
        raise ValueError("video_skip_frames must be >= 1")
    if cfg.eval_video_grid_rows < 1 or cfg.eval_video_grid_cols < 1:
        raise ValueError("eval_video_grid_rows and eval_video_grid_cols must be >= 1")
    if cfg.eval_video_fps < 1:
        raise ValueError("eval_video_fps must be >= 1")
    if cfg.sac_min_buffer_size < 1 or cfg.sac_max_buffer_size < 1:
        raise ValueError("sac_min_buffer_size and sac_max_buffer_size must be >= 1")
    if cfg.sac_max_buffer_size < cfg.sac_min_buffer_size:
        raise ValueError("sac_max_buffer_size must be >= sac_min_buffer_size")
    if cfg.encoder != 1:
        raise ValueError("mass currently requires --encoder 1; encoder 0 is not supported")
    if cfg.alpha <= 0:
        raise ValueError("alpha must be > 0")
    if cfg.cov_encoder_type == "checkpoint" and not cfg.cov_encoder_path:
        raise ValueError(
            "--cov_encoder_path is required when --cov_encoder_type checkpoint. "
            "Create it first with train_cov_encoder_distill.py."
        )
    if cfg.cov_encoder_type == "resnet-101" and not cfg.cov_resnet_path:
        raise ValueError("cov_resnet_path must be set when cov_encoder_type=resnet-101")
    if cfg.cov_encoder_type == "dinov3" and not cfg.cov_dino_path:
        raise ValueError("cov_dino_path must be set when cov_encoder_type=dinov3")
    if cfg.mass_c < 1 or cfg.mass_psi < 1:
        raise ValueError("mass_c and mass_psi must be >= 1")
    if cfg.mass_short_size < 1 or cfg.mass_long_size < 1:
        raise ValueError("mass short/long sizes must be >= 1")
    if cfg.mass_w_short < 0 or cfg.mass_w_long < 0:
        raise ValueError("mass weights must be non-negative")
    if cfg.mass_w_short + cfg.mass_w_long <= 0:
        raise ValueError("at least one MASS weight must be positive")
    if cfg.mass_encode_batch_size < 1:
        raise ValueError("mass_encode_batch_size must be >= 1")
    if cfg.online_cov_update:
        raise NotImplementedError(
            "online_cov_update=True is intentionally not enabled in v1. "
            "The default frozen coverage encoder keeps B_short/B_long coordinates stable."
        )
