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


@dataclass
class CovEncoderDistillConfig:
    seed: int = 0
    device: str = "cuda"
    task: str = "debug_dummy"
    workspace_root: str = "/share/shangyy"
    log_interval: int = 100

    # Env / image plumbing.
    render_size: int = 64
    framestack: int = 1
    action_repeat: int = 1
    time_limit: int = 200
    flatten_obs: int = 1
    camera: str = "corner"
    dmc_camera: int = -1
    encoder: int = 1

    # Distillation.
    distill_sample_steps: int = 5000
    distill_train_steps: int = 2000
    distill_batch_size: int = 256
    distill_lr: float = 1e-4
    teacher_path: str = "/home/shangyy/models/resnet-101/"
    cov_latent_dim: int = 32
    cov_encoder_save_path: str = ""
    lambda_dist: float = 1.0
    lambda_aug: float = 1.0
    lambda_var: float = 1.0
    lambda_cov: float = 0.04
    lambda_inv: float = 0.1
    cov_var_gamma: float = 1.0

    smoke_test: bool = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=CovEncoderDistillConfig.seed)
    parser.add_argument("--device", type=str, default=CovEncoderDistillConfig.device)
    parser.add_argument("--env", dest="task", type=str, default=CovEncoderDistillConfig.task)
    parser.add_argument("--task", dest="task", type=str)
    parser.add_argument("--workspace_root", type=str, default=CovEncoderDistillConfig.workspace_root)
    parser.add_argument("--log_interval", type=int, default=CovEncoderDistillConfig.log_interval)

    parser.add_argument("--render_size", type=int, default=CovEncoderDistillConfig.render_size)
    parser.add_argument("--framestack", type=int, default=CovEncoderDistillConfig.framestack)
    parser.add_argument("--action_repeat", type=int, default=CovEncoderDistillConfig.action_repeat)
    parser.add_argument("--time_limit", type=int, default=CovEncoderDistillConfig.time_limit)
    parser.add_argument("--flatten_obs", type=int, default=CovEncoderDistillConfig.flatten_obs, choices=[0, 1])
    parser.add_argument("--camera", type=str, default=CovEncoderDistillConfig.camera)
    parser.add_argument("--dmc_camera", type=int, default=CovEncoderDistillConfig.dmc_camera)

    parser.add_argument("--distill_sample_steps", type=int, default=CovEncoderDistillConfig.distill_sample_steps)
    parser.add_argument("--distill_train_steps", type=int, default=CovEncoderDistillConfig.distill_train_steps)
    parser.add_argument("--distill_batch_size", type=int, default=CovEncoderDistillConfig.distill_batch_size)
    parser.add_argument("--distill_lr", type=float, default=CovEncoderDistillConfig.distill_lr)
    parser.add_argument("--teacher_path", type=str, default=CovEncoderDistillConfig.teacher_path)
    parser.add_argument("--cov_latent_dim", type=int, default=CovEncoderDistillConfig.cov_latent_dim)
    parser.add_argument("--cov_encoder_save_path", type=str, default=CovEncoderDistillConfig.cov_encoder_save_path)
    parser.add_argument("--lambda_dist", type=float, default=CovEncoderDistillConfig.lambda_dist)
    parser.add_argument("--lambda_aug", type=float, default=CovEncoderDistillConfig.lambda_aug)
    parser.add_argument("--lambda_var", type=float, default=CovEncoderDistillConfig.lambda_var)
    parser.add_argument("--lambda_cov", type=float, default=CovEncoderDistillConfig.lambda_cov)
    parser.add_argument("--lambda_inv", type=float, default=CovEncoderDistillConfig.lambda_inv)
    parser.add_argument("--cov_var_gamma", type=float, default=CovEncoderDistillConfig.cov_var_gamma)

    parser.add_argument("--smoke_test", type=_str_to_bool, default=CovEncoderDistillConfig.smoke_test)
    return parser


def parse_args(argv=None) -> CovEncoderDistillConfig:
    namespace = build_arg_parser().parse_args(argv)
    cfg = CovEncoderDistillConfig(**vars(namespace))
    _validate(cfg)
    if cfg.smoke_test:
        cfg.distill_batch_size = min(cfg.distill_batch_size, max(1, min(32, cfg.distill_sample_steps)))
    return cfg


def _validate(cfg: CovEncoderDistillConfig) -> None:
    if cfg.distill_sample_steps < 1:
        raise ValueError("distill_sample_steps must be >= 1")
    if cfg.distill_train_steps < 1:
        raise ValueError("distill_train_steps must be >= 1")
    if cfg.distill_batch_size < 1:
        raise ValueError("distill_batch_size must be >= 1")
    if cfg.cov_latent_dim < 1:
        raise ValueError("cov_latent_dim must be >= 1")
