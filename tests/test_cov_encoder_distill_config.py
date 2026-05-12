import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.cov_encoder.config import parse_args as parse_distill_args
from core.cov_encoder.utils import timestamped_distill_work_dir
from core.mass.config import MassPixelConfig
from core.mass.config import parse_args as parse_mass_args
from core.mass.trainer import MassPixelTrainer


def test_distill_parser_accepts_algorithm_agnostic_knobs(tmp_path):
    cfg = parse_distill_args(
        [
            "--env",
            "debug_dummy",
            "--distill_sample_steps",
            "100",
            "--distill_train_steps",
            "10",
            "--distill_batch_size",
            "16",
            "--distill_lr",
            "1e-4",
            "--cov_encoder_save_path",
            str(tmp_path / "cov.pt"),
            "--smoke_test",
            "True",
        ]
    )

    assert cfg.task == "debug_dummy"
    assert cfg.distill_sample_steps == 100
    assert cfg.distill_train_steps == 10
    assert cfg.distill_batch_size == 16
    assert cfg.cov_encoder_save_path.endswith("cov.pt")


def test_default_distill_work_dir_shape(tmp_path):
    work_dir = timestamped_distill_work_dir(str(tmp_path), "debug_dummy", 7)
    assert str(tmp_path) in work_dir
    assert "debug_dummy/cov_encoder_distill/resnet101/" in work_dir
    assert work_dir.endswith("_seed7")


def test_mass_accepts_checkpoint_or_direct_cov_encoder(tmp_path):
    cfg = parse_mass_args(
        [
            "--algo",
            "mass",
            "--env",
            "debug_dummy",
            "--encoder",
            "1",
            "--cov_encoder_type",
            "checkpoint",
            "--cov_encoder_path",
            str(tmp_path / "cov.pt"),
            "--seed_steps",
            "1",
            "--n_epochs",
            "1",
            "--traj_batch_size",
            "2",
            "--trans_minibatch_size",
            "4",
            "--trans_optimization_epochs",
            "3",
            "--sac_min_buffer_size",
            "4",
            "--sac_max_buffer_size",
            "16",
            "--sac_lr_a",
            "2e-4",
            "--sac_lr_q",
            "3e-4",
            "--alpha",
            "0.02",
            "--mass_alpha",
            "1.25",
            "--mass_encode_batch_size",
            "17",
            "--mass_device",
            "cpu",
            "--eval_plot_axis",
            "-50",
            "50",
            "-40",
            "40",
            "--n_parallel",
            "2",
            "--parallel_sampler_num_workers",
            "2",
            "--no-parallel_sampler_enabled",
            "--eval_parallel_sampler_enabled",
            "--eval_video_parallel_sampler_enabled",
            "False",
            "--eval_state_coverage_trajectories",
            "48",
            "--no-replay_staging_enabled",
        ]
    )
    assert cfg.algo == "mass"
    assert cfg.encoder == 1
    assert cfg.cov_encoder_path.endswith("cov.pt")
    assert cfg.n_epochs == 1
    assert cfg.traj_batch_size == 2
    assert cfg.trans_minibatch_size == 4
    assert cfg.trans_optimization_epochs == 3
    assert cfg.sac_min_buffer_size == 4
    assert cfg.sac_max_buffer_size == 16
    assert cfg.sac_lr_a == 2e-4
    assert cfg.sac_lr_q == 3e-4
    assert cfg.alpha == 0.02
    assert cfg.mass_alpha == 1.25
    assert cfg.mass_encode_batch_size == 17
    assert cfg.mass_device == "cpu"
    assert cfg.eval_plot_axis == [-50.0, 50.0, -40.0, 40.0]
    assert cfg.n_parallel == 2
    assert cfg.parallel_sampler_num_workers == 2
    assert cfg.parallel_sampler_enabled is False
    assert cfg.eval_parallel_sampler_enabled is True
    assert cfg.eval_video_parallel_sampler_enabled is False
    assert cfg.eval_state_coverage_trajectories == 48
    assert cfg.replay_staging_enabled is False

    direct_cfg = parse_mass_args(
        [
            "--env",
            "debug_dummy",
            "--cov_encoder_type",
            "resnet-101",
            "--seed_steps",
            "1",
            "--n_epochs",
            "1",
        ]
    )
    assert direct_cfg.cov_encoder_type == "resnet-101"
    assert direct_cfg.cov_encoder_path == ""
    assert direct_cfg.mass_refresh_interval == 5

    dino_alias_cfg = parse_mass_args(
        [
            "--env",
            "debug_dummy",
            "--cov_encoder_type",
            "dino-v3",
            "--seed_steps",
            "1",
            "--n_epochs",
            "1",
        ]
    )
    assert dino_alias_cfg.cov_encoder_type == "dinov3"

    source = inspect.getsource(MassPixelTrainer)
    assert "ResNet101Teacher" not in source
    assert "_warmup_coverage_encoder" not in source


def test_mass_legacy_algo_alias_and_encoder_validation():
    alias_cfg = parse_mass_args(
        [
            "--algo",
            "mass_pixel",
            "--env",
            "debug_dummy",
            "--cov_encoder_type",
            "resnet-101",
            "--seed_steps",
            "1",
            "--n_epochs",
            "1",
        ]
    )
    assert alias_cfg.algo == "mass"

    with pytest.raises(ValueError, match="mass currently requires --encoder 1"):
        parse_mass_args(
            [
                "--algo",
                "mass",
                "--env",
                "debug_dummy",
                "--encoder",
                "0",
                "--cov_encoder_type",
                "resnet-101",
            ]
        )


def test_mass_metra_config_wires_sac_alpha_lr_eval_axis_and_replay_staging():
    cfg = MassPixelConfig(
        task="debug_dummy",
        cov_encoder_type="resnet-101",
        alpha=0.03,
        sac_lr_a=1e-9,
        eval_plot_axis=[-1.0, 1.0, -2.0, 2.0],
        replay_staging_enabled=False,
        replay_staging_pin_memory=False,
        mass_encode_batch_size=11,
        mass_device="cpu",
    )
    trainer = MassPixelTrainer.__new__(MassPixelTrainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")
    trainer.work_dir = "/tmp/mass-test"

    metra_cfg = trainer._build_metra_config()

    assert metra_cfg.net.encoder == 1
    assert metra_cfg.algo.alpha == 0.03
    assert metra_cfg.train.sac_lr_a == 1e-9
    assert metra_cfg.log.eval_plot_axis == [-1.0, 1.0, -2.0, 2.0]
    assert metra_cfg.replay_staging_enabled is False
    assert metra_cfg.replay_staging_pin_memory is False
    assert metra_cfg.train.replay_staging_enabled is False
    assert metra_cfg.train.replay_staging_pin_memory is False

    assert trainer._resolve_mass_device("auto") == torch.device("cpu")
    assert trainer._resolve_mass_device("cpu") == torch.device("cpu")
