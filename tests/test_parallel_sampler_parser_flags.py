import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from config.parser import get_parser, make_config_from_args


def test_parallel_speedup_flags_default_true_when_omitted():
    cfg = make_config_from_args(get_parser().parse_args([]))

    assert cfg.train.parallel_sampler_enabled is True
    assert cfg.train.eval_parallel_sampler_enabled is True
    assert cfg.train.async_video_encoding is True
    assert cfg.train.replay_staging_enabled is True


def test_parallel_speedup_flags_true_when_present():
    args = get_parser().parse_args([
        "--parallel_sampler_enabled",
        "--eval_parallel_sampler_enabled",
        "--async_video_encoding",
        "--replay_staging_enabled",
    ])
    cfg = make_config_from_args(args)

    assert cfg.train.parallel_sampler_enabled is True
    assert cfg.train.eval_parallel_sampler_enabled is True
    assert cfg.train.async_video_encoding is True
    assert cfg.train.replay_staging_enabled is True


def test_parallel_speedup_flags_can_be_disabled_explicitly():
    args = get_parser().parse_args([
        "--no-parallel_sampler_enabled",
        "--no-eval_parallel_sampler_enabled",
        "--no-async_video_encoding",
        "--no-replay_staging_enabled",
    ])
    cfg = make_config_from_args(args)

    assert cfg.train.parallel_sampler_enabled is False
    assert cfg.train.eval_parallel_sampler_enabled is False
    assert cfg.train.async_video_encoding is False
    assert cfg.train.replay_staging_enabled is False
