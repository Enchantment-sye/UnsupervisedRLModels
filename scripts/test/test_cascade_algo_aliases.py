import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
TRAIN_DIR = os.path.join(REPO_ROOT, "scripts", "train")
for path in (REPO_ROOT, SRC_DIR, TRAIN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.experimental.idk_csd_config import IdkCsdConfig
from core.experimental.idk_csd_variant import IdkCsdVariant
from core.metra_config import MetraConfig, get_parser, make_config_from_args
from core.metra_variants import (
    CsdVariant,
    DadsVariant,
    DiaynVariant,
    IksdVariant,
    LsdVariant,
    MetraVariant,
    VariantFactory,
)
from core.stage_contract import get_base_algo_name, should_build_skill_dynamics
from workspace_common import WorkspaceContext


def build_cfg(algo, extra_args=None, cls=None):
    parser = get_parser()
    argv = ["--algo", algo, "--dim_skill", "2"]
    if extra_args:
        argv.extend(extra_args)
    args = parser.parse_args(argv)
    cfg_cls = cls if cls is not None else (IdkCsdConfig if get_base_algo_name(algo) == "idk_csd" else MetraConfig)
    return make_config_from_args(args, cls=cfg_cls)


def assert_variant(cfg, expected_cls):
    variant = VariantFactory.create(cfg)
    assert isinstance(variant, expected_cls), f"expected {expected_cls.__name__}, got {type(variant).__name__}"


def test_alias_config_and_variants():
    cases = [
        ("metra_cascade", "metra", MetraVariant, {"dual_dist": "one"}),
        ("diayn_cascade", "diayn", DiaynVariant, {"inner": 0, "dual_reg": 0}),
        ("lsd_cascade", "lsd", LsdVariant, {"dual_dist": "l2"}),
        ("csd_cascade", "csd", CsdVariant, {"dual_dist": "s2_from_s"}),
        ("iksd_cascade", "iksd", IksdVariant, {"dual_dist": "kernel_sim_dist", "use_kme": True}),
        ("dads_cascade", "dads", DadsVariant, {}),
        ("idk_csd_cascade", "idk_csd", IdkCsdVariant, {}),
    ]

    for algo_name, base_algo, variant_cls, expected_values in cases:
        cfg = build_cfg(algo_name)
        assert cfg.algo.algo == algo_name
        assert get_base_algo_name(cfg) == base_algo
        assert cfg.cascade.use_cascade is True
        for key, expected in expected_values.items():
            assert getattr(cfg.algo, key) == expected, f"{algo_name} expected {key}={expected!r}"
        assert_variant(cfg, variant_cls)


def test_dads_hierarchical_phi_rejected():
    parser = get_parser()
    args = parser.parse_args([
        "--algo", "dads_cascade",
        "--dim_skill", "2",
        "--use_hierarchical_skill",
        "--use_hierarchical_phi",
    ])
    try:
        make_config_from_args(args)
    except ValueError as exc:
        assert "not dads" in str(exc)
    else:
        raise AssertionError("dads_cascade should reject use_hierarchical_phi")


def test_dads_skill_dynamics_branch():
    cfg = build_cfg("dads_cascade")
    assert should_build_skill_dynamics(cfg) is True


def test_iksd_workspace_path_retains_subsample_component():
    cfg = build_cfg(
        "iksd_cascade",
        extra_args=["--task", "tmp_cascade_alias_task", "--idk_subsample_size", "123"],
    )
    ctx = WorkspaceContext.create(cfg)
    try:
        expected = os.path.join("tmp_cascade_alias_task", "iksd_cascade", "pre_training", "freeze_visual", "123")
        assert expected in ctx.work_dir, ctx.work_dir
    finally:
        shutil.rmtree(ctx.work_dir, ignore_errors=True)


def main():
    test_alias_config_and_variants()
    test_dads_hierarchical_phi_rejected()
    test_dads_skill_dynamics_branch()
    test_iksd_workspace_path_retains_subsample_component()
    print("cascade alias smoke tests passed")


if __name__ == "__main__":
    main()
