import os
import sys

from torch.utils.tensorboard import SummaryWriter


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

from audit_metra_tb_tags import audit_logdir, main
from metra_log_specs import LEGACY_ALIAS_TAGS, get_tag_spec


def _write_scalars(logdir, tags):
    writer = SummaryWriter(log_dir=str(logdir))
    for step, tag in enumerate(sorted(tags)):
        writer.add_scalar(tag, float(step + 1), step)
    writer.flush()
    writer.close()


def test_audit_finds_missing_required_tags(tmp_path):
    spec = get_tag_spec("kitchen")
    partial_required_tags = {
        "TrainSp/METRA/Alpha",
        "TrainSp/METRA/AlphaLr",
        "eval/KitchenOverall",
        "eval/avg_completed_tasks",
        "METRA/PureRewardMean",
    }
    _write_scalars(tmp_path, partial_required_tags)

    result = audit_logdir(tmp_path, "kitchen")

    assert "TrainSp/METRA/PureRewardMean" in result["missing_required_tags"]
    assert "eval/KitchenAvgCompletedTasksPerTraj" in result["missing_required_tags"]
    assert "METRA/PureRewardMean" in result["legacy_alias_tags"]
    assert "METRA/PureRewardMean" not in result["missing_optional_tags"]
    assert "eval/avg_completed_tasks" in result["legacy_alias_tags"]
    assert (
        result["legacy_alias_tags"]["eval/avg_completed_tasks"]
        == LEGACY_ALIAS_TAGS["eval/avg_completed_tasks"]
    )
    assert set(result["missing_required_tags"]) == spec.required_tags - partial_required_tags
    assert "eval/avg_completed_tasks" not in result["extra_tags"]


def test_strict_mode_succeeds_when_all_required_tags_exist(tmp_path):
    spec = get_tag_spec("ant")
    _write_scalars(tmp_path, spec.required_tags)

    result = audit_logdir(tmp_path, "ant")

    assert result["missing_required_tags"] == []
    assert "TrainSp/METRA/Alpha" in result["found_scalar_tags"]
    assert "METRA/Alpha" not in spec.required_tags
    assert "METRA/Alpha" not in result["missing_optional_tags"]
    assert "eval/undiscounted_return" not in result["missing_optional_tags"]
    assert "eval/TotalEnvSteps" not in result["missing_optional_tags"]
    assert main(["--logdir", str(tmp_path), "--env", "ant", "--strict", "1"]) == 0
