import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath("src"))

from src.core.task_adapter import SkillDiscoveryTaskAdapter


class _FakeLocomotionEnv:
    def calc_eval_metrics(self, trajectories, is_option_trajectories=False):
        assert is_option_trajectories is True
        return {
            "MjNumTrajs": 2,
            "MjAvgTrajLen": 4.5,
            "MjNumCoords": 11,
            "MjNumUniqueCoords": 7,
            "PolicyStateCoverageXYBins": 7,
            "PolicyFinalXYDispMean": 3.25,
            "PolicyFinalXYDispMax": 6.5,
            "PolicyXRange": 9.0,
            "PolicyYRange": 0.0,
            "PolicyMeanSpeed": 1.75,
        }


class _FakeKitchenEnv:
    def calc_eval_metrics(self, trajectories, is_option_trajectories=False):
        return {
            "KitchenOverall": 1.0,
            "KitchenPolicyTaskCoverage": 0.5,
        }


def _adapter(task, env):
    return SkillDiscoveryTaskAdapter(
        SimpleNamespace(task=task),
        env,
        agent=None,
        work_dir=None,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )


def test_policy_coverage_forwards_all_locomotion_tags_for_cheetah():
    adapter = _adapter("dmc_cheetah_run_forward", _FakeLocomotionEnv())

    metrics = adapter.compute_policy_coverage_metrics([{}])

    assert metrics == {
        "MjNumTrajs": 2.0,
        "MjAvgTrajLen": 4.5,
        "MjNumCoords": 11.0,
        "MjNumUniqueCoords": 7.0,
        "PolicyStateCoverageXYBins": 7.0,
        "PolicyFinalXYDispMean": 3.25,
        "PolicyFinalXYDispMax": 6.5,
        "PolicyXRange": 9.0,
        "PolicyYRange": 0.0,
        "PolicyMeanSpeed": 1.75,
    }


def test_policy_coverage_ignores_non_locomotion_eval_metrics():
    adapter = _adapter("d4rl_kitchen", _FakeKitchenEnv())

    assert adapter.compute_policy_coverage_metrics([{}]) == {}


def test_policy_coverage_collects_deterministic_eval_trajectories():
    adapter = _adapter("ant", _FakeLocomotionEnv())
    adapter.cfg.num_random_trajectories = 3
    adapter.cfg.seed = 10
    captured = {}

    def _capture_collect(extras, **kwargs):
        captured["extras"] = extras
        captured.update(kwargs)
        return ["trajectory"]

    adapter._build_quantitative_eval_extras = lambda num_eval_trajs: [None] * num_eval_trajs
    adapter.collect_policy_trajectories = _capture_collect

    assert adapter.collect_policy_coverage_trajectories(total_epoch=7) == ["trajectory"]
    assert captured["extras"] == [None, None, None]
    assert captured["deterministic_policy"] is True
    assert captured["rollout_seed"] == 117
    assert captured["state_record_pixeled"] is False
