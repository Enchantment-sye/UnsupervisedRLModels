import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from core import metra_viz
from core.task_adapter import SkillDiscoveryTaskAdapter


class _FakeWriter:
    def __init__(self):
        self.figures = []
        self.has_legends = []

    def add_figure(self, tag, fig, step):
        self.figures.append((tag, int(step)))
        self.has_legends.append(any(ax.get_legend() is not None for ax in fig.axes))


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, message, *args, **kwargs):
        if args:
            message = message % args
        self.warnings.append(message)


def _trajectory(offset):
    coords = np.asarray(
        [[offset, 0.0], [offset + 0.5, 0.25]],
        dtype=np.float32,
    )
    next_coords = np.asarray(
        [[offset + 0.5, 0.25], [offset + 1.0, 0.5]],
        dtype=np.float32,
    )
    return {
        "env_infos": {
            "coordinates": coords,
            "next_coordinates": next_coords,
        }
    }


def _adapter(tmp_path, task="dmc_quadruped_run_forward_color"):
    cfg = SimpleNamespace(
        task=task,
        stage="pre_training",
        seed=5,
        discrete=True,
        dim_skill=2,
        unit_length=True,
        use_hierarchical_skill=False,
        num_skill_levels=1,
        eval_plot_axis=None,
        eval_skill_xy_plot=True,
        eval_skill_xy_plot_rollouts_per_skill=3,
    )
    return SkillDiscoveryTaskAdapter(
        cfg,
        env=None,
        agent=None,
        work_dir=str(tmp_path),
        logger=_FakeLogger(),
    )


def test_extract_xy_trajectory_includes_final_next_coordinate():
    xy = metra_viz.extract_xy_trajectory(_trajectory(2.0))

    assert xy.shape == (3, 2)
    np.testing.assert_allclose(xy[0], [2.0, 0.0])
    np.testing.assert_allclose(xy[-1], [3.0, 0.5])


def test_plot_skill_xy_trajectories_skips_missing_coordinates(tmp_path):
    writer = _FakeWriter()
    logger = _FakeLogger()

    count = metra_viz.plot_skill_xy_trajectories(
        [{"env_infos": {}}],
        n_trajs_per_skill=3,
        snapshot_dir=str(tmp_path),
        writer=writer,
        step_itr=7,
        logger=logger,
    )

    assert count == 0
    assert writer.figures == []
    assert logger.warnings == ["Skill XY trajectory plot skipped: no valid coordinates were collected."]


def test_task_adapter_collects_three_xy_rollouts_per_discrete_skill(tmp_path):
    adapter = _adapter(tmp_path, task="dmc_humanoid_run_forward_color")
    writer = _FakeWriter()
    captured = {}

    def collect(extras, **kwargs):
        captured["extras"] = extras
        captured.update(kwargs)
        return [_trajectory(float(idx)) for idx in range(len(extras))]

    adapter.collect_policy_trajectories = collect

    adapter._maybe_log_skill_xy_trajectories(step_itr=9, total_epoch=4, writer=writer)

    assert len(captured["extras"]) == 6
    np.testing.assert_allclose(captured["extras"][0]["skill"], [1.0, 0.0])
    np.testing.assert_allclose(captured["extras"][2]["skill"], [1.0, 0.0])
    np.testing.assert_allclose(captured["extras"][3]["skill"], [0.0, 1.0])
    assert captured["deterministic_policy"] is True
    assert captured["rollout_seed"] == 700009
    assert captured["state_record_pixeled"] is False
    assert writer.figures == [("SkillXYTrajPlot", 9)]
    assert writer.has_legends == [False]


def test_task_adapter_does_not_collect_xy_plot_for_non_locomotion_task(tmp_path):
    adapter = _adapter(tmp_path, task="metaworld_reach")
    captured = {}

    def collect(extras, **kwargs):
        captured["called"] = True
        return []

    adapter.collect_policy_trajectories = collect

    adapter._maybe_log_skill_xy_trajectories(step_itr=1, total_epoch=2, writer=_FakeWriter())

    assert captured == {}


def test_task_adapter_allows_ant_tasks(tmp_path):
    adapter = _adapter(tmp_path, task="maze_ant_large")

    assert adapter._task_supports_skill_xy_plot() is True
