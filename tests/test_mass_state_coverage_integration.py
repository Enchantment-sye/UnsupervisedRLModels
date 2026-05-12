import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src"))

import core.mass.trainer as mass_trainer_mod
from core.mass.trainer import MassPixelTrainer


class _DummyCoverageTracker:
    def compute_policy_metrics(self, paths):
        return {"PolicyStateCoverageXYBins": float(len(paths))}

    def compute_queue_metrics(self):
        return {"QueueStateCoverageXYBins": 11.0}

    def compute_total_metrics(self):
        return {"TotalStateCoverageXYBins": 17.0}


class _DummyWriter:
    def __init__(self):
        self.scalars = {}
        self.flushed = False

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (float(value), int(step))

    def flush(self):
        self.flushed = True


class _DummyMass:
    def __init__(self):
        self.refresh_calls = []
        self.added_z = []

    def rolling_refresh(self, refresh_num):
        self.refresh_calls.append(refresh_num)
        return {"refresh_count": float(len(self.refresh_calls))}

    def add_z(self, z):
        self.added_z.append(z)


class _DummyFigManager:
    def __init__(self, *args, **kwargs):
        self.ax = object()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_mass_eval_coverage_logs_tracker_metrics_without_touching_mass():
    tracker = _DummyCoverageTracker()
    writer = _DummyWriter()
    trainer = MassPixelTrainer.__new__(MassPixelTrainer)
    trainer.coverage_tracker = tracker
    trainer.writer = writer
    trainer.global_step = 123
    trainer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    trainer.mass = SimpleNamespace(
        add_z=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mass.add_z called"))
    )

    trainer._write_eval_coverage_metrics([{"env_infos": {}} for _ in range(48)])

    assert writer.scalars["eval/PolicyStateCoverageXYBins"] == (48.0, 123)
    assert writer.scalars["eval/QueueStateCoverageXYBins"] == (11.0, 123)
    assert writer.scalars["eval/TotalStateCoverageXYBins"] == (17.0, 123)
    assert writer.scalars["eval/NumStateCoverageTrajectories"] == (48.0, 123)
    assert writer.flushed is True


def test_mass_refresh_happens_on_epoch_interval_only():
    trainer = MassPixelTrainer.__new__(MassPixelTrainer)
    trainer.cfg = SimpleNamespace(mass_refresh_interval=5, mass_refresh_num=8)
    trainer.mass = _DummyMass()
    trainer.global_step = 123
    trainer._written_mass_stats = []
    trainer._write_mass_stats = lambda stats, step: trainer._written_mass_stats.append((stats, step))

    trainer.epoch = 4
    trainer._maybe_refresh_mass_for_epoch()
    assert trainer.mass.refresh_calls == []

    trainer.epoch = 5
    trainer._maybe_refresh_mass_for_epoch()
    assert trainer.mass.refresh_calls == [8]
    assert trainer._written_mass_stats == [({"refresh_count": 1.0}, 123)]


def test_mass_refresh_interval_zero_disables_epoch_refresh():
    trainer = MassPixelTrainer.__new__(MassPixelTrainer)
    trainer.cfg = SimpleNamespace(mass_refresh_interval=0, mass_refresh_num=8)
    trainer.mass = _DummyMass()
    trainer.global_step = 123
    trainer._write_mass_stats = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("_write_mass_stats should not be called")
    )

    trainer.epoch = 5
    trainer._maybe_refresh_mass_for_epoch()

    assert trainer.mass.refresh_calls == []


def test_mass_path_ingest_does_not_refresh_per_step():
    trainer = MassPixelTrainer.__new__(MassPixelTrainer)
    trainer.action_dim = 1
    trainer.global_step = 0
    trainer.metrics = SimpleNamespace(add=lambda **kwargs: None)
    trainer.mass = _DummyMass()
    trainer.reward_adapter = SimpleNamespace(
        compute_step_reward=lambda *args, **kwargs: {
            "r_int": torch.zeros(1, 1),
            "z_next": torch.ones(1, 2),
        }
    )
    trainer._record_reward_metrics = lambda reward_out: None
    trainer._add_transition_array = lambda *args, **kwargs: None

    trainer._ingest_collected_path(
        {
            "observations": np.zeros((2, 3), dtype=np.float32),
            "next_observations": np.zeros((2, 3), dtype=np.float32),
            "actions": np.zeros((2, 1), dtype=np.float32),
            "dones": np.asarray([False, True]),
        }
    )

    assert trainer.global_step == 2
    assert len(trainer.mass.added_z) == 2
    assert trainer.mass.refresh_calls == []


def test_mass_eval_traj_plot_uses_configured_axis(monkeypatch):
    calls = []

    class _PlotEnv:
        def render_trajectories(self, paths, colors, plot_axis, ax):
            calls.append(
                {
                    "paths": paths,
                    "colors_shape": colors.shape,
                    "plot_axis": plot_axis,
                    "ax": ax,
                }
            )

    monkeypatch.setattr(mass_trainer_mod.metra_utils, "FigManager", _DummyFigManager)

    trainer = MassPixelTrainer.__new__(MassPixelTrainer)
    trainer.cfg = SimpleNamespace(seed=7, eval_plot_axis=[-5.0, 5.0, -6.0, 6.0])
    trainer.epoch = 3
    trainer.global_step = 11
    trainer.work_dir = "/tmp/mass-test"
    trainer.writer = _DummyWriter()
    trainer.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    paths = [
        {
            "env_infos": {
                "coordinates": np.zeros((2, 2), dtype=np.float32),
                "next_coordinates": np.zeros((2, 2), dtype=np.float32),
            }
        }
    ]

    trainer._write_eval_traj_plot(_PlotEnv(), paths)

    assert len(calls) == 1
    assert calls[0]["paths"] is paths
    assert calls[0]["colors_shape"] == (1, 4)
    assert calls[0]["plot_axis"] == [-5.0, 5.0, -6.0, 6.0]


def test_mass_eval_traj_plot_fail_open_without_render_method():
    warnings = []
    trainer = MassPixelTrainer.__new__(MassPixelTrainer)
    trainer.cfg = SimpleNamespace(seed=0, eval_plot_axis=None)
    trainer.epoch = 1
    trainer.global_step = 0
    trainer.work_dir = "/tmp/mass-test"
    trainer.writer = _DummyWriter()
    trainer.logger = SimpleNamespace(
        warning=lambda *args, **kwargs: warnings.append(args[0] % args[1:] if args[1:] else args[0])
    )

    trainer._write_eval_traj_plot(object(), [{"env_infos": {}}])

    assert warnings
    assert "render_trajectories" in warnings[0]
