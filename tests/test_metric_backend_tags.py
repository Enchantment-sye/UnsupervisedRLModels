import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("src"))

from core import metra_viz


class _FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))


class _NoOpIsomap:
    def __init__(self, *args, **kwargs):
        pass

    def fit_transform(self, values):
        return values


def _install_fake_isomap(monkeypatch):
    sklearn_module = ModuleType("sklearn")
    sklearn_module.__path__ = []
    manifold_module = ModuleType("sklearn.manifold")
    manifold_module.Isomap = _NoOpIsomap
    sklearn_module.manifold = manifold_module
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_module)
    monkeypatch.setitem(sys.modules, "sklearn.manifold", manifold_module)


def _agent(*, ikse):
    return SimpleNamespace(
        dim_skill=2,
        discrete=True,
        unit_length=True,
        traj_encoder=object(),
        ikse=ikse,
        metric_num_sampled_points=2,
        dbi_num_rollouts_per_skill=3,
        num_random_trajectories=2,
        encoder=0,
        seed=11,
        device="cpu",
    )


def _record():
    return {
        "raw_points": np.zeros((2, 2), dtype=np.float32),
        "encoded_points": np.zeros((2, 2), dtype=np.float32),
        "query_positions": np.asarray([0, 1], dtype=np.int64),
        "skill_idx": 0,
    }


def _patch_common_collection(monkeypatch):
    monkeypatch.setattr(
        metra_viz,
        "_collect_metric_records",
        lambda *args, **kwargs: ([], [_record()]),
    )
    monkeypatch.setattr(metra_viz, "save_image_grid", lambda *args, **kwargs: None)


def _scalar_map(writer):
    return {tag: value for tag, value, _ in writer.scalars}


def test_temporal_backend_writes_semantic_and_legacy_tags(monkeypatch, tmp_path):
    _install_fake_isomap(monkeypatch)
    _patch_common_collection(monkeypatch)
    monkeypatch.setattr(
        metra_viz,
        "_build_temporal_metric_context",
        lambda *args, **kwargs: {"soft_dtw_device": "cpu"},
    )
    monkeypatch.setattr(
        metra_viz,
        "_compute_temporal_entropy_metrics",
        lambda agent, graph_context: (2.0, 3.0),
    )
    monkeypatch.setattr(
        metra_viz,
        "_compute_temporal_dbi_metrics",
        lambda agent, records, graph_context: (4.0, 5.0, np.zeros((2, 2), dtype=np.float32), [2], [0]),
    )

    writer = _FakeWriter()
    metra_viz.plot_trajectories(
        _agent(ikse=False),
        snapshot_dir=str(tmp_path),
        writer=writer,
        logger=SimpleNamespace(info=lambda message: None),
        step_itr=7,
    )

    scalars = _scalar_map(writer)
    assert scalars["eval/Entropy_Raw"] == 2.0
    assert scalars["eval/Entropy_Enc"] == 3.0
    assert scalars["eval/DBI_Raw"] == 4.0
    assert scalars["eval/DBI_Enc"] == 5.0
    assert scalars["eval/Entropy_TemporalParticle"] == 2.0
    assert scalars["eval/DBI_TemporalMedoid"] == 4.0
    assert scalars["eval/Score_TemporalParticle_DBI"] == pytest.approx(2.0 / (4.0 + 1e-8))
    assert scalars["eval/MetricBackend_IsIKSE"] == 0.0
    assert scalars["eval/MetricBackend_UsesTemporalDistance"] == 1.0
    assert scalars["eval/MetricBackend_UsesIsolationKernel"] == 0.0
    assert scalars["eval/MetricBackend_UsesEuclideanDistance"] == 0.0
    assert "eval/DBI_IKCanonical" not in scalars


def test_ikse_backend_writes_semantic_and_legacy_tags(monkeypatch, tmp_path):
    _install_fake_isomap(monkeypatch)
    _patch_common_collection(monkeypatch)
    monkeypatch.setattr(
        metra_viz,
        "_compute_ikse_entropy_metrics",
        lambda agent, records, *, ensemble_size, subsample_size: (6.0, 7.0),
    )
    monkeypatch.setattr(
        metra_viz,
        "_compute_ikse_dbi_metrics",
        lambda agent, records, *, ensemble_size, subsample_size: (8.0, 9.0, np.zeros((2, 2), dtype=np.float32), [2], [0]),
    )

    writer = _FakeWriter()
    metra_viz.plot_trajectories(
        _agent(ikse=True),
        snapshot_dir=str(tmp_path),
        writer=writer,
        logger=SimpleNamespace(info=lambda message: None),
        step_itr=9,
    )

    scalars = _scalar_map(writer)
    assert scalars["eval/Entropy_Raw"] == 6.0
    assert scalars["eval/Entropy_Enc"] == 7.0
    assert scalars["eval/DBI_Raw"] == 8.0
    assert scalars["eval/DBI_Enc"] == 9.0
    assert scalars["eval/Entropy_IKDE_XY"] == 6.0
    assert scalars["eval/DBI_IKMeanRatio_Legacy"] == 8.0
    assert scalars["eval/IKSE_LegacyDBI"] == pytest.approx(6.0 / (8.0 + 1e-8))
    assert scalars["eval/MetricBackend_IsIKSE"] == 1.0
    assert scalars["eval/MetricBackend_UsesTemporalDistance"] == 0.0
    assert scalars["eval/MetricBackend_UsesIsolationKernel"] == 1.0
    assert scalars["eval/MetricBackend_UsesEuclideanDistance"] == 0.0
    assert "eval/DBI_IKCanonical" not in scalars
