import os
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from analysis.ik_knn_planning_sweep import _ik_label, _sort_ik_rows
from analysis.knn_planning import QuerySet, slice_query_set


def test_ik_label_is_stable_and_readable():
    label = _ik_label(200, 64, 0.008)
    assert label == "ik_e200_s64_t0p008"


def test_sort_ik_rows_uses_success_then_suboptimality_then_precision():
    rows = [
        {
            "dataset": "toy",
            "ensemble_size": 100,
            "subsample_size": 8,
            "temperature": 0.01,
            "planning_success_rate": 0.8,
            "path_suboptimality": 1.2,
            "precision_at_k": 0.7,
            "mean_expanded_nodes": 100.0,
        },
        {
            "dataset": "toy",
            "ensemble_size": 100,
            "subsample_size": 4,
            "temperature": 0.01,
            "planning_success_rate": 0.8,
            "path_suboptimality": 1.1,
            "precision_at_k": 0.6,
            "mean_expanded_nodes": 120.0,
        },
        {
            "dataset": "toy",
            "ensemble_size": 100,
            "subsample_size": 2,
            "temperature": 0.01,
            "planning_success_rate": 0.9,
            "path_suboptimality": 2.0,
            "precision_at_k": 0.1,
            "mean_expanded_nodes": 999.0,
        },
    ]
    ranked = _sort_ik_rows(rows)
    assert int(ranked[0]["subsample_size"]) == 2
    assert int(ranked[1]["subsample_size"]) == 4
    assert int(ranked[2]["subsample_size"]) == 8


def test_slice_query_set_preserves_xy_and_order():
    queries = QuerySet(
        start_node_ids=np.asarray([0, 1, 2], dtype=np.int64),
        goal_node_ids=np.asarray([3, 4, 5], dtype=np.int64),
        query_geodesic=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        difficulty=np.asarray(["easy", "medium", "hard"]),
        start_xy=np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        goal_xy=np.asarray([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]], dtype=np.float32),
    )
    sliced = slice_query_set(queries, 2)
    assert sliced.start_node_ids.tolist() == [0, 1]
    assert sliced.goal_node_ids.tolist() == [3, 4]
    assert np.allclose(sliced.start_xy, np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    assert np.allclose(sliced.goal_xy, np.asarray([[3.0, 3.0], [4.0, 4.0]], dtype=np.float32))


def _run_all_tests() -> None:
    tests = [
        test_ik_label_is_stable_and_readable,
        test_sort_ik_rows_uses_success_then_suboptimality_then_precision,
        test_slice_query_set_preserves_xy_and_order,
    ]
    for test_fn in tests:
        test_fn()
    print(f"Passed {len(tests)} ik sweep tests.")


if __name__ == "__main__":
    _run_all_tests()
