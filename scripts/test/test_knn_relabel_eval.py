import os
import sys
import tempfile

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from analysis.knn_relabeling import (
    _resolve_positive_cutoff,
    evaluate_relabel_metrics,
)
from analysis.reachability_alignment import (
    ParsedDataset,
    ReachabilityAnalysisConfig,
    _sample_anchor_indices,
    load_or_sample_candidates,
)
from analysis.similarity_metrics import compute_one_step_dynamics_scores


def _toy_parsed_dataset() -> ParsedDataset:
    goal_xy = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.4, 0.0],
            [0.6, 0.0],
            [0.8, 0.0],
            [10.0, 0.0],
            [10.2, 0.0],
            [10.4, 0.0],
            [10.6, 0.0],
        ],
        dtype=np.float32,
    )
    state_full = np.concatenate(
        [goal_xy, np.zeros((goal_xy.shape[0], 2), dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    return ParsedDataset(
        dataset_id="D4RL/pointmaze/umaze-v2",
        state_full=state_full,
        goal_xy=goal_xy,
        episode_ids=np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        timesteps=np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=np.int32),
        episode_offsets=np.asarray([0, 5, 9], dtype=np.int64),
        episode_lengths=np.asarray([5, 4], dtype=np.int32),
        total_episodes=2,
        median_step_size=0.2,
        p90_nearest_neighbor=0.2,
    )


def _toy_cfg(cache_dir: str) -> ReachabilityAnalysisConfig:
    return ReachabilityAnalysisConfig(
        datasets=["D4RL/pointmaze/umaze-v2"],
        output_dir=cache_dir,
        cache_dir=cache_dir,
        seed=0,
        num_anchors=3,
        num_candidates=16,
        candidate_pool_mode="planning_aligned",
        query_pool_mode="planning_aligned",
        node_stride_pointmaze=2,
        min_anchor_occurrences=1,
        max_anchor_occurrences=8,
    )


def test_positive_cutoff_supports_percentile_and_fixed_modes():
    matrix = np.asarray([[0.0, 0.2, 0.4, 0.6]], dtype=np.float32)
    mode, cutoff = _resolve_positive_cutoff(matrix, "reach_prob", "percentile", 50.0, 0.1)
    assert mode == "percentile"
    assert np.isclose(cutoff, 0.3)

    mode, cutoff = _resolve_positive_cutoff(matrix, "reach_prob", "fixed", 75.0, 0.25)
    assert mode == "fixed"
    assert np.isclose(cutoff, 0.25)

    mode, cutoff = _resolve_positive_cutoff(np.asarray([[0.0, 1.0]], dtype=np.float32), "geodesic", "percentile", 90.0, 0.5)
    assert mode == "native_binary"
    assert np.isclose(cutoff, 0.5)


def test_local_knn_one_step_dynamics_prefers_candidates_near_successor_cloud():
    fit_states = np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    fit_next_states = np.asarray([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    anchors = np.asarray([[0.1, 0.0]], dtype=np.float32)
    candidates = np.asarray([[1.1, 0.0], [2.8, 0.0], [4.2, 0.0]], dtype=np.float32)

    scores = compute_one_step_dynamics_scores(
        fit_states=fit_states,
        fit_next_states=fit_next_states,
        anchor_positions=anchors,
        candidate_positions=candidates,
        backend="local_knn_nextstate",
        local_knn_m=1,
        local_distance_metric="euclidean",
    )
    assert scores.shape == (1, 3)
    assert scores[0, 0] > scores[0, 1]
    assert scores[0, 1] > scores[0, 2]


def test_planning_aligned_candidate_and_anchor_sampling_are_stride_based_and_reproducible():
    parsed = _toy_parsed_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _toy_cfg(tmpdir)
        candidates = load_or_sample_candidates(parsed, cfg, match_radius=0.2)
        expected = np.asarray([0, 2, 4, 5, 7, 8], dtype=np.int64)
        assert np.array_equal(candidates, expected)

        anchors, occurrence_lists = _sample_anchor_indices(parsed, cfg, match_radius=0.2)
        assert anchors.shape[0] == 3
        assert set(int(x) for x in anchors.tolist()).issubset({0, 2, 5, 7})
        assert all(len(occ) >= 1 for occ in occurrence_lists)

        anchors2, occurrence_lists2 = _sample_anchor_indices(parsed, cfg, match_radius=0.2)
        assert np.array_equal(anchors, anchors2)
        assert all(np.array_equal(a, b) for a, b in zip(occurrence_lists, occurrence_lists2))


def test_evaluate_relabel_metrics_uses_binary_cutoff_for_precision_and_recall():
    gt = np.asarray([[0.1, 0.6, 0.7]], dtype=np.float32)
    scores = np.asarray([[0.9, 0.8, 0.1]], dtype=np.float32)
    geo = np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32)
    candidate_xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    anchor_xy = np.asarray([[9.0, 9.0]], dtype=np.float32)
    rows, summary = evaluate_relabel_metrics(
        ground_truth_matrix=gt,
        similarity_scores=scores,
        candidate_positions=candidate_xy,
        anchor_positions=anchor_xy,
        geo_matrix=geo,
        top_k=2,
        anchor_global_indices=np.asarray([10], dtype=np.int64),
        candidate_global_indices=np.asarray([0, 1, 2], dtype=np.int64),
        dataset_name="toy",
        method_name="euclidean",
        ground_truth_type="reach_prob",
        occurrence_counts=np.asarray([1.0], dtype=np.float32),
        horizon=10,
        positive_mode="fixed",
        positive_cutoff=0.5,
        sampling_protocol="planning_aligned",
        dynamics_backend="local_knn_nextstate",
    )
    assert len(rows) == 1
    assert np.isclose(rows[0]["goal_precision_at_k"], 0.5)
    assert np.isclose(rows[0]["recall_at_k"], 0.5)
    assert summary["positive_mode"] == "fixed"
    assert np.isclose(float(summary["positive_cutoff"]), 0.5)


def _run_all_tests() -> None:
    tests = [
        test_positive_cutoff_supports_percentile_and_fixed_modes,
        test_local_knn_one_step_dynamics_prefers_candidates_near_successor_cloud,
        test_planning_aligned_candidate_and_anchor_sampling_are_stride_based_and_reproducible,
        test_evaluate_relabel_metrics_uses_binary_cutoff_for_precision_and_recall,
    ]
    for test_fn in tests:
        test_fn()
    print(f"Passed {len(tests)} knn relabel tests.")


if __name__ == "__main__":
    _run_all_tests()
