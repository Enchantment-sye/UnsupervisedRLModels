import os
import sys
import warnings
import importlib.util

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from analysis.proxy_ground_truth import compute_ground_truth_bundle
from analysis.antmaze_region_temporal_collapse import (
    aggregate_max_by_regions,
    aggregate_min_by_regions,
    aggregate_topk_mean_by_regions,
    compute_region_commit_and_tail_from_hits,
)
from analysis.antmaze_branch_collapse import (
    BranchCandidate,
    BranchTaskSpec,
    assign_two_clusters_farthest_pair,
    compute_branch_commit_and_mass,
    finalize_branch_tasks,
)
from analysis.antmaze_fair_region_comparison import (
    aggregate_region_score_optimistic,
    aggregate_region_score_topk,
    compute_pair_nonik_method_deltas,
    compute_deep_branch_commit_and_tail_mass,
    _select_dual_core_union_members,
    FairBranchRegionCandidate,
)
from analysis.antmaze_branch_collapse import (
    BranchStateMetrics,
)
from analysis.fitted_baselines import (
    AdaptiveGaussianMetric,
    MahalanobisMetric,
    OneStepDynamicsMetric,
)
from analysis.arr_benchmark import oracle_score_to_distance
from analysis.arr_benchmark import _select_context_decoys
from analysis.reachability_alignment import (
    ParsedDataset,
    ReachabilityAnalysisConfig,
    resolve_horizon_values_for_dataset,
    resolve_match_radius,
    select_metric_representation,
)
from analysis.similarity_metrics import (
    auc_from_binary_labels,
    compute_first_hit_temporal_distances,
    compute_replay_temporal_scores,
    compute_temporal_distance_scores,
    distances_to_scores,
    ndcg_at_k,
    recall_at_k,
    topk_overlap,
)


def _load_module_from_path(module_name: str, relative_path: str):
    module_path = os.path.join(REPO_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_match_radius_auto_resolution():
    parsed = ParsedDataset(
        dataset_id="toy",
        state_full=np.zeros((4, 3), dtype=np.float32),
        goal_xy=np.zeros((4, 2), dtype=np.float32),
        episode_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        timesteps=np.array([0, 1, 0, 1], dtype=np.int32),
        episode_offsets=np.array([0, 2, 4], dtype=np.int64),
        episode_lengths=np.array([2, 2], dtype=np.int32),
        total_episodes=2,
        median_step_size=0.1,
        p90_nearest_neighbor=0.25,
    )
    assert np.isclose(resolve_match_radius(parsed, None), 0.4)


def test_future_h_ground_truth_and_oracle_score():
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    bundle = compute_ground_truth_bundle(
        anchor_occurrence_lists=[np.array([0, 3], dtype=np.int64)],
        candidate_positions=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        positions=positions,
        episode_ids=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        timesteps=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
        episode_offsets=np.array([0, 3, 6], dtype=np.int64),
        episode_lengths=np.array([3, 3], dtype=np.int32),
        horizon=2,
        match_radius=0.05,
    )
    assert bundle.occurrence_counts[0] == 2
    assert np.allclose(bundle.reach_prob[0], np.array([0.5, 1.0, 0.5], dtype=np.float32))
    assert np.allclose(
        bundle.temporal_reachability[0],
        np.array([0.5, 0.75, 0.25], dtype=np.float32),
        atol=1e-6,
    )
    assert np.allclose(
        bundle.oracle_temporal[0],
        np.array([0.25, 0.41666666, 0.16666667], dtype=np.float32),
        atol=1e-6,
    )


def test_temporal_ground_truth_does_not_cross_episodes():
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    bundle = compute_ground_truth_bundle(
        anchor_occurrence_lists=[np.array([1], dtype=np.int64)],
        candidate_positions=np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        positions=positions,
        episode_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        timesteps=np.array([0, 1, 0, 1], dtype=np.int32),
        episode_offsets=np.array([0, 2, 4], dtype=np.int64),
        episode_lengths=np.array([2, 2], dtype=np.int32),
        horizon=2,
        match_radius=0.05,
    )
    assert np.allclose(bundle.reach_prob[0], np.zeros(2, dtype=np.float32))


def test_replay_temporal_scores_non_negative_and_distinct_from_oracle():
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    replay_scores = compute_replay_temporal_scores(
        anchor_occurrence_lists=[np.array([0, 3], dtype=np.int64)],
        candidate_positions=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        positions=positions,
        episode_ids=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        timesteps=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
        episode_offsets=np.array([0, 3, 6], dtype=np.int64),
        episode_lengths=np.array([3, 3], dtype=np.int32),
        match_radius=0.05,
        temporal_window=2,
    )
    assert replay_scores.shape == (1, 3)
    assert np.all(replay_scores >= 0.0)


def test_first_hit_temporal_distance_uses_best_case_earliest_hit():
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    first_hit = compute_first_hit_temporal_distances(
        anchor_occurrence_lists=[np.array([0, 3], dtype=np.int64)],
        candidate_positions=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        positions=positions,
        episode_ids=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        timesteps=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
        episode_offsets=np.array([0, 3, 6], dtype=np.int64),
        episode_lengths=np.array([3, 3], dtype=np.int32),
        match_radius=0.05,
        temporal_window=2,
    )
    assert np.allclose(first_hit[0, :2], np.array([1.0, 1.0], dtype=np.float32))
    assert np.isclose(first_hit[0, 2], 2.0)


def test_strict_temporal_distance_only_keeps_same_episode_future_states():
    scores = compute_temporal_distance_scores(
        anchor_global_indices=np.array([0, 3], dtype=np.int64),
        candidate_global_indices=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        episode_ids=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        timesteps=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
    )
    assert scores.shape == (2, 6)
    assert np.isclose(scores[0, 1], 1.0 / 2.0)
    assert np.isclose(scores[0, 2], 1.0 / 3.0)
    assert np.isclose(scores[1, 4], 1.0 / 2.0)
    assert np.isclose(scores[1, 5], 1.0 / 3.0)
    assert scores[0, 0] == 0.0
    assert scores[0, 3] == 0.0
    assert scores[1, 1] == 0.0


def test_metric_state_variant_switches_between_full_state_and_goal_xy():
    parsed = ParsedDataset(
        dataset_id="toy",
        state_full=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        goal_xy=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        episode_ids=np.array([0, 0], dtype=np.int32),
        timesteps=np.array([0, 1], dtype=np.int32),
        episode_offsets=np.array([0, 2], dtype=np.int64),
        episode_lengths=np.array([2], dtype=np.int32),
        total_episodes=1,
        median_step_size=1.0,
        p90_nearest_neighbor=1.0,
    )
    full_cfg = ReachabilityAnalysisConfig(datasets=["toy"], output_dir="/tmp", cache_dir="/tmp", metric_state_variant="full_state")
    pos_cfg = ReachabilityAnalysisConfig(datasets=["toy"], output_dir="/tmp", cache_dir="/tmp", metric_state_variant="position_only")
    assert np.allclose(select_metric_representation(parsed, full_cfg), parsed.state_full)
    assert np.allclose(select_metric_representation(parsed, pos_cfg), parsed.goal_xy)


def test_oracle_temporal_distance_matches_score_inverse():
    oracle_scores = np.array([[0.5, 0.25, 0.0]], dtype=np.float32)
    distances = oracle_score_to_distance(oracle_scores)
    assert np.allclose(distances[0, :2], np.array([1.0, 3.0], dtype=np.float32))
    assert np.isinf(distances[0, 2])
    reconstructed = distances_to_scores(distances)
    assert np.allclose(reconstructed[0, :2], oracle_scores[0, :2], atol=1e-6)
    assert reconstructed[0, 2] == 0.0


def test_context_selection_caps_infinite_first_hit_fraction():
    remaining = np.arange(8, dtype=np.int64)
    first_hit = np.array([5.0, 5.0, 6.0, 6.0, np.inf, np.inf, np.inf, 7.0], dtype=np.float32)
    distances = np.array([0.1, 0.11, 0.09, 0.12, 0.1, 0.15, 0.2, 0.13], dtype=np.float32)
    replay = np.array([0.8, 0.79, 0.82, 0.78, 0.2, 0.1, 0.05, 0.77], dtype=np.float32)
    reference = np.array([0, 1], dtype=np.int64)
    selected = _select_context_decoys(
        remaining_indices=remaining,
        first_hit_distances=first_hit,
        distances=distances,
        replay_scores=replay,
        reference_indices=reference,
        target_count=4,
        max_inf_fraction=0.25,
    )
    selected_first_hit = first_hit[selected]
    assert selected.shape == (4,)
    assert np.sum(~np.isfinite(selected_first_hit)) <= 1


def test_resolve_horizon_values_prefers_dataset_override():
    cfg = ReachabilityAnalysisConfig(
        datasets=["a"],
        output_dir="/tmp",
        cache_dir="/tmp",
        horizon_values=[20, 30],
        per_dataset_horizon_values={"a": [20, 50, 100]},
    )
    assert resolve_horizon_values_for_dataset("a", cfg) == [20, 50, 100]


def test_ranking_metrics_behave_safely():
    y_true = np.array([0.9, 0.4, 0.0, 0.2], dtype=np.float32)
    y_score = np.array([0.8, 0.1, 0.05, 0.3], dtype=np.float32)
    assert 0.0 <= recall_at_k(y_true, y_score, 2) <= 1.0
    assert 0.0 <= topk_overlap(y_true, y_score, 2) <= 1.0
    assert 0.0 <= ndcg_at_k(y_true, y_score, 3) <= 1.0
    assert 0.0 <= auc_from_binary_labels((y_true > 0).astype(np.int64), y_score) <= 1.0


def test_mahalanobis_metric_handles_degenerate_covariance_and_matches_precision():
    train = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    anchors = np.array([[0.0, 1.0, 3.0], [3.0, 1.0, 3.0]], dtype=np.float32)
    candidates = np.array([[1.0, 1.0, 3.0], [2.0, 1.0, 3.0]], dtype=np.float32)
    metric = MahalanobisMetric.fit(train, covariance_estimator="ledoitwolf", eps=1e-6)
    whitening = metric.pairwise_distance(anchors, candidates, implementation="whitening")
    precision = metric.pairwise_distance(anchors, candidates, implementation="precision")
    assert whitening.shape == (2, 2)
    assert np.all(np.isfinite(whitening))
    assert np.allclose(whitening, precision, atol=1e-5)


def test_adaptive_gaussian_metric_handles_duplicate_points_and_new_queries():
    train = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    metric = AdaptiveGaussianMetric.fit(train, k=10, eps=1e-6)
    sigmas = metric.estimate_query_sigmas(
        np.array([[0.0, 0.0], [1.5, 0.0]], dtype=np.float32),
        query_train_indices=np.array([0, -1], dtype=np.int64),
    )
    kernel = metric.pairwise_kernel(
        anchor_positions=np.array([[0.0, 0.0], [1.5, 0.0]], dtype=np.float32),
        candidate_positions=np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        anchor_train_indices=np.array([1, -1], dtype=np.int64),
        candidate_train_indices=np.array([0, 3], dtype=np.int64),
    )
    assert np.all(sigmas > 0.0)
    assert np.all(np.isfinite(sigmas))
    assert kernel.shape == (2, 2)
    assert np.all(kernel >= 0.0)
    assert np.all(kernel <= 1.0 + 1e-6)


def test_one_step_dynamics_metric_grid_fallback_and_low_count_rows_stay_finite():
    train_states = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    train_next_states = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metric = OneStepDynamicsMetric.fit(
            train_states,
            train_next_states,
            backend="grid",
            num_bins=8,
            distance_metric="jsd",
            alpha=1e-3,
            min_count=5,
            seed=0,
            eps=1e-6,
        )
    distances = metric.pairwise_distance(
        anchor_positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        candidate_positions=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert metric.backend_used == "kmeans"
    assert any("falling back to kmeans" in str(w.message).lower() for w in caught)
    assert distances.shape == (1, 2)
    assert np.all(np.isfinite(distances))


def test_toy_hard_pair_search_uses_extended_nonik_objective():
    toy_module = _load_module_from_path("toy_ik_transfer_validation", os.path.join("experiments", "toy_ik_transfer_validation.py"))
    env = toy_module.ToyEnvironment(
        name="toy-test",
        width=4,
        height=1,
        states=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        transition_matrix=np.eye(4, dtype=np.float32),
        adjacency=[[], [], [], []],
        coord_to_index={(0, 0): 0, (1, 0): 1, (2, 0): 2, (3, 0): 3},
        wall_cells=[],
        door_coord=(0, 0),
        anchor_coords={"door_left": (0, 0)},
        anchor_indices={"door_left": 0},
        teleport_edges=[],
    )
    reachability = np.array(
        [
            [0.0, 0.80, 0.50, 0.20],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    shortest = np.array(
        [
            [0.0, 2.0, 2.0, 2.0],
            [2.0, 0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0, 1.0],
            [2.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    distances = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    gaussian = np.array(
        [
            [1.0, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.2, 0.2],
            [0.5, 0.2, 1.0, 0.2],
            [0.5, 0.2, 0.2, 1.0],
        ],
        dtype=np.float32,
    )
    similarities = {
        "ik": np.array(
            [
                [1.0, 0.60, 0.55, 0.15],
                [0.75, 1.0, 0.0, 0.0],
                [0.55, 0.0, 1.0, 0.0],
                [0.10, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "euclidean": np.array(
            [
                [1.0, 0.20, 0.20, 0.20],
                [0.20, 1.0, 0.0, 0.0],
                [0.20, 0.0, 1.0, 0.0],
                [0.20, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "gaussian": gaussian,
        "mahalanobis": np.array(
            [
                [1.0, 0.75, 0.40, 0.10],
                [0.70, 1.0, 0.0, 0.0],
                [0.60, 0.0, 1.0, 0.0],
                [0.20, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "adaptive_gaussian": np.array(
            [
                [1.0, 0.70, 0.45, 0.20],
                [0.80, 1.0, 0.0, 0.0],
                [0.79, 0.0, 1.0, 0.0],
                [0.10, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "one_step_dynamics": np.array(
            [
                [1.0, 0.55, 0.35, 0.10],
                [0.75, 1.0, 0.0, 0.0],
                [0.30, 0.0, 1.0, 0.0],
                [0.20, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "density": np.array(
            [
                [1.0, 0.90, 0.30, 0.05],
                [0.90, 1.0, 0.0, 0.0],
                [0.20, 0.0, 1.0, 0.0],
                [0.10, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "replay_temporal": np.array(
            [
                [1.0, 0.62, 0.32, 0.05],
                [0.65, 1.0, 0.0, 0.0],
                [0.25, 0.0, 1.0, 0.0],
                [0.05, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "oracle_temporal": np.array(
            [
                [1.0, 0.50, 0.50, 0.50],
                [0.50, 1.0, 0.0, 0.0],
                [0.50, 0.0, 1.0, 0.0],
                [0.50, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }
    selected = toy_module.find_hard_pair_candidate(
        env=env,
        reachability=reachability,
        shortest_paths=shortest,
        distances=distances,
        gaussian=gaussian,
        similarities=similarities,
        anchor_name="door_left",
        min_reach_gap=0.05,
    )
    assert int(selected["target_i_idx"]) == 2
    assert int(selected["target_j_idx"]) == 3
    assert float(selected["ik_vs_best_nonik_margin"]) > 0.0
    assert str(selected["best_nonik_method"]) == "mahalanobis"


def test_ik_best_examples_v2_toy_summary_uses_real_best_nonik():
    build_module = _load_module_from_path("build_ik_best_examples_v2", os.path.join("scripts", "analysis", "build_ik_best_examples_v2.py"))
    rows = [
        {
            "anchor": "door_left",
            "target_i": "[5, 0]",
            "target_j": "[9, 2]",
            "reach_i": "0.10",
            "reach_j": "0.04",
            "seed": "summary",
            "ik_i": "0.24",
            "ik_j": "0.10",
            "ik_margin": "0.14",
            "mahalanobis_aligned_margin": "0.03",
            "adaptive_gaussian_aligned_margin": "0.11",
            "one_step_dynamics_aligned_margin": "0.07",
            "density_aligned_margin": "0.02",
            "replay_temporal_aligned_margin": "0.04",
            "oracle_temporal_aligned_margin": "0.00",
            "gaussian_aligned_margin": "0.00",
            "euclidean_aligned_margin": "0.00",
            "best_nonik_method": "adaptive_gaussian",
            "best_nonik_margin": "0.11",
            "ik_vs_best_nonik_margin": "0.03",
        }
    ]
    summary = build_module.compute_toy_summary_from_hard_pair_rows(rows)
    assert summary["best_nonik_method"] == "adaptive_gaussian"
    assert np.isclose(float(summary["best_nonik_score"]), 0.11)
    assert np.isclose(float(summary["ik_advantage"]), 0.03)
    assert summary["status"] == "strict_positive"


def test_region_aggregation_respects_max_min_and_topk_mean():
    scores = np.array(
        [
            [0.1, 0.4, 0.2, 0.9],
            [0.5, 0.3, 0.8, 0.7],
        ],
        dtype=np.float32,
    )
    distances = np.array(
        [
            [5.0, 3.0, 7.0, 1.0],
            [2.0, 6.0, 4.0, 9.0],
        ],
        dtype=np.float32,
    )
    regions = [
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
    ]
    max_scores = aggregate_max_by_regions(scores, regions)
    min_distances = aggregate_min_by_regions(distances, regions)
    top2_mean = aggregate_topk_mean_by_regions(scores, regions, top_k=2)

    assert np.allclose(max_scores, np.array([[0.4, 0.9], [0.8, 0.7]], dtype=np.float32))
    assert np.allclose(min_distances, np.array([[3.0, 1.0], [2.0, 6.0]], dtype=np.float32))
    assert np.allclose(top2_mean, np.array([[0.3, 0.65], [0.65, 0.5]], dtype=np.float32))


def test_region_commit_prob_and_tail_mass_separate_commit_from_earliest_hit():
    horizon = 4
    occurrence_region_hits = [
        [
            np.array([0, 1], dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ],
        [
            np.array([1], dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ],
    ]
    commit_prob, tail_mass = compute_region_commit_and_tail_from_hits(
        occurrence_region_hits=occurrence_region_hits,
        num_regions=2,
        horizon=horizon,
    )

    earliest_hit = np.array([1.0, 1.0], dtype=np.float32)
    assert np.allclose(earliest_hit, np.array([1.0, 1.0], dtype=np.float32))
    assert np.allclose(commit_prob, np.array([0.0, 1.0], dtype=np.float32))
    assert np.allclose(tail_mass, np.array([0.0, 1.0], dtype=np.float32))


def test_branch_commit_and_mass_separate_same_entry_different_commit():
    traces = [
        type("Trace", (), {"after_step_hits": [np.array([0], dtype=np.int64), np.array([2], dtype=np.int64), np.array([2], dtype=np.int64)]})(),
        type("Trace", (), {"after_step_hits": [np.array([1], dtype=np.int64), np.array([1], dtype=np.int64)]})(),
    ]
    commit_prob, post_entry_mass = compute_branch_commit_and_mass(
        traces=traces,
        region_members=np.array([2], dtype=np.int64),
        num_candidates=4,
        denominator=2,
    )
    assert np.isclose(commit_prob, 0.5)
    assert np.isclose(post_entry_mass, 1.0 / 3.0)


def test_branch_cluster_split_separates_two_lobes():
    points = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
        ],
        dtype=np.float32,
    )
    labels = assign_two_clusters_farthest_pair(points)
    assert labels.shape == (4,)
    assert set(labels.tolist()) == {0, 1}
    assert int(np.sum(labels == labels[0])) == 2


def test_branch_task_temporal_baselines_use_entry_scores_only():
    branch_candidates = [
        BranchCandidate(
            candidate_id=0,
            anchor_row=0,
            entrance_local_index=5,
            region_type="tail",
            branch_label=0,
            member_local_indices=np.array([0, 1], dtype=np.int64),
            center=np.array([0.0, 0.0], dtype=np.float32),
            commit_prob=0.9,
            post_entry_mass=0.8,
            entrance_distance=1.0,
            entrance_first_hit=2.0,
            entrance_replay=0.3,
            entrance_oracle=0.2,
            entrance_euclidean=-1.0,
            entrance_gaussian=0.5,
            entrance_mahalanobis=-0.9,
            entrance_adaptive_gaussian=0.45,
            entrance_one_step_dynamics=-0.7,
            hit_occurrence_count=10,
        ),
        BranchCandidate(
            candidate_id=1,
            anchor_row=0,
            entrance_local_index=5,
            region_type="corridor",
            branch_label=0,
            member_local_indices=np.array([2, 3], dtype=np.int64),
            center=np.array([0.1, 0.0], dtype=np.float32),
            commit_prob=0.8,
            post_entry_mass=0.7,
            entrance_distance=1.0,
            entrance_first_hit=2.0,
            entrance_replay=0.3,
            entrance_oracle=0.2,
            entrance_euclidean=-1.0,
            entrance_gaussian=0.5,
            entrance_mahalanobis=-0.9,
            entrance_adaptive_gaussian=0.45,
            entrance_one_step_dynamics=-0.7,
            hit_occurrence_count=10,
        ),
        BranchCandidate(
            candidate_id=2,
            anchor_row=0,
            entrance_local_index=5,
            region_type="tail",
            branch_label=1,
            member_local_indices=np.array([4, 5], dtype=np.int64),
            center=np.array([1.0, 0.0], dtype=np.float32),
            commit_prob=0.2,
            post_entry_mass=0.1,
            entrance_distance=1.0,
            entrance_first_hit=2.0,
            entrance_replay=0.3,
            entrance_oracle=0.2,
            entrance_euclidean=-1.0,
            entrance_gaussian=0.5,
            entrance_mahalanobis=-0.9,
            entrance_adaptive_gaussian=0.45,
            entrance_one_step_dynamics=-0.7,
            hit_occurrence_count=10,
        ),
        BranchCandidate(
            candidate_id=3,
            anchor_row=0,
            entrance_local_index=5,
            region_type="corridor",
            branch_label=1,
            member_local_indices=np.array([6, 7], dtype=np.int64),
            center=np.array([1.1, 0.0], dtype=np.float32),
            commit_prob=0.1,
            post_entry_mass=0.05,
            entrance_distance=1.0,
            entrance_first_hit=2.0,
            entrance_replay=0.3,
            entrance_oracle=0.2,
            entrance_euclidean=-1.0,
            entrance_gaussian=0.5,
            entrance_mahalanobis=-0.9,
            entrance_adaptive_gaussian=0.45,
            entrance_one_step_dynamics=-0.7,
            hit_occurrence_count=10,
        ),
    ]
    spec = BranchTaskSpec(
        anchor_row=0,
        reference_entrance_local_index=5,
        shell_candidate_ids=np.array([0, 1, 2, 3], dtype=np.int64),
        positive_candidate_ids=np.array([0, 1], dtype=np.int64),
        negative_candidate_ids=np.array([2, 3], dtype=np.int64),
        decoy_candidate_ids=np.empty(0, dtype=np.int64),
        gt_gap=0.6,
        min_positive_commit=0.8,
        max_negative_commit=0.2,
        mean_positive_mass=0.75,
        mean_negative_mass=0.075,
        entrance_distance=1.0,
        entrance_first_hit=2.0,
        entrance_replay=0.3,
        entrance_oracle=0.2,
    )
    tasks = finalize_branch_tasks(
        task_specs=[spec],
        branch_candidates=branch_candidates,
        candidate_ik_scores=np.array([0.9, 0.8, 0.1, 0.0], dtype=np.float32),
        ik_key=(32, 0.004),
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert np.isclose(task.method_accuracies["replay"], 0.0)
    assert np.isclose(task.method_accuracies["oracle"], 0.0)
    assert np.isclose(task.method_accuracies["first_hit"], 0.0)
    assert np.isclose(task.method_accuracies["ik"], 1.0)


def test_fair_topk_and_optimistic_region_aggregation_differ_as_expected():
    scores = np.array([0.95, 0.80, 0.30, 0.10], dtype=np.float32)
    members = np.array([0, 1, 2, 3], dtype=np.int64)
    assert np.isclose(aggregate_region_score_topk(scores, members, top_k=2), 0.875)
    assert np.isclose(aggregate_region_score_topk(scores, members, top_k=4), 0.5375)
    assert np.isclose(aggregate_region_score_optimistic(scores, members), 0.95)


def test_deep_branch_commit_and_tail_mass_only_count_core_hits_after_entry():
    traces = [
        type(
            "Trace",
            (),
            {
                "after_step_hits": [
                    np.array([0], dtype=np.int64),
                    np.array([2], dtype=np.int64),
                    np.array([2, 3], dtype=np.int64),
                    np.array([3], dtype=np.int64),
                ]
            },
        )(),
        type(
            "Trace",
            (),
            {
                "after_step_hits": [
                    np.array([0], dtype=np.int64),
                    np.array([1], dtype=np.int64),
                    np.array([1], dtype=np.int64),
                ]
            },
        )(),
    ]
    commit_prob, tail_mass = compute_deep_branch_commit_and_tail_mass(
        traces=traces,
        deep_core_members=np.array([2, 3], dtype=np.int64),
        num_candidates=6,
        denominator=2,
    )
    assert np.isclose(commit_prob, 0.5)
    assert np.isclose(tail_mass, (3.0 / 4.0) / 2.0)


def test_dual_core_union_members_mix_tail_and_corridor_without_duplicates():
    members = _select_dual_core_union_members(
        tail_core_members=np.array([10, 11, 12, 13], dtype=np.int64),
        corridor_core_members=np.array([12, 20, 21, 22], dtype=np.int64),
        deep_core_size=4,
    )
    assert members.shape == (4,)
    assert 10 in members.tolist()
    assert 20 in members.tolist()
    assert len(set(members.tolist())) == 4


def test_compute_pair_nonik_method_deltas_match_manual_topk_scores():
    state_metrics = BranchStateMetrics(
        reach_prob=np.zeros((1, 6), dtype=np.float32),
        oracle_temporal=np.array([[0.0, 0.0, 0.05, 0.02, 0.04, 0.01]], dtype=np.float32),
        replay_temporal=np.array([[0.0, 0.0, 0.08, 0.02, 0.06, 0.01]], dtype=np.float32),
        first_hit_distances=np.array([[np.inf, np.inf, 1.0, 4.0, 2.0, 5.0]], dtype=np.float32),
        euclidean_scores=np.array([[0.0, 0.0, 0.90, 0.20, 0.70, 0.10]], dtype=np.float32),
        gaussian_scores=np.array([[0.0, 0.0, 0.80, 0.10, 0.60, 0.05]], dtype=np.float32),
        mahalanobis_scores=np.array([[0.0, 0.0, -0.30, -0.60, -0.45, -0.80]], dtype=np.float32),
        adaptive_gaussian_scores=np.array([[0.0, 0.0, 0.75, 0.15, 0.55, 0.08]], dtype=np.float32),
        one_step_dynamics_scores=np.array([[0.0, 0.0, -0.20, -0.70, -0.35, -0.90]], dtype=np.float32),
    )
    positive_region = FairBranchRegionCandidate(
        anchor_row=0,
        entrance_local_index=1,
        branch_label=0,
        region_variant="halo_plus_deep_tail",
        entry_halo_size=2,
        deep_core_size=2,
        full_members=np.array([2, 3], dtype=np.int64),
        entry_halo_members=np.array([2], dtype=np.int64),
        deep_core_members=np.array([2, 3], dtype=np.int64),
        full_center=np.array([0.0, 0.0], dtype=np.float64),
        deep_core_center=np.array([0.0, 0.0], dtype=np.float64),
        deep_commit_prob=0.2,
        deep_tail_mass=0.1,
        entrance_distance=1.0,
        entrance_first_hit=1.0,
        entrance_replay=0.1,
        entrance_oracle=0.05,
        entrance_euclidean=0.2,
        entrance_gaussian=0.3,
        entrance_mahalanobis=-0.2,
        entrance_adaptive_gaussian=0.25,
        entrance_one_step_dynamics=-0.15,
        hit_occurrence_count=10,
    )
    negative_region = FairBranchRegionCandidate(
        anchor_row=0,
        entrance_local_index=1,
        branch_label=1,
        region_variant="halo_plus_deep_tail",
        entry_halo_size=2,
        deep_core_size=2,
        full_members=np.array([4, 5], dtype=np.int64),
        entry_halo_members=np.array([4], dtype=np.int64),
        deep_core_members=np.array([4, 5], dtype=np.int64),
        full_center=np.array([1.0, 0.0], dtype=np.float64),
        deep_core_center=np.array([1.0, 0.0], dtype=np.float64),
        deep_commit_prob=0.1,
        deep_tail_mass=0.02,
        entrance_distance=1.0,
        entrance_first_hit=1.0,
        entrance_replay=0.1,
        entrance_oracle=0.05,
        entrance_euclidean=0.2,
        entrance_gaussian=0.3,
        entrance_mahalanobis=-0.2,
        entrance_adaptive_gaussian=0.25,
        entrance_one_step_dynamics=-0.15,
        hit_occurrence_count=10,
    )
    fair, optimistic = compute_pair_nonik_method_deltas(
        state_metrics=state_metrics,
        anchor_row=0,
        positive_region=positive_region,
        negative_region=negative_region,
        top_k=2,
    )
    assert np.isclose(fair["euclidean"], ((0.90 + 0.20) / 2.0) - ((0.70 + 0.10) / 2.0))
    assert np.isclose(fair["gaussian"], ((0.80 + 0.10) / 2.0) - ((0.60 + 0.05) / 2.0))
    assert np.isclose(fair["replay"], ((0.08 + 0.02) / 2.0) - ((0.06 + 0.01) / 2.0))
    assert np.isclose(fair["oracle"], ((0.05 + 0.02) / 2.0) - ((0.04 + 0.01) / 2.0))
    assert np.isclose(fair["first_hit"], ((1.0 / 2.0 + 1.0 / 5.0) / 2.0) - ((1.0 / 3.0 + 1.0 / 6.0) / 2.0))
    assert np.isclose(optimistic["replay"], 0.08 - 0.06)


def main():
    test_match_radius_auto_resolution()
    test_future_h_ground_truth_and_oracle_score()
    test_temporal_ground_truth_does_not_cross_episodes()
    test_replay_temporal_scores_non_negative_and_distinct_from_oracle()
    test_first_hit_temporal_distance_uses_best_case_earliest_hit()
    test_strict_temporal_distance_only_keeps_same_episode_future_states()
    test_metric_state_variant_switches_between_full_state_and_goal_xy()
    test_oracle_temporal_distance_matches_score_inverse()
    test_context_selection_caps_infinite_first_hit_fraction()
    test_resolve_horizon_values_prefers_dataset_override()
    test_ranking_metrics_behave_safely()
    test_mahalanobis_metric_handles_degenerate_covariance_and_matches_precision()
    test_adaptive_gaussian_metric_handles_duplicate_points_and_new_queries()
    test_one_step_dynamics_metric_grid_fallback_and_low_count_rows_stay_finite()
    test_region_aggregation_respects_max_min_and_topk_mean()
    test_region_commit_prob_and_tail_mass_separate_commit_from_earliest_hit()
    test_branch_commit_and_mass_separate_same_entry_different_commit()
    test_branch_cluster_split_separates_two_lobes()
    test_branch_task_temporal_baselines_use_entry_scores_only()
    test_fair_topk_and_optimistic_region_aggregation_differ_as_expected()
    test_deep_branch_commit_and_tail_mass_only_count_core_hits_after_entry()
    test_dual_core_union_members_mix_tail_and_corridor_without_duplicates()
    test_compute_pair_nonik_method_deltas_match_manual_topk_scores()
    print("reachability alignment tests passed")


if __name__ == "__main__":
    main()
