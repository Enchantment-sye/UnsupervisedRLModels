import os
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from analysis.knn_planning import (
    KNNPlanningEvalConfig,
    MethodTopK,
    NodeSet,
    ParsedPlanningDataset,
    _apply_episode_quota_filter,
    _apply_cross_episode_only_filter,
    _apply_state_variant,
    _compute_temporal_distance_topk,
    _merge_topk_with_temporal_bridges,
    _round_robin_indices,
    apply_task_preset,
    multi_source_dijkstra,
)
from analysis.maze_geodesic import build_maze_spec


def _toy_nodes() -> NodeSet:
    temporal_exclusions = (
        np.asarray([1], dtype=np.int64),
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([4], dtype=np.int64),
        np.asarray([3], dtype=np.int64),
    )
    return NodeSet(
        dataset_id="toy",
        effective_stride=1,
        node_global_indices=np.arange(5, dtype=np.int64),
        node_episode_ids=np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        node_timesteps=np.asarray([0, 1, 3, 0, 2], dtype=np.int32),
        node_state_full=np.zeros((5, 2), dtype=np.float32),
        node_xy=np.zeros((5, 2), dtype=np.float32),
        temporal_src=np.asarray([0, 1, 3], dtype=np.int64),
        temporal_dst=np.asarray([1, 2, 4], dtype=np.int64),
        temporal_cost=np.asarray([1.0, 2.0, 2.0], dtype=np.float32),
        temporal_exclusions=temporal_exclusions,
    )


def _toy_parsed_dataset() -> ParsedPlanningDataset:
    maze_map = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    spec = build_maze_spec(
        dataset_id="D4RL/pointmaze/large-v2",
        maze_map=maze_map,
        maze_kind="pointmaze",
        maze_size_scaling=1.0,
    )
    xy = np.asarray(
        [
            spec.rowcol_to_xy((1, 1)),
            spec.rowcol_to_xy((1, 1)) + np.asarray([0.1, 0.0], dtype=np.float32),
            spec.rowcol_to_xy((1, 1)) + np.asarray([0.0, 0.1], dtype=np.float32),
            spec.rowcol_to_xy((1, 1)) + np.asarray([0.1, 0.1], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    state_full = np.asarray(
        [
            [xy[0, 0], xy[0, 1], 0.0, 0.1],
            [xy[1, 0], xy[1, 1], 0.2, 0.3],
            [xy[2, 0], xy[2, 1], -0.1, 0.4],
            [xy[3, 0], xy[3, 1], 0.5, -0.2],
        ],
        dtype=np.float32,
    )
    transition_state_full = state_full[:-1].copy()
    transition_next_state_full = state_full[1:].copy()
    transition_xy = xy[:-1].copy()
    transition_next_xy = xy[1:].copy()
    return ParsedPlanningDataset(
        dataset_id="D4RL/pointmaze/large-v2",
        dataset_slug="d4rl_pointmaze_large_v2",
        maze_spec=spec,
        state_full=state_full,
        xy=xy,
        qpos_xy=xy.copy(),
        episode_ids=np.asarray([0, 0, 1, 1], dtype=np.int32),
        timesteps=np.asarray([0, 1, 0, 1], dtype=np.int32),
        episode_offsets=np.asarray([0, 2, 4], dtype=np.int64),
        episode_lengths=np.asarray([2, 2], dtype=np.int32),
        transition_state_full=transition_state_full,
        transition_next_state_full=transition_next_state_full,
        transition_xy=transition_xy,
        transition_next_xy=transition_next_xy,
    )
def test_pointmaze_geodesic_cross_wall_is_longer_than_euclidean():
    maze_map = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    spec = build_maze_spec(
        dataset_id="D4RL/pointmaze/umaze-v2",
        maze_map=maze_map,
        maze_kind="pointmaze",
        maze_size_scaling=1.0,
    )
    start = spec.rowcol_to_xy((1, 1))
    goal = spec.rowcol_to_xy((3, 1))
    euclidean = float(np.linalg.norm(goal - start))
    geodesic = float(spec.geodesic_for_pairs(start, goal)[0])
    assert np.isclose(euclidean, 2.0)
    assert np.isclose(geodesic, 6.0)
    assert geodesic > euclidean


def test_antmaze_scaling_uses_four_unit_cell_spacing():
    maze_map = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    spec = build_maze_spec(
        dataset_id="D4RL/antmaze/umaze-diverse-v1",
        maze_map=maze_map,
        maze_kind="antmaze",
        maze_size_scaling=4.0,
    )
    a = spec.rowcol_to_xy((1, 1))
    b = spec.rowcol_to_xy((1, 2))
    assert np.allclose(a, np.asarray([-4.0, -4.0], dtype=np.float32))
    assert np.allclose(b, np.asarray([0.0, -4.0], dtype=np.float32))
    assert np.isclose(float(spec.geodesic_for_pairs(a, b)[0]), 4.0)


def test_temporal_distance_uses_same_episode_and_skips_temporal_neighbors():
    nodes = _toy_nodes()

    class _Cfg:
        retrieval_top_k = 2

    result = _compute_temporal_distance_topk(nodes, _Cfg())
    assert result.indices.shape == (5, 2)
    assert result.scores.shape == (5, 2)

    # Node 0 cannot use node 1 because it is a temporal neighbor.
    # The next best candidate in the same episode is node 2 with delta t = 3.
    assert int(result.indices[0, 0]) == 2
    assert np.isclose(float(result.scores[0, 0]), 1.0 / 4.0)

    # Node 3 only has node 4 in the same episode, but it is excluded as a temporal neighbor.
    assert int(result.indices[3, 0]) == -1


def test_multi_source_dijkstra_recovers_lowest_cost_path():
    graph = {
        "edge_targets": (
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([3], dtype=np.int64),
            np.asarray([3], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        ),
        "edge_costs": (
            np.asarray([1.0, 5.0], dtype=np.float32),
            np.asarray([2.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        ),
        "edge_is_retrieval": (
            np.asarray([0, 1], dtype=np.int8),
            np.asarray([0], dtype=np.int8),
            np.asarray([1], dtype=np.int8),
            np.asarray([], dtype=np.int8),
        ),
        "edge_dgt": (
            np.asarray([0.0, 3.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        ),
    }
    target_mask = np.asarray([False, False, False, True])
    result = multi_source_dijkstra(graph, 4, np.asarray([0], dtype=np.int64), target_mask)
    assert result["found"]
    assert np.all(result["path_nodes"] == np.asarray([0, 1, 3], dtype=np.int64))
    assert np.isclose(float(result["path_cost"]), 3.0)
    assert int(result["retrieval_edge_count"]) == 0


def test_round_robin_indices_interleaves_groups():
    ordered = _round_robin_indices([[0, 1, 2], [10, 11], [20, 21]])
    assert ordered == [0, 10, 20, 1, 11, 21, 2]


def test_merge_topk_with_temporal_bridges_prioritizes_temporal_then_ik():
    ik_topk = MethodTopK(
        method="ik",
        indices=np.asarray([[5, 6, 7, 8]], dtype=np.int64),
        scores=np.asarray([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
    )
    temporal_topk = MethodTopK(
        method="temporal_distance",
        indices=np.asarray([[6, 3, 4, 5]], dtype=np.int64),
        scores=np.asarray([[1.0, 0.9, 0.8, 0.7]], dtype=np.float32),
    )
    merged = _merge_topk_with_temporal_bridges(ik_topk, temporal_topk, bridge_k=2, top_k=4)
    assert merged.method == "ik_temporal_bridge"
    assert merged.indices.tolist() == [[6, 3, 5, 7]]


def test_cross_episode_only_filter_drops_same_episode_candidates():
    nodes = _toy_nodes()
    topk = MethodTopK(
        method="ik",
        indices=np.asarray([[1, 3, 4], [0, 3, 4], [1, 3, 4], [4, 0, 1], [3, 0, 1]], dtype=np.int64),
        scores=np.asarray(
            [
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
            ],
            dtype=np.float32,
        ),
    )
    filtered = _apply_cross_episode_only_filter(topk, nodes)
    assert filtered.indices[0].tolist() == [3, 4, -1]
    assert filtered.indices[3].tolist() == [0, 1, -1]


def test_same_episode_quota_one_keeps_at_most_one_same_episode_candidate():
    nodes = _toy_nodes()
    topk = MethodTopK(
        method="ik",
        indices=np.asarray(
            [
                [1, 2, 3, 4],
                [0, 2, 3, 4],
                [1, 0, 3, 4],
                [4, 0, 1, 2],
                [3, 0, 1, 2],
            ],
            dtype=np.int64,
        ),
        scores=np.asarray(
            [
                [0.95, 0.90, 0.85, 0.80],
                [0.95, 0.90, 0.85, 0.80],
                [0.95, 0.90, 0.85, 0.80],
                [0.95, 0.90, 0.85, 0.80],
                [0.95, 0.90, 0.85, 0.80],
            ],
            dtype=np.float32,
        ),
    )
    filtered = _apply_episode_quota_filter(topk, nodes, same_episode_quota=1)
    assert filtered.indices[0].tolist() == [1, 3, 4, -1]
    assert filtered.indices[1].tolist() == [0, 3, 4, -1]
    assert filtered.indices[3].tolist() == [4, 0, 1, 2]


def test_same_episode_quota_zero_matches_cross_episode_only():
    nodes = _toy_nodes()
    topk = MethodTopK(
        method="ik",
        indices=np.asarray([[1, 3, 4], [0, 3, 4], [1, 3, 4], [4, 0, 1], [3, 0, 1]], dtype=np.int64),
        scores=np.asarray(
            [
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
            ],
            dtype=np.float32,
        ),
    )
    quota_filtered = _apply_episode_quota_filter(topk, nodes, same_episode_quota=0)
    cross_filtered = _apply_cross_episode_only_filter(topk, nodes)
    assert np.array_equal(quota_filtered.indices, cross_filtered.indices)
    assert np.allclose(quota_filtered.scores, cross_filtered.scores)


def test_nuisance_v1_state_variant_is_deterministic_and_preserves_xy():
    parsed = _toy_parsed_dataset()
    cfg = KNNPlanningEvalConfig(
        datasets=[parsed.dataset_id],
        output_dir="/tmp/knn_planning_test",
        cache_dir="/tmp/knn_planning_test_cache",
        state_variant="nuisance_v1",
    )
    variant_a = _apply_state_variant(parsed, cfg)
    variant_b = _apply_state_variant(parsed, cfg)
    assert variant_a.state_full.shape[1] == parsed.state_full.shape[1] + 64
    assert variant_a.transition_state_full.shape[1] == parsed.transition_state_full.shape[1] + 64
    assert variant_a.transition_next_state_full.shape[1] == parsed.transition_next_state_full.shape[1] + 64
    assert np.allclose(variant_a.xy, parsed.xy)
    assert np.allclose(variant_a.state_full, variant_b.state_full)


def test_apply_large_v2_ik_favoring_preset_overrides_task_shape():
    cfg = KNNPlanningEvalConfig(
        datasets=["D4RL/pointmaze/umaze-v2", "D4RL/pointmaze/large-v2"],
        output_dir="/tmp/knn_planning_test",
        cache_dir="/tmp/knn_planning_test_cache",
        task_preset="large_v2_ik_favoring",
    )
    preset = apply_task_preset(cfg)
    assert preset.datasets == ["D4RL/pointmaze/large-v2"]
    assert preset.state_variant == "nuisance_v1"
    assert preset.cross_episode_only is True
    assert preset.query_source == "shared_bank"
    assert preset.query_difficulty_filter == "easy"
    assert preset.query_limit == 30
    assert np.isclose(float(preset.pointmaze_h_bridge), 3.0)
    assert np.isclose(float(preset.alpha), 0.8)


def test_apply_large_v2_ik_soft_local_stitching_preset_overrides_task_shape():
    cfg = KNNPlanningEvalConfig(
        datasets=["D4RL/pointmaze/large-v2"],
        output_dir="/tmp/knn_planning_test",
        cache_dir="/tmp/knn_planning_test_cache",
        task_preset="large_v2_ik_soft_local_stitching",
    )
    preset = apply_task_preset(cfg)
    assert preset.datasets == ["D4RL/pointmaze/large-v2"]
    assert preset.state_variant == "nuisance_v1"
    assert preset.cross_episode_only is False
    assert preset.same_episode_quota == 1
    assert preset.query_source == "shared_bank"
    assert preset.query_difficulty_filter == "easy"
    assert preset.query_limit == 30
    assert np.isclose(float(preset.pointmaze_h_bridge), 3.0)
    assert np.isclose(float(preset.alpha), 0.8)


def test_apply_large_v2_ik_soft_local_stitching_preserves_explicit_alpha_override():
    cfg = KNNPlanningEvalConfig(
        datasets=["D4RL/pointmaze/large-v2"],
        output_dir="/tmp/knn_planning_test",
        cache_dir="/tmp/knn_planning_test_cache",
        task_preset="large_v2_ik_soft_local_stitching",
        alpha=1.0,
    )
    preset = apply_task_preset(cfg)
    assert np.isclose(float(preset.alpha), 1.0)
    assert preset.same_episode_quota == 1


def test_apply_antmaze_umaze_detour_focus_preset_overrides_task_shape():
    cfg = KNNPlanningEvalConfig(
        datasets=["D4RL/antmaze/umaze-diverse-v1", "D4RL/pointmaze/large-v2"],
        output_dir="/tmp/knn_planning_test",
        cache_dir="/tmp/knn_planning_test_cache",
        task_preset="antmaze_umaze_detour_focus",
    )
    preset = apply_task_preset(cfg)
    assert preset.datasets == ["D4RL/antmaze/umaze-diverse-v1"]
    assert preset.query_source == "node_sample"
    assert preset.query_difficulty_filter == "all"
    assert preset.query_limit == 200
    assert preset.num_queries == 200
    assert preset.retrieval_top_k == 20
    assert np.isclose(float(preset.antmaze_h_bridge), 15.0)
    assert np.isclose(float(preset.alpha), 0.88)
    assert preset.cross_episode_only is False
    assert preset.same_episode_quota is None
    assert preset.state_variant == "raw"


def _run_all_tests() -> None:
    tests = [
        test_pointmaze_geodesic_cross_wall_is_longer_than_euclidean,
        test_antmaze_scaling_uses_four_unit_cell_spacing,
        test_temporal_distance_uses_same_episode_and_skips_temporal_neighbors,
        test_multi_source_dijkstra_recovers_lowest_cost_path,
        test_round_robin_indices_interleaves_groups,
        test_merge_topk_with_temporal_bridges_prioritizes_temporal_then_ik,
        test_cross_episode_only_filter_drops_same_episode_candidates,
        test_same_episode_quota_one_keeps_at_most_one_same_episode_candidate,
        test_same_episode_quota_zero_matches_cross_episode_only,
        test_nuisance_v1_state_variant_is_deterministic_and_preserves_xy,
        test_apply_large_v2_ik_favoring_preset_overrides_task_shape,
        test_apply_large_v2_ik_soft_local_stitching_preset_overrides_task_shape,
        test_apply_large_v2_ik_soft_local_stitching_preserves_explicit_alpha_override,
    ]
    for test_fn in tests:
        test_fn()
    print(f"Passed {len(tests)} knn planning tests.")


if __name__ == "__main__":
    _run_all_tests()
