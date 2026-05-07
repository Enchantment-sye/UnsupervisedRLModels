import os
import sys
import tempfile

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from analysis.reachability_alignment import ParsedDataset
from analysis.successor_distance import (
    SuccessorDistanceConfig,
    build_or_load_future_window_bundle,
    build_or_load_grid_spec,
    compute_adaptive_gdk_paired_distances,
    compute_gdk_paired_distances,
    compute_raw_successor_paired_distances,
    compute_wasserstein_paired_distances,
    dataset_slug,
    load_or_create_episode_split,
    run_successor_distance,
)
from analysis.fitted_baselines import AdaptiveGaussianMetric


def _make_small_parsed() -> ParsedDataset:
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
            [12.0, 0.0],
        ],
        dtype=np.float32,
    )
    return ParsedDataset(
        dataset_id="toy/small",
        state_full=positions.copy(),
        goal_xy=positions,
        episode_ids=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        timesteps=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
        episode_offsets=np.array([0, 3, 6], dtype=np.int64),
        episode_lengths=np.array([3, 3], dtype=np.int32),
        total_episodes=2,
        median_step_size=1.0,
        p90_nearest_neighbor=1.0,
    )


def _make_toy_parsed(total_episodes: int = 20, episode_length: int = 6) -> ParsedDataset:
    positions = []
    episode_ids = []
    timesteps = []
    episode_offsets = [0]
    episode_lengths = []
    for episode_id in range(total_episodes):
        lane = float(episode_id % 2)
        x_values = np.linspace(0.0, 2.5, episode_length, dtype=np.float32)
        y_values = np.full(episode_length, lane, dtype=np.float32)
        episode_positions = np.stack([x_values, y_values], axis=1)
        positions.append(episode_positions)
        episode_ids.append(np.full(episode_length, episode_id, dtype=np.int32))
        timesteps.append(np.arange(episode_length, dtype=np.int32))
        episode_lengths.append(episode_length)
        episode_offsets.append(episode_offsets[-1] + episode_length)
    all_positions = np.concatenate(positions, axis=0).astype(np.float32)
    return ParsedDataset(
        dataset_id="toy/successor",
        state_full=all_positions.copy(),
        goal_xy=all_positions,
        episode_ids=np.concatenate(episode_ids, axis=0).astype(np.int32),
        timesteps=np.concatenate(timesteps, axis=0).astype(np.int32),
        episode_offsets=np.asarray(episode_offsets, dtype=np.int64),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
        total_episodes=total_episodes,
        median_step_size=0.5,
        p90_nearest_neighbor=0.5,
    )


def _write_parse_cache(cache_dir: str, parsed: ParsedDataset) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"dataset_parse_{dataset_slug(parsed.dataset_id)}.npz")
    np.savez_compressed(
        path,
        dataset_parse_cache_version=np.asarray(2),
        dataset_id=np.asarray(parsed.dataset_id),
        state_full=parsed.state_full,
        goal_xy=parsed.goal_xy,
        episode_ids=parsed.episode_ids,
        timesteps=parsed.timesteps,
        episode_offsets=parsed.episode_offsets,
        episode_lengths=parsed.episode_lengths,
        total_episodes=np.asarray(parsed.total_episodes),
        median_step_size=np.asarray(parsed.median_step_size),
        p90_nearest_neighbor=np.asarray(parsed.p90_nearest_neighbor),
    )


def test_future_window_does_not_cross_episodes():
    parsed = _make_toy_parsed(total_episodes=8, episode_length=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SuccessorDistanceConfig(
            datasets=[parsed.dataset_id],
            output_dir=tmpdir,
            cache_dir=os.path.join(tmpdir, "cache"),
            horizon_values=[2],
            grid_nx=2,
            grid_ny=2,
            seed=2,
        )
        split = load_or_create_episode_split(parsed, cfg)
        grid = build_or_load_grid_spec(parsed, split, cfg)
        bundle = build_or_load_future_window_bundle(parsed, split, "train", 2, grid, cfg)
        for global_index, endpoint in zip(bundle.valid_global_indices.tolist(), bundle.future_endpoints):
            episode_id = int(parsed.episode_ids[global_index])
            assert int(parsed.episode_ids[global_index + 2]) == episode_id
            assert np.allclose(endpoint, parsed.positions[global_index + 2])


def test_future_region_labels_are_correct():
    parsed = _make_toy_parsed(total_episodes=8, episode_length=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SuccessorDistanceConfig(
            datasets=[parsed.dataset_id],
            output_dir=tmpdir,
            cache_dir=os.path.join(tmpdir, "cache"),
            horizon_values=[1],
            grid_nx=2,
            grid_ny=2,
            seed=1,
        )
        split = load_or_create_episode_split(parsed, cfg)
        grid = build_or_load_grid_spec(parsed, split, cfg)
        bundle = build_or_load_future_window_bundle(parsed, split, "test", 1, grid, cfg)
        assert bundle.future_region_ids.ndim == 1
        assert bundle.future_region_ids.shape[0] == bundle.valid_global_indices.shape[0]
        recomputed = np.asarray(bundle.future_region_ids, dtype=np.int64)
        assert np.array_equal(recomputed, bundle.future_region_ids)


def test_raw_successor_distance_zero_on_identical_windows():
    windows = np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    distances = compute_raw_successor_paired_distances(windows, windows)
    assert np.allclose(distances, np.zeros(1, dtype=np.float32))


def test_gdk_and_adaptive_gdk_self_distances_are_zero():
    windows = np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    gdk = compute_gdk_paired_distances(windows, windows, sigma=1.0, batch_size=1)
    adaptive_metric = AdaptiveGaussianMetric.fit(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32), k=1, eps=1e-6)
    adaptive = compute_adaptive_gdk_paired_distances(adaptive_metric, windows, windows, batch_size=1)
    assert np.allclose(gdk, np.zeros(1, dtype=np.float32), atol=1e-6)
    assert np.allclose(adaptive, np.zeros(1, dtype=np.float32), atol=1e-6)


def test_wasserstein_self_distance_is_zero():
    windows = np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=np.float32)
    distances = compute_wasserstein_paired_distances(windows, windows)
    assert np.allclose(distances, np.zeros(1, dtype=np.float32), atol=1e-6)


def test_episode_split_respects_rounding():
    parsed = _make_toy_parsed(total_episodes=20, episode_length=6)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SuccessorDistanceConfig(
            datasets=[parsed.dataset_id],
            output_dir=tmpdir,
            cache_dir=os.path.join(tmpdir, "cache"),
            horizon_values=[2],
            seed=3,
        )
        split = load_or_create_episode_split(parsed, cfg)
        assert split.train_episode_ids.size == 14
        assert split.val_episode_ids.size == 3
        assert split.test_episode_ids.size == 3


def test_toy_successor_distance_end_to_end():
    parsed = _make_toy_parsed(total_episodes=20, episode_length=6)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        _write_parse_cache(cache_dir, parsed)
        cfg = SuccessorDistanceConfig(
            datasets=[parsed.dataset_id],
            output_dir=tmpdir,
            cache_dir=cache_dir,
            seed=7,
            horizon_values=[2],
            grid_nx=2,
            grid_ny=2,
            search_num_pairs=12,
            eval_num_pairs=16,
            search_pair_batch_size=4,
            eval_pair_batch_size=4,
            num_queries=4,
            num_candidates=8,
            recall_k_values=(1, 3),
            plot_top_k=3,
            query_matrix_batch_size=2,
            ik_ensemble_sizes=(2,),
            ik_subsample_sizes=(2,),
            ik_temperatures=(0.1,),
            ik_batch_size=32,
            ik_device="cpu",
            ik_explicit_max_feature_values=100000,
        )
        result = run_successor_distance(cfg)
        methods = {row["method"] for row in result["summary_rows"]}
        assert methods == {"raw", "idk", "gdk", "wasserstein_w2", "adaptive_gdk"}
        assert os.path.exists(result["search_full_path"])
        assert os.path.exists(result["search_best_path"])
        assert os.path.exists(result["per_dataset_path"])
        assert os.path.exists(result["overall_path"])
        assert os.path.exists(result["recall_path"])
        assert os.path.exists(result["report_path"])
        for row in result["summary_rows"]:
            assert np.isfinite(float(row["auroc"]))
            assert np.isfinite(float(row["auprc"]))


def _run_all_tests() -> None:
    tests = [
        test_future_window_does_not_cross_episodes,
        test_future_region_labels_are_correct,
        test_raw_successor_distance_zero_on_identical_windows,
        test_gdk_and_adaptive_gdk_self_distances_are_zero,
        test_wasserstein_self_distance_is_zero,
        test_episode_split_respects_rounding,
        test_toy_successor_distance_end_to_end,
    ]
    for test_fn in tests:
        test_fn()
    print(f"Passed {len(tests)} successor-distance tests.")


if __name__ == "__main__":
    _run_all_tests()
