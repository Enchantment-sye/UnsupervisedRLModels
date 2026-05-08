from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .proxy_ground_truth import GroundTruthBundle, compute_ground_truth_bundle
from .fitted_baselines import sample_transition_pairs
from .similarity_metrics import (
    SimilarityBundle,
    compute_adaptive_gaussian_scores,
    compute_euclidean_scores,
    compute_gaussian_scores,
    compute_ik_scores,
    compute_mahalanobis_scores,
    compute_one_step_dynamics_scores,
    compute_temporal_distance_scores,
    compute_replay_temporal_scores,
    evaluate_alignment,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


BASELINE_CACHE_VERSION = 4
DATASET_PARSE_CACHE_VERSION = 2


def dataset_slug(dataset_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", dataset_id).strip("_").lower()


def save_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass
class ReachabilityAnalysisConfig:
    datasets: list[str]
    output_dir: str
    cache_dir: str
    seed: int = 0
    mode: str = "single"
    horizon: int = 20
    num_anchors: int = 200
    num_candidates: int = 1000
    top_k: int = 20
    match_radius: float | None = None
    candidate_pool_mode: str = "planning_aligned"
    query_pool_mode: str = "planning_aligned"
    node_stride_pointmaze: int = 5
    node_stride_antmaze: int = 5
    candidate_sampling: str = "dedupe"
    min_anchor_occurrences: int = 8
    max_anchor_occurrences: int = 512
    max_anchor_attempts: int = 50000
    ik_ensemble_size: int = 100
    ik_subsample_size: int = 32
    ik_temperature: float = 0.01
    gk_sigma_mode: str = "median_heuristic"
    gk_sigma: float | None = None
    mahalanobis_covariance_estimator: str = "ledoitwolf"
    mahalanobis_implementation: str = "whitening"
    mahalanobis_eps: float = 1e-6
    adaptive_gaussian_k: int = 10
    adaptive_gaussian_eps: float = 1e-6
    adaptive_gaussian_output: str = "kernel"
    dynamics_backend: str = "local_knn_nextstate"
    dynamics_num_bins: int = 64
    dynamics_distance_metric: str = "jsd"
    dynamics_alpha: float = 1e-3
    dynamics_min_count: int = 5
    dynamics_eps: float = 1e-6
    dynamics_local_knn_m: int = 20
    dynamics_local_distance_metric: str = "euclidean"
    dynamics_state_variant: str | None = None
    metric_state_variant: str = "full_state"
    fit_pool_size: int = 50000
    ik_batch_size: int = 4096
    ik_device: str = "auto"
    overwrite_cache: bool = False
    minari_datasets_path: str = "/home/shangyy/.minari/datasets"
    scatter_points: int = 4000

    horizon_values: list[int] | None = None
    per_dataset_horizon_values: dict[str, list[int]] | None = None
    search_num_anchors: int = 64
    search_num_candidates: int = 512
    search_max_anchor_occurrences: int = 128
    final_num_anchors: int = 256
    final_num_candidates: int = 2048
    final_top_k: int = 50
    final_max_anchor_occurrences: int = 256
    ik_subsample_grid: list[int] | None = None
    ik_temperature_grid: list[float] | None = None
    selection_metric: str = "spearman"
    selection_ground_truth: str = "reach_prob"
    selection_split: str = "search_subset"
    replay_temporal_window: int | None = None
    report_oracle_temp: bool = True
    tmux_session_name: str = "reachability_alignment"


@dataclass
class ParsedDataset:
    dataset_id: str
    state_full: np.ndarray
    goal_xy: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray
    episode_offsets: np.ndarray
    episode_lengths: np.ndarray
    total_episodes: int
    median_step_size: float
    p90_nearest_neighbor: float

    @property
    def positions(self) -> np.ndarray:
        return self.goal_xy

    def build_tree(self) -> cKDTree:
        return cKDTree(self.goal_xy)


@dataclass
class EvaluationContext:
    parsed: ParsedDataset
    match_radius: float
    anchor_indices: np.ndarray
    anchor_occurrence_lists: list[np.ndarray]
    candidate_indices: np.ndarray
    ground_truth: GroundTruthBundle


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:12]


def _npz_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def _save_npz(path: str, **kwargs: Any) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **kwargs)


def _load_npz(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _safe_load_npz(path: str) -> dict[str, Any] | None:
    try:
        return _load_npz(path)
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _sample_nearest_neighbor_p90(
    positions: np.ndarray,
    rng: np.random.Generator,
    max_points: int = 4096,
) -> float:
    if positions.shape[0] < 2:
        return 1.0
    if positions.shape[0] > max_points:
        indices = rng.choice(positions.shape[0], size=max_points, replace=False)
        sampled = positions[indices]
    else:
        sampled = positions
    tree = cKDTree(sampled)
    distances, _ = tree.query(sampled, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 1.0
    return float(np.percentile(nearest, 90))


def load_or_parse_dataset(
    dataset_id: str,
    cache_dir: str,
    overwrite_cache: bool,
    minari_datasets_path: str,
    seed: int,
) -> ParsedDataset:
    import minari

    slug = dataset_slug(dataset_id)
    cache_path = os.path.join(cache_dir, f"dataset_parse_{slug}.npz")
    if _npz_exists(cache_path) and not overwrite_cache:
        payload = _safe_load_npz(cache_path)
        required_keys = {
            "dataset_parse_cache_version",
            "dataset_id",
            "state_full",
            "goal_xy",
            "episode_ids",
            "timesteps",
            "episode_offsets",
            "episode_lengths",
            "total_episodes",
            "median_step_size",
            "p90_nearest_neighbor",
        }
        if payload is not None and required_keys.issubset(set(payload.keys())):
            cache_version = int(payload["dataset_parse_cache_version"].item())
            if cache_version != DATASET_PARSE_CACHE_VERSION:
                payload = None
        if payload is not None:
            return ParsedDataset(
                dataset_id=str(payload["dataset_id"].item()),
                state_full=np.asarray(payload["state_full"], dtype=np.float32),
                goal_xy=np.asarray(payload["goal_xy"], dtype=np.float32),
                episode_ids=np.asarray(payload["episode_ids"], dtype=np.int32),
                timesteps=np.asarray(payload["timesteps"], dtype=np.int32),
                episode_offsets=np.asarray(payload["episode_offsets"], dtype=np.int64),
                episode_lengths=np.asarray(payload["episode_lengths"], dtype=np.int32),
                total_episodes=int(payload["total_episodes"].item()),
                median_step_size=float(payload["median_step_size"].item()),
                p90_nearest_neighbor=float(payload["p90_nearest_neighbor"].item()),
            )

    os.environ.setdefault("MINARI_DATASETS_PATH", minari_datasets_path)
    dataset = minari.load_dataset(dataset_id)

    state_full_blocks = []
    goal_xy_blocks = []
    episode_ids = []
    timesteps = []
    episode_offsets = [0]
    episode_lengths = []
    step_norms = []

    for episode_id in range(dataset.total_episodes):
        episode = dataset[episode_id]
        observations = episode.observations
        state_full = np.asarray(observations["observation"], dtype=np.float32)
        goal_xy = np.asarray(observations["achieved_goal"], dtype=np.float32)
        state_full_blocks.append(state_full)
        goal_xy_blocks.append(goal_xy)
        episode_lengths.append(int(goal_xy.shape[0]))
        episode_ids.append(np.full(goal_xy.shape[0], episode_id, dtype=np.int32))
        timesteps.append(np.arange(goal_xy.shape[0], dtype=np.int32))
        episode_offsets.append(episode_offsets[-1] + int(goal_xy.shape[0]))
        if goal_xy.shape[0] > 1:
            step_norms.append(np.linalg.norm(goal_xy[1:] - goal_xy[:-1], axis=1))

    state_full = np.concatenate(state_full_blocks, axis=0).astype(np.float32)
    goal_xy = np.concatenate(goal_xy_blocks, axis=0).astype(np.float32)
    episode_ids_array = np.concatenate(episode_ids, axis=0).astype(np.int32)
    timesteps_array = np.concatenate(timesteps, axis=0).astype(np.int32)
    episode_offsets_array = np.asarray(episode_offsets, dtype=np.int64)
    episode_lengths_array = np.asarray(episode_lengths, dtype=np.int32)
    all_step_norms = np.concatenate(step_norms, axis=0) if step_norms else np.asarray([1.0], dtype=np.float32)
    median_step_size = float(np.median(all_step_norms))
    rng = np.random.default_rng(seed)
    p90_nearest_neighbor = _sample_nearest_neighbor_p90(goal_xy, rng)

    _save_npz(
        cache_path,
        dataset_parse_cache_version=np.asarray(DATASET_PARSE_CACHE_VERSION),
        dataset_id=np.asarray(dataset_id),
        state_full=state_full,
        goal_xy=goal_xy,
        episode_ids=episode_ids_array,
        timesteps=timesteps_array,
        episode_offsets=episode_offsets_array,
        episode_lengths=episode_lengths_array,
        total_episodes=np.asarray(dataset.total_episodes),
        median_step_size=np.asarray(median_step_size),
        p90_nearest_neighbor=np.asarray(p90_nearest_neighbor),
    )

    return ParsedDataset(
        dataset_id=dataset_id,
        state_full=state_full,
        goal_xy=goal_xy,
        episode_ids=episode_ids_array,
        timesteps=timesteps_array,
        episode_offsets=episode_offsets_array,
        episode_lengths=episode_lengths_array,
        total_episodes=dataset.total_episodes,
        median_step_size=median_step_size,
        p90_nearest_neighbor=p90_nearest_neighbor,
    )


def resolve_match_radius(parsed: ParsedDataset, requested_radius: float | None) -> float:
    if requested_radius is not None and requested_radius > 0:
        return float(requested_radius)
    auto_radius = max(4.0 * parsed.median_step_size, parsed.p90_nearest_neighbor)
    return float(max(auto_radius, 1e-4))


def select_metric_representation(
    parsed: ParsedDataset,
    cfg: ReachabilityAnalysisConfig,
    *,
    variant: str | None = None,
) -> np.ndarray:
    variant_name = str(variant or getattr(cfg, "metric_state_variant", "full_state") or "full_state").strip().lower()
    if variant_name in {"position_only", "goal_xy", "xy"}:
        return parsed.goal_xy
    if variant_name in {"full_state", "full", "state_full"}:
        return parsed.state_full
    raise ValueError(f"Unsupported metric_state_variant: {variant_name}")


def _is_antmaze(dataset_id: str) -> bool:
    return "antmaze" in str(dataset_id).lower()


def _resolve_node_stride(parsed: ParsedDataset, cfg: ReachabilityAnalysisConfig) -> int:
    return int(cfg.node_stride_antmaze if _is_antmaze(parsed.dataset_id) else cfg.node_stride_pointmaze)


def _sample_episode_indices(length: int, stride: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    selected = list(range(0, int(length), max(int(stride), 1)))
    if selected[-1] != int(length) - 1:
        selected.append(int(length) - 1)
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _candidate_quantized_sample(
    positions: np.ndarray,
    num_candidates: int,
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    order = rng.permutation(positions.shape[0])
    quantize = max(radius, 1e-6)
    selected = []
    seen_keys: set[tuple[int, int]] = set()
    for index in order:
        point = positions[index]
        key = tuple(np.floor(point / quantize).astype(np.int64).tolist())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(int(index))
        if len(selected) >= num_candidates:
            break
    if len(selected) < num_candidates:
        selected_set = set(selected)
        for index in order:
            if int(index) in selected_set:
                continue
            selected.append(int(index))
            if len(selected) >= num_candidates:
                break
    return np.asarray(selected[:num_candidates], dtype=np.int64)


def _load_or_build_planning_aligned_node_pool(
    parsed: ParsedDataset,
    cfg: ReachabilityAnalysisConfig,
) -> np.ndarray:
    payload = {
        "dataset": parsed.dataset_id,
        "seed": cfg.seed,
        "node_stride_pointmaze": int(cfg.node_stride_pointmaze),
        "node_stride_antmaze": int(cfg.node_stride_antmaze),
        "candidate_pool_mode": str(cfg.candidate_pool_mode),
        "query_pool_mode": str(cfg.query_pool_mode),
    }
    cache_name = f"planning_nodes_{dataset_slug(parsed.dataset_id)}_{_hash_payload(payload)}.npz"
    cache_path = os.path.join(cfg.cache_dir, cache_name)
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return np.asarray(cached["node_global_indices"], dtype=np.int64)

    stride = max(_resolve_node_stride(parsed, cfg), 1)
    per_episode = []
    for episode_id, length in enumerate(parsed.episode_lengths.tolist()):
        local = _sample_episode_indices(int(length), stride)
        global_indices = parsed.episode_offsets[episode_id] + local
        per_episode.append(np.asarray(global_indices, dtype=np.int64))

    node_global_indices = np.concatenate(per_episode, axis=0).astype(np.int64)
    _save_npz(
        cache_path,
        node_global_indices=node_global_indices,
        effective_stride=np.asarray(stride),
    )
    return node_global_indices


def load_or_sample_candidates(
    parsed: ParsedDataset,
    cfg: ReachabilityAnalysisConfig,
    match_radius: float,
) -> np.ndarray:
    payload = {
        "dataset": parsed.dataset_id,
        "seed": cfg.seed,
        "num_candidates": cfg.num_candidates,
        "candidate_pool_mode": cfg.candidate_pool_mode,
        "node_stride_pointmaze": int(cfg.node_stride_pointmaze),
        "node_stride_antmaze": int(cfg.node_stride_antmaze),
        "candidate_sampling": cfg.candidate_sampling,
        "match_radius": match_radius,
    }
    cache_name = f"candidates_{dataset_slug(parsed.dataset_id)}_{_hash_payload(payload)}.npz"
    cache_path = os.path.join(cfg.cache_dir, cache_name)
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return np.asarray(cached["candidate_indices"], dtype=np.int64)

    rng = np.random.default_rng(cfg.seed + 17)
    pool_mode = str(cfg.candidate_pool_mode or "planning_aligned").strip().lower()
    if pool_mode == "planning_aligned":
        node_pool = _load_or_build_planning_aligned_node_pool(parsed, cfg)
        if node_pool.shape[0] <= int(cfg.num_candidates):
            candidate_indices = np.asarray(node_pool, dtype=np.int64)
        else:
            chosen = rng.choice(
                node_pool,
                size=int(cfg.num_candidates),
                replace=False,
            )
            candidate_indices = np.sort(np.asarray(chosen, dtype=np.int64))
    elif cfg.candidate_sampling == "random":
        count = min(cfg.num_candidates, parsed.positions.shape[0])
        candidate_indices = rng.choice(parsed.positions.shape[0], size=count, replace=False).astype(np.int64)
    else:
        candidate_indices = _candidate_quantized_sample(
            positions=parsed.positions,
            num_candidates=min(cfg.num_candidates, parsed.positions.shape[0]),
            radius=match_radius,
            rng=rng,
        )
    _save_npz(cache_path, candidate_indices=candidate_indices)
    return candidate_indices


def _filter_future_available(parsed: ParsedDataset, indices: np.ndarray) -> np.ndarray:
    candidate_indices = np.asarray(indices, dtype=np.int64)
    future_available = parsed.timesteps[candidate_indices] < (
        parsed.episode_lengths[parsed.episode_ids[candidate_indices]] - 1
    )
    return candidate_indices[future_available]


def _sample_anchor_indices(
    parsed: ParsedDataset,
    cfg: ReachabilityAnalysisConfig,
    match_radius: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    payload = {
        "dataset": parsed.dataset_id,
        "seed": cfg.seed,
        "num_anchors": cfg.num_anchors,
        "query_pool_mode": cfg.query_pool_mode,
        "node_stride_pointmaze": int(cfg.node_stride_pointmaze),
        "node_stride_antmaze": int(cfg.node_stride_antmaze),
        "match_radius": match_radius,
        "min_anchor_occurrences": cfg.min_anchor_occurrences,
        "max_anchor_occurrences": cfg.max_anchor_occurrences,
        "horizon": cfg.horizon,
    }
    cache_name = f"anchors_{dataset_slug(parsed.dataset_id)}_{_hash_payload(payload)}.npz"
    cache_path = os.path.join(cfg.cache_dir, cache_name)
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            anchor_indices = np.asarray(cached["anchor_indices"], dtype=np.int64)
            anchor_occurrence_lists = [np.asarray(row, dtype=np.int64) for row in cached["anchor_occurrences"]]
            return anchor_indices, anchor_occurrence_lists

    positions_tree = parsed.build_tree()
    rng = np.random.default_rng(cfg.seed)
    pool_mode = str(cfg.query_pool_mode or "planning_aligned").strip().lower()
    if pool_mode == "planning_aligned":
        valid_anchor_candidates = _filter_future_available(
            parsed,
            _load_or_build_planning_aligned_node_pool(parsed, cfg),
        )
    else:
        valid_anchor_candidates = _filter_future_available(
            parsed,
            np.arange(parsed.positions.shape[0], dtype=np.int64),
        )
    grouped_by_episode: list[np.ndarray] = []
    for episode_id in range(parsed.total_episodes):
        grouped_by_episode.append(valid_anchor_candidates[parsed.episode_ids[valid_anchor_candidates] == episode_id])

    anchor_indices: list[int] = []
    occurrence_lists: list[np.ndarray] = []
    seen_anchor_indices: set[int] = set()
    attempts = 0
    episode_order = rng.permutation(parsed.total_episodes)

    while len(anchor_indices) < cfg.num_anchors and attempts < cfg.max_anchor_attempts:
        episode_id = int(episode_order[attempts % len(episode_order)])
        episode_candidates = grouped_by_episode[episode_id]
        attempts += 1
        if episode_candidates.size == 0:
            continue
        anchor_index = int(episode_candidates[rng.integers(0, episode_candidates.size)])
        if anchor_index in seen_anchor_indices:
            continue
        occurrences = np.asarray(
            positions_tree.query_ball_point(parsed.positions[anchor_index], r=match_radius),
            dtype=np.int64,
        )
        future_occurrences = occurrences[
            parsed.timesteps[occurrences] < (parsed.episode_lengths[parsed.episode_ids[occurrences]] - 1)
        ]
        if future_occurrences.size < cfg.min_anchor_occurrences:
            continue
        if future_occurrences.size > cfg.max_anchor_occurrences:
            future_occurrences = np.sort(
                rng.choice(
                    future_occurrences,
                    size=cfg.max_anchor_occurrences,
                    replace=False,
                ).astype(np.int64)
            )
        seen_anchor_indices.add(anchor_index)
        anchor_indices.append(anchor_index)
        occurrence_lists.append(np.sort(future_occurrences))

    if not anchor_indices:
        raise RuntimeError(f"Failed to sample any valid anchors for {parsed.dataset_id}")

    anchor_indices_array = np.asarray(anchor_indices, dtype=np.int64)
    occurrence_object = np.asarray(occurrence_lists, dtype=object)
    _save_npz(cache_path, anchor_indices=anchor_indices_array, anchor_occurrences=occurrence_object)
    return anchor_indices_array, occurrence_lists


def _fit_pool_positions(
    parsed: ParsedDataset,
    cfg: ReachabilityAnalysisConfig,
    *,
    exclude_indices: np.ndarray | None = None,
    variant: str | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 123)
    metric_values = select_metric_representation(parsed, cfg, variant=variant)
    available_indices = np.arange(metric_values.shape[0], dtype=np.int64)
    if exclude_indices is not None:
        excluded = np.zeros(metric_values.shape[0], dtype=bool)
        excluded[np.asarray(exclude_indices, dtype=np.int64)] = True
        available_indices = available_indices[~excluded]
    if available_indices.size == 0:
        available_indices = np.arange(metric_values.shape[0], dtype=np.int64)
    pool_size = min(int(cfg.fit_pool_size), int(available_indices.size))
    indices = rng.choice(available_indices, size=pool_size, replace=False)
    return metric_values[indices]


def _fit_transition_pairs(
    parsed: ParsedDataset,
    cfg: ReachabilityAnalysisConfig,
    *,
    exclude_indices: np.ndarray | None = None,
    variant: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    metric_values = select_metric_representation(parsed, cfg, variant=variant)
    fit_states, fit_next_states = sample_transition_pairs(
        metric_values,
        parsed.episode_ids,
        parsed.timesteps,
        parsed.episode_lengths,
        max_pairs=cfg.fit_pool_size,
        seed=cfg.seed + 211,
        exclude_indices=exclude_indices,
    )
    if fit_states.shape[0] == 0:
        fallback = _fit_pool_positions(parsed, cfg, exclude_indices=exclude_indices, variant=variant)
        return fallback, fallback.copy()
    return fit_states, fit_next_states


def _device_string(cfg: ReachabilityAnalysisConfig) -> str:
    if cfg.ik_device != "auto":
        return cfg.ik_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def prepare_evaluation_context(parsed: ParsedDataset, cfg: ReachabilityAnalysisConfig) -> EvaluationContext:
    match_radius = resolve_match_radius(parsed, cfg.match_radius)
    candidate_indices = load_or_sample_candidates(parsed, cfg, match_radius)
    anchor_indices, anchor_occurrence_lists = _sample_anchor_indices(parsed, cfg, match_radius)
    ground_truth = compute_or_load_ground_truth(
        parsed=parsed,
        cfg=cfg,
        match_radius=match_radius,
        anchor_indices=anchor_indices,
        anchor_occurrence_lists=anchor_occurrence_lists,
        candidate_indices=candidate_indices,
    )
    return EvaluationContext(
        parsed=parsed,
        match_radius=match_radius,
        anchor_indices=anchor_indices,
        anchor_occurrence_lists=anchor_occurrence_lists,
        candidate_indices=candidate_indices,
        ground_truth=ground_truth,
    )


def compute_or_load_ground_truth(
    parsed: ParsedDataset,
    cfg: ReachabilityAnalysisConfig,
    match_radius: float,
    anchor_indices: np.ndarray,
    anchor_occurrence_lists: list[np.ndarray],
    candidate_indices: np.ndarray,
) -> GroundTruthBundle:
    payload = {
        "dataset": parsed.dataset_id,
        "seed": cfg.seed,
        "num_anchors": int(anchor_indices.shape[0]),
        "num_candidates": int(candidate_indices.shape[0]),
        "horizon": cfg.horizon,
        "match_radius": match_radius,
        "max_anchor_occurrences": cfg.max_anchor_occurrences,
    }
    cache_name = f"ground_truth_{dataset_slug(parsed.dataset_id)}_{_hash_payload(payload)}.npz"
    cache_path = os.path.join(cfg.cache_dir, cache_name)
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return GroundTruthBundle(
                reach_prob=np.asarray(cached["reach_prob"], dtype=np.float32),
                temporal_reachability=np.asarray(cached["temporal_reachability"], dtype=np.float32),
                oracle_temporal=np.asarray(cached["oracle_temporal"], dtype=np.float32),
                occurrence_counts=np.asarray(cached["occurrence_counts"], dtype=np.float32),
            )

    candidate_positions = parsed.positions[candidate_indices]
    bundle = compute_ground_truth_bundle(
        anchor_occurrence_lists=anchor_occurrence_lists,
        candidate_positions=candidate_positions,
        positions=parsed.positions,
        episode_ids=parsed.episode_ids,
        timesteps=parsed.timesteps,
        episode_offsets=parsed.episode_offsets,
        episode_lengths=parsed.episode_lengths,
        horizon=cfg.horizon,
        match_radius=match_radius,
    )
    _save_npz(
        cache_path,
        reach_prob=bundle.reach_prob,
        temporal_reachability=bundle.temporal_reachability,
        oracle_temporal=bundle.oracle_temporal,
        occurrence_counts=bundle.occurrence_counts,
    )
    return bundle


def compute_or_load_baseline_scores(
    context: EvaluationContext,
    cfg: ReachabilityAnalysisConfig,
) -> SimilarityBundle:
    payload = {
        "cache_version": BASELINE_CACHE_VERSION,
        "dataset": context.parsed.dataset_id,
        "seed": cfg.seed,
        "num_anchors": int(context.anchor_indices.shape[0]),
        "num_candidates": int(context.candidate_indices.shape[0]),
        "match_radius": context.match_radius,
        "gk_sigma_mode": cfg.gk_sigma_mode,
        "gk_sigma": cfg.gk_sigma,
        "mahalanobis_covariance_estimator": cfg.mahalanobis_covariance_estimator,
        "mahalanobis_implementation": cfg.mahalanobis_implementation,
        "mahalanobis_eps": cfg.mahalanobis_eps,
        "adaptive_gaussian_k": cfg.adaptive_gaussian_k,
        "adaptive_gaussian_eps": cfg.adaptive_gaussian_eps,
        "adaptive_gaussian_output": cfg.adaptive_gaussian_output,
        "dynamics_backend": cfg.dynamics_backend,
        "dynamics_num_bins": cfg.dynamics_num_bins,
        "dynamics_distance_metric": cfg.dynamics_distance_metric,
        "dynamics_alpha": cfg.dynamics_alpha,
        "dynamics_min_count": cfg.dynamics_min_count,
        "dynamics_eps": cfg.dynamics_eps,
        "dynamics_local_knn_m": cfg.dynamics_local_knn_m,
        "dynamics_local_distance_metric": cfg.dynamics_local_distance_metric,
        "dynamics_state_variant": cfg.dynamics_state_variant,
        "fit_pool_size": cfg.fit_pool_size,
        "metric_state_variant": cfg.metric_state_variant,
        "replay_temporal_window": resolve_replay_temporal_window(cfg),
        "horizon": cfg.horizon,
    }
    cache_name = f"baseline_scores_{dataset_slug(context.parsed.dataset_id)}_{_hash_payload(payload)}.npz"
    cache_path = os.path.join(cfg.cache_dir, cache_name)
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        required_keys = {
            "euclidean",
            "gaussian",
            "mahalanobis",
            "adaptive_gaussian",
            "temporal_distance",
            "one_step_dynamics",
            "replay_temporal",
            "oracle_temporal",
        }
        if cached is not None and required_keys.issubset(set(cached.keys())):
            return SimilarityBundle(
                euclidean=np.asarray(cached["euclidean"], dtype=np.float32),
                gaussian=np.asarray(cached["gaussian"], dtype=np.float32),
                mahalanobis=np.asarray(cached["mahalanobis"], dtype=np.float32),
                adaptive_gaussian=np.asarray(cached["adaptive_gaussian"], dtype=np.float32),
                temporal_distance=np.asarray(cached["temporal_distance"], dtype=np.float32),
                one_step_dynamics=np.asarray(cached["one_step_dynamics"], dtype=np.float32),
                replay_temporal=np.asarray(cached["replay_temporal"], dtype=np.float32),
                oracle_temporal=np.asarray(cached["oracle_temporal"], dtype=np.float32),
            )

    metric_values = select_metric_representation(context.parsed, cfg)
    anchor_positions = metric_values[context.anchor_indices]
    candidate_positions = metric_values[context.candidate_indices]
    anchor_goal_xy = context.parsed.goal_xy[context.anchor_indices]
    candidate_goal_xy = context.parsed.goal_xy[context.candidate_indices]
    exclude_indices = np.unique(np.concatenate([context.anchor_indices, context.candidate_indices])).astype(np.int64)
    fit_positions = _fit_pool_positions(context.parsed, cfg, exclude_indices=exclude_indices)
    dynamics_variant = cfg.dynamics_state_variant or cfg.metric_state_variant
    dynamics_metric_values = select_metric_representation(
        context.parsed,
        cfg,
        variant=dynamics_variant,
    )
    dynamics_anchor_positions = dynamics_metric_values[context.anchor_indices]
    dynamics_candidate_positions = dynamics_metric_values[context.candidate_indices]
    fit_states, fit_next_states = _fit_transition_pairs(
        context.parsed,
        cfg,
        exclude_indices=exclude_indices,
        variant=dynamics_variant,
    )
    euclidean = compute_euclidean_scores(anchor_positions, candidate_positions)
    gaussian = compute_gaussian_scores(
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        sigma_mode=cfg.gk_sigma_mode,
        sigma_value=cfg.gk_sigma,
        fallback_sigma=max(context.match_radius, context.parsed.median_step_size, 1e-4),
    )
    mahalanobis = compute_mahalanobis_scores(
        fit_positions=fit_positions,
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        covariance_estimator=cfg.mahalanobis_covariance_estimator,
        implementation=cfg.mahalanobis_implementation,
        eps=cfg.mahalanobis_eps,
    )
    adaptive_gaussian = compute_adaptive_gaussian_scores(
        fit_positions=fit_positions,
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        k=cfg.adaptive_gaussian_k,
        eps=cfg.adaptive_gaussian_eps,
        output=cfg.adaptive_gaussian_output,
    )
    one_step_dynamics = compute_one_step_dynamics_scores(
        fit_states=fit_states,
        fit_next_states=fit_next_states,
        anchor_positions=dynamics_anchor_positions,
        candidate_positions=dynamics_candidate_positions,
        backend=cfg.dynamics_backend,
        num_bins=cfg.dynamics_num_bins,
        distance_metric=cfg.dynamics_distance_metric,
        alpha=cfg.dynamics_alpha,
        min_count=cfg.dynamics_min_count,
        seed=cfg.seed,
        eps=cfg.dynamics_eps,
        local_knn_m=cfg.dynamics_local_knn_m,
        local_distance_metric=cfg.dynamics_local_distance_metric,
    )
    temporal_distance = compute_temporal_distance_scores(
        anchor_global_indices=context.anchor_indices,
        candidate_global_indices=context.candidate_indices,
        episode_ids=context.parsed.episode_ids,
        timesteps=context.parsed.timesteps,
    )
    replay_temporal = compute_replay_temporal_scores(
        anchor_occurrence_lists=context.anchor_occurrence_lists,
        candidate_positions=candidate_goal_xy,
        positions=context.parsed.goal_xy,
        episode_ids=context.parsed.episode_ids,
        timesteps=context.parsed.timesteps,
        episode_offsets=context.parsed.episode_offsets,
        episode_lengths=context.parsed.episode_lengths,
        match_radius=context.match_radius,
        temporal_window=resolve_replay_temporal_window(cfg),
    )
    oracle_temporal = np.asarray(context.ground_truth.oracle_temporal, dtype=np.float32)
    bundle = SimilarityBundle(
        euclidean=euclidean,
        gaussian=gaussian,
        mahalanobis=mahalanobis,
        adaptive_gaussian=adaptive_gaussian,
        temporal_distance=temporal_distance,
        one_step_dynamics=one_step_dynamics,
        replay_temporal=replay_temporal,
        oracle_temporal=oracle_temporal,
    )
    _save_npz(
        cache_path,
        euclidean=bundle.euclidean,
        gaussian=bundle.gaussian,
        mahalanobis=bundle.mahalanobis,
        adaptive_gaussian=bundle.adaptive_gaussian,
        temporal_distance=bundle.temporal_distance,
        one_step_dynamics=bundle.one_step_dynamics,
        replay_temporal=bundle.replay_temporal,
        oracle_temporal=bundle.oracle_temporal,
    )
    return bundle


def compute_or_load_ik_score_matrix(
    context: EvaluationContext,
    cfg: ReachabilityAnalysisConfig,
    subsample_size: int,
    temperature: float,
) -> np.ndarray:
    payload = {
        "dataset": context.parsed.dataset_id,
        "seed": cfg.seed,
        "num_anchors": int(context.anchor_indices.shape[0]),
        "num_candidates": int(context.candidate_indices.shape[0]),
        "match_radius": context.match_radius,
        "ik_ensemble_size": cfg.ik_ensemble_size,
        "ik_subsample_size": int(subsample_size),
        "ik_temperature": float(temperature),
        "fit_pool_size": cfg.fit_pool_size,
        "metric_state_variant": cfg.metric_state_variant,
    }
    cache_name = f"ik_scores_{dataset_slug(context.parsed.dataset_id)}_{_hash_payload(payload)}.npz"
    cache_path = os.path.join(cfg.cache_dir, cache_name)
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return np.asarray(cached["ik"], dtype=np.float32)

    metric_values = select_metric_representation(context.parsed, cfg)
    anchor_positions = metric_values[context.anchor_indices]
    candidate_positions = metric_values[context.candidate_indices]
    ik_scores = compute_ik_scores(
        fit_positions=_fit_pool_positions(context.parsed, cfg),
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        ensemble_size=cfg.ik_ensemble_size,
        subsample_size=subsample_size,
        temperature=temperature,
        device=_device_string(cfg),
        batch_size=cfg.ik_batch_size,
    )
    _save_npz(cache_path, ik=ik_scores)
    return ik_scores


def _build_summary_rows(
    dataset_id: str,
    cfg: ReachabilityAnalysisConfig,
    context: EvaluationContext,
    sim_map: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gt_map = {
        "reach_prob": context.ground_truth.reach_prob,
        "temporal_reachability": context.ground_truth.temporal_reachability,
    }
    summary_rows: list[dict[str, Any]] = []
    per_anchor_rows: list[dict[str, Any]] = []
    for gt_name, gt_matrix in gt_map.items():
        for method_name, sim_matrix in sim_map.items():
            anchor_rows, summary = evaluate_alignment(
                ground_truth=gt_matrix,
                similarity_scores=sim_matrix,
                top_k=cfg.top_k,
                anchor_global_indices=context.anchor_indices,
                candidate_global_indices=context.candidate_indices,
                dataset_name=dataset_id,
                method_name=method_name,
                ground_truth_type=gt_name,
                occurrence_counts=context.ground_truth.occurrence_counts,
            )
            for row in anchor_rows:
                row["horizon"] = cfg.horizon
                row["top_k"] = cfg.top_k
                row["match_radius"] = context.match_radius
                per_anchor_rows.append(row)
            summary["horizon"] = cfg.horizon
            summary["top_k"] = cfg.top_k
            summary["match_radius"] = context.match_radius
            summary_rows.append(summary)
    return summary_rows, per_anchor_rows


def analyze_single_dataset(
    dataset_id: str,
    cfg: ReachabilityAnalysisConfig,
) -> dict[str, Any]:
    parsed = load_or_parse_dataset(
        dataset_id=dataset_id,
        cache_dir=cfg.cache_dir,
        overwrite_cache=cfg.overwrite_cache,
        minari_datasets_path=cfg.minari_datasets_path,
        seed=cfg.seed,
    )
    context = prepare_evaluation_context(parsed, cfg)
    baselines = compute_or_load_baseline_scores(context, cfg)
    ik_scores = compute_or_load_ik_score_matrix(context, cfg, cfg.ik_subsample_size, cfg.ik_temperature)

    sim_map = {
        "euclidean": baselines.euclidean,
        "gaussian": baselines.gaussian,
        "mahalanobis": baselines.mahalanobis,
        "adaptive_gaussian": baselines.adaptive_gaussian,
        "ik": ik_scores,
        "one_step_dynamics": baselines.one_step_dynamics,
        "temporal_distance": baselines.temporal_distance,
        "replay_temporal": baselines.replay_temporal,
    }
    if cfg.report_oracle_temp:
        sim_map["oracle_temporal"] = baselines.oracle_temporal

    summary_rows, per_anchor_rows = _build_summary_rows(dataset_id, cfg, context, sim_map)
    return {
        "dataset": dataset_id,
        "dataset_slug": dataset_slug(dataset_id),
        "match_radius": context.match_radius,
        "summary_rows": summary_rows,
        "per_anchor_rows": per_anchor_rows,
        "anchor_indices": context.anchor_indices,
        "candidate_indices": context.candidate_indices,
        "parsed": parsed,
        "ground_truth": context.ground_truth,
        "similarities": SimilarityBundle(
            euclidean=baselines.euclidean,
            gaussian=baselines.gaussian,
            mahalanobis=baselines.mahalanobis,
            adaptive_gaussian=baselines.adaptive_gaussian,
            temporal_distance=baselines.temporal_distance,
            one_step_dynamics=baselines.one_step_dynamics,
            replay_temporal=baselines.replay_temporal,
            oracle_temporal=baselines.oracle_temporal,
            ik=ik_scores,
        ),
        "config": asdict(cfg),
    }


def resolve_replay_temporal_window(cfg: ReachabilityAnalysisConfig) -> int:
    if cfg.replay_temporal_window is None or cfg.replay_temporal_window <= 0:
        return int(cfg.horizon)
    return int(cfg.replay_temporal_window)


def resolve_horizon_values_for_dataset(
    dataset_id: str,
    cfg: ReachabilityAnalysisConfig,
) -> list[int]:
    if cfg.per_dataset_horizon_values:
        if dataset_id in cfg.per_dataset_horizon_values:
            return list(cfg.per_dataset_horizon_values[dataset_id])
        slug = dataset_slug(dataset_id)
        if slug in cfg.per_dataset_horizon_values:
            return list(cfg.per_dataset_horizon_values[slug])
    if cfg.horizon_values:
        return list(cfg.horizon_values)
    return [int(cfg.horizon)]


def _search_cfg_for_horizon(cfg: ReachabilityAnalysisConfig, horizon: int) -> ReachabilityAnalysisConfig:
    return replace(
        cfg,
        horizon=horizon,
        num_anchors=cfg.search_num_anchors,
        num_candidates=cfg.search_num_candidates,
        top_k=cfg.final_top_k,
        max_anchor_occurrences=cfg.search_max_anchor_occurrences,
    )


def _final_cfg_for_best_row(cfg: ReachabilityAnalysisConfig, best_row: dict[str, Any]) -> ReachabilityAnalysisConfig:
    return replace(
        cfg,
        horizon=int(best_row["horizon"]),
        num_anchors=cfg.final_num_anchors,
        num_candidates=cfg.final_num_candidates,
        top_k=cfg.final_top_k,
        max_anchor_occurrences=cfg.final_max_anchor_occurrences,
        ik_subsample_size=int(best_row["ik_subsample_size"]),
        ik_temperature=float(best_row["ik_temperature"]),
    )


def _selection_summary_key(row: dict[str, Any], cfg: ReachabilityAnalysisConfig) -> tuple[float, float, float, float, float]:
    metric_key = f"{cfg.selection_metric}_mean"
    primary = float(row.get(metric_key, 0.0))
    ndcg = float(row.get("ndcg_at_k_mean", 0.0))
    pearson = float(row.get("pearson_mean", 0.0))
    smaller_subsample = -float(row.get("ik_subsample_size", 0.0))
    smaller_temperature = -float(row.get("ik_temperature", 0.0))
    return (primary, ndcg, pearson, smaller_subsample, smaller_temperature)


def select_best_search_row(dataset_rows: list[dict[str, Any]], cfg: ReachabilityAnalysisConfig) -> dict[str, Any]:
    if not dataset_rows:
        raise RuntimeError("No search rows to select from")
    return max(dataset_rows, key=lambda row: _selection_summary_key(row, cfg))


def _mean_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["ground_truth_type"]), str(row["method"]))
        groups.setdefault(key, []).append(row)

    merged_rows = []
    for (ground_truth_type, method), group_rows in sorted(groups.items()):
        merged_rows.append(
            {
                "dataset": "overall",
                "ground_truth_type": ground_truth_type,
                "method": method,
                "num_datasets": len(group_rows),
                "num_anchors": int(np.sum([int(row["num_anchors"]) for row in group_rows])),
                "spearman_mean": float(np.mean([float(row["spearman_mean"]) for row in group_rows])),
                "pearson_mean": float(np.mean([float(row["pearson_mean"]) for row in group_rows])),
                "recall_at_k_mean": float(np.mean([float(row["recall_at_k_mean"]) for row in group_rows])),
                "topk_overlap_mean": float(np.mean([float(row["topk_overlap_mean"]) for row in group_rows])),
                "ndcg_at_k_mean": float(np.mean([float(row["ndcg_at_k_mean"]) for row in group_rows])),
                "auc_mean": float(np.mean([float(row["auc_mean"]) for row in group_rows])),
            }
        )
    return merged_rows


def _scatter_sample_indices(total_points: int, max_points: int) -> np.ndarray:
    if total_points <= max_points:
        return np.arange(total_points, dtype=np.int64)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(total_points, size=max_points, replace=False))


def plot_dataset_bars(result: dict[str, Any], figures_dir: str) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main_rows = [row for row in result["summary_rows"] if row["ground_truth_type"] == "reach_prob"]
    methods = [row["method"] for row in main_rows]
    spearman = [float(row["spearman_mean"]) for row in main_rows]
    ndcg = [float(row["ndcg_at_k_mean"]) for row in main_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(methods, spearman, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#9467BD"][: len(methods)])
    axes[0].set_title("Spearman")
    axes[0].set_ylim(min(-0.1, min(spearman) - 0.05), max(1.0, max(spearman) + 0.05))
    axes[1].bar(methods, ndcg, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#9467BD"][: len(methods)])
    axes[1].set_title(f"NDCG@{result['summary_rows'][0]['top_k']}")
    axes[1].set_ylim(0.0, max(1.0, max(ndcg) + 0.05))
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(result["dataset"])
    fig.tight_layout()

    out_path = os.path.join(figures_dir, f"{result['dataset_slug']}_bars.png")
    ensure_dir(figures_dir)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dataset_scatter(result: dict[str, Any], figures_dir: str) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.asarray(result["ground_truth"].reach_prob[0], dtype=np.float64)
    ik = np.asarray(result["similarities"].ik[0], dtype=np.float64)
    baseline = np.asarray(result["similarities"].replay_temporal[0], dtype=np.float64)
    indices = _scatter_sample_indices(gt.shape[0], result["config"]["scatter_points"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(gt[indices], ik[indices], s=8, alpha=0.5, color="#4C78A8")
    axes[0].set_title("Ground Truth vs IK")
    axes[0].set_xlabel("Reach Probability")
    axes[0].set_ylabel("IK Score")

    axes[1].scatter(gt[indices], baseline[indices], s=8, alpha=0.5, color="#54A24B")
    axes[1].set_title("Ground Truth vs Replay-temp")
    axes[1].set_xlabel("Reach Probability")
    axes[1].set_ylabel("Replay-temp Score")

    fig.suptitle(f"{result['dataset']} (anchor 0)")
    fig.tight_layout()
    out_path = os.path.join(figures_dir, f"{result['dataset_slug']}_scatter.png")
    ensure_dir(figures_dir)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_overall_summary(overall_rows: list[dict[str, Any]], figures_dir: str) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main_rows = [row for row in overall_rows if row["ground_truth_type"] == "reach_prob"]
    methods = [row["method"] for row in main_rows]
    scores = [float(row["spearman_mean"]) for row in main_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(methods, scores, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#9467BD"][: len(methods)])
    ax.set_ylabel("Avg Spearman")
    ax.set_title("Overall Reachability Alignment")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()

    out_path = os.path.join(figures_dir, "overall_summary.png")
    ensure_dir(figures_dir)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_search_heatmap(
    dataset_id: str,
    dataset_rows: list[dict[str, Any]],
    figures_dir: str,
    best_horizon: int,
) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_rows = [row for row in dataset_rows if int(row["horizon"]) == int(best_horizon)]
    subsamples = sorted({int(row["ik_subsample_size"]) for row in best_rows})
    temperatures = sorted({float(row["ik_temperature"]) for row in best_rows})
    matrix = np.full((len(subsamples), len(temperatures)), np.nan, dtype=np.float64)

    subsample_to_idx = {value: idx for idx, value in enumerate(subsamples)}
    temp_to_idx = {value: idx for idx, value in enumerate(temperatures)}
    for row in best_rows:
        matrix[subsample_to_idx[int(row["ik_subsample_size"])], temp_to_idx[float(row["ik_temperature"])]] = float(
            row["spearman_mean"]
        )

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(temperatures)))
    ax.set_xticklabels([str(value) for value in temperatures], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(subsamples)))
    ax.set_yticklabels([str(value) for value in subsamples])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Subsample size")
    ax.set_title(f"{dataset_id} IK Search Heatmap (best H={best_horizon})")
    fig.colorbar(im, ax=ax, label="Spearman on reach_prob")
    fig.tight_layout()

    out_path = os.path.join(figures_dir, f"{dataset_slug(dataset_id)}_ik_heatmap_best_h.png")
    ensure_dir(figures_dir)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_horizon_trace(
    dataset_id: str,
    dataset_rows: list[dict[str, Any]],
    figures_dir: str,
) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in dataset_rows:
        grouped.setdefault(int(row["horizon"]), []).append(row)
    horizons = sorted(grouped.keys())
    best_scores = [max(float(row["spearman_mean"]) for row in grouped[h]) for h in horizons]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(horizons, best_scores, marker="o", color="#4C78A8")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Best IK Spearman")
    ax.set_title(f"{dataset_id} H Search Trace")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path = os.path.join(figures_dir, f"{dataset_slug(dataset_id)}_h_trace.png")
    ensure_dir(figures_dir)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_report(
    cfg: ReachabilityAnalysisConfig,
    dataset_results: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    report_path: str,
    figure_paths: dict[str, Any],
    best_rows: list[dict[str, Any]] | None = None,
    search_paths: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Reachability Alignment Report",
        "",
        "## Setup",
        "",
        f"- Datasets: {', '.join(cfg.datasets)}",
        f"- Final top-k: {cfg.final_top_k}",
        f"- Final num anchors per dataset: {cfg.final_num_anchors}",
        f"- Final num candidates per dataset: {cfg.final_num_candidates}",
        f"- Search num anchors per dataset: {cfg.search_num_anchors}",
        f"- Search num candidates per dataset: {cfg.search_num_candidates}",
        f"- Candidate sampling: {cfg.candidate_sampling}",
        f"- Max sampled occurrences per anchor (search/final): {cfg.search_max_anchor_occurrences}/{cfg.final_max_anchor_occurrences}",
        f"- Minari datasets path: `{cfg.minari_datasets_path}`",
        f"- Selection metric: `{cfg.selection_metric}` on `{cfg.selection_ground_truth}`",
        "",
        "## Ground Truth",
        "",
        "- Main ground truth: empirical future-H reach probability based on radius-matched future visits.",
        "- Secondary ground truth: temporal reachability score using first-hit weighting `1 / tau` within the same trajectory.",
        "",
        "## Similarity Definitions",
        "",
        "- Euclidean: negative 2D position distance.",
        "- Gaussian kernel: legacy repository Gaussian baseline kept for historical comparability.",
        "- Mahalanobis: global covariance-whitened distance fitted on training states only.",
        "- Adaptive Gaussian: local-bandwidth Gaussian using training-set k-NN scales.",
        "- Isolation kernel: repository `SoftIsolationKernel` fitted on sampled offline positions.",
        "- One-step dynamics: reward-free one-step next-state-distribution distance fitted on training transitions only.",
        "- Replay-temp: lag-weighted replay co-visitation affinity accumulated over future windows and visitation-normalized.",
        "- Oracle-temp: future-window earliest-hit upper bound using `1 / (1 + tau_first)`.",
        "",
    ]

    if best_rows:
        lines.extend(
            [
                "## Best Config Per Dataset",
                "",
                "| Dataset | Best H | Best Subsample | Best Temperature | Spearman | NDCG@k | Pearson |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in best_rows:
            lines.append(
                f"| {row['dataset']} | {row['horizon']} | {row['ik_subsample_size']} | "
                f"{row['ik_temperature']} | {row['spearman_mean']:.4f} | "
                f"{row['ndcg_at_k_mean']:.4f} | {row['pearson_mean']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Dataset Notes",
            "",
        ]
    )
    for result in dataset_results:
        parsed: ParsedDataset = result["parsed"]
        lines.extend(
            [
                f"### {result['dataset']}",
                "",
                f"- Total episodes: {parsed.total_episodes}",
                f"- Parsed states: {parsed.positions.shape[0]}",
                f"- Median step size: {parsed.median_step_size:.6f}",
                f"- P90 nearest-neighbor distance: {parsed.p90_nearest_neighbor:.6f}",
                f"- Match radius used: {result['match_radius']:.6f}",
                f"- Usable anchors sampled: {len(result['anchor_indices'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## Overall Summary",
            "",
            "| Ground Truth | Method | Spearman | Pearson | Recall@k | Top-k Overlap | NDCG@k | AUC |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overall_rows:
        lines.append(
            f"| {row['ground_truth_type']} | {row['method']} | "
            f"{row['spearman_mean']:.4f} | {row['pearson_mean']:.4f} | "
            f"{row['recall_at_k_mean']:.4f} | {row['topk_overlap_mean']:.4f} | "
            f"{row['ndcg_at_k_mean']:.4f} | {row['auc_mean']:.4f} |"
        )

    if search_paths is not None:
        lines.extend(
            [
                "",
                "## Search Outputs",
                "",
                f"- Full search table: `{search_paths['full_search_table_path']}`",
                f"- Best config table: `{search_paths['best_config_path']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- Overall summary: `{figure_paths['overall']}`",
        ]
    )
    for dataset_id, paths in figure_paths["datasets"].items():
        lines.append(f"- {dataset_id}: `{paths[0]}`, `{paths[1]}`")
    if "search" in figure_paths:
        for dataset_id, paths in figure_paths["search"].items():
            lines.append(f"- Search {dataset_id}: `{paths[0]}`, `{paths[1]}`")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The main analysis uses 2D positions only, not full observations.",
            "- Radius matching is an empirical proxy for repeated states and may blur fine-grained state differences.",
            "- Anchor occurrence sets are subsampled to a fixed maximum for tractable offline analysis.",
            "- Oracle-temp is an upper-bound reference and should not be interpreted as a fair replay-only baseline.",
        ]
    )

    ensure_dir(os.path.dirname(report_path))
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")


def run_search(cfg: ReachabilityAnalysisConfig) -> dict[str, Any]:
    search_dir = os.path.join(cfg.output_dir, "search")
    figures_dir = os.path.join(search_dir, "figures")
    ensure_dir(search_dir)
    ensure_dir(figures_dir)

    full_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    figure_paths: dict[str, list[str]] = {}

    for dataset_id in cfg.datasets:
        parsed = load_or_parse_dataset(
            dataset_id=dataset_id,
            cache_dir=cfg.cache_dir,
            overwrite_cache=cfg.overwrite_cache,
            minari_datasets_path=cfg.minari_datasets_path,
            seed=cfg.seed,
        )
        dataset_rows: list[dict[str, Any]] = []
        for horizon in resolve_horizon_values_for_dataset(dataset_id, cfg):
            search_cfg = _search_cfg_for_horizon(cfg, horizon)
            context = prepare_evaluation_context(parsed, search_cfg)
            for subsample_size in cfg.ik_subsample_grid or [cfg.ik_subsample_size]:
                for temperature in cfg.ik_temperature_grid or [cfg.ik_temperature]:
                    ik_scores = compute_or_load_ik_score_matrix(
                        context=context,
                        cfg=search_cfg,
                        subsample_size=int(subsample_size),
                        temperature=float(temperature),
                    )
                    _, summary = evaluate_alignment(
                        ground_truth=context.ground_truth.reach_prob,
                        similarity_scores=ik_scores,
                        top_k=search_cfg.top_k,
                        anchor_global_indices=context.anchor_indices,
                        candidate_global_indices=context.candidate_indices,
                        dataset_name=dataset_id,
                        method_name="ik",
                        ground_truth_type="reach_prob",
                        occurrence_counts=context.ground_truth.occurrence_counts,
                    )
                    row = {
                        "dataset": dataset_id,
                        "horizon": int(horizon),
                        "ik_subsample_size": int(subsample_size),
                        "ik_temperature": float(temperature),
                        "search_num_anchors": int(search_cfg.num_anchors),
                        "search_num_candidates": int(search_cfg.num_candidates),
                        "match_radius": float(context.match_radius),
                        "selection_metric": cfg.selection_metric,
                        "selection_ground_truth": cfg.selection_ground_truth,
                        "selection_split": cfg.selection_split,
                        **summary,
                    }
                    full_rows.append(row)
                    dataset_rows.append(row)

        best_row = select_best_search_row(dataset_rows, cfg)
        best_rows.append(best_row)
        figure_paths[dataset_id] = [
            plot_search_heatmap(dataset_id, dataset_rows, figures_dir, best_horizon=int(best_row["horizon"])),
            plot_horizon_trace(dataset_id, dataset_rows, figures_dir),
        ]

    full_search_table_path = os.path.join(search_dir, "full_search_table.csv")
    best_config_path = os.path.join(search_dir, "best_config_per_dataset.csv")
    save_csv(
        full_search_table_path,
        full_rows,
        [
            "dataset",
            "ground_truth_type",
            "method",
            "horizon",
            "ik_subsample_size",
            "ik_temperature",
            "search_num_anchors",
            "search_num_candidates",
            "match_radius",
            "selection_metric",
            "selection_ground_truth",
            "selection_split",
            "num_anchors",
            "spearman_mean",
            "pearson_mean",
            "recall_at_k_mean",
            "topk_overlap_mean",
            "ndcg_at_k_mean",
            "auc_mean",
        ],
    )
    save_csv(
        best_config_path,
        best_rows,
        [
            "dataset",
            "ground_truth_type",
            "method",
            "horizon",
            "ik_subsample_size",
            "ik_temperature",
            "search_num_anchors",
            "search_num_candidates",
            "match_radius",
            "selection_metric",
            "selection_ground_truth",
            "selection_split",
            "num_anchors",
            "spearman_mean",
            "pearson_mean",
            "recall_at_k_mean",
            "topk_overlap_mean",
            "ndcg_at_k_mean",
            "auc_mean",
        ],
    )
    return {
        "full_rows": full_rows,
        "best_rows": best_rows,
        "full_search_table_path": full_search_table_path,
        "best_config_path": best_config_path,
        "figure_paths": figure_paths,
    }


def _load_best_rows(best_config_path: str) -> list[dict[str, Any]]:
    rows = []
    with open(best_config_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No best config rows found in {best_config_path}")
    return rows


def run_final_evaluation(
    cfg: ReachabilityAnalysisConfig,
    best_rows: list[dict[str, Any]],
    search_paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_results = []
    all_summary_rows = []
    all_per_anchor_rows = []
    figures_dir = os.path.join(cfg.output_dir, "figures")
    tables_dir = os.path.join(cfg.output_dir, "tables")
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)
    figure_paths: dict[str, Any] = {"datasets": {}}

    best_by_dataset = {row["dataset"]: row for row in best_rows}
    for dataset_id in cfg.datasets:
        best_row = best_by_dataset[dataset_id]
        final_cfg = _final_cfg_for_best_row(cfg, best_row)
        result = analyze_single_dataset(dataset_id, final_cfg)
        for row in result["summary_rows"]:
            row["best_ik_subsample_size"] = int(best_row["ik_subsample_size"])
            row["best_ik_temperature"] = float(best_row["ik_temperature"])
        for row in result["per_anchor_rows"]:
            row["best_ik_subsample_size"] = int(best_row["ik_subsample_size"])
            row["best_ik_temperature"] = float(best_row["ik_temperature"])

        dataset_results.append(result)
        all_summary_rows.extend(result["summary_rows"])
        all_per_anchor_rows.extend(result["per_anchor_rows"])

        figure_paths["datasets"][dataset_id] = [
            plot_dataset_bars(result, figures_dir),
            plot_dataset_scatter(result, figures_dir),
        ]

        per_dataset_path = os.path.join(tables_dir, f"{dataset_slug(dataset_id)}_metrics.csv")
        save_csv(
            per_dataset_path,
            result["summary_rows"],
            [
                "dataset",
                "ground_truth_type",
                "method",
                "num_anchors",
                "horizon",
                "top_k",
                "match_radius",
                "best_ik_subsample_size",
                "best_ik_temperature",
                "spearman_mean",
                "pearson_mean",
                "recall_at_k_mean",
                "topk_overlap_mean",
                "ndcg_at_k_mean",
                "auc_mean",
            ],
        )

    overall_rows = _mean_summary_rows(all_summary_rows)
    overall_table_path = os.path.join(tables_dir, "overall_summary.csv")
    per_anchor_table_path = os.path.join(tables_dir, "per_anchor_metrics.csv")
    per_dataset_table_path = os.path.join(tables_dir, "per_dataset_metrics.csv")

    save_csv(
        per_dataset_table_path,
        all_summary_rows,
        [
            "dataset",
            "ground_truth_type",
            "method",
            "num_anchors",
            "horizon",
            "top_k",
            "match_radius",
            "best_ik_subsample_size",
            "best_ik_temperature",
            "spearman_mean",
            "pearson_mean",
            "recall_at_k_mean",
            "topk_overlap_mean",
            "ndcg_at_k_mean",
            "auc_mean",
        ],
    )
    save_csv(
        overall_table_path,
        overall_rows,
        [
            "dataset",
            "ground_truth_type",
            "method",
            "num_datasets",
            "num_anchors",
            "spearman_mean",
            "pearson_mean",
            "recall_at_k_mean",
            "topk_overlap_mean",
            "ndcg_at_k_mean",
            "auc_mean",
        ],
    )
    save_csv(
        per_anchor_table_path,
        all_per_anchor_rows,
        [
            "dataset",
            "ground_truth_type",
            "method",
            "anchor_row",
            "anchor_index",
            "occurrence_count",
            "horizon",
            "top_k",
            "match_radius",
            "best_ik_subsample_size",
            "best_ik_temperature",
            "spearman",
            "pearson",
            "recall_at_k",
            "topk_overlap",
            "ndcg_at_k",
            "auc",
        ],
    )

    figure_paths["overall"] = plot_overall_summary(overall_rows, figures_dir)
    if search_paths is not None:
        figure_paths["search"] = search_paths["figure_paths"]
    report_path = os.path.join(cfg.output_dir, "report.md")
    build_report(
        cfg=cfg,
        dataset_results=dataset_results,
        overall_rows=overall_rows,
        report_path=report_path,
        figure_paths=figure_paths,
        best_rows=best_rows,
        search_paths=search_paths,
    )
    return {
        "dataset_results": dataset_results,
        "overall_rows": overall_rows,
        "all_summary_rows": all_summary_rows,
        "all_per_anchor_rows": all_per_anchor_rows,
        "overall_table_path": overall_table_path,
        "per_anchor_table_path": per_anchor_table_path,
        "per_dataset_table_path": per_dataset_table_path,
        "report_path": report_path,
        "figure_paths": figure_paths,
    }


def analyze_datasets(cfg: ReachabilityAnalysisConfig) -> dict[str, Any]:
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir)

    if cfg.mode == "search":
        return run_search(cfg)
    if cfg.mode == "final":
        best_config_path = os.path.join(cfg.output_dir, "search", "best_config_per_dataset.csv")
        best_rows = _load_best_rows(best_config_path)
        search_paths = {
            "best_config_path": best_config_path,
            "full_search_table_path": os.path.join(cfg.output_dir, "search", "full_search_table.csv"),
            "figure_paths": {
                dataset_id: [
                    os.path.join(cfg.output_dir, "search", "figures", f"{dataset_slug(dataset_id)}_ik_heatmap_best_h.png"),
                    os.path.join(cfg.output_dir, "search", "figures", f"{dataset_slug(dataset_id)}_h_trace.png"),
                ]
                for dataset_id in cfg.datasets
            },
        }
        return run_final_evaluation(cfg, best_rows=best_rows, search_paths=search_paths)
    if cfg.mode == "full":
        search_paths = run_search(cfg)
        return run_final_evaluation(cfg, best_rows=search_paths["best_rows"], search_paths=search_paths)

    dataset_results = []
    all_summary_rows = []
    all_per_anchor_rows = []
    for dataset_id in cfg.datasets:
        result = analyze_single_dataset(dataset_id, cfg)
        dataset_results.append(result)
        all_summary_rows.extend(result["summary_rows"])
        all_per_anchor_rows.extend(result["per_anchor_rows"])
    return {
        "dataset_results": dataset_results,
        "all_summary_rows": all_summary_rows,
        "all_per_anchor_rows": all_per_anchor_rows,
    }
