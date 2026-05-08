from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .fitted_baselines import sample_transition_pairs
from .proxy_ground_truth import compute_ground_truth_bundle, filter_occurrences_with_future
from .similarity_metrics import (
    compute_adaptive_gaussian_scores,
    compute_first_hit_temporal_distances,
    compute_gaussian_scores,
    compute_mahalanobis_scores,
    compute_one_step_dynamics_scores,
    compute_replay_temporal_scores,
    distances_to_scores,
    safe_pearson,
    safe_spearman,
)


IK_GRID: list[tuple[int, float]] = [
    (32, 0.004),
    (32, 0.008),
    (32, 0.02),
    (128, 0.004),
    (128, 0.008),
    (128, 0.02),
    (1024, 0.004),
    (1024, 0.008),
    (1024, 0.02),
    (2048, 0.004),
    (2048, 0.008),
    (2048, 0.02),
]

STATE_METRIC_CACHE_VERSION = 2


METHOD_ORDER = [
    "euclidean",
    "gaussian",
    "mahalanobis",
    "adaptive_gaussian",
    "first_hit",
    "one_step_dynamics",
    "replay",
    "oracle",
    "ik",
]
METHOD_LABELS = {
    "euclidean": "Euclidean",
    "gaussian": "GK-fixed",
    "mahalanobis": "Mahalanobis",
    "adaptive_gaussian": "GK-adaptive",
    "first_hit": "First-hit",
    "one_step_dynamics": "1-step Dyn",
    "replay": "Replay-temp",
    "oracle": "Oracle-temp",
    "ik": "IK",
}
METHOD_COLORS = {
    "euclidean": "#5B6C8F",
    "gaussian": "#7A90B6",
    "mahalanobis": "#5AA9A4",
    "adaptive_gaussian": "#4E8A59",
    "first_hit": "#C7724A",
    "one_step_dynamics": "#C9A227",
    "replay": "#BE8A3A",
    "oracle": "#B24E60",
    "ik": "#237A57",
}


@dataclass(frozen=True)
class AntMazeSourceConfig:
    dataset_id: str = "D4RL/antmaze/umaze-diverse-v1"
    slug: str = "d4rl_antmaze_umaze_diverse_v1"
    match_radius: float = 0.08586683747003457
    num_anchors: int = 256
    num_candidates: int = 2048
    parse_file: str = "dataset_parse_d4rl_antmaze_umaze_diverse_v1.npz"
    anchors_file: str = "anchors_d4rl_antmaze_umaze_diverse_v1_18b9926743d8.npz"
    candidates_file: str = "candidates_d4rl_antmaze_umaze_diverse_v1_6ef7cff2707d.npz"
    seed: int = 0
    ik_ensemble_size: int = 100
    fit_pool_size: int = 50000
    gk_sigma_mode: str = "adaptive"
    gk_sigma: float | None = None
    mahalanobis_covariance_estimator: str = "ledoitwolf"
    mahalanobis_implementation: str = "whitening"
    mahalanobis_eps: float = 1e-6
    adaptive_gaussian_k: int = 10
    adaptive_gaussian_eps: float = 1e-6
    dynamics_backend: str = "kmeans"
    dynamics_num_bins: int = 64
    dynamics_distance_metric: str = "jsd"
    dynamics_alpha: float = 1e-3
    dynamics_min_count: int = 5
    dynamics_eps: float = 1e-6


@dataclass
class AntMazeSourceBundle:
    config: AntMazeSourceConfig
    positions: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray
    episode_offsets: np.ndarray
    episode_lengths: np.ndarray
    median_step_size: float
    anchor_indices: np.ndarray
    anchor_occurrence_lists: list[np.ndarray]
    candidate_indices: np.ndarray
    anchor_positions: np.ndarray
    candidate_positions: np.ndarray


@dataclass
class RegionCollection:
    radius_multiplier: float
    radius: float
    region_members: list[np.ndarray]
    seed_local_indices: np.ndarray
    centers: np.ndarray
    member_counts: np.ndarray
    candidate_to_regions: list[tuple[int, ...]]


@dataclass
class StateMetricBundle:
    horizon: int
    reach_prob: np.ndarray
    oracle_temporal: np.ndarray
    replay_temporal: np.ndarray
    first_hit_distances: np.ndarray
    euclidean_scores: np.ndarray
    gaussian_scores: np.ndarray
    mahalanobis_scores: np.ndarray
    adaptive_gaussian_scores: np.ndarray
    one_step_dynamics_scores: np.ndarray
    ik_scores: dict[tuple[int, float], np.ndarray]


@dataclass
class RegionMetricBundle:
    horizon: int
    regions: RegionCollection
    center_distances: np.ndarray
    commit_prob: np.ndarray
    tail_mass: np.ndarray
    euclidean_scores: np.ndarray
    gaussian_scores: np.ndarray
    mahalanobis_scores: np.ndarray
    adaptive_gaussian_scores: np.ndarray
    first_hit_distances: np.ndarray
    one_step_dynamics_scores: np.ndarray
    replay_scores: np.ndarray
    oracle_scores: np.ndarray
    ik_scores: dict[tuple[int, float], np.ndarray]


@dataclass
class RegionTask:
    anchor_row: int
    reference_region_id: int
    band_region_ids: np.ndarray
    positive_region_ids: np.ndarray
    negative_region_ids: np.ndarray
    context_region_ids: np.ndarray
    ik_key: tuple[int, float]
    gt_variant: str
    gt_gap: float
    min_positive_gt: float
    max_negative_gt: float
    band_size: int
    method_accuracies: dict[str, float]

    @property
    def ik_accuracy(self) -> float:
        return float(self.method_accuracies["ik"])

    @property
    def best_nonik_accuracy(self) -> float:
        return float(max(self.method_accuracies[name] for name in METHOD_ORDER if name != "ik"))

    @property
    def strict_advantage(self) -> float:
        return float(self.ik_accuracy - self.best_nonik_accuracy)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:12]


def _load_npz(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_antmaze_source_bundle(
    source_cache_dir: str,
    config: AntMazeSourceConfig | None = None,
) -> AntMazeSourceBundle:
    cfg = config or AntMazeSourceConfig()
    parsed = _load_npz(os.path.join(source_cache_dir, cfg.parse_file))
    anchors = _load_npz(os.path.join(source_cache_dir, cfg.anchors_file))
    candidates = _load_npz(os.path.join(source_cache_dir, cfg.candidates_file))

    positions = np.asarray(parsed["positions"], dtype=np.float32)
    anchor_indices = np.asarray(anchors["anchor_indices"], dtype=np.int64)
    candidate_indices = np.asarray(candidates["candidate_indices"], dtype=np.int64)

    occurrence_lists = [np.asarray(row, dtype=np.int64) for row in anchors["anchor_occurrences"]]
    return AntMazeSourceBundle(
        config=cfg,
        positions=positions,
        episode_ids=np.asarray(parsed["episode_ids"], dtype=np.int32),
        timesteps=np.asarray(parsed["timesteps"], dtype=np.int32),
        episode_offsets=np.asarray(parsed["episode_offsets"], dtype=np.int64),
        episode_lengths=np.asarray(parsed["episode_lengths"], dtype=np.int32),
        median_step_size=float(parsed["median_step_size"]),
        anchor_indices=anchor_indices,
        anchor_occurrence_lists=occurrence_lists,
        candidate_indices=candidate_indices,
        anchor_positions=np.asarray(positions[anchor_indices], dtype=np.float32),
        candidate_positions=np.asarray(positions[candidate_indices], dtype=np.float32),
    )


def aggregate_max_by_regions(score_matrix: np.ndarray, region_members: list[np.ndarray]) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=np.float32)
    result = np.empty((scores.shape[0], len(region_members)), dtype=np.float32)
    for region_id, members in enumerate(region_members):
        result[:, region_id] = np.max(scores[:, members], axis=1)
    return result


def aggregate_min_by_regions(value_matrix: np.ndarray, region_members: list[np.ndarray]) -> np.ndarray:
    values = np.asarray(value_matrix, dtype=np.float32)
    result = np.empty((values.shape[0], len(region_members)), dtype=np.float32)
    for region_id, members in enumerate(region_members):
        result[:, region_id] = np.min(values[:, members], axis=1)
    return result


def aggregate_topk_mean_by_regions(score_matrix: np.ndarray, region_members: list[np.ndarray], top_k: int) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=np.float32)
    result = np.empty((scores.shape[0], len(region_members)), dtype=np.float32)
    for region_id, members in enumerate(region_members):
        region_scores = np.asarray(scores[:, members], dtype=np.float32)
        k = min(int(top_k), int(region_scores.shape[1]))
        if k <= 0:
            raise ValueError("top_k must be positive")
        partitioned = np.partition(region_scores, kth=region_scores.shape[1] - k, axis=1)
        result[:, region_id] = np.mean(partitioned[:, -k:], axis=1)
    return result


def compute_region_commit_and_tail_from_hits(
    occurrence_region_hits: list[list[np.ndarray]],
    num_regions: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not occurrence_region_hits:
        zeros = np.zeros(num_regions, dtype=np.float32)
        return zeros, zeros.copy()

    tail_half = int(math.ceil(float(horizon) / 2.0))
    commit_sum = np.zeros(num_regions, dtype=np.float64)
    tail_sum = np.zeros(num_regions, dtype=np.float64)

    for hit_steps in occurrence_region_hits:
        max_tau = len(hit_steps)
        if max_tau <= 0:
            continue
        tail_window = min(tail_half, max_tau)
        tail_start = max(0, max_tau - tail_window)

        any_hit = np.zeros(num_regions, dtype=bool)
        tail_counts = np.zeros(num_regions, dtype=np.int16)

        for tau_index, region_ids in enumerate(hit_steps):
            if region_ids.size == 0:
                continue
            any_hit[region_ids] = True
            if tau_index >= tail_start:
                tail_counts[region_ids] += 1

        commit_sum += np.asarray(any_hit & (tail_counts >= 2), dtype=np.float64)
        tail_sum += tail_counts.astype(np.float64) / float(max(tail_window, 1))

    denominator = float(len(occurrence_region_hits))
    return (commit_sum / denominator).astype(np.float32), (tail_sum / denominator).astype(np.float32)


def build_region_collection(
    candidate_positions: np.ndarray,
    match_radius: float,
    radius_multiplier: float,
    min_members: int = 6,
    max_members: int = 48,
) -> RegionCollection:
    radius = float(radius_multiplier) * float(match_radius)
    tree = cKDTree(candidate_positions)
    deduped: dict[tuple[int, ...], int] = {}
    region_members: list[np.ndarray] = []
    seed_local_indices: list[int] = []
    centers: list[np.ndarray] = []
    member_counts: list[int] = []

    for seed_local_index in range(candidate_positions.shape[0]):
        members = np.asarray(tree.query_ball_point(candidate_positions[seed_local_index], r=radius), dtype=np.int64)
        if members.size < min_members or members.size > max_members:
            continue
        members.sort()
        key = tuple(int(x) for x in members.tolist())
        if key in deduped:
            continue
        deduped[key] = len(region_members)
        region_members.append(members)
        seed_local_indices.append(int(seed_local_index))
        centers.append(np.mean(candidate_positions[members], axis=0, dtype=np.float64))
        member_counts.append(int(members.size))

    candidate_to_regions: list[list[int]] = [[] for _ in range(candidate_positions.shape[0])]
    for region_id, members in enumerate(region_members):
        for member in members.tolist():
            candidate_to_regions[int(member)].append(int(region_id))

    return RegionCollection(
        radius_multiplier=float(radius_multiplier),
        radius=float(radius),
        region_members=region_members,
        seed_local_indices=np.asarray(seed_local_indices, dtype=np.int64),
        centers=np.asarray(centers, dtype=np.float32),
        member_counts=np.asarray(member_counts, dtype=np.int32),
        candidate_to_regions=[tuple(region_ids) for region_ids in candidate_to_regions],
    )


def _ik_cache_path(source_cache_dir: str, cfg: AntMazeSourceConfig, subsample_size: int, temperature: float) -> str:
    payload = {
        "dataset": cfg.dataset_id,
        "seed": cfg.seed,
        "num_anchors": cfg.num_anchors,
        "num_candidates": cfg.num_candidates,
        "match_radius": cfg.match_radius,
        "ik_ensemble_size": cfg.ik_ensemble_size,
        "ik_subsample_size": int(subsample_size),
        "ik_temperature": float(temperature),
        "fit_pool_size": cfg.fit_pool_size,
    }
    return os.path.join(
        source_cache_dir,
        f"ik_scores_{cfg.slug}_{_hash_payload(payload)}.npz",
    )


def _state_cache_path(cache_dir: str, cfg: AntMazeSourceConfig, horizon: int) -> str:
    payload = {
        "cache_version": STATE_METRIC_CACHE_VERSION,
        "dataset": cfg.dataset_id,
        "num_anchors": cfg.num_anchors,
        "num_candidates": cfg.num_candidates,
        "match_radius": cfg.match_radius,
        "horizon": int(horizon),
    }
    return os.path.join(cache_dir, f"antmaze_region_state_scores_{cfg.slug}_{_hash_payload(payload)}.npz")


def _fit_positions_for_source(
    source: AntMazeSourceBundle,
    *,
    exclude_indices: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(int(source.config.seed) + 123)
    available_indices = np.arange(source.positions.shape[0], dtype=np.int64)
    if exclude_indices is not None:
        excluded = np.zeros(source.positions.shape[0], dtype=bool)
        excluded[np.asarray(exclude_indices, dtype=np.int64)] = True
        available_indices = available_indices[~excluded]
    if available_indices.size == 0:
        available_indices = np.arange(source.positions.shape[0], dtype=np.int64)
    pool_size = min(int(source.config.fit_pool_size), int(available_indices.size))
    indices = rng.choice(available_indices, size=pool_size, replace=False)
    return np.asarray(source.positions[indices], dtype=np.float32)


def _fit_transition_pairs_for_source(
    source: AntMazeSourceBundle,
    *,
    exclude_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fit_states, fit_next_states = sample_transition_pairs(
        source.positions,
        source.episode_ids,
        source.timesteps,
        source.episode_lengths,
        max_pairs=int(source.config.fit_pool_size),
        seed=int(source.config.seed) + 211,
        exclude_indices=exclude_indices,
    )
    if fit_states.shape[0] == 0:
        fallback = _fit_positions_for_source(source, exclude_indices=exclude_indices)
        return fallback, fallback.copy()
    return np.asarray(fit_states, dtype=np.float32), np.asarray(fit_next_states, dtype=np.float32)


def compute_or_load_state_metric_bundle(
    source: AntMazeSourceBundle,
    source_cache_dir: str,
    cache_dir: str,
    horizon: int,
) -> StateMetricBundle:
    ensure_dir(cache_dir)
    state_cache = _state_cache_path(cache_dir, source.config, horizon)
    if os.path.exists(state_cache):
        cached = _load_npz(state_cache)
        ik_scores = {
            (int(subsample), float(temp)): np.asarray(cached[f"ik_{subsample}_{str(temp).replace('.', 'p')}"], dtype=np.float32)
            for subsample, temp in IK_GRID
        }
        return StateMetricBundle(
            horizon=int(horizon),
            reach_prob=np.asarray(cached["reach_prob"], dtype=np.float32),
            oracle_temporal=np.asarray(cached["oracle_temporal"], dtype=np.float32),
            replay_temporal=np.asarray(cached["replay_temporal"], dtype=np.float32),
            first_hit_distances=np.asarray(cached["first_hit_distances"], dtype=np.float32),
            euclidean_scores=np.asarray(cached["euclidean_scores"], dtype=np.float32),
            gaussian_scores=np.asarray(cached["gaussian_scores"], dtype=np.float32),
            mahalanobis_scores=np.asarray(cached["mahalanobis_scores"], dtype=np.float32),
            adaptive_gaussian_scores=np.asarray(cached["adaptive_gaussian_scores"], dtype=np.float32),
            one_step_dynamics_scores=np.asarray(cached["one_step_dynamics_scores"], dtype=np.float32),
            ik_scores=ik_scores,
        )

    gt_bundle = compute_ground_truth_bundle(
        anchor_occurrence_lists=source.anchor_occurrence_lists,
        candidate_positions=source.candidate_positions,
        positions=source.positions,
        episode_ids=source.episode_ids,
        timesteps=source.timesteps,
        episode_offsets=source.episode_offsets,
        episode_lengths=source.episode_lengths,
        horizon=int(horizon),
        match_radius=source.config.match_radius,
    )
    replay_temporal = compute_replay_temporal_scores(
        anchor_occurrence_lists=source.anchor_occurrence_lists,
        candidate_positions=source.candidate_positions,
        positions=source.positions,
        episode_ids=source.episode_ids,
        timesteps=source.timesteps,
        episode_offsets=source.episode_offsets,
        episode_lengths=source.episode_lengths,
        match_radius=source.config.match_radius,
        temporal_window=int(horizon),
    )
    first_hit = compute_first_hit_temporal_distances(
        anchor_occurrence_lists=source.anchor_occurrence_lists,
        candidate_positions=source.candidate_positions,
        positions=source.positions,
        episode_ids=source.episode_ids,
        timesteps=source.timesteps,
        episode_offsets=source.episode_offsets,
        episode_lengths=source.episode_lengths,
        match_radius=source.config.match_radius,
        temporal_window=int(horizon),
    )
    euclidean = (-cdist(source.anchor_positions, source.candidate_positions, metric="euclidean")).astype(np.float32)
    gaussian = compute_gaussian_scores(
        anchor_positions=source.anchor_positions,
        candidate_positions=source.candidate_positions,
        sigma_mode=source.config.gk_sigma_mode,
        sigma_value=source.config.gk_sigma,
        fallback_sigma=max(source.config.match_radius, float(source.median_step_size), 1e-4),
    )
    exclude_indices = np.unique(np.concatenate([source.anchor_indices, source.candidate_indices])).astype(np.int64)
    fit_positions = _fit_positions_for_source(source, exclude_indices=exclude_indices)
    fit_states, fit_next_states = _fit_transition_pairs_for_source(source, exclude_indices=exclude_indices)
    mahalanobis = compute_mahalanobis_scores(
        fit_positions=fit_positions,
        anchor_positions=source.anchor_positions,
        candidate_positions=source.candidate_positions,
        covariance_estimator=source.config.mahalanobis_covariance_estimator,
        implementation=source.config.mahalanobis_implementation,
        eps=source.config.mahalanobis_eps,
    )
    adaptive_gaussian = compute_adaptive_gaussian_scores(
        fit_positions=fit_positions,
        anchor_positions=source.anchor_positions,
        candidate_positions=source.candidate_positions,
        k=source.config.adaptive_gaussian_k,
        eps=source.config.adaptive_gaussian_eps,
    )
    one_step_dynamics = compute_one_step_dynamics_scores(
        fit_states=fit_states,
        fit_next_states=fit_next_states,
        anchor_positions=source.anchor_positions,
        candidate_positions=source.candidate_positions,
        backend=source.config.dynamics_backend,
        num_bins=source.config.dynamics_num_bins,
        distance_metric=source.config.dynamics_distance_metric,
        alpha=source.config.dynamics_alpha,
        min_count=source.config.dynamics_min_count,
        seed=source.config.seed,
        eps=source.config.dynamics_eps,
    )

    ik_scores: dict[tuple[int, float], np.ndarray] = {}
    for subsample_size, temperature in IK_GRID:
        path = _ik_cache_path(source_cache_dir, source.config, subsample_size, temperature)
        ik_scores[(subsample_size, temperature)] = np.asarray(_load_npz(path)["ik"], dtype=np.float32)

    save_payload: dict[str, Any] = {
        "reach_prob": gt_bundle.reach_prob,
        "oracle_temporal": gt_bundle.oracle_temporal,
        "replay_temporal": replay_temporal,
        "first_hit_distances": first_hit,
        "euclidean_scores": euclidean,
        "gaussian_scores": gaussian,
        "mahalanobis_scores": mahalanobis,
        "adaptive_gaussian_scores": adaptive_gaussian,
        "one_step_dynamics_scores": one_step_dynamics,
    }
    for subsample_size, temperature in IK_GRID:
        key = f"ik_{subsample_size}_{str(temperature).replace('.', 'p')}"
        save_payload[key] = ik_scores[(subsample_size, temperature)]
    np.savez_compressed(state_cache, **save_payload)

    return StateMetricBundle(
        horizon=int(horizon),
        reach_prob=np.asarray(gt_bundle.reach_prob, dtype=np.float32),
        oracle_temporal=np.asarray(gt_bundle.oracle_temporal, dtype=np.float32),
        replay_temporal=np.asarray(replay_temporal, dtype=np.float32),
        first_hit_distances=np.asarray(first_hit, dtype=np.float32),
        euclidean_scores=np.asarray(euclidean, dtype=np.float32),
        gaussian_scores=np.asarray(gaussian, dtype=np.float32),
        mahalanobis_scores=np.asarray(mahalanobis, dtype=np.float32),
        adaptive_gaussian_scores=np.asarray(adaptive_gaussian, dtype=np.float32),
        one_step_dynamics_scores=np.asarray(one_step_dynamics, dtype=np.float32),
        ik_scores=ik_scores,
    )


def _region_gt_cache_path(cache_dir: str, cfg: AntMazeSourceConfig, radius_multiplier: float, horizon: int) -> str:
    payload = {
        "dataset": cfg.dataset_id,
        "radius_multiplier": float(radius_multiplier),
        "horizon": int(horizon),
        "match_radius": cfg.match_radius,
        "num_anchors": cfg.num_anchors,
        "num_candidates": cfg.num_candidates,
    }
    return os.path.join(cache_dir, f"antmaze_region_gt_{cfg.slug}_{_hash_payload(payload)}.npz")


def _regions_hit_for_candidates(hit_candidate_ids: list[int], candidate_to_regions: list[tuple[int, ...]]) -> np.ndarray:
    if not hit_candidate_ids:
        return np.empty(0, dtype=np.int64)
    region_ids: set[int] = set()
    for candidate_id in hit_candidate_ids:
        region_ids.update(candidate_to_regions[int(candidate_id)])
    if not region_ids:
        return np.empty(0, dtype=np.int64)
    return np.asarray(sorted(region_ids), dtype=np.int64)


def _compute_missing_region_ground_truths(
    source: AntMazeSourceBundle,
    regions_by_multiplier: dict[float, RegionCollection],
    missing_pairs: list[tuple[float, int]],
) -> dict[tuple[float, int], tuple[np.ndarray, np.ndarray]]:
    horizons = sorted({int(horizon) for _, horizon in missing_pairs})
    radius_keys = sorted({float(multiplier) for multiplier, _ in missing_pairs})
    tail_half = {horizon: int(math.ceil(float(horizon) / 2.0)) for horizon in horizons}
    max_horizon = max(horizons)
    candidate_tree = cKDTree(source.candidate_positions)

    results: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
    commit_accumulators: dict[tuple[float, int], np.ndarray] = {}
    tail_accumulators: dict[tuple[float, int], np.ndarray] = {}

    for multiplier in radius_keys:
        num_regions = len(regions_by_multiplier[multiplier].region_members)
        for horizon in horizons:
            if (multiplier, horizon) not in missing_pairs:
                continue
            commit_accumulators[(multiplier, horizon)] = np.zeros(
                (source.anchor_indices.shape[0], num_regions),
                dtype=np.float32,
            )
            tail_accumulators[(multiplier, horizon)] = np.zeros(
                (source.anchor_indices.shape[0], num_regions),
                dtype=np.float32,
            )

    for anchor_row, occurrence_indices in enumerate(source.anchor_occurrence_lists):
        valid_occurrences = filter_occurrences_with_future(
            occurrence_indices=np.asarray(occurrence_indices, dtype=np.int64),
            timesteps=source.timesteps,
            episode_lengths=source.episode_lengths,
            episode_ids=source.episode_ids,
        )
        if valid_occurrences.size == 0:
            continue

        occurrence_hits: dict[tuple[float, int], list[list[np.ndarray]]] = {
            key: [] for key in commit_accumulators
        }

        for global_index in valid_occurrences.tolist():
            episode_id = int(source.episode_ids[global_index])
            timestep = int(source.timesteps[global_index])
            episode_start = int(source.episode_offsets[episode_id])
            remaining_steps = int(source.episode_lengths[episode_id] - timestep - 1)
            horizon_limits = {horizon: min(int(horizon), remaining_steps) for horizon in horizons}
            max_tau = max(horizon_limits.values())
            if max_tau <= 0:
                continue

            per_pair_hits: dict[tuple[float, int], list[np.ndarray]] = {
                (multiplier, horizon): []
                for multiplier in radius_keys
                for horizon in horizons
                if (multiplier, horizon) in commit_accumulators
            }

            for tau in range(1, max_tau + 1):
                future_global_index = episode_start + timestep + tau
                future_position = source.positions[future_global_index]
                hit_candidate_ids = candidate_tree.query_ball_point(future_position, r=source.config.match_radius)
                if hit_candidate_ids:
                    region_hits_by_multiplier = {
                        multiplier: _regions_hit_for_candidates(
                            hit_candidate_ids=hit_candidate_ids,
                            candidate_to_regions=regions_by_multiplier[multiplier].candidate_to_regions,
                        )
                        for multiplier in radius_keys
                    }
                else:
                    region_hits_by_multiplier = {
                        multiplier: np.empty(0, dtype=np.int64) for multiplier in radius_keys
                    }

                for horizon, limit in horizon_limits.items():
                    if tau > limit:
                        continue
                    for multiplier in radius_keys:
                        key = (multiplier, horizon)
                        if key not in per_pair_hits:
                            continue
                        per_pair_hits[key].append(region_hits_by_multiplier[multiplier])

            for key, hit_steps in per_pair_hits.items():
                occurrence_hits[key].append(hit_steps)

        for key, hit_lists in occurrence_hits.items():
            if not hit_lists:
                continue
            num_regions = len(regions_by_multiplier[key[0]].region_members)
            commit_row, tail_row = compute_region_commit_and_tail_from_hits(
                occurrence_region_hits=hit_lists,
                num_regions=num_regions,
                horizon=int(key[1]),
            )
            commit_accumulators[key][anchor_row] = commit_row
            tail_accumulators[key][anchor_row] = tail_row

    for key in commit_accumulators:
        results[key] = (commit_accumulators[key], tail_accumulators[key])
    return results


def compute_or_load_region_ground_truths(
    source: AntMazeSourceBundle,
    cache_dir: str,
    regions_by_multiplier: dict[float, RegionCollection],
    horizons: list[int],
) -> dict[tuple[float, int], tuple[np.ndarray, np.ndarray]]:
    ensure_dir(cache_dir)
    loaded: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
    missing_pairs: list[tuple[float, int]] = []

    for multiplier, regions in regions_by_multiplier.items():
        for horizon in horizons:
            path = _region_gt_cache_path(cache_dir, source.config, multiplier, horizon)
            if os.path.exists(path):
                cached = _load_npz(path)
                loaded[(multiplier, horizon)] = (
                    np.asarray(cached["commit_prob"], dtype=np.float32),
                    np.asarray(cached["tail_mass"], dtype=np.float32),
                )
            else:
                missing_pairs.append((multiplier, int(horizon)))

    if missing_pairs:
        computed = _compute_missing_region_ground_truths(
            source=source,
            regions_by_multiplier=regions_by_multiplier,
            missing_pairs=missing_pairs,
        )
        for key, (commit_prob, tail_mass) in computed.items():
            path = _region_gt_cache_path(cache_dir, source.config, key[0], key[1])
            np.savez_compressed(path, commit_prob=commit_prob, tail_mass=tail_mass)
            loaded[key] = (commit_prob, tail_mass)

    return loaded


def build_region_metric_bundle(
    source: AntMazeSourceBundle,
    state_metrics: StateMetricBundle,
    regions: RegionCollection,
    commit_prob: np.ndarray,
    tail_mass: np.ndarray,
) -> RegionMetricBundle:
    center_distances = cdist(source.anchor_positions, regions.centers, metric="euclidean").astype(np.float32)
    euclidean_scores = aggregate_max_by_regions(state_metrics.euclidean_scores, regions.region_members)
    gaussian_scores = aggregate_max_by_regions(state_metrics.gaussian_scores, regions.region_members)
    mahalanobis_scores = aggregate_max_by_regions(state_metrics.mahalanobis_scores, regions.region_members)
    adaptive_gaussian_scores = aggregate_max_by_regions(state_metrics.adaptive_gaussian_scores, regions.region_members)
    first_hit_distances = aggregate_min_by_regions(state_metrics.first_hit_distances, regions.region_members)
    one_step_dynamics_scores = aggregate_max_by_regions(state_metrics.one_step_dynamics_scores, regions.region_members)
    replay_scores = aggregate_max_by_regions(state_metrics.replay_temporal, regions.region_members)
    oracle_scores = aggregate_max_by_regions(state_metrics.oracle_temporal, regions.region_members)
    ik_scores = {
        ik_key: aggregate_topk_mean_by_regions(state_metrics.ik_scores[ik_key], regions.region_members, top_k=4)
        for ik_key in IK_GRID
    }
    return RegionMetricBundle(
        horizon=int(state_metrics.horizon),
        regions=regions,
        center_distances=center_distances,
        commit_prob=np.asarray(commit_prob, dtype=np.float32),
        tail_mass=np.asarray(tail_mass, dtype=np.float32),
        euclidean_scores=euclidean_scores,
        gaussian_scores=gaussian_scores,
        mahalanobis_scores=mahalanobis_scores,
        adaptive_gaussian_scores=adaptive_gaussian_scores,
        first_hit_distances=first_hit_distances,
        one_step_dynamics_scores=one_step_dynamics_scores,
        replay_scores=replay_scores,
        oracle_scores=oracle_scores,
        ik_scores=ik_scores,
    )


def collect_temporal_bands(
    bundle: RegionMetricBundle,
    anchor_rows: np.ndarray,
    dist_tol: float,
    band_tol: float,
    min_band_size: int = 24,
    max_band_size: int = 48,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    results: dict[int, list[tuple[int, np.ndarray]]] = {}
    for anchor_row in anchor_rows.tolist():
        center = np.asarray(bundle.center_distances[anchor_row], dtype=np.float64)
        tau = np.asarray(bundle.first_hit_distances[anchor_row], dtype=np.float64)
        replay = np.asarray(bundle.replay_scores[anchor_row], dtype=np.float64)
        oracle = np.asarray(bundle.oracle_scores[anchor_row], dtype=np.float64)
        finite = np.isfinite(tau)
        region_ids = np.flatnonzero(finite)
        if region_ids.size < min_band_size:
            results[int(anchor_row)] = []
            continue

        features = np.stack(
            [
                center[region_ids] / float(dist_tol),
                tau[region_ids] / 2.0,
                replay[region_ids] / float(band_tol),
                oracle[region_ids] / float(band_tol),
            ],
            axis=1,
        )
        tree = cKDTree(features)
        seen_bands: set[tuple[int, ...]] = set()
        anchor_bands: list[tuple[int, np.ndarray]] = []
        for local_offset, reference_region_id in enumerate(region_ids.tolist()):
            neighbors = tree.query_ball_point(features[local_offset], r=1.0, p=np.inf)
            band = np.asarray(region_ids[np.asarray(neighbors, dtype=np.int64)], dtype=np.int64)
            if band.size < min_band_size or band.size > max_band_size:
                continue
            band.sort()
            band_key = tuple(int(x) for x in band.tolist())
            if band_key in seen_bands:
                continue
            seen_bands.add(band_key)
            anchor_bands.append((int(reference_region_id), band))
        results[int(anchor_row)] = anchor_bands
    return results


def _middle_context_indices(sorted_band: np.ndarray, negative_count: int, positive_count: int, context_count: int) -> np.ndarray | None:
    middle = np.asarray(sorted_band[negative_count:-positive_count], dtype=np.int64)
    if middle.size < context_count:
        return None
    start = (middle.size - context_count) // 2
    return np.asarray(middle[start : start + context_count], dtype=np.int64)


def pair_accuracy(scores: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> float:
    if positives.size == 0 or negatives.size == 0:
        return 0.0
    return float((scores[positives][:, None] > scores[negatives][None, :]).mean())


def method_accuracy_map(
    bundle: RegionMetricBundle,
    anchor_row: int,
    positive_region_ids: np.ndarray,
    negative_region_ids: np.ndarray,
    ik_key: tuple[int, float],
) -> dict[str, float]:
    first_hit_scores = distances_to_scores(bundle.first_hit_distances[anchor_row])
    method_map = {
        "euclidean": np.asarray(bundle.euclidean_scores[anchor_row], dtype=np.float64),
        "gaussian": np.asarray(bundle.gaussian_scores[anchor_row], dtype=np.float64),
        "mahalanobis": np.asarray(bundle.mahalanobis_scores[anchor_row], dtype=np.float64),
        "adaptive_gaussian": np.asarray(bundle.adaptive_gaussian_scores[anchor_row], dtype=np.float64),
        "first_hit": np.asarray(first_hit_scores, dtype=np.float64),
        "one_step_dynamics": np.asarray(bundle.one_step_dynamics_scores[anchor_row], dtype=np.float64),
        "replay": np.asarray(bundle.replay_scores[anchor_row], dtype=np.float64),
        "oracle": np.asarray(bundle.oracle_scores[anchor_row], dtype=np.float64),
        "ik": np.asarray(bundle.ik_scores[ik_key][anchor_row], dtype=np.float64),
    }
    return {
        name: pair_accuracy(scores, positive_region_ids, negative_region_ids)
        for name, scores in method_map.items()
    }


def build_region_tasks(
    bundle: RegionMetricBundle,
    bands_by_anchor: dict[int, list[tuple[int, np.ndarray]]],
    anchor_rows: np.ndarray,
    gt_variant: str,
    ik_key: tuple[int, float],
    positive_count: int = 4,
    negative_count: int = 8,
    context_count: int = 12,
    gt_gap_threshold: float = 0.15,
) -> list[RegionTask]:
    if gt_variant not in {"commit_prob", "tail_mass"}:
        raise ValueError(f"Unsupported gt_variant: {gt_variant}")

    gt_matrix = bundle.commit_prob if gt_variant == "commit_prob" else bundle.tail_mass
    tasks: list[RegionTask] = []
    for anchor_row in anchor_rows.tolist():
        gt_row = np.asarray(gt_matrix[int(anchor_row)], dtype=np.float64)
        for reference_region_id, band_region_ids in bands_by_anchor.get(int(anchor_row), []):
            sorted_band = np.asarray(
                band_region_ids[np.argsort(gt_row[band_region_ids])],
                dtype=np.int64,
            )
            if sorted_band.size < positive_count + negative_count + context_count:
                continue
            positive_region_ids = np.asarray(sorted_band[-positive_count:], dtype=np.int64)
            negative_region_ids = np.asarray(sorted_band[:negative_count], dtype=np.int64)
            min_positive_gt = float(np.min(gt_row[positive_region_ids]))
            max_negative_gt = float(np.max(gt_row[negative_region_ids]))
            gt_gap = min_positive_gt - max_negative_gt
            if gt_gap < float(gt_gap_threshold):
                continue
            context_region_ids = _middle_context_indices(
                sorted_band=sorted_band,
                negative_count=negative_count,
                positive_count=positive_count,
                context_count=context_count,
            )
            if context_region_ids is None:
                continue
            method_accuracies = method_accuracy_map(
                bundle=bundle,
                anchor_row=int(anchor_row),
                positive_region_ids=positive_region_ids,
                negative_region_ids=negative_region_ids,
                ik_key=ik_key,
            )
            tasks.append(
                RegionTask(
                    anchor_row=int(anchor_row),
                    reference_region_id=int(reference_region_id),
                    band_region_ids=np.asarray(band_region_ids, dtype=np.int64),
                    positive_region_ids=positive_region_ids,
                    negative_region_ids=negative_region_ids,
                    context_region_ids=context_region_ids,
                    ik_key=ik_key,
                    gt_variant=gt_variant,
                    gt_gap=float(gt_gap),
                    min_positive_gt=min_positive_gt,
                    max_negative_gt=max_negative_gt,
                    band_size=int(band_region_ids.size),
                    method_accuracies=method_accuracies,
                )
            )
    return tasks


def select_non_overlapping_tasks(
    tasks: list[RegionTask],
    region_centers: np.ndarray,
    top_k: int = 5,
    min_center_separation: float = 0.0,
) -> list[RegionTask]:
    ordered = sorted(
        tasks,
        key=lambda task: (
            task.strict_advantage,
            task.ik_accuracy,
            -task.best_nonik_accuracy,
            task.gt_gap,
            -float(task.band_size),
        ),
        reverse=True,
    )
    selected: list[RegionTask] = []
    used_anchors: set[int] = set()
    used_regions: set[int] = set()

    for task in ordered:
        if task.anchor_row in used_anchors:
            continue
        if any(int(region_id) in used_regions for region_id in task.band_region_ids.tolist()):
            continue
        reference_center = np.asarray(region_centers[task.reference_region_id], dtype=np.float64)
        if min_center_separation > 0.0:
            too_close = False
            for selected_task in selected:
                selected_center = np.asarray(region_centers[selected_task.reference_region_id], dtype=np.float64)
                if np.linalg.norm(reference_center - selected_center) < float(min_center_separation):
                    too_close = True
                    break
            if too_close:
                continue
        selected.append(task)
        used_anchors.add(int(task.anchor_row))
        used_regions.update(int(region_id) for region_id in task.band_region_ids.tolist())
        if len(selected) >= int(top_k):
            break
    return selected


def summarize_selected_tasks(tasks: list[RegionTask]) -> dict[str, float]:
    if not tasks:
        return {
            "task_count": 0.0,
            "mean_strict_advantage": -1.0,
            "strict_positive_count": 0.0,
            "mean_gt_gap": 0.0,
            "mean_band_size": 0.0,
            "mean_ik_accuracy": 0.0,
            "mean_best_nonik_accuracy": 0.0,
        }
    return {
        "task_count": float(len(tasks)),
        "mean_strict_advantage": float(np.mean([task.strict_advantage for task in tasks])),
        "strict_positive_count": float(sum(1 for task in tasks if task.strict_advantage > 1e-9)),
        "mean_gt_gap": float(np.mean([task.gt_gap for task in tasks])),
        "mean_band_size": float(np.mean([task.band_size for task in tasks])),
        "mean_ik_accuracy": float(np.mean([task.ik_accuracy for task in tasks])),
        "mean_best_nonik_accuracy": float(np.mean([task.best_nonik_accuracy for task in tasks])),
    }


def evaluate_methods_on_tasks(
    bundle: RegionMetricBundle,
    tasks: list[RegionTask],
) -> list[dict[str, Any]]:
    gt_matrix = bundle.commit_prob if (tasks and tasks[0].gt_variant == "commit_prob") else bundle.tail_mass
    rows: list[dict[str, Any]] = []
    for task_id, task in enumerate(tasks):
        gt_row = np.asarray(gt_matrix[task.anchor_row, task.band_region_ids], dtype=np.float64)
        method_scores = {
            "euclidean": np.asarray(bundle.euclidean_scores[task.anchor_row, task.band_region_ids], dtype=np.float64),
            "gaussian": np.asarray(bundle.gaussian_scores[task.anchor_row, task.band_region_ids], dtype=np.float64),
            "mahalanobis": np.asarray(bundle.mahalanobis_scores[task.anchor_row, task.band_region_ids], dtype=np.float64),
            "adaptive_gaussian": np.asarray(bundle.adaptive_gaussian_scores[task.anchor_row, task.band_region_ids], dtype=np.float64),
            "first_hit": np.asarray(distances_to_scores(bundle.first_hit_distances[task.anchor_row])[task.band_region_ids], dtype=np.float64),
            "one_step_dynamics": np.asarray(bundle.one_step_dynamics_scores[task.anchor_row, task.band_region_ids], dtype=np.float64),
            "replay": np.asarray(bundle.replay_scores[task.anchor_row, task.band_region_ids], dtype=np.float64),
            "oracle": np.asarray(bundle.oracle_scores[task.anchor_row, task.band_region_ids], dtype=np.float64),
            "ik": np.asarray(bundle.ik_scores[task.ik_key][task.anchor_row, task.band_region_ids], dtype=np.float64),
        }
        band_lookup = {int(region_id): offset for offset, region_id in enumerate(task.band_region_ids.tolist())}
        positives = np.asarray([band_lookup[int(region_id)] for region_id in task.positive_region_ids.tolist()], dtype=np.int64)
        negatives = np.asarray([band_lookup[int(region_id)] for region_id in task.negative_region_ids.tolist()], dtype=np.int64)
        binary = np.zeros(task.band_region_ids.size, dtype=np.int64)
        binary[positives] = 1

        for method_name in METHOD_ORDER:
            scores = method_scores[method_name]
            rows.append(
                {
                    "task_id": int(task_id),
                    "anchor_row": int(task.anchor_row),
                    "reference_region_id": int(task.reference_region_id),
                    "gt_variant": task.gt_variant,
                    "ik_subsample_size": int(task.ik_key[0]),
                    "ik_temperature": float(task.ik_key[1]),
                    "method": method_name,
                    "hard_pair_accuracy": pair_accuracy(scores, positives, negatives),
                    "spearman": safe_spearman(scores, gt_row),
                    "pearson": safe_pearson(scores, gt_row),
                    "band_size": int(task.band_size),
                    "gt_gap": float(task.gt_gap),
                    "num_positives": int(task.positive_region_ids.size),
                    "num_negatives": int(task.negative_region_ids.size),
                }
            )
    return rows


def task_rows_for_csv(
    source: AntMazeSourceBundle,
    bundle: RegionMetricBundle,
    tasks: list[RegionTask],
) -> list[dict[str, Any]]:
    gt_matrix = bundle.commit_prob if (tasks and tasks[0].gt_variant == "commit_prob") else bundle.tail_mass
    rows: list[dict[str, Any]] = []
    for task_id, task in enumerate(tasks):
        role_lookup = {int(region_id): "context" for region_id in task.context_region_ids.tolist()}
        for region_id in task.positive_region_ids.tolist():
            role_lookup[int(region_id)] = "hard_positive"
        for region_id in task.negative_region_ids.tolist():
            role_lookup[int(region_id)] = "hard_negative"

        for region_id in task.band_region_ids.tolist():
            members = bundle.regions.region_members[int(region_id)]
            seed_local_index = int(bundle.regions.seed_local_indices[int(region_id)])
            for member_local_index in members.tolist():
                member_global_index = int(source.candidate_indices[int(member_local_index)])
                rows.append(
                    {
                        "task_id": int(task_id),
                        "anchor_row": int(task.anchor_row),
                        "anchor_global_index": int(source.anchor_indices[int(task.anchor_row)]),
                        "reference_region_id": int(task.reference_region_id),
                        "region_id": int(region_id),
                        "region_role": role_lookup.get(int(region_id), "band_extra"),
                        "seed_local_index": seed_local_index,
                        "seed_global_index": int(source.candidate_indices[seed_local_index]),
                        "region_center_x": float(bundle.regions.centers[int(region_id), 0]),
                        "region_center_y": float(bundle.regions.centers[int(region_id), 1]),
                        "member_count": int(bundle.regions.member_counts[int(region_id)]),
                        "member_local_index": int(member_local_index),
                        "member_global_index": member_global_index,
                        "member_x": float(source.positions[member_global_index, 0]),
                        "member_y": float(source.positions[member_global_index, 1]),
                        "center_distance": float(bundle.center_distances[int(task.anchor_row), int(region_id)]),
                        "commit_prob": float(bundle.commit_prob[int(task.anchor_row), int(region_id)]),
                        "tail_mass": float(bundle.tail_mass[int(task.anchor_row), int(region_id)]),
                        "euclidean_score": float(bundle.euclidean_scores[int(task.anchor_row), int(region_id)]),
                        "gaussian_score": float(bundle.gaussian_scores[int(task.anchor_row), int(region_id)]),
                        "mahalanobis_score": float(bundle.mahalanobis_scores[int(task.anchor_row), int(region_id)]),
                        "adaptive_gaussian_score": float(bundle.adaptive_gaussian_scores[int(task.anchor_row), int(region_id)]),
                        "first_hit_distance": float(bundle.first_hit_distances[int(task.anchor_row), int(region_id)]),
                        "one_step_dynamics_score": float(bundle.one_step_dynamics_scores[int(task.anchor_row), int(region_id)]),
                        "replay_score": float(bundle.replay_scores[int(task.anchor_row), int(region_id)]),
                        "oracle_score": float(bundle.oracle_scores[int(task.anchor_row), int(region_id)]),
                        "ik_score": float(bundle.ik_scores[task.ik_key][int(task.anchor_row), int(region_id)]),
                        "task_gt_value": float(gt_matrix[int(task.anchor_row), int(region_id)]),
                        "task_gt_gap": float(task.gt_gap),
                    }
                )
    return rows
