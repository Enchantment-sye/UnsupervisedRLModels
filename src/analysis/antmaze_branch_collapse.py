from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from .antmaze_region_temporal_collapse import IK_GRID, METHOD_COLORS, METHOD_LABELS, METHOD_ORDER
from .fitted_baselines import sample_transition_pairs
from .proxy_ground_truth import compute_ground_truth_bundle
from .reachability_alignment import (
    ParsedDataset,
    ReachabilityAnalysisConfig,
    _sample_anchor_indices,
    load_or_sample_candidates,
)
from .similarity_metrics import (
    compute_adaptive_gaussian_scores,
    compute_euclidean_scores,
    compute_first_hit_temporal_distances,
    compute_gaussian_scores,
    compute_ik_scores,
    compute_mahalanobis_scores,
    compute_one_step_dynamics_scores,
    compute_replay_temporal_scores,
    safe_pearson,
    safe_spearman,
)


BRANCH_STATE_CACHE_VERSION = 2


@dataclass(frozen=True)
class AntMazeBranchConfig:
    dataset_id: str = "D4RL/antmaze/umaze-diverse-v1"
    slug: str = "d4rl_antmaze_umaze_diverse_v1"
    parse_file: str = "dataset_parse_d4rl_antmaze_umaze_diverse_v1.npz"
    base_match_radius: float = 0.08586683747003457
    fit_pool_size: int = 50000
    ik_ensemble_size: int = 100
    ik_batch_size: int = 4096
    ik_device: str = "cpu"


@dataclass(frozen=True)
class BranchContextSpec:
    stage_name: str
    seed: int
    num_anchors: int
    num_candidates: int
    candidate_sampling: str
    match_radius_scale: float
    horizon: int


@dataclass
class BranchSearchContext:
    config: AntMazeBranchConfig
    spec: BranchContextSpec
    parsed: ParsedDataset
    match_radius: float
    anchor_indices: np.ndarray
    anchor_occurrence_lists: list[np.ndarray]
    candidate_indices: np.ndarray
    anchor_positions: np.ndarray
    candidate_positions: np.ndarray

    @property
    def context_key(self) -> dict[str, Any]:
        return {
            "dataset": self.config.dataset_id,
            "stage_name": self.spec.stage_name,
            "seed": int(self.spec.seed),
            "num_anchors": int(self.spec.num_anchors),
            "num_candidates": int(self.spec.num_candidates),
            "candidate_sampling": str(self.spec.candidate_sampling),
            "match_radius": float(self.match_radius),
            "horizon": int(self.spec.horizon),
        }


@dataclass
class BranchStateMetrics:
    reach_prob: np.ndarray
    oracle_temporal: np.ndarray
    replay_temporal: np.ndarray
    first_hit_distances: np.ndarray
    euclidean_scores: np.ndarray
    gaussian_scores: np.ndarray
    mahalanobis_scores: np.ndarray
    adaptive_gaussian_scores: np.ndarray
    one_step_dynamics_scores: np.ndarray


@dataclass
class PostEntryTrace:
    after_step_hits: list[np.ndarray]


@dataclass
class BranchCandidate:
    candidate_id: int
    anchor_row: int
    entrance_local_index: int
    region_type: str
    branch_label: int
    member_local_indices: np.ndarray
    center: np.ndarray
    commit_prob: float
    post_entry_mass: float
    entrance_distance: float
    entrance_first_hit: float
    entrance_replay: float
    entrance_oracle: float
    entrance_euclidean: float
    entrance_gaussian: float
    entrance_mahalanobis: float
    entrance_adaptive_gaussian: float
    entrance_one_step_dynamics: float
    hit_occurrence_count: int


@dataclass
class BranchTaskSpec:
    anchor_row: int
    reference_entrance_local_index: int
    shell_candidate_ids: np.ndarray
    positive_candidate_ids: np.ndarray
    negative_candidate_ids: np.ndarray
    decoy_candidate_ids: np.ndarray
    gt_gap: float
    min_positive_commit: float
    max_negative_commit: float
    mean_positive_mass: float
    mean_negative_mass: float
    entrance_distance: float
    entrance_first_hit: float
    entrance_replay: float
    entrance_oracle: float


@dataclass
class BranchTask:
    spec: BranchTaskSpec
    ik_key: tuple[int, float]
    method_accuracies: dict[str, float]
    method_spearman: dict[str, float]
    method_pearson: dict[str, float]

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


def load_cached_antmaze_parsed(
    source_cache_dir: str,
    cfg: AntMazeBranchConfig | None = None,
) -> ParsedDataset:
    config = cfg or AntMazeBranchConfig()
    payload = _load_npz(os.path.join(source_cache_dir, config.parse_file))
    state_full = np.asarray(payload["state_full"], dtype=np.float32) if "state_full" in payload else np.asarray(payload["positions"], dtype=np.float32)
    goal_xy = np.asarray(payload["goal_xy"], dtype=np.float32) if "goal_xy" in payload else np.asarray(payload["positions"], dtype=np.float32)
    return ParsedDataset(
        dataset_id=str(payload["dataset_id"].item()),
        state_full=state_full,
        goal_xy=goal_xy,
        episode_ids=np.asarray(payload["episode_ids"], dtype=np.int32),
        timesteps=np.asarray(payload["timesteps"], dtype=np.int32),
        episode_offsets=np.asarray(payload["episode_offsets"], dtype=np.int64),
        episode_lengths=np.asarray(payload["episode_lengths"], dtype=np.int32),
        total_episodes=int(payload["total_episodes"].item()),
        median_step_size=float(payload["median_step_size"].item()),
        p90_nearest_neighbor=float(payload["p90_nearest_neighbor"].item()),
    )


def prepare_branch_context(
    parsed: ParsedDataset,
    source_cache_dir: str,
    spec: BranchContextSpec,
    cfg: AntMazeBranchConfig | None = None,
) -> BranchSearchContext:
    config = cfg or AntMazeBranchConfig()
    candidate_sampling = "dedupe" if spec.candidate_sampling in {"quantized", "dedupe"} else "random"
    match_radius = float(config.base_match_radius * float(spec.match_radius_scale))
    ra_cfg = ReachabilityAnalysisConfig(
        datasets=[config.dataset_id],
        output_dir=source_cache_dir,
        cache_dir=source_cache_dir,
        seed=int(spec.seed),
        horizon=int(spec.horizon),
        num_anchors=int(spec.num_anchors),
        num_candidates=int(spec.num_candidates),
        match_radius=match_radius,
        candidate_sampling=candidate_sampling,
        max_anchor_occurrences=min(256, max(128, int(spec.num_anchors))),
        fit_pool_size=int(config.fit_pool_size),
        ik_ensemble_size=int(config.ik_ensemble_size),
        ik_batch_size=int(config.ik_batch_size),
        ik_device=str(config.ik_device),
    )
    candidate_indices = load_or_sample_candidates(parsed, ra_cfg, match_radius)
    anchor_indices, anchor_occurrence_lists = _sample_anchor_indices(parsed, ra_cfg, match_radius)
    return BranchSearchContext(
        config=config,
        spec=spec,
        parsed=parsed,
        match_radius=match_radius,
        anchor_indices=np.asarray(anchor_indices, dtype=np.int64),
        anchor_occurrence_lists=[np.asarray(row, dtype=np.int64) for row in anchor_occurrence_lists],
        candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
        anchor_positions=np.asarray(parsed.positions[anchor_indices], dtype=np.float32),
        candidate_positions=np.asarray(parsed.positions[candidate_indices], dtype=np.float32),
    )


def _state_metric_cache_path(cache_dir: str, context: BranchSearchContext) -> str:
    payload = {
        **context.context_key,
        "cache_version": BRANCH_STATE_CACHE_VERSION,
    }
    return os.path.join(
        cache_dir,
        f"antmaze_branch_state_{context.config.slug}_{_hash_payload(payload)}.npz",
    )


def compute_or_load_branch_state_metrics(
    context: BranchSearchContext,
    cache_dir: str,
) -> BranchStateMetrics:
    ensure_dir(cache_dir)
    cache_path = _state_metric_cache_path(cache_dir, context)
    if os.path.exists(cache_path):
        cached = _load_npz(cache_path)
        return BranchStateMetrics(
            reach_prob=np.asarray(cached["reach_prob"], dtype=np.float32),
            oracle_temporal=np.asarray(cached["oracle_temporal"], dtype=np.float32),
            replay_temporal=np.asarray(cached["replay_temporal"], dtype=np.float32),
            first_hit_distances=np.asarray(cached["first_hit_distances"], dtype=np.float32),
            euclidean_scores=np.asarray(cached["euclidean_scores"], dtype=np.float32),
            gaussian_scores=np.asarray(cached["gaussian_scores"], dtype=np.float32),
            mahalanobis_scores=np.asarray(cached["mahalanobis_scores"], dtype=np.float32),
            adaptive_gaussian_scores=np.asarray(cached["adaptive_gaussian_scores"], dtype=np.float32),
            one_step_dynamics_scores=np.asarray(cached["one_step_dynamics_scores"], dtype=np.float32),
        )

    gt_bundle = compute_ground_truth_bundle(
        anchor_occurrence_lists=context.anchor_occurrence_lists,
        candidate_positions=context.candidate_positions,
        positions=context.parsed.positions,
        episode_ids=context.parsed.episode_ids,
        timesteps=context.parsed.timesteps,
        episode_offsets=context.parsed.episode_offsets,
        episode_lengths=context.parsed.episode_lengths,
        horizon=int(context.spec.horizon),
        match_radius=context.match_radius,
    )
    replay_temporal = compute_replay_temporal_scores(
        anchor_occurrence_lists=context.anchor_occurrence_lists,
        candidate_positions=context.candidate_positions,
        positions=context.parsed.positions,
        episode_ids=context.parsed.episode_ids,
        timesteps=context.parsed.timesteps,
        episode_offsets=context.parsed.episode_offsets,
        episode_lengths=context.parsed.episode_lengths,
        match_radius=context.match_radius,
        temporal_window=int(context.spec.horizon),
    )
    first_hit = compute_first_hit_temporal_distances(
        anchor_occurrence_lists=context.anchor_occurrence_lists,
        candidate_positions=context.candidate_positions,
        positions=context.parsed.positions,
        episode_ids=context.parsed.episode_ids,
        timesteps=context.parsed.timesteps,
        episode_offsets=context.parsed.episode_offsets,
        episode_lengths=context.parsed.episode_lengths,
        match_radius=context.match_radius,
        temporal_window=int(context.spec.horizon),
    )
    euclidean_scores = compute_euclidean_scores(context.anchor_positions, context.candidate_positions)
    gaussian_scores = compute_gaussian_scores(
        anchor_positions=context.anchor_positions,
        candidate_positions=context.candidate_positions,
        sigma_mode="adaptive",
        sigma_value=None,
        fallback_sigma=max(context.match_radius, float(context.parsed.median_step_size), 1e-4),
    )
    exclude_indices = np.unique(np.concatenate([context.anchor_indices, context.candidate_indices])).astype(np.int64)
    fit_positions = _fit_pool_positions(
        context.parsed,
        context.spec.seed,
        context.config.fit_pool_size,
        exclude_indices=exclude_indices,
    )
    fit_states, fit_next_states = _fit_transition_pairs(
        context.parsed,
        context.spec.seed,
        context.config.fit_pool_size,
        exclude_indices=exclude_indices,
    )
    mahalanobis_scores = compute_mahalanobis_scores(
        fit_positions=fit_positions,
        anchor_positions=context.anchor_positions,
        candidate_positions=context.candidate_positions,
    )
    adaptive_gaussian_scores = compute_adaptive_gaussian_scores(
        fit_positions=fit_positions,
        anchor_positions=context.anchor_positions,
        candidate_positions=context.candidate_positions,
    )
    one_step_dynamics_scores = compute_one_step_dynamics_scores(
        fit_states=fit_states,
        fit_next_states=fit_next_states,
        anchor_positions=context.anchor_positions,
        candidate_positions=context.candidate_positions,
        seed=context.spec.seed,
    )
    np.savez_compressed(
        cache_path,
        reach_prob=gt_bundle.reach_prob,
        oracle_temporal=gt_bundle.oracle_temporal,
        replay_temporal=replay_temporal,
        first_hit_distances=first_hit,
        euclidean_scores=euclidean_scores,
        gaussian_scores=gaussian_scores,
        mahalanobis_scores=mahalanobis_scores,
        adaptive_gaussian_scores=adaptive_gaussian_scores,
        one_step_dynamics_scores=one_step_dynamics_scores,
    )
    return BranchStateMetrics(
        reach_prob=np.asarray(gt_bundle.reach_prob, dtype=np.float32),
        oracle_temporal=np.asarray(gt_bundle.oracle_temporal, dtype=np.float32),
        replay_temporal=np.asarray(replay_temporal, dtype=np.float32),
        first_hit_distances=np.asarray(first_hit, dtype=np.float32),
        euclidean_scores=np.asarray(euclidean_scores, dtype=np.float32),
        gaussian_scores=np.asarray(gaussian_scores, dtype=np.float32),
        mahalanobis_scores=np.asarray(mahalanobis_scores, dtype=np.float32),
        adaptive_gaussian_scores=np.asarray(adaptive_gaussian_scores, dtype=np.float32),
        one_step_dynamics_scores=np.asarray(one_step_dynamics_scores, dtype=np.float32),
    )


def _fit_pool_positions(
    parsed: ParsedDataset,
    seed: int,
    fit_pool_size: int,
    *,
    exclude_indices: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 123)
    available_indices = np.arange(parsed.positions.shape[0], dtype=np.int64)
    if exclude_indices is not None:
        excluded = np.zeros(parsed.positions.shape[0], dtype=bool)
        excluded[np.asarray(exclude_indices, dtype=np.int64)] = True
        available_indices = available_indices[~excluded]
    if available_indices.size == 0:
        available_indices = np.arange(parsed.positions.shape[0], dtype=np.int64)
    pool_size = min(int(fit_pool_size), int(available_indices.size))
    indices = rng.choice(available_indices, size=pool_size, replace=False)
    return np.asarray(parsed.positions[indices], dtype=np.float32)


def _fit_transition_pairs(
    parsed: ParsedDataset,
    seed: int,
    fit_pool_size: int,
    *,
    exclude_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fit_states, fit_next_states = sample_transition_pairs(
        parsed.positions,
        parsed.episode_ids,
        parsed.timesteps,
        parsed.episode_lengths,
        max_pairs=int(fit_pool_size),
        seed=int(seed) + 211,
        exclude_indices=exclude_indices,
    )
    if fit_states.shape[0] == 0:
        fallback = _fit_pool_positions(parsed, seed, fit_pool_size, exclude_indices=exclude_indices)
        return fallback, fallback.copy()
    return np.asarray(fit_states, dtype=np.float32), np.asarray(fit_next_states, dtype=np.float32)


def _ik_cache_path(
    cache_dir: str,
    context: BranchSearchContext,
    ik_key: tuple[int, float],
) -> str:
    payload = {
        **context.context_key,
        "ik_subsample_size": int(ik_key[0]),
        "ik_temperature": float(ik_key[1]),
        "ik_ensemble_size": int(context.config.ik_ensemble_size),
        "fit_pool_size": int(context.config.fit_pool_size),
    }
    return os.path.join(cache_dir, f"antmaze_branch_ik_{context.config.slug}_{_hash_payload(payload)}.npz")


def compute_or_load_ik_matrix(
    context: BranchSearchContext,
    cache_dir: str,
    ik_key: tuple[int, float],
) -> np.ndarray:
    ensure_dir(cache_dir)
    cache_path = _ik_cache_path(cache_dir, context, ik_key)
    if os.path.exists(cache_path):
        return np.asarray(_load_npz(cache_path)["ik"], dtype=np.float32)
    device = str(context.config.ik_device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ik_scores = compute_ik_scores(
        fit_positions=_fit_pool_positions(context.parsed, context.spec.seed, context.config.fit_pool_size),
        anchor_positions=context.anchor_positions,
        candidate_positions=context.candidate_positions,
        ensemble_size=int(context.config.ik_ensemble_size),
        subsample_size=int(ik_key[0]),
        temperature=float(ik_key[1]),
        device=device,
        batch_size=int(context.config.ik_batch_size),
    )
    np.savez_compressed(cache_path, ik=ik_scores)
    return np.asarray(ik_scores, dtype=np.float32)


def build_candidate_radius_graph(
    candidate_positions: np.ndarray,
    radius: float,
    max_neighbors: int = 24,
) -> list[np.ndarray]:
    tree = cKDTree(candidate_positions)
    neighbors: list[np.ndarray] = []
    for index, point in enumerate(candidate_positions):
        hits = np.asarray(tree.query_ball_point(point, r=radius), dtype=np.int64)
        hits = hits[hits != index]
        if hits.size == 0:
            neighbors.append(np.empty(0, dtype=np.int64))
            continue
        distances = np.linalg.norm(candidate_positions[hits] - point, axis=1)
        order = np.argsort(distances)
        neighbors.append(np.asarray(hits[order[:max_neighbors]], dtype=np.int64))
    return neighbors


def assign_two_clusters_farthest_pair(points: np.ndarray, weights: np.ndarray | None = None, max_iters: int = 8) -> np.ndarray:
    if points.shape[0] < 2:
        raise ValueError("Need at least two points to split clusters.")
    weights_array = np.ones(points.shape[0], dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    distances = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    centers = np.asarray([points[i], points[j]], dtype=np.float64)
    labels = np.zeros(points.shape[0], dtype=np.int64)
    for _ in range(max_iters):
        sq_dist = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(sq_dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in (0, 1):
            mask = labels == cluster_id
            if not np.any(mask):
                continue
            centers[cluster_id] = np.average(points[mask], axis=0, weights=weights_array[mask])
    return labels


def _bfs_parents(neighbors: list[np.ndarray], source: int, max_depth: int) -> tuple[np.ndarray, np.ndarray]:
    parents = np.full(len(neighbors), -1, dtype=np.int64)
    depths = np.full(len(neighbors), -1, dtype=np.int64)
    queue: deque[int] = deque([int(source)])
    parents[int(source)] = int(source)
    depths[int(source)] = 0
    while queue:
        node = int(queue.popleft())
        if depths[node] >= int(max_depth):
            continue
        for neighbor in neighbors[node].tolist():
            if depths[int(neighbor)] >= 0:
                continue
            parents[int(neighbor)] = node
            depths[int(neighbor)] = depths[node] + 1
            queue.append(int(neighbor))
    return parents, depths


def path_from_parents(parents: np.ndarray, source: int, target: int) -> np.ndarray:
    if target < 0 or target >= parents.shape[0] or parents[target] < 0:
        return np.empty(0, dtype=np.int64)
    path = [int(target)]
    node = int(target)
    while node != int(source):
        node = int(parents[node])
        if node < 0:
            return np.empty(0, dtype=np.int64)
        path.append(node)
    path.reverse()
    return np.asarray(path, dtype=np.int64)


def compute_branch_commit_and_mass(
    traces: list[PostEntryTrace],
    region_members: np.ndarray,
    num_candidates: int,
    denominator: int,
) -> tuple[float, float]:
    if denominator <= 0:
        return 0.0, 0.0
    mask = np.zeros(num_candidates, dtype=bool)
    mask[np.asarray(region_members, dtype=np.int64)] = True
    commit_sum = 0.0
    mass_sum = 0.0
    for trace in traces:
        if not trace.after_step_hits:
            continue
        hit_count = 0
        for step_hits in trace.after_step_hits:
            if step_hits.size == 0:
                continue
            if np.any(mask[step_hits]):
                hit_count += 1
        if hit_count >= 2:
            commit_sum += 1.0
        mass_sum += float(hit_count) / float(len(trace.after_step_hits))
    return float(commit_sum / float(denominator)), float(mass_sum / float(denominator))


def collect_post_entry_traces(
    context: BranchSearchContext,
    anchor_row: int,
    entrance_local_index: int,
    candidate_tree: cKDTree,
    entrance_tau_limit: int,
) -> list[PostEntryTrace]:
    entrance_pos = np.asarray(context.candidate_positions[int(entrance_local_index)], dtype=np.float64)
    traces: list[PostEntryTrace] = []
    for global_index in context.anchor_occurrence_lists[int(anchor_row)].tolist():
        episode_id = int(context.parsed.episode_ids[global_index])
        timestep = int(context.parsed.timesteps[global_index])
        episode_start = int(context.parsed.episode_offsets[episode_id])
        remaining_steps = int(context.parsed.episode_lengths[episode_id] - timestep - 1)
        max_tau = min(int(context.spec.horizon), remaining_steps)
        hit_tau: int | None = None
        after_hits: list[np.ndarray] = []
        for tau in range(1, max_tau + 1):
            future_position = context.parsed.positions[episode_start + timestep + tau]
            if hit_tau is None and np.linalg.norm(future_position - entrance_pos) <= context.match_radius:
                hit_tau = tau
                if hit_tau > int(entrance_tau_limit):
                    hit_tau = None
                    break
                continue
            if hit_tau is not None and tau > hit_tau:
                step_hits = np.asarray(candidate_tree.query_ball_point(future_position, r=context.match_radius), dtype=np.int64)
                after_hits.append(step_hits)
        if hit_tau is not None:
            traces.append(PostEntryTrace(after_step_hits=after_hits))
    return traces


def _tail_cluster_members(
    cluster_indices: np.ndarray,
    counts: np.ndarray,
    max_members: int,
) -> np.ndarray:
    ordered = np.asarray(cluster_indices[np.argsort(counts[cluster_indices])[::-1]], dtype=np.int64)
    return np.asarray(ordered[:max_members], dtype=np.int64)


def _corridor_members_from_cluster(
    entrance_local_index: int,
    cluster_indices: np.ndarray,
    counts: np.ndarray,
    parents: np.ndarray,
    depths: np.ndarray,
    max_members: int,
) -> np.ndarray:
    reachable = np.asarray([idx for idx in cluster_indices.tolist() if depths[int(idx)] >= 0], dtype=np.int64)
    if reachable.size == 0:
        return np.empty(0, dtype=np.int64)
    endpoint = int(reachable[np.argmax(depths[reachable])])
    path = path_from_parents(parents, int(entrance_local_index), endpoint)
    if path.size < 2:
        return np.empty(0, dtype=np.int64)
    support = _tail_cluster_members(reachable, counts, max_members=max_members)
    merged = []
    seen: set[int] = set()
    for index in path.tolist() + support.tolist():
        if int(index) in seen:
            continue
        seen.add(int(index))
        merged.append(int(index))
        if len(merged) >= max_members:
            break
    return np.asarray(merged, dtype=np.int64)


def build_branch_candidates_for_anchor(
    context: BranchSearchContext,
    state_metrics: BranchStateMetrics,
    neighbors: list[np.ndarray],
    candidate_tree: cKDTree,
    anchor_row: int,
    branch_width_multiplier: float,
    entrance_tau_limit: int,
    min_hit_occurrences: int = 8,
    top_future_candidates: int = 24,
    tail_region_size: int = 12,
    corridor_region_size: int = 16,
) -> list[BranchCandidate]:
    tau_row = np.asarray(state_metrics.first_hit_distances[int(anchor_row)], dtype=np.float64)
    replay_row = np.asarray(state_metrics.replay_temporal[int(anchor_row)], dtype=np.float64)
    oracle_row = np.asarray(state_metrics.oracle_temporal[int(anchor_row)], dtype=np.float64)
    euclidean_row = np.asarray(state_metrics.euclidean_scores[int(anchor_row)], dtype=np.float64)
    gaussian_row = np.asarray(state_metrics.gaussian_scores[int(anchor_row)], dtype=np.float64)
    mahalanobis_row = np.asarray(state_metrics.mahalanobis_scores[int(anchor_row)], dtype=np.float64)
    adaptive_gaussian_row = np.asarray(state_metrics.adaptive_gaussian_scores[int(anchor_row)], dtype=np.float64)
    one_step_dynamics_row = np.asarray(state_metrics.one_step_dynamics_scores[int(anchor_row)], dtype=np.float64)
    entrance_distance_row = -euclidean_row
    anchor_pos = np.asarray(context.anchor_positions[int(anchor_row)], dtype=np.float64)
    anchor_side_mask = np.linalg.norm(context.candidate_positions - anchor_pos, axis=1) <= (1.5 * context.match_radius)

    entrance_candidates = np.flatnonzero(
        np.isfinite(tau_row)
        & (tau_row >= 1.0)
        & (tau_row <= float(entrance_tau_limit))
        & (replay_row > 0.01)
        & (entrance_distance_row > 2.0 * context.match_radius)
        & (entrance_distance_row < max(1.25, 12.0 * context.match_radius))
    )
    if entrance_candidates.size == 0:
        return []
    if entrance_candidates.size > 32:
        keep = np.argsort(replay_row[entrance_candidates])[::-1][:32]
        entrance_candidates = np.asarray(entrance_candidates[keep], dtype=np.int64)

    branch_candidates: list[BranchCandidate] = []
    for entrance_local_index in entrance_candidates.tolist():
        traces = collect_post_entry_traces(
            context=context,
            anchor_row=int(anchor_row),
            entrance_local_index=int(entrance_local_index),
            candidate_tree=candidate_tree,
            entrance_tau_limit=int(entrance_tau_limit),
        )
        if len(traces) < int(min_hit_occurrences):
            continue

        counts = np.zeros(context.candidate_indices.shape[0], dtype=np.float64)
        entrance_shell = np.linalg.norm(
            context.candidate_positions - context.candidate_positions[int(entrance_local_index)],
            axis=1,
        ) <= context.match_radius
        for trace in traces:
            for step_hits in trace.after_step_hits:
                if step_hits.size == 0:
                    continue
                counts[step_hits] += 1.0
        counts[int(entrance_local_index)] = 0.0
        counts[anchor_side_mask] = 0.0
        counts[entrance_shell] = 0.0

        top_ids = np.argsort(counts)[::-1][:top_future_candidates]
        top_ids = np.asarray(top_ids[counts[top_ids] > 0], dtype=np.int64)
        if top_ids.size < 8:
            continue

        labels = assign_two_clusters_farthest_pair(
            context.candidate_positions[top_ids],
            weights=np.asarray(counts[top_ids], dtype=np.float64),
        )
        cluster_a = np.asarray(top_ids[labels == 0], dtype=np.int64)
        cluster_b = np.asarray(top_ids[labels == 1], dtype=np.int64)
        if cluster_a.size < 4 or cluster_b.size < 4:
            continue
        center_gap = np.linalg.norm(
            np.mean(context.candidate_positions[cluster_a], axis=0) - np.mean(context.candidate_positions[cluster_b], axis=0)
        )
        if center_gap < (2.0 * context.match_radius):
            continue

        parents, depths = _bfs_parents(
            neighbors=neighbors,
            source=int(entrance_local_index),
            max_depth=max(4, int(math.ceil(4.0 * branch_width_multiplier))),
        )

        for branch_label, cluster_indices in enumerate([cluster_a, cluster_b]):
            tail_members = _tail_cluster_members(cluster_indices, counts, max_members=tail_region_size)
            if tail_members.size >= 4:
                commit_prob, post_entry_mass = compute_branch_commit_and_mass(
                    traces=traces,
                    region_members=tail_members,
                    num_candidates=context.candidate_indices.shape[0],
                    denominator=len(context.anchor_occurrence_lists[int(anchor_row)]),
                )
                branch_candidates.append(
                    BranchCandidate(
                        candidate_id=len(branch_candidates),
                        anchor_row=int(anchor_row),
                        entrance_local_index=int(entrance_local_index),
                        region_type="trajectory_tail_region",
                        branch_label=int(branch_label),
                        member_local_indices=tail_members,
                        center=np.mean(context.candidate_positions[tail_members], axis=0, dtype=np.float64),
                        commit_prob=float(commit_prob),
                        post_entry_mass=float(post_entry_mass),
                        entrance_distance=float(entrance_distance_row[int(entrance_local_index)]),
                        entrance_first_hit=float(tau_row[int(entrance_local_index)]),
                        entrance_replay=float(replay_row[int(entrance_local_index)]),
                        entrance_oracle=float(oracle_row[int(entrance_local_index)]),
                        entrance_euclidean=float(euclidean_row[int(entrance_local_index)]),
                        entrance_gaussian=float(gaussian_row[int(entrance_local_index)]),
                        entrance_mahalanobis=float(mahalanobis_row[int(entrance_local_index)]),
                        entrance_adaptive_gaussian=float(adaptive_gaussian_row[int(entrance_local_index)]),
                        entrance_one_step_dynamics=float(one_step_dynamics_row[int(entrance_local_index)]),
                        hit_occurrence_count=int(len(traces)),
                    )
                )

            corridor_members = _corridor_members_from_cluster(
                entrance_local_index=int(entrance_local_index),
                cluster_indices=cluster_indices,
                counts=counts,
                parents=parents,
                depths=depths,
                max_members=corridor_region_size,
            )
            if corridor_members.size >= 4:
                commit_prob, post_entry_mass = compute_branch_commit_and_mass(
                    traces=traces,
                    region_members=corridor_members,
                    num_candidates=context.candidate_indices.shape[0],
                    denominator=len(context.anchor_occurrence_lists[int(anchor_row)]),
                )
                branch_candidates.append(
                    BranchCandidate(
                        candidate_id=len(branch_candidates),
                        anchor_row=int(anchor_row),
                        entrance_local_index=int(entrance_local_index),
                        region_type="corridor_segment",
                        branch_label=int(branch_label),
                        member_local_indices=corridor_members,
                        center=np.mean(context.candidate_positions[corridor_members], axis=0, dtype=np.float64),
                        commit_prob=float(commit_prob),
                        post_entry_mass=float(post_entry_mass),
                        entrance_distance=float(entrance_distance_row[int(entrance_local_index)]),
                        entrance_first_hit=float(tau_row[int(entrance_local_index)]),
                        entrance_replay=float(replay_row[int(entrance_local_index)]),
                        entrance_oracle=float(oracle_row[int(entrance_local_index)]),
                        entrance_euclidean=float(euclidean_row[int(entrance_local_index)]),
                        entrance_gaussian=float(gaussian_row[int(entrance_local_index)]),
                        entrance_mahalanobis=float(mahalanobis_row[int(entrance_local_index)]),
                        entrance_adaptive_gaussian=float(adaptive_gaussian_row[int(entrance_local_index)]),
                        entrance_one_step_dynamics=float(one_step_dynamics_row[int(entrance_local_index)]),
                        hit_occurrence_count=int(len(traces)),
                    )
                )

    return branch_candidates


def build_branch_candidates(
    context: BranchSearchContext,
    state_metrics: BranchStateMetrics,
    branch_width_multiplier: float,
    anchor_rows: np.ndarray | None = None,
) -> list[BranchCandidate]:
    rows = np.arange(context.anchor_indices.shape[0], dtype=np.int64) if anchor_rows is None else np.asarray(anchor_rows, dtype=np.int64)
    graph_radius = float(branch_width_multiplier) * float(context.match_radius)
    neighbors = build_candidate_radius_graph(context.candidate_positions, radius=graph_radius)
    candidate_tree = cKDTree(context.candidate_positions)
    entrance_tau_limit = max(2, min(8, int(context.spec.horizon // 4)))
    all_candidates: list[BranchCandidate] = []
    for anchor_row in rows.tolist():
        anchor_candidates = build_branch_candidates_for_anchor(
            context=context,
            state_metrics=state_metrics,
            neighbors=neighbors,
            candidate_tree=candidate_tree,
            anchor_row=int(anchor_row),
            branch_width_multiplier=float(branch_width_multiplier),
            entrance_tau_limit=entrance_tau_limit,
        )
        for candidate in anchor_candidates:
            candidate.candidate_id = len(all_candidates)
            all_candidates.append(candidate)
    return all_candidates


def _candidate_method_score(candidate: BranchCandidate, method: str) -> float:
    if method == "euclidean":
        return float(candidate.entrance_euclidean)
    if method == "gaussian":
        return float(candidate.entrance_gaussian)
    if method == "mahalanobis":
        return float(candidate.entrance_mahalanobis)
    if method == "adaptive_gaussian":
        return float(candidate.entrance_adaptive_gaussian)
    if method == "first_hit":
        tau = float(candidate.entrance_first_hit)
        return 1.0 / (1.0 + tau) if np.isfinite(tau) else 0.0
    if method == "one_step_dynamics":
        return float(candidate.entrance_one_step_dynamics)
    if method == "replay":
        return float(candidate.entrance_replay)
    if method == "oracle":
        return float(candidate.entrance_oracle)
    raise KeyError(method)


def _pair_accuracy_from_scores(scores: np.ndarray, positive_ids: np.ndarray, negative_ids: np.ndarray) -> float:
    if positive_ids.size == 0 or negative_ids.size == 0:
        return 0.0
    return float((scores[positive_ids][:, None] > scores[negative_ids][None, :]).mean())


def build_branch_task_specs(
    branch_candidates: list[BranchCandidate],
    dist_tol: float,
    epsilon: float,
    gt_gap_threshold: float,
    decoy_count: int = 8,
) -> list[BranchTaskSpec]:
    candidates_by_anchor: dict[int, list[BranchCandidate]] = defaultdict(list)
    for candidate in branch_candidates:
        candidates_by_anchor[int(candidate.anchor_row)].append(candidate)

    task_specs: list[BranchTaskSpec] = []
    for anchor_row, anchor_candidates in candidates_by_anchor.items():
        by_entrance: dict[int, list[BranchCandidate]] = defaultdict(list)
        for candidate in anchor_candidates:
            by_entrance[int(candidate.entrance_local_index)].append(candidate)
        for entrance_local_index, same_entrance_candidates in by_entrance.items():
            if len(same_entrance_candidates) < 4:
                continue
            ordered_same = sorted(
                same_entrance_candidates,
                key=lambda item: (item.commit_prob, item.post_entry_mass, item.region_type),
            )
            negatives = ordered_same[:2]
            positives = ordered_same[-2:]
            min_positive_commit = min(float(candidate.commit_prob) for candidate in positives)
            max_negative_commit = max(float(candidate.commit_prob) for candidate in negatives)
            gt_gap = float(min_positive_commit - max_negative_commit)
            if gt_gap < float(gt_gap_threshold):
                continue

            mean_positive_mass = float(np.mean([candidate.post_entry_mass for candidate in positives]))
            mean_negative_mass = float(np.mean([candidate.post_entry_mass for candidate in negatives]))
            if mean_positive_mass <= mean_negative_mass:
                continue

            reference = same_entrance_candidates[0]
            shell = [
                candidate
                for candidate in anchor_candidates
                if abs(candidate.entrance_distance - reference.entrance_distance) <= float(dist_tol)
                and abs(candidate.entrance_first_hit - reference.entrance_first_hit) <= 1.0
                and abs(candidate.entrance_replay - reference.entrance_replay) <= float(epsilon)
                and abs(candidate.entrance_oracle - reference.entrance_oracle) <= float(epsilon)
            ]
            shell_ids = {int(candidate.candidate_id) for candidate in shell}
            main_ids = {int(candidate.candidate_id) for candidate in positives + negatives}
            decoy_pool = [
                candidate
                for candidate in shell
                if int(candidate.candidate_id) not in main_ids
            ]
            if len(decoy_pool) < int(decoy_count):
                continue
            decoy_pool.sort(
                key=lambda item: (
                    abs(item.commit_prob - 0.5 * (min_positive_commit + max_negative_commit)),
                    abs(item.entrance_distance - reference.entrance_distance),
                    abs(item.entrance_replay - reference.entrance_replay),
                    abs(item.entrance_oracle - reference.entrance_oracle),
                )
            )
            decoys = decoy_pool[:decoy_count]
            shell_candidate_ids = np.asarray(
                [int(candidate.candidate_id) for candidate in positives + negatives + decoys],
                dtype=np.int64,
            )
            task_specs.append(
                BranchTaskSpec(
                    anchor_row=int(anchor_row),
                    reference_entrance_local_index=int(entrance_local_index),
                    shell_candidate_ids=shell_candidate_ids,
                    positive_candidate_ids=np.asarray([int(candidate.candidate_id) for candidate in positives], dtype=np.int64),
                    negative_candidate_ids=np.asarray([int(candidate.candidate_id) for candidate in negatives], dtype=np.int64),
                    decoy_candidate_ids=np.asarray([int(candidate.candidate_id) for candidate in decoys], dtype=np.int64),
                    gt_gap=float(gt_gap),
                    min_positive_commit=float(min_positive_commit),
                    max_negative_commit=float(max_negative_commit),
                    mean_positive_mass=float(mean_positive_mass),
                    mean_negative_mass=float(mean_negative_mass),
                    entrance_distance=float(reference.entrance_distance),
                    entrance_first_hit=float(reference.entrance_first_hit),
                    entrance_replay=float(reference.entrance_replay),
                    entrance_oracle=float(reference.entrance_oracle),
                )
            )
    return task_specs


def finalize_branch_tasks(
    task_specs: list[BranchTaskSpec],
    branch_candidates: list[BranchCandidate],
    candidate_ik_scores: np.ndarray,
    ik_key: tuple[int, float],
) -> list[BranchTask]:
    tasks: list[BranchTask] = []
    for spec in task_specs:
        method_accuracies: dict[str, float] = {}
        method_spearman: dict[str, float] = {}
        method_pearson: dict[str, float] = {}
        gt_values = np.asarray([branch_candidates[int(idx)].commit_prob for idx in spec.shell_candidate_ids.tolist()], dtype=np.float64)

        method_score_rows: dict[str, np.ndarray] = {
            "euclidean": np.asarray([branch_candidates[int(idx)].entrance_euclidean for idx in spec.shell_candidate_ids.tolist()], dtype=np.float64),
            "gaussian": np.asarray([branch_candidates[int(idx)].entrance_gaussian for idx in spec.shell_candidate_ids.tolist()], dtype=np.float64),
            "mahalanobis": np.asarray([branch_candidates[int(idx)].entrance_mahalanobis for idx in spec.shell_candidate_ids.tolist()], dtype=np.float64),
            "adaptive_gaussian": np.asarray([branch_candidates[int(idx)].entrance_adaptive_gaussian for idx in spec.shell_candidate_ids.tolist()], dtype=np.float64),
            "first_hit": np.asarray(
                [
                    1.0 / (1.0 + branch_candidates[int(idx)].entrance_first_hit)
                    if np.isfinite(branch_candidates[int(idx)].entrance_first_hit)
                    else 0.0
                    for idx in spec.shell_candidate_ids.tolist()
                ],
                dtype=np.float64,
            ),
            "one_step_dynamics": np.asarray([branch_candidates[int(idx)].entrance_one_step_dynamics for idx in spec.shell_candidate_ids.tolist()], dtype=np.float64),
            "replay": np.asarray([branch_candidates[int(idx)].entrance_replay for idx in spec.shell_candidate_ids.tolist()], dtype=np.float64),
            "oracle": np.asarray([branch_candidates[int(idx)].entrance_oracle for idx in spec.shell_candidate_ids.tolist()], dtype=np.float64),
            "ik": np.asarray([candidate_ik_scores[int(idx)] for idx in spec.shell_candidate_ids.tolist()], dtype=np.float64),
        }
        positive_lookup = {int(idx): offset for offset, idx in enumerate(spec.shell_candidate_ids.tolist())}
        positive_offsets = np.asarray([positive_lookup[int(idx)] for idx in spec.positive_candidate_ids.tolist()], dtype=np.int64)
        negative_offsets = np.asarray([positive_lookup[int(idx)] for idx in spec.negative_candidate_ids.tolist()], dtype=np.int64)
        for method in METHOD_ORDER:
            scores = method_score_rows["oracle" if method == "oracle" else method]
            method_accuracies[method] = _pair_accuracy_from_scores(scores, positive_offsets, negative_offsets)
            method_spearman[method] = safe_spearman(scores, gt_values)
            method_pearson[method] = safe_pearson(scores, gt_values)
        tasks.append(
            BranchTask(
                spec=spec,
                ik_key=ik_key,
                method_accuracies=method_accuracies,
                method_spearman=method_spearman,
                method_pearson=method_pearson,
            )
        )
    return tasks


def compute_candidate_ik_scores(
    branch_candidates: list[BranchCandidate],
    ik_matrix: np.ndarray,
) -> np.ndarray:
    if not branch_candidates:
        return np.empty(0, dtype=np.float32)
    scores = np.zeros(len(branch_candidates), dtype=np.float32)
    for candidate in branch_candidates:
        region_scores = np.asarray(ik_matrix[int(candidate.anchor_row), candidate.member_local_indices], dtype=np.float32)
        k = min(4, int(region_scores.size))
        scores[int(candidate.candidate_id)] = float(np.mean(np.sort(region_scores)[-k:]))
    return scores


def select_top_branch_tasks(tasks: list[BranchTask], top_k: int = 5) -> list[BranchTask]:
    ordered = sorted(
        tasks,
        key=lambda task: (
            task.strict_advantage,
            task.ik_accuracy,
            -task.best_nonik_accuracy,
            task.spec.gt_gap,
            task.spec.mean_positive_mass - task.spec.mean_negative_mass,
        ),
        reverse=True,
    )
    selected: list[BranchTask] = []
    used_anchors: set[int] = set()
    for task in ordered:
        if int(task.spec.anchor_row) in used_anchors:
            continue
        selected.append(task)
        used_anchors.add(int(task.spec.anchor_row))
        if len(selected) >= int(top_k):
            break
    return selected


def summarize_branch_tasks(tasks: list[BranchTask]) -> dict[str, float]:
    if not tasks:
        return {
            "task_count": 0.0,
            "mean_strict_advantage": -1.0,
            "strict_positive_count": 0.0,
            "mean_gt_gap": 0.0,
            "mean_ik_accuracy": 0.0,
            "mean_best_nonik_accuracy": 0.0,
            "mean_positive_mass_gap": 0.0,
        }
    return {
        "task_count": float(len(tasks)),
        "mean_strict_advantage": float(np.mean([task.strict_advantage for task in tasks])),
        "strict_positive_count": float(sum(1 for task in tasks if task.strict_advantage > 1e-9)),
        "mean_gt_gap": float(np.mean([task.spec.gt_gap for task in tasks])),
        "mean_ik_accuracy": float(np.mean([task.ik_accuracy for task in tasks])),
        "mean_best_nonik_accuracy": float(np.mean([task.best_nonik_accuracy for task in tasks])),
        "mean_positive_mass_gap": float(np.mean([task.spec.mean_positive_mass - task.spec.mean_negative_mass for task in tasks])),
    }


def branch_task_rows_for_csv(tasks: list[BranchTask]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, task in enumerate(tasks):
        row = {
            "task_id": int(task_id),
            "anchor_row": int(task.spec.anchor_row),
            "reference_entrance_local_index": int(task.spec.reference_entrance_local_index),
            "ik_subsample_size": int(task.ik_key[0]),
            "ik_temperature": float(task.ik_key[1]),
            "gt_gap": float(task.spec.gt_gap),
            "min_positive_commit": float(task.spec.min_positive_commit),
            "max_negative_commit": float(task.spec.max_negative_commit),
            "mean_positive_mass": float(task.spec.mean_positive_mass),
            "mean_negative_mass": float(task.spec.mean_negative_mass),
            "strict_advantage": float(task.strict_advantage),
            "ik_accuracy": float(task.ik_accuracy),
            "best_nonik_accuracy": float(task.best_nonik_accuracy),
            "entrance_distance": float(task.spec.entrance_distance),
            "entrance_first_hit": float(task.spec.entrance_first_hit),
            "entrance_replay": float(task.spec.entrance_replay),
            "entrance_oracle": float(task.spec.entrance_oracle),
            "num_shell_candidates": int(task.spec.shell_candidate_ids.size),
            "num_decoys": int(task.spec.decoy_candidate_ids.size),
        }
        for method in METHOD_ORDER:
            row[f"{method}_accuracy"] = float(task.method_accuracies[method])
            row[f"{method}_spearman"] = float(task.method_spearman[method])
        rows.append(row)
    return rows


def branch_member_rows_for_csv(
    branch_candidates: list[BranchCandidate],
    tasks: list[BranchTask],
    context: BranchSearchContext,
    candidate_ik_scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, task in enumerate(tasks):
        positive_ids = set(int(idx) for idx in task.spec.positive_candidate_ids.tolist())
        negative_ids = set(int(idx) for idx in task.spec.negative_candidate_ids.tolist())
        decoy_ids = set(int(idx) for idx in task.spec.decoy_candidate_ids.tolist())
        for candidate_id in task.spec.shell_candidate_ids.tolist():
            candidate = branch_candidates[int(candidate_id)]
            if int(candidate_id) in positive_ids:
                role = "hard_positive"
            elif int(candidate_id) in negative_ids:
                role = "hard_negative"
            elif int(candidate_id) in decoy_ids:
                role = "decoy"
            else:
                role = "shell_extra"
            for member_local_index in candidate.member_local_indices.tolist():
                global_index = int(context.candidate_indices[int(member_local_index)])
                rows.append(
                    {
                        "task_id": int(task_id),
                        "anchor_row": int(task.spec.anchor_row),
                        "anchor_global_index": int(context.anchor_indices[int(task.spec.anchor_row)]),
                        "branch_candidate_id": int(candidate_id),
                        "role": role,
                        "region_type": candidate.region_type,
                        "branch_label": int(candidate.branch_label),
                        "entrance_local_index": int(candidate.entrance_local_index),
                        "entrance_global_index": int(context.candidate_indices[int(candidate.entrance_local_index)]),
                        "entrance_x": float(context.candidate_positions[int(candidate.entrance_local_index), 0]),
                        "entrance_y": float(context.candidate_positions[int(candidate.entrance_local_index), 1]),
                        "candidate_center_x": float(candidate.center[0]),
                        "candidate_center_y": float(candidate.center[1]),
                        "commit_prob": float(candidate.commit_prob),
                        "post_entry_mass": float(candidate.post_entry_mass),
                        "entrance_replay": float(candidate.entrance_replay),
                        "entrance_oracle": float(candidate.entrance_oracle),
                        "entrance_first_hit": float(candidate.entrance_first_hit),
                        "entrance_distance": float(candidate.entrance_distance),
                        "ik_region_score": float(candidate_ik_scores[int(candidate_id)]),
                        "member_local_index": int(member_local_index),
                        "member_global_index": global_index,
                        "member_x": float(context.candidate_positions[int(member_local_index), 0]),
                        "member_y": float(context.candidate_positions[int(member_local_index), 1]),
                    }
                )
    return rows
