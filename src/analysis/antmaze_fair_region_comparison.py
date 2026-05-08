from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from .antmaze_branch_collapse import (
    IK_GRID,
    AntMazeBranchConfig,
    BranchContextSpec,
    BranchSearchContext,
    BranchStateMetrics,
    PostEntryTrace,
    _bfs_parents,
    _fit_pool_positions,
    _tail_cluster_members,
    assign_two_clusters_farthest_pair,
    build_candidate_radius_graph,
    collect_post_entry_traces,
    compute_branch_commit_and_mass,
    ensure_dir,
    load_cached_antmaze_parsed,
    path_from_parents,
    prepare_branch_context,
)
from .antmaze_region_temporal_collapse import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER
from .similarity_metrics import compute_ik_scores


FAIR_TOPK_VALUES = (2, 4, 8)
FAIR_HORIZONS = (40, 80, 120, 160)
FAIR_BRANCH_WIDTH_MULTIPLIERS = (1.5, 2.0, 2.5)
FAIR_ENTRY_HALO_SIZES = (2, 4)
FAIR_DEEP_CORE_SIZES = (8, 12, 16)
FAIR_DEFAULT_REGION_VARIANTS = (
    "trajectory_tail_region",
    "corridor_segment",
    "entry_halo_plus_tail",
    "entry_halo_plus_corridor",
)
STRICT_POSITIVE_REGION_VARIANTS = (
    "deep_tail_region",
    "deep_corridor_region",
    "halo_plus_deep_tail",
    "halo_plus_deep_corridor",
    "dual_core_union",
)


@dataclass(frozen=True)
class FairRegionSearchConfig:
    horizons: tuple[int, ...] = FAIR_HORIZONS
    branch_width_multipliers: tuple[float, ...] = FAIR_BRANCH_WIDTH_MULTIPLIERS
    entry_halo_sizes: tuple[int, ...] = FAIR_ENTRY_HALO_SIZES
    deep_core_sizes: tuple[int, ...] = FAIR_DEEP_CORE_SIZES
    topk_values: tuple[int, ...] = FAIR_TOPK_VALUES
    top_future_candidates: int = 40
    min_hit_occurrences: int = 8
    entrance_limit_count: int = 32
    gt_commit_gap_threshold: float = 0.05
    gt_tail_gap_threshold: float = 0.02
    ik_delta_threshold: float = 0.10
    strict_adv_threshold: float = 0.05
    anchor_side_multiplier: float = 1.5
    entrance_tau_min: int = 2
    entrance_tau_max: int = 8
    entrance_far_multiplier: float = 2.0
    entrance_near_core_multiplier: float = 1.25
    entrance_deep_core_multiplier: float = 1.75
    region_variants: tuple[str, ...] = FAIR_DEFAULT_REGION_VARIANTS


@dataclass
class FairEntrancePrototype:
    anchor_row: int
    entrance_local_index: int
    traces: list[PostEntryTrace]
    counts: np.ndarray
    cluster_members: dict[int, np.ndarray]
    parents: np.ndarray
    depths: np.ndarray
    entrance_distance: float
    entrance_first_hit: float
    entrance_replay: float
    entrance_oracle: float
    entrance_euclidean: float
    entrance_gaussian: float
    entrance_mahalanobis: float
    entrance_adaptive_gaussian: float
    entrance_one_step_dynamics: float

    @property
    def hit_occurrence_count(self) -> int:
        return int(len(self.traces))


@dataclass
class FairBranchRegionCandidate:
    anchor_row: int
    entrance_local_index: int
    branch_label: int
    region_variant: str
    entry_halo_size: int
    deep_core_size: int
    full_members: np.ndarray
    entry_halo_members: np.ndarray
    deep_core_members: np.ndarray
    full_center: np.ndarray
    deep_core_center: np.ndarray
    deep_commit_prob: float
    deep_tail_mass: float
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
class FairCounterexample:
    success: bool
    context: BranchSearchContext
    config: FairRegionSearchConfig
    branch_width_multiplier: float
    top_k: int
    ik_key: tuple[int, float]
    positive_region: FairBranchRegionCandidate
    negative_region: FairBranchRegionCandidate
    fair_method_deltas: dict[str, float]
    optimistic_method_deltas: dict[str, float]
    strict_advantage: float
    deep_commit_gap: float
    deep_tail_gap: float
    num_pairs_evaluated: int
    num_pairs_passing_gt: int
    total_prototypes: int
    total_regions_considered: int


@dataclass
class FairPrescreenCandidate:
    context: BranchSearchContext
    config: FairRegionSearchConfig
    branch_width_multiplier: float
    top_k: int
    positive_region: FairBranchRegionCandidate
    negative_region: FairBranchRegionCandidate
    fair_method_deltas: dict[str, float]
    optimistic_method_deltas: dict[str, float]
    deep_commit_gap: float
    deep_tail_gap: float


def split_anchor_rows(num_anchors: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_anchors)
    split_index = num_anchors // 2
    return (
        np.sort(permutation[:split_index]).astype(np.int64),
        np.sort(permutation[split_index:]).astype(np.int64),
    )


def first_hit_scores_from_distances(first_hit_row: np.ndarray) -> np.ndarray:
    tau = np.asarray(first_hit_row, dtype=np.float32)
    scores = np.zeros_like(tau, dtype=np.float32)
    finite = np.isfinite(tau)
    scores[finite] = 1.0 / (1.0 + tau[finite])
    return scores


def aggregate_region_score_topk(score_row: np.ndarray, members: np.ndarray, top_k: int) -> float:
    member_ids = np.asarray(members, dtype=np.int64)
    if member_ids.size == 0:
        raise ValueError("members must be non-empty")
    values = np.asarray(score_row[member_ids], dtype=np.float64)
    k = min(int(top_k), int(values.size))
    if k <= 0:
        raise ValueError("top_k must be positive")
    return float(np.mean(np.sort(values)[-k:]))


def aggregate_region_score_optimistic(score_row: np.ndarray, members: np.ndarray) -> float:
    member_ids = np.asarray(members, dtype=np.int64)
    if member_ids.size == 0:
        raise ValueError("members must be non-empty")
    return float(np.max(np.asarray(score_row[member_ids], dtype=np.float64)))


def compute_deep_branch_commit_and_tail_mass(
    traces: list[PostEntryTrace],
    deep_core_members: np.ndarray,
    num_candidates: int,
    denominator: int,
) -> tuple[float, float]:
    return compute_branch_commit_and_mass(
        traces=traces,
        region_members=np.asarray(deep_core_members, dtype=np.int64),
        num_candidates=int(num_candidates),
        denominator=int(denominator),
    )


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:12]


def _shared_ik_cache_path(
    cache_dir: str,
    context: BranchSearchContext,
    ik_key: tuple[int, float],
) -> str:
    payload = {
        "dataset": context.config.dataset_id,
        "seed": int(context.spec.seed),
        "num_anchors": int(context.spec.num_anchors),
        "num_candidates": int(context.spec.num_candidates),
        "candidate_sampling": str(context.spec.candidate_sampling),
        "match_radius": float(context.match_radius),
        "anchor_hash": hashlib.md5(np.asarray(context.anchor_indices, dtype=np.int64).tobytes()).hexdigest()[:12],
        "candidate_hash": hashlib.md5(np.asarray(context.candidate_indices, dtype=np.int64).tobytes()).hexdigest()[:12],
        "ik_subsample_size": int(ik_key[0]),
        "ik_temperature": float(ik_key[1]),
        "ik_ensemble_size": int(context.config.ik_ensemble_size),
        "fit_pool_size": int(context.config.fit_pool_size),
    }
    return os.path.join(cache_dir, f"antmaze_fair_ik_{context.config.slug}_{_hash_payload(payload)}.npz")


def compute_or_load_shared_ik_matrix(
    context: BranchSearchContext,
    cache_dir: str,
    ik_key: tuple[int, float],
) -> np.ndarray:
    ensure_dir(cache_dir)
    cache_path = _shared_ik_cache_path(cache_dir, context, ik_key)
    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=True) as data:
            return np.asarray(data["ik"], dtype=np.float32)
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


def load_current_cache_context(
    source_cache_dir: str,
    horizon: int,
    cfg: AntMazeBranchConfig | None = None,
) -> tuple[Any, BranchSearchContext]:
    config = cfg or AntMazeBranchConfig()
    parsed = load_cached_antmaze_parsed(source_cache_dir, config)
    context = prepare_branch_context(
        parsed=parsed,
        source_cache_dir=source_cache_dir,
        spec=BranchContextSpec(
            stage_name="current_cache",
            seed=0,
            num_anchors=256,
            num_candidates=2048,
            candidate_sampling="dedupe",
            match_radius_scale=1.0,
            horizon=int(horizon),
        ),
        cfg=config,
    )
    return parsed, context


def _unique_preserve_order(indices: list[int]) -> np.ndarray:
    seen: set[int] = set()
    ordered: list[int] = []
    for index in indices:
        value = int(index)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return np.asarray(ordered, dtype=np.int64)


def _entry_halo_from_cluster(
    candidate_positions: np.ndarray,
    entrance_local_index: int,
    cluster_indices: np.ndarray,
    entry_halo_size: int,
) -> np.ndarray:
    members = np.asarray(cluster_indices, dtype=np.int64)
    if members.size == 0:
        return np.empty(0, dtype=np.int64)
    entrance_pos = np.asarray(candidate_positions[int(entrance_local_index)], dtype=np.float64)
    distances = np.linalg.norm(candidate_positions[members] - entrance_pos, axis=1)
    order = np.argsort(distances)
    return np.asarray(members[order[: min(int(entry_halo_size), int(members.size))]], dtype=np.int64)


def _select_tail_core_members(
    candidate_positions: np.ndarray,
    entrance_local_index: int,
    cluster_indices: np.ndarray,
    counts: np.ndarray,
    match_radius: float,
    entry_halo_members: np.ndarray,
    deep_core_size: int,
    config: FairRegionSearchConfig,
) -> np.ndarray:
    members = np.asarray(cluster_indices, dtype=np.int64)
    if members.size == 0:
        return np.empty(0, dtype=np.int64)
    entrance_pos = np.asarray(candidate_positions[int(entrance_local_index)], dtype=np.float64)
    distances = np.linalg.norm(candidate_positions[members] - entrance_pos, axis=1)
    distance_lookup = {int(member): float(distance) for member, distance in zip(members.tolist(), distances.tolist(), strict=True)}
    halo_set = {int(index) for index in np.asarray(entry_halo_members, dtype=np.int64).tolist()}
    eligible = [
        int(member)
        for member, distance in zip(members.tolist(), distances.tolist(), strict=True)
        if int(member) not in halo_set and distance > float(config.entrance_deep_core_multiplier) * float(match_radius)
    ]
    if len(eligible) < int(deep_core_size):
        eligible = [
            int(member)
            for member, distance in zip(members.tolist(), distances.tolist(), strict=True)
            if int(member) not in halo_set and distance > float(match_radius)
        ]
    if len(eligible) < int(deep_core_size):
        return np.empty(0, dtype=np.int64)
    eligible_array = np.asarray(eligible, dtype=np.int64)
    eligible_distances = np.asarray([distance_lookup[int(member)] for member in eligible_array.tolist()], dtype=np.float64)
    order = np.lexsort((-np.asarray(counts[eligible_array], dtype=np.float64), -eligible_distances))
    ranked = eligible_array[order]
    return np.asarray(ranked[: int(deep_core_size)], dtype=np.int64)


def _select_corridor_core_members(
    candidate_positions: np.ndarray,
    entrance_local_index: int,
    cluster_indices: np.ndarray,
    counts: np.ndarray,
    parents: np.ndarray,
    depths: np.ndarray,
    match_radius: float,
    entry_halo_members: np.ndarray,
    deep_core_size: int,
    config: FairRegionSearchConfig,
) -> np.ndarray:
    cluster = np.asarray(cluster_indices, dtype=np.int64)
    reachable = np.asarray([idx for idx in cluster.tolist() if depths[int(idx)] >= 0], dtype=np.int64)
    if reachable.size == 0:
        return np.empty(0, dtype=np.int64)
    endpoint = int(reachable[np.argmax(depths[reachable])])
    path = path_from_parents(parents, int(entrance_local_index), endpoint)
    if path.size < 3:
        return np.empty(0, dtype=np.int64)

    entrance_pos = np.asarray(candidate_positions[int(entrance_local_index)], dtype=np.float64)
    halo_set = {int(index) for index in np.asarray(entry_halo_members, dtype=np.int64).tolist()}
    corridor_pool = _unique_preserve_order(path[1:].tolist() + _tail_cluster_members(reachable, counts, max_members=deep_core_size + 6).tolist())
    if corridor_pool.size == 0:
        return np.empty(0, dtype=np.int64)
    distances = np.linalg.norm(candidate_positions[corridor_pool] - entrance_pos, axis=1)
    eligible = np.asarray(
        [
            int(member)
            for member, distance in zip(corridor_pool.tolist(), distances.tolist(), strict=True)
            if int(member) not in halo_set and distance > float(config.entrance_deep_core_multiplier) * float(match_radius)
        ],
        dtype=np.int64,
    )
    if eligible.size < int(deep_core_size):
        eligible = np.asarray(
            [
                int(member)
                for member, distance in zip(corridor_pool.tolist(), distances.tolist(), strict=True)
                if int(member) not in halo_set and distance > float(match_radius)
            ],
            dtype=np.int64,
        )
    if eligible.size < int(deep_core_size):
        return np.empty(0, dtype=np.int64)
    eligible_depths = np.asarray(depths[eligible], dtype=np.float64)
    eligible_counts = np.asarray(counts[eligible], dtype=np.float64)
    eligible_distances = np.linalg.norm(candidate_positions[eligible] - entrance_pos, axis=1)
    order = np.lexsort((-eligible_distances, -eligible_counts, -eligible_depths))
    ranked = eligible[order]
    return np.asarray(ranked[: int(deep_core_size)], dtype=np.int64)


def _select_dual_core_union_members(
    tail_core_members: np.ndarray,
    corridor_core_members: np.ndarray,
    deep_core_size: int,
) -> np.ndarray:
    if int(deep_core_size) <= 0:
        return np.empty(0, dtype=np.int64)
    tail_members = np.asarray(tail_core_members, dtype=np.int64)
    corridor_members = np.asarray(corridor_core_members, dtype=np.int64)
    if tail_members.size == 0 or corridor_members.size == 0:
        return np.empty(0, dtype=np.int64)
    first_count = max(1, int(math.ceil(float(deep_core_size) / 2.0)))
    second_count = max(1, int(deep_core_size) - first_count)
    ordered = (
        tail_members[: min(first_count, int(tail_members.size))].tolist()
        + corridor_members[: min(second_count, int(corridor_members.size))].tolist()
        + tail_members[min(first_count, int(tail_members.size)) :].tolist()
        + corridor_members[min(second_count, int(corridor_members.size)) :].tolist()
    )
    merged = _unique_preserve_order([int(index) for index in ordered])
    if merged.size < int(deep_core_size):
        return np.empty(0, dtype=np.int64)
    return np.asarray(merged[: int(deep_core_size)], dtype=np.int64)


def build_entrance_prototypes_for_anchor(
    context: BranchSearchContext,
    state_metrics: BranchStateMetrics,
    neighbors: list[np.ndarray],
    candidate_tree: cKDTree,
    anchor_row: int,
    branch_width_multiplier: float,
    config: FairRegionSearchConfig | None = None,
) -> list[FairEntrancePrototype]:
    cfg = config or FairRegionSearchConfig()
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
    anchor_side_mask = np.linalg.norm(context.candidate_positions - anchor_pos, axis=1) <= (
        float(cfg.anchor_side_multiplier) * float(context.match_radius)
    )
    entrance_tau_limit = max(int(cfg.entrance_tau_min), min(int(cfg.entrance_tau_max), int(context.spec.horizon // 4)))
    entrance_candidates = np.flatnonzero(
        np.isfinite(tau_row)
        & (tau_row >= 1.0)
        & (tau_row <= float(entrance_tau_limit))
        & (replay_row > 0.01)
        & (entrance_distance_row > float(cfg.entrance_far_multiplier) * float(context.match_radius))
        & (entrance_distance_row < max(1.25, 12.0 * float(context.match_radius)))
    )
    if entrance_candidates.size == 0:
        return []
    if entrance_candidates.size > int(cfg.entrance_limit_count):
        keep = np.argsort(replay_row[entrance_candidates])[::-1][: int(cfg.entrance_limit_count)]
        entrance_candidates = np.asarray(entrance_candidates[keep], dtype=np.int64)

    prototypes: list[FairEntrancePrototype] = []
    for entrance_local_index in entrance_candidates.tolist():
        traces = collect_post_entry_traces(
            context=context,
            anchor_row=int(anchor_row),
            entrance_local_index=int(entrance_local_index),
            candidate_tree=candidate_tree,
            entrance_tau_limit=int(entrance_tau_limit),
        )
        if len(traces) < int(cfg.min_hit_occurrences):
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

        top_ids = np.argsort(counts)[::-1][: int(cfg.top_future_candidates)]
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
        if min(int(cluster_a.size), int(cluster_b.size)) < min(int(value) for value in cfg.deep_core_sizes):
            continue
        center_gap = np.linalg.norm(
            np.mean(context.candidate_positions[cluster_a], axis=0) - np.mean(context.candidate_positions[cluster_b], axis=0)
        )
        if center_gap < (2.0 * context.match_radius):
            continue
        parents, depths = _bfs_parents(
            neighbors=neighbors,
            source=int(entrance_local_index),
            max_depth=max(4, int(math.ceil(4.0 * float(branch_width_multiplier)))),
        )
        prototypes.append(
            FairEntrancePrototype(
                anchor_row=int(anchor_row),
                entrance_local_index=int(entrance_local_index),
                traces=traces,
                counts=np.asarray(counts, dtype=np.float32),
                cluster_members={
                    0: cluster_a,
                    1: cluster_b,
                },
                parents=parents,
                depths=depths,
                entrance_distance=float(entrance_distance_row[int(entrance_local_index)]),
                entrance_first_hit=float(tau_row[int(entrance_local_index)]),
                entrance_replay=float(replay_row[int(entrance_local_index)]),
                entrance_oracle=float(oracle_row[int(entrance_local_index)]),
                entrance_euclidean=float(euclidean_row[int(entrance_local_index)]),
                entrance_gaussian=float(gaussian_row[int(entrance_local_index)]),
                entrance_mahalanobis=float(mahalanobis_row[int(entrance_local_index)]),
                entrance_adaptive_gaussian=float(adaptive_gaussian_row[int(entrance_local_index)]),
                entrance_one_step_dynamics=float(one_step_dynamics_row[int(entrance_local_index)]),
            )
        )
    return prototypes


def build_fair_region_candidates_from_prototype(
    context: BranchSearchContext,
    prototype: FairEntrancePrototype,
    entry_halo_size: int,
    deep_core_size: int,
    config: FairRegionSearchConfig | None = None,
) -> list[FairBranchRegionCandidate]:
    cfg = config or FairRegionSearchConfig()
    regions: list[FairBranchRegionCandidate] = []
    denominator = len(context.anchor_occurrence_lists[int(prototype.anchor_row)])
    for branch_label in (0, 1):
        cluster_members = np.asarray(prototype.cluster_members.get(int(branch_label), np.empty(0, dtype=np.int64)), dtype=np.int64)
        if cluster_members.size == 0:
            continue
        entry_halo_members = _entry_halo_from_cluster(
            candidate_positions=context.candidate_positions,
            entrance_local_index=int(prototype.entrance_local_index),
            cluster_indices=cluster_members,
            entry_halo_size=int(entry_halo_size),
        )
        if entry_halo_members.size < int(entry_halo_size):
            continue

        tail_core_members = _select_tail_core_members(
            candidate_positions=context.candidate_positions,
            entrance_local_index=int(prototype.entrance_local_index),
            cluster_indices=cluster_members,
            counts=np.asarray(prototype.counts, dtype=np.float64),
            match_radius=float(context.match_radius),
            entry_halo_members=entry_halo_members,
            deep_core_size=int(deep_core_size),
            config=cfg,
        )
        corridor_core_members = _select_corridor_core_members(
            candidate_positions=context.candidate_positions,
            entrance_local_index=int(prototype.entrance_local_index),
            cluster_indices=cluster_members,
            counts=np.asarray(prototype.counts, dtype=np.float64),
            parents=prototype.parents,
            depths=prototype.depths,
            match_radius=float(context.match_radius),
            entry_halo_members=entry_halo_members,
            deep_core_size=int(deep_core_size),
            config=cfg,
        )
        dual_core_members = _select_dual_core_union_members(
            tail_core_members=tail_core_members,
            corridor_core_members=corridor_core_members,
            deep_core_size=int(deep_core_size),
        )

        region_variants = {
            "trajectory_tail_region": tail_core_members,
            "corridor_segment": corridor_core_members,
            "entry_halo_plus_tail": _unique_preserve_order(entry_halo_members.tolist() + tail_core_members.tolist()),
            "entry_halo_plus_corridor": _unique_preserve_order(entry_halo_members.tolist() + corridor_core_members.tolist()),
            "deep_tail_region": tail_core_members,
            "deep_corridor_region": corridor_core_members,
            "halo_plus_deep_tail": _unique_preserve_order(entry_halo_members.tolist() + tail_core_members.tolist()),
            "halo_plus_deep_corridor": _unique_preserve_order(entry_halo_members.tolist() + corridor_core_members.tolist()),
            "dual_core_union": dual_core_members,
        }
        deep_core_lookup = {
            "trajectory_tail_region": tail_core_members,
            "corridor_segment": corridor_core_members,
            "entry_halo_plus_tail": tail_core_members,
            "entry_halo_plus_corridor": corridor_core_members,
            "deep_tail_region": tail_core_members,
            "deep_corridor_region": corridor_core_members,
            "halo_plus_deep_tail": tail_core_members,
            "halo_plus_deep_corridor": corridor_core_members,
            "dual_core_union": dual_core_members,
        }

        for region_variant in tuple(cfg.region_variants):
            if region_variant not in region_variants:
                continue
            full_members = region_variants[region_variant]
            deep_core_members = np.asarray(deep_core_lookup[region_variant], dtype=np.int64)
            if deep_core_members.size < int(deep_core_size):
                continue
            if full_members.size == 0:
                continue
            deep_commit_prob, deep_tail_mass = compute_deep_branch_commit_and_tail_mass(
                traces=prototype.traces,
                deep_core_members=deep_core_members,
                num_candidates=context.candidate_indices.shape[0],
                denominator=denominator,
            )
            regions.append(
                FairBranchRegionCandidate(
                    anchor_row=int(prototype.anchor_row),
                    entrance_local_index=int(prototype.entrance_local_index),
                    branch_label=int(branch_label),
                    region_variant=region_variant,
                    entry_halo_size=int(entry_halo_size),
                    deep_core_size=int(deep_core_size),
                    full_members=np.asarray(full_members, dtype=np.int64),
                    entry_halo_members=np.asarray(entry_halo_members, dtype=np.int64),
                    deep_core_members=np.asarray(deep_core_members, dtype=np.int64),
                    full_center=np.mean(context.candidate_positions[np.asarray(full_members, dtype=np.int64)], axis=0, dtype=np.float64),
                    deep_core_center=np.mean(context.candidate_positions[np.asarray(deep_core_members, dtype=np.int64)], axis=0, dtype=np.float64),
                    deep_commit_prob=float(deep_commit_prob),
                    deep_tail_mass=float(deep_tail_mass),
                    entrance_distance=float(prototype.entrance_distance),
                    entrance_first_hit=float(prototype.entrance_first_hit),
                    entrance_replay=float(prototype.entrance_replay),
                    entrance_oracle=float(prototype.entrance_oracle),
                    entrance_euclidean=float(prototype.entrance_euclidean),
                    entrance_gaussian=float(prototype.entrance_gaussian),
                    entrance_mahalanobis=float(prototype.entrance_mahalanobis),
                    entrance_adaptive_gaussian=float(prototype.entrance_adaptive_gaussian),
                    entrance_one_step_dynamics=float(prototype.entrance_one_step_dynamics),
                    hit_occurrence_count=int(prototype.hit_occurrence_count),
                )
            )
    return regions


def _pair_method_rows(
    state_metrics: BranchStateMetrics,
    anchor_row: int,
) -> dict[str, np.ndarray]:
    return {
        "euclidean": np.asarray(state_metrics.euclidean_scores[int(anchor_row)], dtype=np.float64),
        "gaussian": np.asarray(state_metrics.gaussian_scores[int(anchor_row)], dtype=np.float64),
        "mahalanobis": np.asarray(state_metrics.mahalanobis_scores[int(anchor_row)], dtype=np.float64),
        "adaptive_gaussian": np.asarray(state_metrics.adaptive_gaussian_scores[int(anchor_row)], dtype=np.float64),
        "first_hit": np.asarray(first_hit_scores_from_distances(state_metrics.first_hit_distances[int(anchor_row)]), dtype=np.float64),
        "one_step_dynamics": np.asarray(state_metrics.one_step_dynamics_scores[int(anchor_row)], dtype=np.float64),
        "replay": np.asarray(state_metrics.replay_temporal[int(anchor_row)], dtype=np.float64),
        "oracle": np.asarray(state_metrics.oracle_temporal[int(anchor_row)], dtype=np.float64),
    }


def compute_pair_nonik_method_deltas(
    state_metrics: BranchStateMetrics,
    anchor_row: int,
    positive_region: FairBranchRegionCandidate,
    negative_region: FairBranchRegionCandidate,
    top_k: int,
) -> tuple[dict[str, float], dict[str, float]]:
    rows = _pair_method_rows(state_metrics, anchor_row)
    fair: dict[str, float] = {}
    optimistic: dict[str, float] = {}
    for method_name, score_row in rows.items():
        fair[method_name] = float(
            aggregate_region_score_topk(score_row, positive_region.full_members, top_k)
            - aggregate_region_score_topk(score_row, negative_region.full_members, top_k)
        )
        optimistic[method_name] = float(
            aggregate_region_score_optimistic(score_row, positive_region.full_members)
            - aggregate_region_score_optimistic(score_row, negative_region.full_members)
        )
    return fair, optimistic


def compute_pair_method_deltas(
    state_metrics: BranchStateMetrics,
    anchor_row: int,
    positive_region: FairBranchRegionCandidate,
    negative_region: FairBranchRegionCandidate,
    ik_matrix: np.ndarray,
    top_k: int,
) -> tuple[dict[str, float], dict[str, float]]:
    fair, optimistic = compute_pair_nonik_method_deltas(
        state_metrics=state_metrics,
        anchor_row=anchor_row,
        positive_region=positive_region,
        negative_region=negative_region,
        top_k=top_k,
    )
    ik_row = np.asarray(ik_matrix[int(anchor_row)], dtype=np.float64)
    fair["ik"] = float(
        aggregate_region_score_topk(ik_row, positive_region.full_members, top_k)
        - aggregate_region_score_topk(ik_row, negative_region.full_members, top_k)
    )
    optimistic["ik"] = fair["ik"]
    return fair, optimistic


def build_region_member_rows(example: FairCounterexample) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, region in [("positive", example.positive_region), ("negative", example.negative_region)]:
        halo_set = {int(index) for index in region.entry_halo_members.tolist()}
        core_set = {int(index) for index in region.deep_core_members.tolist()}
        full_set = {int(index) for index in region.full_members.tolist()}
        for member_local_index in region.full_members.tolist():
            label = "deep_core" if int(member_local_index) in core_set else "entry_halo" if int(member_local_index) in halo_set else "full_only"
            rows.append(
                {
                    "role": role,
                    "anchor_row": int(region.anchor_row),
                    "entrance_local_index": int(region.entrance_local_index),
                    "branch_label": int(region.branch_label),
                    "region_variant": region.region_variant,
                    "entry_halo_size": int(region.entry_halo_size),
                    "deep_core_size": int(region.deep_core_size),
                    "member_local_index": int(member_local_index),
                    "member_global_index": int(example.context.candidate_indices[int(member_local_index)]),
                    "member_x": float(example.context.candidate_positions[int(member_local_index), 0]),
                    "member_y": float(example.context.candidate_positions[int(member_local_index), 1]),
                    "member_label": label,
                    "in_entry_halo": int(int(member_local_index) in halo_set),
                    "in_deep_core": int(int(member_local_index) in core_set),
                    "in_full_region": int(int(member_local_index) in full_set),
                }
            )
    return rows
