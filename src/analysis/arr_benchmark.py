from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .reachability_alignment import (
    ReachabilityAnalysisConfig,
    _final_cfg_for_best_row,
    _hash_payload,
    _load_best_rows,
    _npz_exists,
    _save_npz,
    _safe_load_npz,
    compute_or_load_baseline_scores,
    compute_or_load_ik_score_matrix,
    dataset_slug,
    ensure_dir,
    load_or_parse_dataset,
    prepare_evaluation_context,
    save_csv,
)
from .similarity_metrics import (
    compute_first_hit_temporal_distances,
    distances_to_scores,
    ndcg_at_k,
    safe_pearson,
    safe_spearman,
)


ARR_METHOD_COLORS = {
    "euclidean": "#4C78A8",
    "gaussian": "#F58518",
    "mahalanobis": "#72B7B2",
    "adaptive_gaussian": "#54A24B",
    "first_hit_temporal_distance": "#E45756",
    "one_step_dynamics": "#EECA3B",
    "oracle_temporal_distance": "#B279A2",
    "replay_temporal": "#FF9DA6",
    "ik": "#7F7F7F",
}


@dataclass
class ARRBenchmarkConfig:
    datasets: list[str]
    output_dir: str
    cache_dir: str
    best_config_path: str
    seed: int = 0
    num_benchmark_anchors: int = 50
    candidate_pool_size: int = 64
    hard_positive_count: int = 4
    hard_negative_count: int = 12
    context_decoy_count: int = 48
    min_pairs_per_anchor: int = 2
    top_k: int = 10
    search_split_fraction: float = 0.5
    ik_ensemble_size: int = 100
    ik_subsample_grid: list[int] | None = None
    ik_temperature_grid: list[float] | None = None
    minari_datasets_path: str = "/home/shangyy/.minari/datasets"
    overwrite_cache: bool = False
    fit_pool_size: int = 50000
    ik_batch_size: int = 4096
    ik_device: str = "auto"
    first_hit_window: int | None = None
    scatter_points: int = 4000
    max_context_inf_fraction: float = 0.25
    write_report: bool = False


@dataclass
class ARRTask:
    dataset: str
    anchor_row: int
    anchor_global_index: int
    anchor_occurrence_count: int
    candidate_local_indices: np.ndarray
    hard_positive_local_indices: np.ndarray
    hard_negative_local_indices: np.ndarray
    context_local_indices: np.ndarray
    pair_positive_local_indices: np.ndarray
    pair_negative_local_indices: np.ndarray
    fallback_level: int
    mean_pair_gt_gap: float
    max_pair_gt_gap: float


def _dataset_dist_tolerance(dataset_id: str) -> float:
    if "pointmaze/umaze" in dataset_id.lower():
        return 0.005
    return 0.01


def _threshold_schedule(dataset_id: str) -> list[dict[str, float]]:
    base_dist = _dataset_dist_tolerance(dataset_id)
    is_antmaze = "antmaze" in dataset_id.lower()
    reach_gap = 0.15 if is_antmaze else 0.25
    first_hit_fallback = 1.0 if is_antmaze else 0.0
    schedule = [
        {"dist": base_dist, "first_hit": 0.0, "oracle_score": 0.02, "reach": reach_gap},
        {"dist": base_dist, "first_hit": first_hit_fallback, "oracle_score": 0.02, "reach": reach_gap},
        {"dist": base_dist * 1.5, "first_hit": max(first_hit_fallback, 1.0), "oracle_score": 0.03, "reach": reach_gap},
        {"dist": base_dist * 2.0, "first_hit": max(first_hit_fallback, 2.0), "oracle_score": 0.04, "reach": max(reach_gap - 0.05, 0.10)},
    ]
    if is_antmaze:
        schedule.append({"dist": base_dist * 2.5, "first_hit": 2.0, "oracle_score": 0.05, "reach": 0.12})
    return schedule


def oracle_score_to_distance(score_matrix: np.ndarray) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=np.float64)
    distances = np.full(scores.shape, np.inf, dtype=np.float64)
    mask = scores > 1e-12
    distances[mask] = (1.0 / scores[mask]) - 1.0
    return distances.astype(np.float32)


def _safe_task_recall(binary_labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    positives = np.flatnonzero(binary_labels > 0)
    if positives.size == 0:
        return 0.0
    top_k = min(int(k), int(binary_labels.size))
    pred_idx = np.argsort(scores)[::-1][:top_k]
    hits = len(set(int(x) for x in positives) & set(int(x) for x in pred_idx))
    return float(hits / float(positives.size))


def _pair_accuracy(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    if positive_scores.size == 0:
        return 0.0
    return float(np.mean((positive_scores > negative_scores).astype(np.float64)))


def _candidate_shell_order(
    local_indices: np.ndarray,
    distances: np.ndarray,
    replay_scores: np.ndarray,
    reference_indices: np.ndarray,
) -> np.ndarray:
    if local_indices.size == 0:
        return local_indices
    if reference_indices.size == 0:
        return local_indices
    center_dist = float(np.median(distances[reference_indices]))
    center_replay = float(np.median(replay_scores[reference_indices]))
    ranking = np.lexsort(
        (
            np.abs(replay_scores[local_indices] - center_replay),
            np.abs(distances[local_indices] - center_dist),
        )
    )
    return local_indices[ranking]


def _context_shell_rank(
    local_indices: np.ndarray,
    distances: np.ndarray,
    replay_scores: np.ndarray,
    first_hit_distances: np.ndarray,
    reference_indices: np.ndarray,
) -> np.ndarray:
    if local_indices.size == 0:
        return local_indices
    finite_ref = reference_indices[np.isfinite(first_hit_distances[reference_indices])]
    if finite_ref.size == 0:
        return _candidate_shell_order(local_indices, distances, replay_scores, reference_indices)

    ref_first_hit = first_hit_distances[finite_ref]
    ref_dist = float(np.median(distances[finite_ref]))
    ref_replay = float(np.median(replay_scores[finite_ref]))

    tau_diff = np.zeros(local_indices.shape[0], dtype=np.float64)
    finite_local = np.isfinite(first_hit_distances[local_indices])
    tau_diff[~finite_local] = np.inf
    if np.any(finite_local):
        local_tau = first_hit_distances[local_indices[finite_local]][:, None]
        tau_diff[finite_local] = np.min(np.abs(local_tau - ref_first_hit[None, :]), axis=1)

    shell_bucket = np.full(local_indices.shape[0], 3.0, dtype=np.float64)
    shell_bucket[np.isinf(tau_diff)] = 4.0
    shell_bucket[tau_diff <= 1.0] = 2.0
    shell_bucket[tau_diff == 0.0] = 1.0

    ranking = np.lexsort(
        (
            np.abs(replay_scores[local_indices] - ref_replay),
            np.abs(distances[local_indices] - ref_dist),
            tau_diff,
            shell_bucket,
        )
    )
    return local_indices[ranking]


def _select_context_decoys(
    remaining_indices: np.ndarray,
    first_hit_distances: np.ndarray,
    distances: np.ndarray,
    replay_scores: np.ndarray,
    reference_indices: np.ndarray,
    target_count: int,
    max_inf_fraction: float,
) -> np.ndarray:
    if target_count <= 0 or remaining_indices.size == 0:
        return np.zeros(0, dtype=np.int64)

    ordered = _context_shell_rank(
        remaining_indices,
        distances=distances,
        replay_scores=replay_scores,
        first_hit_distances=first_hit_distances,
        reference_indices=reference_indices,
    )
    inf_cap = int(np.floor(target_count * max_inf_fraction))
    finite_selected: list[int] = []
    inf_selected: list[int] = []

    for idx in ordered.tolist():
        is_inf = not np.isfinite(first_hit_distances[int(idx)])
        if is_inf:
            if len(inf_selected) >= inf_cap:
                continue
            inf_selected.append(int(idx))
        else:
            finite_selected.append(int(idx))
        if len(finite_selected) + len(inf_selected) >= target_count:
            break

    if len(finite_selected) + len(inf_selected) < target_count:
        for idx in ordered.tolist():
            idx = int(idx)
            if idx in finite_selected or idx in inf_selected:
                continue
            if not np.isfinite(first_hit_distances[idx]) and len(inf_selected) >= inf_cap:
                continue
            if np.isfinite(first_hit_distances[idx]):
                finite_selected.append(idx)
            else:
                inf_selected.append(idx)
            if len(finite_selected) + len(inf_selected) >= target_count:
                break

    return np.asarray((finite_selected + inf_selected)[:target_count], dtype=np.int64)


def _mine_pairs_for_anchor(
    reach_prob: np.ndarray,
    euclidean_distances: np.ndarray,
    first_hit_distances: np.ndarray,
    oracle_scores: np.ndarray,
    replay_scores: np.ndarray,
    ik_scores: np.ndarray,
    schedule: list[dict[str, float]],
    min_pairs_per_anchor: int,
) -> tuple[list[dict[str, Any]], int] | tuple[None, None]:
    finite_first_hit = np.isfinite(first_hit_distances)
    valid = finite_first_hit & np.isfinite(euclidean_distances)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        return None, None

    ordered = valid_indices[np.argsort(euclidean_distances[valid_indices])]
    for level, thresholds in enumerate(schedule):
        pair_rows: list[dict[str, Any]] = []
        for offset_i, i in enumerate(ordered[:-1]):
            for j in ordered[offset_i + 1 : offset_i + 41]:
                dist_gap = float(abs(euclidean_distances[j] - euclidean_distances[i]))
                if dist_gap > thresholds["dist"]:
                    break
                first_hit_gap = float(abs(first_hit_distances[j] - first_hit_distances[i]))
                if first_hit_gap > thresholds["first_hit"]:
                    continue
                oracle_gap = float(abs(oracle_scores[j] - oracle_scores[i]))
                if oracle_gap > thresholds["oracle_score"]:
                    continue
                if abs(float(reach_prob[j] - reach_prob[i])) < thresholds["reach"]:
                    continue

                positive = int(j if reach_prob[j] >= reach_prob[i] else i)
                negative = int(i if positive == j else j)
                ik_margin = float(ik_scores[positive] - ik_scores[negative])
                if ik_margin <= 0:
                    continue
                positive_first_hit_score = 0.0
                negative_first_hit_score = 0.0
                if np.isfinite(first_hit_distances[positive]):
                    positive_first_hit_score = 1.0 / (1.0 + float(first_hit_distances[positive]))
                if np.isfinite(first_hit_distances[negative]):
                    negative_first_hit_score = 1.0 / (1.0 + float(first_hit_distances[negative]))
                pair_rows.append(
                    {
                        "positive": positive,
                        "negative": negative,
                        "gt_gap": float(reach_prob[positive] - reach_prob[negative]),
                        "dist_gap": dist_gap,
                        "first_hit_gap": first_hit_gap,
                        "oracle_score_gap": oracle_gap,
                        "replay_gap": float(abs(replay_scores[positive] - replay_scores[negative])),
                        "ik_margin": ik_margin,
                        "first_hit_failure": float(positive_first_hit_score <= negative_first_hit_score),
                    }
                )

        if len(pair_rows) >= min_pairs_per_anchor:
            pair_rows.sort(
                key=lambda row: (
                    float(row["first_hit_failure"]),
                    float(row["gt_gap"]),
                    -float(row["replay_gap"]),
                    -float(row["dist_gap"]),
                    -float(row["first_hit_gap"]),
                    -float(row["oracle_score_gap"]),
                    float(row["ik_margin"]),
                ),
                reverse=True,
            )
            return pair_rows, level

    return None, None


def _build_task_from_pairs(
    dataset_id: str,
    anchor_row: int,
    anchor_index: int,
    occurrence_count: int,
    pair_rows: list[dict[str, Any]],
    fallback_level: int,
    reach_prob: np.ndarray,
    distances: np.ndarray,
    first_hit_distances: np.ndarray,
    replay_scores: np.ndarray,
    cfg: ARRBenchmarkConfig,
) -> ARRTask | None:
    selected_pairs: list[dict[str, Any]] = []
    selected_positive_set: set[int] = set()
    selected_negative_set: set[int] = set()

    def _consume_pairs(source_pairs: list[dict[str, Any]]) -> None:
        for pair in source_pairs:
            pos = int(pair["positive"])
            neg = int(pair["negative"])
            if pos in selected_positive_set or neg in selected_negative_set:
                continue
            selected_pairs.append(pair)
            selected_positive_set.add(pos)
            selected_negative_set.add(neg)
            if len(selected_positive_set) >= cfg.hard_positive_count:
                break

    failure_pairs = [pair for pair in pair_rows if float(pair.get("first_hit_failure", 0.0)) >= 1.0]
    non_failure_pairs = [pair for pair in pair_rows if float(pair.get("first_hit_failure", 0.0)) < 1.0]
    _consume_pairs(failure_pairs)
    if len(selected_positive_set) < cfg.hard_positive_count:
        _consume_pairs(non_failure_pairs)

    if len(selected_pairs) < cfg.min_pairs_per_anchor:
        return None

    additional_negatives: list[int] = []
    for pair in pair_rows:
        neg = int(pair["negative"])
        if neg in selected_negative_set or neg in selected_positive_set:
            continue
        additional_negatives.append(neg)
        selected_negative_set.add(neg)
        if len(selected_negative_set) >= cfg.hard_negative_count:
            break

    if len(selected_negative_set) < cfg.hard_negative_count:
        positive_indices = np.asarray(sorted(selected_positive_set), dtype=np.int64)
        remaining = np.setdiff1d(
            np.arange(reach_prob.shape[0], dtype=np.int64),
            np.asarray(sorted(selected_positive_set | selected_negative_set), dtype=np.int64),
            assume_unique=False,
        )
        remaining = remaining[reach_prob[remaining] <= max(float(np.median(reach_prob[positive_indices])) - 0.05, 0.0)]
        ordered = _candidate_shell_order(
            remaining,
            distances,
            replay_scores,
            positive_indices,
        )
        for idx in ordered:
            if int(idx) in selected_negative_set:
                continue
            selected_negative_set.add(int(idx))
            if len(selected_negative_set) >= cfg.hard_negative_count:
                break

    positive_local = np.asarray(sorted(selected_positive_set), dtype=np.int64)
    negative_local = np.asarray(sorted(selected_negative_set), dtype=np.int64)
    if positive_local.size < min(cfg.hard_positive_count, cfg.min_pairs_per_anchor):
        return None
    if negative_local.size < cfg.min_pairs_per_anchor:
        return None

    selected_set = set(int(x) for x in positive_local.tolist() + negative_local.tolist())
    remaining_all = np.setdiff1d(
        np.arange(reach_prob.shape[0], dtype=np.int64),
        np.asarray(sorted(selected_set), dtype=np.int64),
        assume_unique=False,
    )
    reference = np.concatenate([positive_local, negative_local], axis=0)
    context_local = _select_context_decoys(
        remaining_indices=remaining_all,
        first_hit_distances=first_hit_distances,
        distances=distances,
        replay_scores=replay_scores,
        reference_indices=reference,
        target_count=cfg.context_decoy_count,
        max_inf_fraction=cfg.max_context_inf_fraction,
    )

    candidate_local = np.concatenate([positive_local, negative_local, context_local], axis=0)
    if candidate_local.size > cfg.candidate_pool_size:
        candidate_local = candidate_local[: cfg.candidate_pool_size]
        keep_set = set(int(x) for x in candidate_local.tolist())
        positive_local = np.asarray([idx for idx in positive_local if int(idx) in keep_set], dtype=np.int64)
        negative_local = np.asarray([idx for idx in negative_local if int(idx) in keep_set], dtype=np.int64)
        context_local = np.asarray([idx for idx in context_local if int(idx) in keep_set], dtype=np.int64)

    pair_positive = []
    pair_negative = []
    keep_set = set(int(x) for x in candidate_local.tolist())
    for pair in selected_pairs:
        pos = int(pair["positive"])
        neg = int(pair["negative"])
        if pos in keep_set and neg in keep_set:
            pair_positive.append(pos)
            pair_negative.append(neg)

    if len(pair_positive) < cfg.min_pairs_per_anchor:
        return None

    pair_gt_gaps = [float(pair["gt_gap"]) for pair in selected_pairs]
    return ARRTask(
        dataset=dataset_id,
        anchor_row=int(anchor_row),
        anchor_global_index=int(anchor_index),
        anchor_occurrence_count=int(occurrence_count),
        candidate_local_indices=np.asarray(candidate_local, dtype=np.int64),
        hard_positive_local_indices=np.asarray(positive_local, dtype=np.int64),
        hard_negative_local_indices=np.asarray(negative_local, dtype=np.int64),
        context_local_indices=np.asarray(context_local, dtype=np.int64),
        pair_positive_local_indices=np.asarray(pair_positive, dtype=np.int64),
        pair_negative_local_indices=np.asarray(pair_negative, dtype=np.int64),
        fallback_level=int(fallback_level),
        mean_pair_gt_gap=float(np.mean(pair_gt_gaps)),
        max_pair_gt_gap=float(np.max(pair_gt_gaps)),
    )


def _task_cache_path(dataset_id: str, cfg: ARRBenchmarkConfig, base_row: dict[str, Any]) -> str:
    payload = {
        "dataset": dataset_id,
        "seed": cfg.seed,
        "candidate_pool_size": cfg.candidate_pool_size,
        "num_benchmark_anchors": cfg.num_benchmark_anchors,
        "hard_positive_count": cfg.hard_positive_count,
        "hard_negative_count": cfg.hard_negative_count,
        "context_decoy_count": cfg.context_decoy_count,
        "min_pairs_per_anchor": cfg.min_pairs_per_anchor,
        "horizon": int(base_row["horizon"]),
        "ik_subsample_size": int(base_row["ik_subsample_size"]),
        "ik_temperature": float(base_row["ik_temperature"]),
        "first_hit_window": int(cfg.first_hit_window or int(base_row["horizon"])),
        "max_context_inf_fraction": float(cfg.max_context_inf_fraction),
        "task_version": 6,
    }
    return os.path.join(cfg.cache_dir, f"arr_tasks_{dataset_slug(dataset_id)}_{_hash_payload(payload)}.npz")


def _serialize_tasks(tasks: list[ARRTask]) -> np.ndarray:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows.append(
            {
                "dataset": task.dataset,
                "anchor_row": task.anchor_row,
                "anchor_global_index": task.anchor_global_index,
                "anchor_occurrence_count": task.anchor_occurrence_count,
                "candidate_local_indices": task.candidate_local_indices,
                "hard_positive_local_indices": task.hard_positive_local_indices,
                "hard_negative_local_indices": task.hard_negative_local_indices,
                "context_local_indices": task.context_local_indices,
                "pair_positive_local_indices": task.pair_positive_local_indices,
                "pair_negative_local_indices": task.pair_negative_local_indices,
                "fallback_level": task.fallback_level,
                "mean_pair_gt_gap": task.mean_pair_gt_gap,
                "max_pair_gt_gap": task.max_pair_gt_gap,
            }
        )
    return np.asarray(rows, dtype=object)


def _deserialize_tasks(blob: np.ndarray) -> list[ARRTask]:
    tasks: list[ARRTask] = []
    for raw in blob.tolist():
        tasks.append(
            ARRTask(
                dataset=str(raw["dataset"]),
                anchor_row=int(raw["anchor_row"]),
                anchor_global_index=int(raw["anchor_global_index"]),
                anchor_occurrence_count=int(raw["anchor_occurrence_count"]),
                candidate_local_indices=np.asarray(raw["candidate_local_indices"], dtype=np.int64),
                hard_positive_local_indices=np.asarray(raw["hard_positive_local_indices"], dtype=np.int64),
                hard_negative_local_indices=np.asarray(raw["hard_negative_local_indices"], dtype=np.int64),
                context_local_indices=np.asarray(raw["context_local_indices"], dtype=np.int64),
                pair_positive_local_indices=np.asarray(raw["pair_positive_local_indices"], dtype=np.int64),
                pair_negative_local_indices=np.asarray(raw["pair_negative_local_indices"], dtype=np.int64),
                fallback_level=int(raw["fallback_level"]),
                mean_pair_gt_gap=float(raw["mean_pair_gt_gap"]),
                max_pair_gt_gap=float(raw["max_pair_gt_gap"]),
            )
        )
    return tasks


def compute_or_load_first_hit_distances(
    dataset_id: str,
    context: Any,
    cfg: ARRBenchmarkConfig,
    horizon: int,
) -> np.ndarray:
    payload = {
        "dataset": dataset_id,
        "seed": cfg.seed,
        "num_anchors": int(context.anchor_indices.shape[0]),
        "num_candidates": int(context.candidate_indices.shape[0]),
        "match_radius": float(context.match_radius),
        "temporal_window": int(cfg.first_hit_window or horizon),
    }
    path = os.path.join(cfg.cache_dir, f"arr_first_hit_{dataset_slug(dataset_id)}_{_hash_payload(payload)}.npz")
    if _npz_exists(path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(path)
        if cached is not None:
            return np.asarray(cached["first_hit_distances"], dtype=np.float32)

    first_hit = compute_first_hit_temporal_distances(
        anchor_occurrence_lists=context.anchor_occurrence_lists,
        candidate_positions=context.parsed.positions[context.candidate_indices],
        positions=context.parsed.positions,
        episode_ids=context.parsed.episode_ids,
        timesteps=context.parsed.timesteps,
        episode_offsets=context.parsed.episode_offsets,
        episode_lengths=context.parsed.episode_lengths,
        match_radius=context.match_radius,
        temporal_window=int(cfg.first_hit_window or horizon),
    )
    _save_npz(path, first_hit_distances=first_hit)
    return first_hit


def mine_arr_tasks_for_dataset(
    dataset_id: str,
    best_row: dict[str, Any],
    cfg: ARRBenchmarkConfig,
) -> dict[str, Any]:
    task_cache = _task_cache_path(dataset_id, cfg, best_row)
    formal_cfg = ReachabilityAnalysisConfig(
        datasets=[dataset_id],
        output_dir=cfg.output_dir,
        cache_dir=cfg.cache_dir,
        seed=cfg.seed,
        final_num_anchors=256,
        final_num_candidates=2048,
        final_top_k=50,
        final_max_anchor_occurrences=256,
        fit_pool_size=cfg.fit_pool_size,
        ik_ensemble_size=cfg.ik_ensemble_size,
        ik_batch_size=cfg.ik_batch_size,
        ik_device=cfg.ik_device,
        report_oracle_temp=True,
        minari_datasets_path=cfg.minari_datasets_path,
        overwrite_cache=cfg.overwrite_cache,
    )
    final_cfg = _final_cfg_for_best_row(formal_cfg, best_row)
    parsed = load_or_parse_dataset(
        dataset_id=dataset_id,
        cache_dir=cfg.cache_dir,
        overwrite_cache=cfg.overwrite_cache,
        minari_datasets_path=cfg.minari_datasets_path,
        seed=cfg.seed,
    )
    context = prepare_evaluation_context(parsed, final_cfg)
    baselines = compute_or_load_baseline_scores(context, final_cfg)
    reference_ik = compute_or_load_ik_score_matrix(
        context=context,
        cfg=final_cfg,
        subsample_size=int(best_row["ik_subsample_size"]),
        temperature=float(best_row["ik_temperature"]),
    )
    first_hit_distances = compute_or_load_first_hit_distances(dataset_id, context, cfg, int(best_row["horizon"]))
    oracle_distances = oracle_score_to_distance(context.ground_truth.oracle_temporal)

    if _npz_exists(task_cache) and not cfg.overwrite_cache:
        payload = _safe_load_npz(task_cache)
        tasks = _deserialize_tasks(np.asarray(payload["tasks"], dtype=object)) if payload is not None else []
    else:
        tasks: list[ARRTask] = []

    if not tasks:
        schedule = _threshold_schedule(dataset_id)
        for anchor_row in range(context.anchor_indices.shape[0]):
            reach_prob = np.asarray(context.ground_truth.reach_prob[anchor_row], dtype=np.float64)
            euclidean_distances = -np.asarray(baselines.euclidean[anchor_row], dtype=np.float64)
            first_hit_row = np.asarray(first_hit_distances[anchor_row], dtype=np.float64)
            oracle_score_row = np.asarray(context.ground_truth.oracle_temporal[anchor_row], dtype=np.float64)
            replay_row = np.asarray(baselines.replay_temporal[anchor_row], dtype=np.float64)
            ik_row = np.asarray(reference_ik[anchor_row], dtype=np.float64)
            self_mask = context.candidate_indices == context.anchor_indices[anchor_row]
            reach_prob[self_mask] = 0.0
            euclidean_distances[self_mask] = np.inf
            first_hit_row[self_mask] = np.inf
            oracle_score_row[self_mask] = 0.0

            pair_rows, fallback_level = _mine_pairs_for_anchor(
                reach_prob=reach_prob,
                euclidean_distances=euclidean_distances,
                first_hit_distances=first_hit_row,
                oracle_scores=oracle_score_row,
                replay_scores=replay_row,
                ik_scores=ik_row,
                schedule=schedule,
                min_pairs_per_anchor=cfg.min_pairs_per_anchor,
            )
            if pair_rows is None:
                continue
            task = _build_task_from_pairs(
                dataset_id=dataset_id,
                anchor_row=anchor_row,
                anchor_index=int(context.anchor_indices[anchor_row]),
                occurrence_count=int(context.ground_truth.occurrence_counts[anchor_row]),
                pair_rows=pair_rows,
                fallback_level=int(fallback_level),
                reach_prob=reach_prob,
                distances=euclidean_distances,
                first_hit_distances=first_hit_row,
                replay_scores=replay_row,
                cfg=cfg,
            )
            if task is not None:
                tasks.append(task)

        tasks.sort(key=lambda task: (task.mean_pair_gt_gap, task.max_pair_gt_gap), reverse=True)
        tasks = tasks[: cfg.num_benchmark_anchors]
        _save_npz(task_cache, tasks=_serialize_tasks(tasks))

    return {
        "dataset": dataset_id,
        "best_row": best_row,
        "parsed": parsed,
        "context": context,
        "baselines": baselines,
        "reference_ik": reference_ik,
        "first_hit_distances": first_hit_distances,
        "oracle_distances": oracle_distances,
        "tasks": tasks,
        "config": asdict(final_cfg),
    }


def _split_tasks(tasks: list[ARRTask], seed: int, fraction: float) -> tuple[list[ARRTask], list[ARRTask]]:
    if len(tasks) <= 1:
        return tasks, tasks
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tasks))
    split_index = max(1, min(len(tasks) - 1, int(round(len(tasks) * fraction))))
    search_ids = set(int(x) for x in order[:split_index])
    search_tasks = [task for idx, task in enumerate(tasks) if idx in search_ids]
    eval_tasks = [task for idx, task in enumerate(tasks) if idx not in search_ids]
    return search_tasks, eval_tasks


def _task_rows_for_csv(dataset_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    context = dataset_result["context"]
    baselines = dataset_result["baselines"]
    first_hit = dataset_result["first_hit_distances"]
    oracle_dist = dataset_result["oracle_distances"]
    ik = dataset_result["reference_ik"]
    for task_id, task in enumerate(dataset_result["tasks"]):
        pair_group = {}
        for group_id, (pos_idx, neg_idx) in enumerate(
            zip(task.pair_positive_local_indices.tolist(), task.pair_negative_local_indices.tolist())
        ):
            pair_group[int(pos_idx)] = group_id
            pair_group[int(neg_idx)] = group_id

        for local_idx in task.candidate_local_indices.tolist():
            role = "context"
            if int(local_idx) in set(task.hard_positive_local_indices.tolist()):
                role = "hard_positive"
            elif int(local_idx) in set(task.hard_negative_local_indices.tolist()):
                role = "hard_negative"
            global_idx = int(context.candidate_indices[int(local_idx)])
            rows.append(
                {
                    "dataset": task.dataset,
                    "task_id": int(task_id),
                    "anchor_row": int(task.anchor_row),
                    "anchor_global_index": int(task.anchor_global_index),
                    "anchor_occurrence_count": int(task.anchor_occurrence_count),
                    "candidate_local_index": int(local_idx),
                    "candidate_global_index": global_idx,
                    "candidate_x": float(context.parsed.positions[global_idx, 0]),
                    "candidate_y": float(context.parsed.positions[global_idx, 1]),
                    "role": role,
                    "pair_group": pair_group.get(int(local_idx), -1),
                    "reach_prob": float(dataset_result["context"].ground_truth.reach_prob[task.anchor_row, local_idx]),
                    "euclidean_distance": float(-baselines.euclidean[task.anchor_row, local_idx]),
                    "mahalanobis_distance": float(-baselines.mahalanobis[task.anchor_row, local_idx]),
                    "adaptive_gaussian": float(baselines.adaptive_gaussian[task.anchor_row, local_idx]),
                    "first_hit_temporal_distance": float(first_hit[task.anchor_row, local_idx]),
                    "one_step_dynamics_distance": float(-baselines.one_step_dynamics[task.anchor_row, local_idx]),
                    "oracle_temporal_distance": float(oracle_dist[task.anchor_row, local_idx]),
                    "oracle_temporal_score": float(dataset_result["context"].ground_truth.oracle_temporal[task.anchor_row, local_idx]),
                    "replay_temporal": float(baselines.replay_temporal[task.anchor_row, local_idx]),
                    "reference_ik": float(ik[task.anchor_row, local_idx]),
                    "fallback_level": int(task.fallback_level),
                    "mean_pair_gt_gap": float(task.mean_pair_gt_gap),
                }
            )
    return rows


def _evaluate_method_on_tasks(
    dataset_id: str,
    tasks: list[ARRTask],
    scores_matrix: np.ndarray,
    reach_matrix: np.ndarray,
    split_name: str,
    method_name: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_task_rows: list[dict[str, Any]] = []
    for task_id, task in enumerate(tasks):
        task_indices = task.candidate_local_indices
        gt = np.asarray(reach_matrix[task.anchor_row, task_indices], dtype=np.float64)
        scores = np.asarray(scores_matrix[task.anchor_row, task_indices], dtype=np.float64)
        positive_set = set(int(x) for x in task.hard_positive_local_indices.tolist())
        binary = np.asarray([1 if int(idx) in positive_set else 0 for idx in task_indices.tolist()], dtype=np.int64)

        task_local_lookup = {int(idx): offset for offset, idx in enumerate(task_indices.tolist())}
        pos_scores = np.asarray(
            [scores[task_local_lookup[int(idx)]] for idx in task.pair_positive_local_indices.tolist()],
            dtype=np.float64,
        )
        neg_scores = np.asarray(
            [scores[task_local_lookup[int(idx)]] for idx in task.pair_negative_local_indices.tolist()],
            dtype=np.float64,
        )
        row = {
            "dataset": dataset_id,
            "split": split_name,
            "method": method_name,
            "task_id": int(task_id),
            "anchor_row": int(task.anchor_row),
            "hard_pair_accuracy": _pair_accuracy(pos_scores, neg_scores),
            "group_recall_at_k": _safe_task_recall(binary, scores, top_k),
            "spearman": safe_spearman(scores, gt),
            "pearson": safe_pearson(scores, gt),
            "ndcg_at_k": ndcg_at_k(gt, scores, top_k),
            "num_candidates": int(task_indices.size),
            "num_pairs": int(task.pair_positive_local_indices.size),
            "num_positives": int(task.hard_positive_local_indices.size),
        }
        per_task_rows.append(row)

    if not per_task_rows:
        summary = {
            "dataset": dataset_id,
            "split": split_name,
            "method": method_name,
            "num_tasks": 0,
            "hard_pair_accuracy_mean": 0.0,
            "group_recall_at_k_mean": 0.0,
            "spearman_mean": 0.0,
            "pearson_mean": 0.0,
            "ndcg_at_k_mean": 0.0,
        }
        return per_task_rows, summary

    def _mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in per_task_rows]))

    return per_task_rows, {
        "dataset": dataset_id,
        "split": split_name,
        "method": method_name,
        "num_tasks": len(per_task_rows),
        "hard_pair_accuracy_mean": _mean("hard_pair_accuracy"),
        "group_recall_at_k_mean": _mean("group_recall_at_k"),
        "spearman_mean": _mean("spearman"),
        "pearson_mean": _mean("pearson"),
        "ndcg_at_k_mean": _mean("ndcg_at_k"),
    }


def _ik_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["hard_pair_accuracy_mean"]),
        float(row["group_recall_at_k_mean"]),
        float(row["spearman_mean"]),
        float(row["ndcg_at_k_mean"]),
        -float(row["ik_subsample_size"]),
        -float(row["ik_temperature"]),
    )


def _method_scores_map(dataset_result: dict[str, Any], ik_scores: np.ndarray) -> dict[str, np.ndarray]:
    first_hit_scores = distances_to_scores(dataset_result["first_hit_distances"])
    oracle_scores = distances_to_scores(dataset_result["oracle_distances"])
    baselines = dataset_result["baselines"]
    return {
        "euclidean": np.asarray(baselines.euclidean, dtype=np.float32),
        "gaussian": np.asarray(baselines.gaussian, dtype=np.float32),
        "mahalanobis": np.asarray(baselines.mahalanobis, dtype=np.float32),
        "adaptive_gaussian": np.asarray(baselines.adaptive_gaussian, dtype=np.float32),
        "first_hit_temporal_distance": first_hit_scores,
        "one_step_dynamics": np.asarray(baselines.one_step_dynamics, dtype=np.float32),
        "oracle_temporal_distance": oracle_scores,
        "replay_temporal": np.asarray(baselines.replay_temporal, dtype=np.float32),
        "ik": np.asarray(ik_scores, dtype=np.float32),
    }


def _run_ik_search_for_dataset(dataset_result: dict[str, Any], cfg: ARRBenchmarkConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tasks = dataset_result["tasks"]
    search_tasks, _ = _split_tasks(tasks, seed=cfg.seed, fraction=cfg.search_split_fraction)
    reach_matrix = dataset_result["context"].ground_truth.reach_prob

    search_rows: list[dict[str, Any]] = []
    for subsample_size in cfg.ik_subsample_grid or [32]:
        for temperature in cfg.ik_temperature_grid or [0.01]:
            ik_scores = compute_or_load_ik_score_matrix(
                context=dataset_result["context"],
                cfg=ReachabilityAnalysisConfig(
                    datasets=[dataset_result["dataset"]],
                    output_dir=cfg.output_dir,
                    cache_dir=cfg.cache_dir,
                    seed=cfg.seed,
                    fit_pool_size=cfg.fit_pool_size,
                    ik_ensemble_size=cfg.ik_ensemble_size,
                    ik_batch_size=cfg.ik_batch_size,
                    ik_device=cfg.ik_device,
                    overwrite_cache=cfg.overwrite_cache,
                    minari_datasets_path=cfg.minari_datasets_path,
                ),
                subsample_size=int(subsample_size),
                temperature=float(temperature),
            )
            _, summary = _evaluate_method_on_tasks(
                dataset_id=dataset_result["dataset"],
                tasks=search_tasks,
                scores_matrix=ik_scores,
                reach_matrix=reach_matrix,
                split_name="search",
                method_name="ik",
                top_k=cfg.top_k,
            )
            row = {
                **summary,
                "ik_subsample_size": int(subsample_size),
                "ik_temperature": float(temperature),
            }
            search_rows.append(row)

    best_row = max(search_rows, key=_ik_selection_key)
    return best_row, search_rows


def _plot_multi_decoy_task(dataset_result: dict[str, Any], task: ARRTask, output_path: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parsed = dataset_result["parsed"]
    context = dataset_result["context"]
    task_global = context.candidate_indices[task.candidate_local_indices]
    pos_global = context.candidate_indices[task.hard_positive_local_indices]
    neg_global = context.candidate_indices[task.hard_negative_local_indices]
    ctx_global = context.candidate_indices[task.context_local_indices]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(parsed.positions[:, 0], parsed.positions[:, 1], s=3, alpha=0.08, color="#bbbbbb")
    if ctx_global.size:
        ax.scatter(parsed.positions[ctx_global, 0], parsed.positions[ctx_global, 1], s=18, alpha=0.5, color="#9aa0a6", label="context")
    if neg_global.size:
        ax.scatter(parsed.positions[neg_global, 0], parsed.positions[neg_global, 1], s=35, alpha=0.9, color="#d64f4f", label="hard negatives")
    if pos_global.size:
        ax.scatter(parsed.positions[pos_global, 0], parsed.positions[pos_global, 1], s=35, alpha=0.9, color="#2b8a3e", label="hard positives")
    anchor_pos = parsed.positions[task.anchor_global_index]
    ax.scatter(anchor_pos[0], anchor_pos[1], s=180, marker="*", color="#111111", label="anchor")

    for pos_idx, neg_idx in zip(task.pair_positive_local_indices.tolist(), task.pair_negative_local_indices.tolist()):
        pos = parsed.positions[context.candidate_indices[int(pos_idx)]]
        neg = parsed.positions[context.candidate_indices[int(neg_idx)]]
        ax.plot([anchor_pos[0], pos[0]], [anchor_pos[1], pos[1]], color="#2b8a3e", alpha=0.35, linewidth=1.0)
        ax.plot([anchor_pos[0], neg[0]], [anchor_pos[1], neg[1]], color="#d64f4f", alpha=0.35, linewidth=1.0)

    ax.set_title(f"{dataset_result['dataset']} multi-decoy hard task")
    ax.legend(frameon=False, loc="best")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_first_hit_failure(dataset_result: dict[str, Any], task: ARRTask, output_path: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = dataset_result["context"].ground_truth.reach_prob[task.anchor_row]
    first_hit = dataset_result["first_hit_distances"][task.anchor_row]
    labels = []
    gt_vals = []
    first_vals = []
    for pair_id, (pos_idx, neg_idx) in enumerate(
        zip(task.pair_positive_local_indices.tolist(), task.pair_negative_local_indices.tolist())
    ):
        labels.extend([f"P{pair_id+1}", f"N{pair_id+1}"])
        gt_vals.extend([float(gt[int(pos_idx)]), float(gt[int(neg_idx)])])
        first_vals.extend([float(first_hit[int(pos_idx)]), float(first_hit[int(neg_idx)])])

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].bar(x, gt_vals, color=["#2b8a3e" if label.startswith("P") else "#d64f4f" for label in labels])
    axes[0].set_ylabel("reach_prob")
    axes[0].set_title("Matched pairs: same/near same first-hit, different reachability")
    axes[1].bar(x, first_vals, color=["#2b8a3e" if label.startswith("P") else "#d64f4f" for label in labels])
    axes[1].set_ylabel("first-hit dist")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_oracle_mismatch(dataset_result: dict[str, Any], task: ARRTask, output_path: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = dataset_result["context"].ground_truth.reach_prob[task.anchor_row, task.candidate_local_indices]
    oracle_dist = dataset_result["oracle_distances"][task.anchor_row, task.candidate_local_indices]
    pos_set = set(int(x) for x in task.hard_positive_local_indices.tolist())
    neg_set = set(int(x) for x in task.hard_negative_local_indices.tolist())
    colors = []
    for idx in task.candidate_local_indices.tolist():
        if int(idx) in pos_set:
            colors.append("#2b8a3e")
        elif int(idx) in neg_set:
            colors.append("#d64f4f")
        else:
            colors.append("#9aa0a6")

    finite_oracle = np.where(np.isfinite(oracle_dist), oracle_dist, np.nan)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(finite_oracle, gt, s=24, alpha=0.75, c=colors)
    ax.set_xlabel("oracle-temporal distance")
    ax.set_ylabel("reach_prob")
    ax.set_title(f"{dataset_result['dataset']} oracle-temporal mismatch")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_method_bars(summary_rows: list[dict[str, Any]], dataset_id: str, split_name: str, output_path: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in summary_rows if row["dataset"] == dataset_id and row["split"] == split_name]
    methods = [row["method"] for row in rows]
    hard_pair = [float(row["hard_pair_accuracy_mean"]) for row in rows]
    recall = [float(row["group_recall_at_k_mean"]) for row in rows]
    spearman = [float(row["spearman_mean"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    values = [hard_pair, recall, spearman]
    titles = ["Hard-Pair Accuracy", "Group Recall@k", "Spearman"]
    for ax, metric_values, title in zip(axes, values, titles):
        ax.bar(methods, metric_values, color=[ARR_METHOD_COLORS.get(method, "#7f7f7f") for method in methods])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0.0, max(1.0, max(metric_values) + 0.05))

    fig.suptitle(f"{dataset_id} ARR benchmark ({split_name})")
    fig.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_arr_report(
    cfg: ARRBenchmarkConfig,
    dataset_results: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    best_ik_rows: list[dict[str, Any]],
    report_path: str,
) -> None:
    lines = [
        "# Ambiguous Reachability Retrieval (ARR) Report",
        "",
        "## Setup",
        "",
        f"- Datasets: {', '.join(cfg.datasets)}",
        f"- Benchmark anchors per dataset (target): {cfg.num_benchmark_anchors}",
        f"- Candidate pool size: {cfg.candidate_pool_size}",
        f"- Hard positives / negatives / context: {cfg.hard_positive_count}/{cfg.hard_negative_count}/{cfg.context_decoy_count}",
        f"- IK ensemble size: {cfg.ik_ensemble_size}",
        f"- IK search grid sizes: {len(cfg.ik_subsample_grid or [])} subsamples x {len(cfg.ik_temperature_grid or [])} temperatures",
        "",
        "## Task Definition",
        "",
        "- ARR mines anchors whose local candidate sets are geometrically ambiguous but have sharply different empirical reachability.",
        "- `first_hit_temporal_distance` uses the best-case earliest hit step across anchor occurrences and intentionally amplifies the weakness of earliest-hit-only temporal proxies.",
        "- `oracle_temporal_distance` converts the aggregated oracle temporal score `mean(1 / (1 + tau_first))` into a distance-like form; it is future-aware but still only captures earliest-hit information, not full reachability mass.",
        "- `replay_temporal` remains a stronger replay baseline and is reported separately from the main criticized temporal distance.",
        "- `mahalanobis` is a global covariance-whitened baseline; `adaptive_gaussian` is the new local-bandwidth Gaussian; `one_step_dynamics` is a reward-free one-step dynamics-aware baseline.",
        "",
        "## Best IK Per Dataset",
        "",
        "| Dataset | H | Best Subsample | Best Temperature | Search Hard-Pair Acc | Search Recall@k | Search Spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best_ik_rows:
        lines.append(
            f"| {row['dataset']} | {row['horizon']} | {row['ik_subsample_size']} | {row['ik_temperature']} | "
            f"{row['hard_pair_accuracy_mean']:.4f} | {row['group_recall_at_k_mean']:.4f} | {row['spearman_mean']:.4f} |"
        )
    lines.extend(["", "## Dataset Notes", ""])
    for result in dataset_results:
        lines.extend(
            [
                f"### {result['dataset']}",
                "",
                f"- Retained ARR tasks: {len(result['tasks'])}",
                f"- Median fallback level: {int(np.median([task.fallback_level for task in result['tasks']])) if result['tasks'] else -1}",
                f"- Mean matched GT gap: {float(np.mean([task.mean_pair_gt_gap for task in result['tasks']])) if result['tasks'] else 0.0:.4f}",
                f"- Formal best-H reused: {result['best_row']['horizon']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Summary",
            "",
            "| Dataset | Split | Method | Hard-Pair Acc | Group Recall@k | Spearman | Pearson | NDCG@k |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['method']} | {row['hard_pair_accuracy_mean']:.4f} | "
            f"{row['group_recall_at_k_mean']:.4f} | {row['spearman_mean']:.4f} | "
            f"{row['pearson_mean']:.4f} | {row['ndcg_at_k_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- ARR is intentionally constructed to stress earliest-hit temporal distances; it should not be read as a claim that every temporal method fails on every task.",
            "- `oracle_temporal_distance` is reported as a future-aware reference, not as the main ground truth.",
            "- The benchmark fixes tasks before ARR-specific IK search; IK is searched only on the ARR search split to reduce overfitting risk.",
        ]
    )

    ensure_dir(os.path.dirname(report_path))
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")


def run_arr_benchmark(cfg: ARRBenchmarkConfig) -> dict[str, Any]:
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir)
    best_rows = _load_best_rows(cfg.best_config_path)
    best_by_dataset = {row["dataset"]: row for row in best_rows}

    dataset_results = []
    benchmark_task_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    ik_search_rows: list[dict[str, Any]] = []
    best_ik_rows: list[dict[str, Any]] = []
    figure_paths: dict[str, dict[str, str]] = {}

    for dataset_id in cfg.datasets:
        dataset_result = mine_arr_tasks_for_dataset(dataset_id, best_by_dataset[dataset_id], cfg)
        best_ik_row, search_rows = _run_ik_search_for_dataset(dataset_result, cfg)
        best_ik_row = dict(best_ik_row)
        best_ik_row["horizon"] = int(dataset_result["best_row"]["horizon"])
        best_ik_rows.append(best_ik_row)
        ik_search_rows.extend(search_rows)

        best_ik_scores = compute_or_load_ik_score_matrix(
            context=dataset_result["context"],
            cfg=ReachabilityAnalysisConfig(
                datasets=[dataset_id],
                output_dir=cfg.output_dir,
                cache_dir=cfg.cache_dir,
                seed=cfg.seed,
                fit_pool_size=cfg.fit_pool_size,
                ik_ensemble_size=cfg.ik_ensemble_size,
                ik_batch_size=cfg.ik_batch_size,
                ik_device=cfg.ik_device,
                overwrite_cache=cfg.overwrite_cache,
                minari_datasets_path=cfg.minari_datasets_path,
            ),
            subsample_size=int(best_ik_row["ik_subsample_size"]),
            temperature=float(best_ik_row["ik_temperature"]),
        )
        score_map = _method_scores_map(dataset_result, best_ik_scores)
        search_tasks, eval_tasks = _split_tasks(dataset_result["tasks"], seed=cfg.seed, fraction=cfg.search_split_fraction)
        reach_matrix = dataset_result["context"].ground_truth.reach_prob
        for split_name, split_tasks in (("search", search_tasks), ("eval", eval_tasks), ("all", dataset_result["tasks"])):
            for method_name, score_matrix in score_map.items():
                task_rows, summary = _evaluate_method_on_tasks(
                    dataset_id=dataset_id,
                    tasks=split_tasks,
                    scores_matrix=score_matrix,
                    reach_matrix=reach_matrix,
                    split_name=split_name,
                    method_name=method_name,
                    top_k=cfg.top_k,
                )
                per_task_rows.extend(task_rows)
                summary_rows.append(summary)

        dataset_results.append(dataset_result)
        benchmark_task_rows.extend(_task_rows_for_csv(dataset_result))

        figures_dir = os.path.join(cfg.output_dir, "figures")
        ensure_dir(figures_dir)
        showcase_task = dataset_result["tasks"][0] if dataset_result["tasks"] else None
        figure_paths[dataset_id] = {}
        if showcase_task is not None:
            multi_path = os.path.join(figures_dir, f"{dataset_slug(dataset_id)}_multi_decoy.png")
            first_path = os.path.join(figures_dir, f"{dataset_slug(dataset_id)}_first_hit_failure.png")
            oracle_path = os.path.join(figures_dir, f"{dataset_slug(dataset_id)}_oracle_mismatch.png")
            _plot_multi_decoy_task(dataset_result, showcase_task, multi_path)
            _plot_first_hit_failure(dataset_result, showcase_task, first_path)
            _plot_oracle_mismatch(dataset_result, showcase_task, oracle_path)
            figure_paths[dataset_id]["multi_decoy"] = multi_path
            figure_paths[dataset_id]["first_hit_failure"] = first_path
            figure_paths[dataset_id]["oracle_mismatch"] = oracle_path
        bar_path = os.path.join(figures_dir, f"{dataset_slug(dataset_id)}_method_bars_eval.png")
        _plot_method_bars(summary_rows, dataset_id, "eval", bar_path)
        figure_paths[dataset_id]["method_bars_eval"] = bar_path

    tables_dir = os.path.join(cfg.output_dir, "tables")
    ensure_dir(tables_dir)
    benchmark_tasks_path = os.path.join(tables_dir, "benchmark_tasks.csv")
    per_task_path = os.path.join(tables_dir, "benchmark_per_task_metrics.csv")
    benchmark_summary_path = os.path.join(tables_dir, "benchmark_summary.csv")
    ik_search_path = os.path.join(tables_dir, "ik_search_summary.csv")
    best_ik_path = os.path.join(tables_dir, "ik_best_per_dataset.csv")
    report_path = os.path.join(cfg.output_dir, "arr_report.md")

    save_csv(
        benchmark_tasks_path,
        benchmark_task_rows,
        [
            "dataset",
            "task_id",
            "anchor_row",
            "anchor_global_index",
            "anchor_occurrence_count",
            "candidate_local_index",
            "candidate_global_index",
            "candidate_x",
            "candidate_y",
            "role",
            "pair_group",
            "reach_prob",
            "euclidean_distance",
            "mahalanobis_distance",
            "adaptive_gaussian",
            "first_hit_temporal_distance",
            "one_step_dynamics_distance",
            "oracle_temporal_distance",
            "oracle_temporal_score",
            "replay_temporal",
            "reference_ik",
            "fallback_level",
            "mean_pair_gt_gap",
        ],
    )
    save_csv(
        per_task_path,
        per_task_rows,
        [
            "dataset",
            "split",
            "method",
            "task_id",
            "anchor_row",
            "hard_pair_accuracy",
            "group_recall_at_k",
            "spearman",
            "pearson",
            "ndcg_at_k",
            "num_candidates",
            "num_pairs",
            "num_positives",
        ],
    )
    save_csv(
        benchmark_summary_path,
        summary_rows,
        [
            "dataset",
            "split",
            "method",
            "num_tasks",
            "hard_pair_accuracy_mean",
            "group_recall_at_k_mean",
            "spearman_mean",
            "pearson_mean",
            "ndcg_at_k_mean",
        ],
    )
    save_csv(
        ik_search_path,
        ik_search_rows,
        [
            "dataset",
            "split",
            "method",
            "ik_subsample_size",
            "ik_temperature",
            "num_tasks",
            "hard_pair_accuracy_mean",
            "group_recall_at_k_mean",
            "spearman_mean",
            "pearson_mean",
            "ndcg_at_k_mean",
        ],
    )
    save_csv(
        best_ik_path,
        best_ik_rows,
        [
            "dataset",
            "horizon",
            "split",
            "method",
            "ik_subsample_size",
            "ik_temperature",
            "num_tasks",
            "hard_pair_accuracy_mean",
            "group_recall_at_k_mean",
            "spearman_mean",
            "pearson_mean",
            "ndcg_at_k_mean",
        ],
    )
    if cfg.write_report:
        build_arr_report(cfg, dataset_results, summary_rows, best_ik_rows, report_path)
    else:
        report_path = ""

    metadata_path = os.path.join(cfg.output_dir, "arr_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "config": asdict(cfg),
                "datasets": cfg.datasets,
                "num_dataset_results": len(dataset_results),
                "figure_paths": figure_paths,
            },
            metadata_file,
            indent=2,
        )

    return {
        "dataset_results": dataset_results,
        "benchmark_tasks_path": benchmark_tasks_path,
        "per_task_path": per_task_path,
        "benchmark_summary_path": benchmark_summary_path,
        "ik_search_path": ik_search_path,
        "best_ik_path": best_ik_path,
        "report_path": report_path,
        "figure_paths": figure_paths,
        "metadata_path": metadata_path,
    }
