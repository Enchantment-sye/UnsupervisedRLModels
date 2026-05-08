"""
kNN-based Relabeling Benchmark Module
======================================

Wraps the existing ``reachability_alignment`` infrastructure and adds the
metrics/visualisations specific to the relabeling evaluation protocol:

  For each query state s_t, retrieve top-k candidate goals from the offline
  buffer using each of 7 similarity functions, then measure whether those
  retrieved goals are truly reachable (H-step empirical ground truth).

New additions on top of the existing pipeline
----------------------------------------------
- ``goal_precision_at_k``        – fraction of top-k retrieved that are reachable
- ``mean_gt_score_at_k``         – mean reach_prob of top-k retrieved
- ``mean_geodesic_dist_at_k``    – mean geodesic distance of top-k retrieved from anchor
- ``diversity_at_k``             – avg pairwise L2 distance among top-k retrieved
- Geodesic-based binary GT       – y is positive iff geo(anchor, y) <= H_goal
- Min-time-gap filter            – mask trivially close candidates (same episode, |dt| < gap)
- Maze visualisation             – top-k goals overlaid on 2D maze scatter
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .reachability_alignment import (
    EvaluationContext,
    ParsedDataset,
    ReachabilityAnalysisConfig,
    compute_or_load_baseline_scores,
    compute_or_load_ik_score_matrix,
    dataset_slug,
    ensure_dir,
    load_or_parse_dataset,
    prepare_evaluation_context,
)
from .maze_geodesic import MazeSpec, load_maze_spec
from .similarity_metrics import (
    auc_from_binary_labels,
    ndcg_at_k,
    recall_at_k,
    safe_pearson,
    safe_spearman,
    topk_overlap,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KNNRelabelConfig:
    """
    Configuration for the kNN relabeling benchmark.

    Holds all fields needed to construct a ``ReachabilityAnalysisConfig``
    plus relabeling-specific extensions.
    """

    # ---- I/O ----
    datasets: list[str]
    output_dir: str
    cache_dir: str
    seed: int = 0
    minari_datasets_path: str = "/home/shangyy/.minari/datasets"
    overwrite_cache: bool = False

    # ---- Sampling ----
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
    fit_pool_size: int = 50000

    # ---- Similarity hyperparameters ----
    ik_ensemble_size: int = 100
    ik_subsample_size: int = 32
    ik_temperature: float = 0.01
    ik_batch_size: int = 4096
    ik_device: str = "auto"
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
    replay_temporal_window: int | None = None

    # ---- Relabeling-specific ----
    min_time_gap: int = 5
    """Exclude candidate goals from the same episode whose timestep differs by less than this."""
    min_goal_dist: float = 0.0
    """Minimum Euclidean distance from anchor; 0.0 disables the filter."""
    geodesic_h_goal_pointmaze: float = 10.0
    """Geodesic-distance threshold for the binary secondary GT on pointmaze datasets."""
    geodesic_h_goal_antmaze: float = 15.0
    """Geodesic-distance threshold for the binary secondary GT on antmaze datasets."""
    reachability_positive_mode: str = "percentile"
    reachability_positive_percentile: float = 75.0
    reachability_positive_threshold: float = 0.1
    plot_query_examples: int = 3
    """Number of anchor states for which to draw top-k goal visualisations."""

    def to_reach_cfg(self) -> ReachabilityAnalysisConfig:
        """Build a ReachabilityAnalysisConfig from the shared fields."""
        return ReachabilityAnalysisConfig(
            datasets=self.datasets,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            seed=self.seed,
            mode="single",
            horizon=self.horizon,
            num_anchors=self.num_anchors,
            num_candidates=self.num_candidates,
            top_k=self.top_k,
            match_radius=self.match_radius,
            candidate_pool_mode=self.candidate_pool_mode,
            query_pool_mode=self.query_pool_mode,
            node_stride_pointmaze=self.node_stride_pointmaze,
            node_stride_antmaze=self.node_stride_antmaze,
            candidate_sampling=self.candidate_sampling,
            min_anchor_occurrences=self.min_anchor_occurrences,
            max_anchor_occurrences=self.max_anchor_occurrences,
            max_anchor_attempts=self.max_anchor_attempts,
            fit_pool_size=self.fit_pool_size,
            ik_ensemble_size=self.ik_ensemble_size,
            ik_subsample_size=self.ik_subsample_size,
            ik_temperature=self.ik_temperature,
            ik_batch_size=self.ik_batch_size,
            ik_device=self.ik_device,
            gk_sigma_mode=self.gk_sigma_mode,
            gk_sigma=self.gk_sigma,
            mahalanobis_covariance_estimator=self.mahalanobis_covariance_estimator,
            mahalanobis_implementation=self.mahalanobis_implementation,
            mahalanobis_eps=self.mahalanobis_eps,
            adaptive_gaussian_k=self.adaptive_gaussian_k,
            adaptive_gaussian_eps=self.adaptive_gaussian_eps,
            adaptive_gaussian_output=self.adaptive_gaussian_output,
            dynamics_backend=self.dynamics_backend,
            dynamics_num_bins=self.dynamics_num_bins,
            dynamics_distance_metric=self.dynamics_distance_metric,
            dynamics_alpha=self.dynamics_alpha,
            dynamics_min_count=self.dynamics_min_count,
            dynamics_eps=self.dynamics_eps,
            dynamics_local_knn_m=self.dynamics_local_knn_m,
            dynamics_local_distance_metric=self.dynamics_local_distance_metric,
            dynamics_state_variant=self.dynamics_state_variant,
            metric_state_variant=self.metric_state_variant,
            replay_temporal_window=self.replay_temporal_window,
            overwrite_cache=self.overwrite_cache,
            minari_datasets_path=self.minari_datasets_path,
            scatter_points=4000,
        )


# ---------------------------------------------------------------------------
# New metric helpers
# ---------------------------------------------------------------------------

def goal_precision_at_k(y_true_binary: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Precision@k = fraction of top-k retrieved candidates that are true positives.

    Unlike topk_overlap (intersection of top-k true vs top-k pred), this uses
    the predicted ranking exclusively.
    """
    labels = np.asarray(y_true_binary, dtype=np.int64)
    top_k = min(k, int(labels.size))
    if top_k == 0:
        return 0.0
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))[:top_k]
    positives = float(np.sum(labels[order] > 0))
    return positives / float(top_k)


def _resolve_positive_cutoff(
    ground_truth_matrix: np.ndarray,
    ground_truth_type: str,
    positive_mode: str,
    positive_percentile: float,
    positive_threshold: float,
) -> tuple[str, float]:
    gt_name = str(ground_truth_type)
    if gt_name == "geodesic":
        return "native_binary", 0.5

    mode = str(positive_mode or "percentile").strip().lower()
    if mode == "fixed":
        return mode, float(positive_threshold)
    finite = np.asarray(ground_truth_matrix, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "percentile", 0.0
    percentile = float(np.clip(positive_percentile, 0.0, 100.0))
    return "percentile", float(np.percentile(finite, percentile))


def mean_gt_score_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Mean ground-truth reachability score of the top-k retrieved candidates."""
    top_k = min(k, int(y_true.size))
    if top_k == 0:
        return 0.0
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))[:top_k]
    return float(np.mean(np.asarray(y_true, dtype=np.float64)[order]))


def _mean_geodesic_at_k(geo_row: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Mean geodesic distance of the top-k retrieved candidates from the anchor."""
    top_k = min(k, int(geo_row.size))
    if top_k == 0:
        return float("inf")
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))[:top_k]
    geo_vals = np.asarray(geo_row, dtype=np.float64)[order]
    finite = geo_vals[np.isfinite(geo_vals)]
    if finite.size == 0:
        return float("inf")
    return float(np.mean(finite))


def diversity_at_k(candidate_positions: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Average pairwise Euclidean distance among the top-k retrieved goal positions.

    Higher → retrieved goals are spatially spread out (less collapse).
    """
    from scipy.spatial.distance import pdist

    top_k = min(k, int(candidate_positions.shape[0]))
    if top_k < 2:
        return 0.0
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))[:top_k]
    positions = np.asarray(candidate_positions[order], dtype=np.float64)
    dists = pdist(positions, metric="euclidean")
    if dists.size == 0:
        return 0.0
    return float(np.mean(dists))


def unique_goal_ratio_at_k(candidate_positions: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of unique spatial goals among the top-k retrieved candidates."""
    top_k = min(k, int(candidate_positions.shape[0]))
    if top_k == 0:
        return 0.0
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))[:top_k]
    positions = np.asarray(candidate_positions[order], dtype=np.float64)
    rounded = np.round(positions, decimals=6)
    unique = np.unique(rounded, axis=0)
    return float(unique.shape[0] / float(top_k))


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def compute_geodesic_gt(
    anchor_xy: np.ndarray,
    candidate_xy: np.ndarray,
    maze_spec: MazeSpec,
    h_goal: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the geodesic-based binary ground truth.

    Returns
    -------
    geo_binary : (N_anchors, N_candidates) float32
        1.0 if geodesic(anchor_i, cand_j) <= h_goal, else 0.0.
    geo_matrix : (N_anchors, N_candidates) float32
        Raw geodesic distances.
    """
    geo_matrix = maze_spec.geodesic_distances(
        np.asarray(anchor_xy, dtype=np.float32),
        np.asarray(candidate_xy, dtype=np.float32),
    ).astype(np.float32)
    geo_binary = (geo_matrix <= float(h_goal)).astype(np.float32)
    return geo_binary, geo_matrix


# ---------------------------------------------------------------------------
# Trivial-goal filter
# ---------------------------------------------------------------------------

def apply_min_time_gap_filter(
    scores: np.ndarray,
    anchor_global_indices: np.ndarray,
    candidate_global_indices: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
    min_time_gap: int,
) -> np.ndarray:
    """
    Set score[i, j] = -inf when anchor i and candidate j are in the same episode
    and |timestep_i - timestep_j| < min_time_gap.

    This prevents trivially close states from being retrieved as goals.
    """
    if min_time_gap <= 0:
        return scores
    filtered = np.array(scores, dtype=np.float32, copy=True)
    anchor_ep = episode_ids[anchor_global_indices]   # (N_a,)
    anchor_ts = timesteps[anchor_global_indices].astype(np.int64)
    cand_ep = episode_ids[candidate_global_indices]   # (N_c,)
    cand_ts = timesteps[candidate_global_indices].astype(np.int64)

    same_ep = anchor_ep[:, None] == cand_ep[None, :]  # (N_a, N_c)
    time_diff = np.abs(anchor_ts[:, None] - cand_ts[None, :])  # (N_a, N_c)
    trivial = same_ep & (time_diff < int(min_time_gap))
    filtered[trivial] = -np.inf
    return filtered


def apply_min_goal_dist_filter(
    scores: np.ndarray,
    anchor_xy: np.ndarray,
    candidate_xy: np.ndarray,
    min_goal_dist: float,
) -> np.ndarray:
    """Set score[i, j] = -inf when Euclidean(anchor_i, cand_j) < min_goal_dist."""
    if min_goal_dist <= 0.0:
        return scores
    from scipy.spatial.distance import cdist
    dists = cdist(anchor_xy, candidate_xy, metric="euclidean")  # (N_a, N_c)
    filtered = np.array(scores, dtype=np.float32, copy=True)
    filtered[dists < float(min_goal_dist)] = -np.inf
    return filtered


# ---------------------------------------------------------------------------
# Extended evaluation
# ---------------------------------------------------------------------------

def evaluate_relabel_metrics(
    ground_truth_matrix: np.ndarray,
    similarity_scores: np.ndarray,
    candidate_positions: np.ndarray,
    anchor_positions: np.ndarray,
    geo_matrix: np.ndarray,
    top_k: int,
    anchor_global_indices: np.ndarray,
    candidate_global_indices: np.ndarray,
    dataset_name: str,
    method_name: str,
    ground_truth_type: str,
    occurrence_counts: np.ndarray,
    horizon: int,
    positive_mode: str,
    positive_cutoff: float,
    sampling_protocol: str,
    dynamics_backend: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Per-anchor relabeling metrics including the new goal_precision@k,
    mean GT reachability, mean geodesic distance, and diversity.

    Parameters
    ----------
    geo_matrix : (N_anchors, N_candidates) float32
        Raw geodesic distances from each anchor to each candidate.
        Used to compute ``mean_geodesic_dist`` regardless of which GT type is active.
    """
    per_anchor_rows: list[dict[str, Any]] = []
    cutoff = float(positive_cutoff)
    mode = str(positive_mode)

    for anchor_row_id in range(ground_truth_matrix.shape[0]):
        gt_row = np.asarray(ground_truth_matrix[anchor_row_id], dtype=np.float64)
        score_row = np.asarray(similarity_scores[anchor_row_id], dtype=np.float64)
        geo_row = np.asarray(geo_matrix[anchor_row_id], dtype=np.float64)

        # Exclude self (same global index)
        valid_mask = np.ones(int(gt_row.size), dtype=bool)
        self_positions = np.flatnonzero(
            candidate_global_indices == anchor_global_indices[anchor_row_id]
        )
        valid_mask[self_positions] = False

        gt_valid = gt_row[valid_mask]
        score_valid = score_row[valid_mask]
        cand_pos_valid = candidate_positions[valid_mask]
        geo_valid = geo_row[valid_mask]

        if gt_valid.size == 0:
            continue

        binary_labels = (gt_valid >= cutoff).astype(np.int64)
        row: dict[str, Any] = {
            "dataset": dataset_name,
            "ground_truth_type": ground_truth_type,
            "method": method_name,
            "sampling_protocol": sampling_protocol,
            "positive_mode": mode,
            "positive_cutoff": cutoff,
            "dynamics_backend": dynamics_backend,
            "anchor_row": int(anchor_row_id),
            "anchor_index": int(anchor_global_indices[anchor_row_id]),
            "occurrence_count": float(
                occurrence_counts[anchor_row_id]
                if anchor_row_id < int(occurrence_counts.size)
                else 0.0
            ),
            "horizon": int(horizon),
            "top_k": int(top_k),
            # Existing metrics
            "spearman": safe_spearman(score_valid, gt_valid),
            "pearson": safe_pearson(score_valid, gt_valid),
            "recall_at_k": recall_at_k(binary_labels, score_valid, top_k),
            "topk_overlap": topk_overlap(gt_valid, score_valid, top_k),
            "ndcg_at_k": ndcg_at_k(gt_valid, score_valid, top_k),
            "auc": auc_from_binary_labels(binary_labels, score_valid),
            # New relabeling metrics
            "goal_precision_at_k": goal_precision_at_k(binary_labels, score_valid, top_k),
            "mean_gt_reachability": mean_gt_score_at_k(gt_valid, score_valid, top_k),
            "mean_geodesic_dist": _mean_geodesic_at_k(geo_valid, score_valid, top_k),
            "diversity": diversity_at_k(cand_pos_valid, score_valid, top_k),
            "unique_goal_ratio": unique_goal_ratio_at_k(cand_pos_valid, score_valid, top_k),
        }
        per_anchor_rows.append(row)

    _ZERO_SUMMARY: dict[str, Any] = dict(
        dataset=dataset_name,
        ground_truth_type=ground_truth_type,
        method=method_name,
        sampling_protocol=sampling_protocol,
        positive_mode=mode,
        positive_cutoff=cutoff,
        dynamics_backend=dynamics_backend,
        horizon=int(horizon),
        top_k=int(top_k),
        num_anchors=0,
        spearman_mean=0.0,
        pearson_mean=0.0,
        recall_at_k_mean=0.0,
        topk_overlap_mean=0.0,
        ndcg_at_k_mean=0.0,
        auc_mean=0.5,
        goal_precision_at_k_mean=0.0,
        mean_gt_reachability_mean=0.0,
        mean_geodesic_dist_mean=float("inf"),
        diversity_mean=0.0,
        unique_goal_ratio_mean=0.0,
    )
    if not per_anchor_rows:
        return per_anchor_rows, _ZERO_SUMMARY

    def _mean(key: str) -> float:
        vals = [
            float(r[key])
            for r in per_anchor_rows
            if np.isfinite(float(r[key]))
        ]
        return float(np.mean(vals)) if vals else 0.0

    def _mean_allow_inf(key: str) -> float:
        vals = [float(r[key]) for r in per_anchor_rows if float(r[key]) < 1e18]
        return float(np.mean(vals)) if vals else float("inf")

    summary: dict[str, Any] = dict(
        dataset=dataset_name,
        ground_truth_type=ground_truth_type,
        method=method_name,
        sampling_protocol=sampling_protocol,
        positive_mode=mode,
        positive_cutoff=cutoff,
        dynamics_backend=dynamics_backend,
        horizon=int(horizon),
        top_k=int(top_k),
        num_anchors=len(per_anchor_rows),
        spearman_mean=_mean("spearman"),
        pearson_mean=_mean("pearson"),
        recall_at_k_mean=_mean("recall_at_k"),
        topk_overlap_mean=_mean("topk_overlap"),
        ndcg_at_k_mean=_mean("ndcg_at_k"),
        auc_mean=_mean("auc"),
        goal_precision_at_k_mean=_mean("goal_precision_at_k"),
        mean_gt_reachability_mean=_mean("mean_gt_reachability"),
        mean_geodesic_dist_mean=_mean_allow_inf("mean_geodesic_dist"),
        diversity_mean=_mean("diversity"),
        unique_goal_ratio_mean=_mean("unique_goal_ratio"),
    )
    return per_anchor_rows, summary


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_relabel_benchmark(
    cfg: KNNRelabelConfig,
    dataset_id: str,
) -> dict[str, Any]:
    """
    Full per-dataset relabeling benchmark pipeline.

    1. Load Minari dataset → ParsedDataset
    2. Sample anchors (query states) + candidates (goal pool)
    3. Compute H-step empirical ground truth (reach_prob) and geodesic GT
    4. Compute all 7 similarity scores
    5. Apply min_time_gap and min_goal_dist filters
    6. Evaluate all methods × both GT types
    7. Return structured result dict

    Returns
    -------
    dict with keys:
        dataset, dataset_slug, match_radius,
        summary_rows, per_anchor_rows,
        anchor_indices, candidate_indices,
        parsed, ground_truth, geo_matrix,
        similarities (dict method→scores), maze_spec, config
    """
    reach_cfg = cfg.to_reach_cfg()

    print(f"[relabel] Loading dataset: {dataset_id}")
    parsed = load_or_parse_dataset(
        dataset_id=dataset_id,
        cache_dir=cfg.cache_dir,
        overwrite_cache=cfg.overwrite_cache,
        minari_datasets_path=cfg.minari_datasets_path,
        seed=cfg.seed,
    )

    print(f"[relabel] {dataset_id}: {parsed.positions.shape[0]} states, "
          f"{parsed.total_episodes} episodes")

    context: EvaluationContext = prepare_evaluation_context(parsed, reach_cfg)
    print(f"[relabel] Anchors={context.anchor_indices.shape[0]}, "
          f"Candidates={context.candidate_indices.shape[0]}, "
          f"match_radius={context.match_radius:.4f}")

    print(f"[relabel] Computing baseline similarity scores…")
    baselines = compute_or_load_baseline_scores(context, reach_cfg)
    print(f"[relabel] Computing IK scores…")
    ik_scores = compute_or_load_ik_score_matrix(
        context, reach_cfg,
        reach_cfg.ik_subsample_size,
        reach_cfg.ik_temperature,
    )

    # Build sim_map with the 7 methods
    sim_map: dict[str, np.ndarray] = {
        "euclidean": np.asarray(baselines.euclidean, dtype=np.float32),
        "gaussian": np.asarray(baselines.gaussian, dtype=np.float32),
        "mahalanobis": np.asarray(baselines.mahalanobis, dtype=np.float32),
        "adaptive_gaussian": np.asarray(baselines.adaptive_gaussian, dtype=np.float32),
        "ik": np.asarray(ik_scores, dtype=np.float32),
        "one_step_dynamics": np.asarray(baselines.one_step_dynamics, dtype=np.float32),
        "temporal_distance": np.asarray(baselines.temporal_distance, dtype=np.float32),
    }

    anchor_xy = parsed.positions[context.anchor_indices]
    candidate_xy = parsed.positions[context.candidate_indices]

    # Apply trivial-goal filters
    if cfg.min_time_gap > 0:
        print(f"[relabel] Applying min_time_gap={cfg.min_time_gap} filter…")
        filtered: dict[str, np.ndarray] = {}
        for method, scores in sim_map.items():
            filtered[method] = apply_min_time_gap_filter(
                scores=scores,
                anchor_global_indices=context.anchor_indices,
                candidate_global_indices=context.candidate_indices,
                episode_ids=parsed.episode_ids,
                timesteps=parsed.timesteps,
                min_time_gap=cfg.min_time_gap,
            )
        sim_map = filtered

    if cfg.min_goal_dist > 0.0:
        print(f"[relabel] Applying min_goal_dist={cfg.min_goal_dist} filter…")
        filtered2: dict[str, np.ndarray] = {}
        for method, scores in sim_map.items():
            filtered2[method] = apply_min_goal_dist_filter(
                scores=scores,
                anchor_xy=anchor_xy,
                candidate_xy=candidate_xy,
                min_goal_dist=cfg.min_goal_dist,
            )
        sim_map = filtered2

    # Load maze spec for geodesic GT
    print(f"[relabel] Loading maze spec for geodesic ground truth…")
    maze_spec = load_maze_spec(
        dataset_id=dataset_id,
        minari_root=cfg.minari_datasets_path,
        cache_dir=cfg.cache_dir,
    )

    # Compute geodesic GT
    is_antmaze = "antmaze" in dataset_id.lower()
    h_goal = cfg.geodesic_h_goal_antmaze if is_antmaze else cfg.geodesic_h_goal_pointmaze
    print(f"[relabel] Computing geodesic GT (h_goal={h_goal})…")
    geo_binary, geo_matrix = compute_geodesic_gt(anchor_xy, candidate_xy, maze_spec, h_goal)

    # Ground truth map
    gt_map: dict[str, np.ndarray] = {
        "reach_prob": np.asarray(context.ground_truth.reach_prob, dtype=np.float32),
        "geodesic": geo_binary,
    }
    positive_cutoffs = {
        gt_name: _resolve_positive_cutoff(
            gt_matrix,
            gt_name,
            cfg.reachability_positive_mode,
            cfg.reachability_positive_percentile,
            cfg.reachability_positive_threshold,
        )
        for gt_name, gt_matrix in gt_map.items()
    }

    # Evaluate
    print(f"[relabel] Evaluating {len(sim_map)} methods × {len(gt_map)} GT types…")
    summary_rows: list[dict[str, Any]] = []
    per_anchor_rows: list[dict[str, Any]] = []

    for gt_name, gt_matrix in gt_map.items():
        positive_mode, positive_cutoff = positive_cutoffs[gt_name]
        for method_name, sim_matrix in sim_map.items():
            anchor_rows, summary = evaluate_relabel_metrics(
                ground_truth_matrix=gt_matrix,
                similarity_scores=sim_matrix,
                candidate_positions=candidate_xy,
                anchor_positions=anchor_xy,
                geo_matrix=geo_matrix,
                top_k=cfg.top_k,
                anchor_global_indices=context.anchor_indices,
                candidate_global_indices=context.candidate_indices,
                dataset_name=dataset_id,
                method_name=method_name,
                ground_truth_type=gt_name,
                occurrence_counts=context.ground_truth.occurrence_counts,
                horizon=cfg.horizon,
                positive_mode=positive_mode,
                positive_cutoff=positive_cutoff,
                sampling_protocol=str(cfg.query_pool_mode),
                dynamics_backend=str(cfg.dynamics_backend),
            )
            summary_rows.append(summary)
            per_anchor_rows.extend(anchor_rows)

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
        "geo_matrix": geo_matrix,
        "similarities": sim_map,
        "maze_spec": maze_spec,
        "h_goal": h_goal,
        "config": {
            "dataset": dataset_id,
            "min_time_gap": cfg.min_time_gap,
            "min_goal_dist": cfg.min_goal_dist,
            "geodesic_h_goal": h_goal,
            "horizon": cfg.horizon,
            "top_k": cfg.top_k,
            "sampling_protocol": str(cfg.query_pool_mode),
            "candidate_pool_mode": str(cfg.candidate_pool_mode),
            "reachability_positive_mode": str(cfg.reachability_positive_mode),
            "reachability_positive_percentile": float(cfg.reachability_positive_percentile),
            "reachability_positive_threshold": float(cfg.reachability_positive_threshold),
            "dynamics_backend": str(cfg.dynamics_backend),
            "ik_subsample_size": cfg.ik_subsample_size,
            "ik_temperature": cfg.ik_temperature,
        },
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_METHOD_COLORS = {
    "euclidean": "#4C78A8",
    "gaussian": "#F58518",
    "mahalanobis": "#54A24B",
    "adaptive_gaussian": "#E45756",
    "ik": "#9467BD",
    "one_step_dynamics": "#8C564B",
    "temporal_distance": "#E377C2",
}

_ORDERED_METHODS = list(_METHOD_COLORS.keys())


def plot_relabel_bars(
    summary_rows: list[dict[str, Any]],
    figures_dir: str,
    dataset_slug_str: str,
    gt_type: str = "reach_prob",
) -> str:
    """
    3-panel bar chart: Spearman | NDCG@k | Goal Precision@k.
    One figure per dataset × GT type combination.
    """
    rows = [r for r in summary_rows if r["ground_truth_type"] == gt_type]
    if not rows:
        return ""

    # Order methods consistently
    ordered = sorted(rows, key=lambda r: _ORDERED_METHODS.index(r["method"])
                     if r["method"] in _ORDERED_METHODS else 99)
    methods = [r["method"] for r in ordered]
    colors = [_METHOD_COLORS.get(m, "#888888") for m in methods]
    spearman = [float(r["spearman_mean"]) for r in ordered]
    ndcg = [float(r["ndcg_at_k_mean"]) for r in ordered]
    precision = [float(r["goal_precision_at_k_mean"]) for r in ordered]
    top_k = int(ordered[0]["top_k"]) if ordered else 20

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, values, title in zip(
        axes,
        [spearman, ndcg, precision],
        [f"Spearman (H={ordered[0]['horizon']})",
         f"NDCG@{top_k}",
         f"Goal Precision@{top_k}"],
    ):
        ax.bar(methods, values, color=colors)
        ax.set_title(title, fontsize=11)
        ymin = min(min(values) - 0.05, -0.05)
        ax.set_ylim(ymin, max(1.0, max(values) + 0.05))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(0, color="gray", linewidth=0.5)

    dataset_display = dataset_slug_str.replace("_", "/")
    fig.suptitle(f"{dataset_display}  [GT: {gt_type}]", fontsize=12)
    fig.tight_layout()

    ensure_dir(figures_dir)
    out_path = os.path.join(figures_dir, f"{dataset_slug_str}_{gt_type}_relabel_bars.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_gt_scatter(
    gt_matrix: np.ndarray,
    sim_bundle: dict[str, np.ndarray],
    anchor_row_id: int,
    figures_dir: str,
    dataset_slug_str: str,
    gt_type: str = "reach_prob",
) -> str:
    """GT score vs similarity score scatter for one anchor, all methods."""
    methods = list(sim_bundle.keys())
    n = len(methods)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    gt = np.asarray(gt_matrix[anchor_row_id], dtype=np.float64)
    for idx, method in enumerate(methods):
        ax = axes[idx // ncols][idx % ncols]
        scores = np.asarray(sim_bundle[method][anchor_row_id], dtype=np.float64)
        finite = np.isfinite(scores) & np.isfinite(gt)
        if finite.sum() > 1:
            ax.scatter(gt[finite], scores[finite], s=5, alpha=0.4,
                       color=_METHOD_COLORS.get(method, "#888888"))
        ax.set_title(method, fontsize=9)
        ax.set_xlabel(f"GT {gt_type}", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide extra axes
    for idx in range(len(methods), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"{dataset_slug_str} – anchor {anchor_row_id} – {gt_type}", fontsize=10)
    fig.tight_layout()

    ensure_dir(figures_dir)
    out_path = os.path.join(
        figures_dir, f"{dataset_slug_str}_{gt_type}_scatter_anchor{anchor_row_id}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_topk_goals_on_maze(
    anchor_xy: np.ndarray,
    candidate_xy: np.ndarray,
    sim_bundle: dict[str, np.ndarray],
    anchor_row_id: int,
    top_k: int,
    figures_dir: str,
    dataset_slug_str: str,
    maze_spec: MazeSpec | None = None,
) -> str:
    """
    For one anchor state, visualise top-k retrieved goals for each method on the maze.

    Layout: one subplot per method.  Background = all candidate positions (gray),
    top-k retrieved = red circles, anchor = blue star.
    """
    methods = list(sim_bundle.keys())
    n = len(methods)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    anchor_pos = np.asarray(anchor_xy[anchor_row_id], dtype=np.float32)

    for idx, method in enumerate(methods):
        ax = axes[idx // ncols][idx % ncols]
        scores = np.asarray(sim_bundle[method][anchor_row_id], dtype=np.float64)
        finite_mask = np.isfinite(scores)
        top_indices = np.argsort(-scores)[:top_k]

        # Background
        ax.scatter(
            candidate_xy[:, 0], candidate_xy[:, 1],
            s=4, alpha=0.25, color="#aaaaaa", zorder=1,
        )
        # Top-k retrieved
        ax.scatter(
            candidate_xy[top_indices, 0], candidate_xy[top_indices, 1],
            s=30, alpha=0.85, color="#E45756", zorder=3, label=f"top-{top_k}",
        )
        # Anchor
        ax.scatter(
            anchor_pos[0], anchor_pos[1],
            s=120, marker="*", color="#4C78A8", zorder=4, label="anchor",
        )

        ax.set_title(method, fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for idx in range(len(methods), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"{dataset_slug_str} – anchor {anchor_row_id} top-{top_k} goals per method",
        fontsize=10,
    )
    fig.tight_layout()

    ensure_dir(figures_dir)
    out_path = os.path.join(
        figures_dir, f"{dataset_slug_str}_topk_goals_anchor{anchor_row_id}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Table I/O helpers
# ---------------------------------------------------------------------------

SUMMARY_FIELDNAMES = [
    "dataset", "method", "ground_truth_type", "sampling_protocol",
    "positive_mode", "positive_cutoff", "dynamics_backend",
    "horizon", "top_k", "num_anchors",
    "spearman_mean", "pearson_mean", "ndcg_at_k_mean",
    "goal_precision_at_k_mean", "recall_at_k_mean",
    "mean_gt_reachability_mean", "mean_geodesic_dist_mean",
    "diversity_mean", "unique_goal_ratio_mean", "auc_mean", "topk_overlap_mean",
]

PER_ANCHOR_FIELDNAMES = [
    "dataset", "method", "ground_truth_type", "sampling_protocol",
    "positive_mode", "positive_cutoff", "dynamics_backend",
    "horizon", "top_k",
    "anchor_row", "anchor_index", "occurrence_count",
    "spearman", "pearson", "ndcg_at_k",
    "goal_precision_at_k", "recall_at_k",
    "mean_gt_reachability", "mean_geodesic_dist",
    "diversity", "unique_goal_ratio", "auc", "topk_overlap",
]


def save_csv(
    path: str,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# IK Hyperparameter Sweep
# ---------------------------------------------------------------------------

IK_SWEEP_FIELDNAMES = [
    "dataset", "stage", "ik_label",
    "ik_ensemble_size", "ik_subsample_size", "ik_temperature",
    "num_anchors", "num_candidates", "top_k", "horizon",
    "ground_truth_type", "sampling_protocol", "positive_mode",
    "positive_cutoff", "dynamics_backend",
    "spearman_mean", "pearson_mean", "ndcg_at_k_mean",
    "goal_precision_at_k_mean", "recall_at_k_mean",
    "mean_gt_reachability_mean", "mean_geodesic_dist_mean",
    "diversity_mean", "unique_goal_ratio_mean", "auc_mean",
]

BASELINE_SWEEP_FIELDNAMES = [
    "dataset", "stage", "method",
    "num_anchors", "num_candidates", "top_k", "horizon",
    "ground_truth_type", "sampling_protocol", "positive_mode",
    "positive_cutoff", "dynamics_backend",
    "spearman_mean", "pearson_mean", "ndcg_at_k_mean",
    "goal_precision_at_k_mean", "recall_at_k_mean",
    "mean_gt_reachability_mean", "mean_geodesic_dist_mean",
    "diversity_mean", "unique_goal_ratio_mean", "auc_mean",
]


def _fmt_temp(temp: float) -> str:
    """Format a temperature float into a compact label-safe string."""
    s = f"{temp:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _fmt_ik_label(ensemble: int, subsample: int, temp: float) -> str:
    return f"ik_e{ensemble}_s{subsample}_t{_fmt_temp(temp)}"


def run_ik_sweep_for_dataset(
    cfg: KNNRelabelConfig,
    dataset_id: str,
    ensemble_sizes: list[int],
    subsample_sizes: list[int],
    temperatures: list[float],
    stage_label: str = "search",
    gt_type: str = "reach_prob",
    selection_metric: str = "spearman_mean",
    shortlist_ik_labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run an IK hyperparameter sweep for one dataset at the current budget
    defined by ``cfg.num_anchors`` / ``cfg.num_candidates``.

    Parameters
    ----------
    shortlist_ik_labels
        If provided, only evaluate IK configs whose label is in this list.
        Used in Stage 2 to re-evaluate the top-N from Stage 1 at full budget.

    Returns
    -------
    dict with:
      "ik_rows"        – one summary row per IK config
      "baseline_rows"  – one summary row per non-IK method (euclidean, etc.)
      "context"        – the EvaluationContext used (shared)
      "dataset_id"     – str
      "parsed"         – ParsedDataset
    """
    import itertools
    from dataclasses import replace as dc_replace
    from .reachability_alignment import compute_or_load_ik_score_matrix

    reach_cfg = cfg.to_reach_cfg()

    print(f"[ik-sweep/{stage_label}] {dataset_id}: "
          f"anchors={cfg.num_anchors} candidates={cfg.num_candidates}")
    parsed = load_or_parse_dataset(
        dataset_id=dataset_id,
        cache_dir=cfg.cache_dir,
        overwrite_cache=cfg.overwrite_cache,
        minari_datasets_path=cfg.minari_datasets_path,
        seed=cfg.seed,
    )

    context = prepare_evaluation_context(parsed, reach_cfg)
    anchor_xy = parsed.positions[context.anchor_indices]
    candidate_xy = parsed.positions[context.candidate_indices]

    # Baseline scores (computed once, independent of IK params)
    baselines = compute_or_load_baseline_scores(context, reach_cfg)
    baseline_sim_map: dict[str, np.ndarray] = {
        "euclidean": np.asarray(baselines.euclidean, dtype=np.float32),
        "gaussian": np.asarray(baselines.gaussian, dtype=np.float32),
        "mahalanobis": np.asarray(baselines.mahalanobis, dtype=np.float32),
        "adaptive_gaussian": np.asarray(baselines.adaptive_gaussian, dtype=np.float32),
        "one_step_dynamics": np.asarray(baselines.one_step_dynamics, dtype=np.float32),
        "temporal_distance": np.asarray(baselines.temporal_distance, dtype=np.float32),
    }
    if cfg.min_time_gap > 0:
        baseline_sim_map = {
            m: apply_min_time_gap_filter(
                scores=s,
                anchor_global_indices=context.anchor_indices,
                candidate_global_indices=context.candidate_indices,
                episode_ids=parsed.episode_ids,
                timesteps=parsed.timesteps,
                min_time_gap=cfg.min_time_gap,
            )
            for m, s in baseline_sim_map.items()
        }
    if cfg.min_goal_dist > 0.0:
        baseline_sim_map = {
            m: apply_min_goal_dist_filter(s, anchor_xy, candidate_xy, cfg.min_goal_dist)
            for m, s in baseline_sim_map.items()
        }

    # Geodesic GT and ground truth
    is_antmaze = "antmaze" in dataset_id.lower()
    h_goal = cfg.geodesic_h_goal_antmaze if is_antmaze else cfg.geodesic_h_goal_pointmaze
    maze_spec = load_maze_spec(
        dataset_id=dataset_id,
        minari_root=cfg.minari_datasets_path,
        cache_dir=cfg.cache_dir,
    )
    geo_binary, geo_matrix = compute_geodesic_gt(anchor_xy, candidate_xy, maze_spec, h_goal)

    gt_map: dict[str, np.ndarray] = {
        "reach_prob": np.asarray(context.ground_truth.reach_prob, dtype=np.float32),
        "geodesic": geo_binary,
    }
    gt_matrix = gt_map.get(gt_type, gt_map["reach_prob"])
    positive_mode, positive_cutoff = _resolve_positive_cutoff(
        gt_matrix,
        gt_type,
        cfg.reachability_positive_mode,
        cfg.reachability_positive_percentile,
        cfg.reachability_positive_threshold,
    )

    # Shared eval kwargs
    _eval_kw = dict(
        candidate_positions=candidate_xy,
        anchor_positions=anchor_xy,
        geo_matrix=geo_matrix,
        top_k=cfg.top_k,
        anchor_global_indices=context.anchor_indices,
        candidate_global_indices=context.candidate_indices,
        dataset_name=dataset_id,
        ground_truth_type=gt_type,
        occurrence_counts=context.ground_truth.occurrence_counts,
        horizon=cfg.horizon,
        positive_mode=positive_mode,
        positive_cutoff=positive_cutoff,
        sampling_protocol=str(cfg.query_pool_mode),
        dynamics_backend=str(cfg.dynamics_backend),
    )

    # Evaluate non-IK baselines
    baseline_rows: list[dict[str, Any]] = []
    for method_name, sim_matrix in baseline_sim_map.items():
        _, summary = evaluate_relabel_metrics(
            ground_truth_matrix=gt_matrix,
            similarity_scores=sim_matrix,
            method_name=method_name,
            **_eval_kw,
        )
        summary["stage"] = stage_label
        summary["num_anchors"] = int(context.anchor_indices.shape[0])
        summary["num_candidates"] = int(context.candidate_indices.shape[0])
        baseline_rows.append(summary)

    # Build shortlist set for filtering
    shortlist_set: set[str] | None = (
        set(shortlist_ik_labels) if shortlist_ik_labels is not None else None
    )

    # IK sweep
    configs = list(itertools.product(ensemble_sizes, subsample_sizes, temperatures))
    if shortlist_set is not None:
        configs = [
            (e, s, t) for (e, s, t) in configs
            if _fmt_ik_label(e, s, t) in shortlist_set
        ]

    total = len(configs)
    print(f"[ik-sweep/{stage_label}] {total} IK configs to evaluate")

    ik_rows: list[dict[str, Any]] = []
    for idx, (ensemble, subsample, temp) in enumerate(configs):
        label = _fmt_ik_label(ensemble, subsample, temp)
        print(f"[ik-sweep/{stage_label}] {idx+1}/{total}: {label}", end="\r", flush=True)

        # Modified reach_cfg with this ensemble size (required for cache key).
        # Also compute an adaptive batch size so that the intermediate distance
        # matrix (batch × ensemble × subsample) stays under ~1 GB on GPU,
        # preventing CUDA OOM for large subsample sizes.
        #
        #   memory ≈ batch_size × ensemble × subsample × 4 bytes
        #   → batch_size = 1 GB / (ensemble × subsample × 4)
        _target_bytes = int(1.5 * 1024 ** 3)   # 1.5 GB headroom
        _adaptive_batch = max(1, _target_bytes // (ensemble * subsample * 4))
        _adaptive_batch = min(_adaptive_batch, cfg.ik_batch_size)
        sweep_reach_cfg = dc_replace(
            reach_cfg,
            ik_ensemble_size=ensemble,
            ik_batch_size=_adaptive_batch,   # not part of cache key — safe to change
        )

        try:
            ik_scores = compute_or_load_ik_score_matrix(
                context, sweep_reach_cfg, subsample, temp
            )
        except Exception as exc:
            # Fallback: halve batch size once and retry, then give up
            _half_cfg = dc_replace(sweep_reach_cfg, ik_batch_size=max(1, _adaptive_batch // 4))
            try:
                ik_scores = compute_or_load_ik_score_matrix(
                    context, _half_cfg, subsample, temp
                )
            except Exception as exc2:
                _is_oom = "out of memory" in str(exc2).lower() or "OutOfMemoryError" in type(exc2).__name__
                tag = "OOM" if _is_oom else "ERR"
                print(f"\n[ik-sweep/{stage_label}] {tag} skipping {label}: {exc2}", flush=True)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                nan_summary: dict[str, Any] = {
                    k: float("nan") for k in IK_SWEEP_FIELDNAMES
                    if k not in ("dataset", "method", "ik_label",
                                 "ik_ensemble_size", "ik_subsample_size",
                                 "ik_temperature", "stage",
                                 "num_anchors", "num_candidates",
                                 "ground_truth_type", "sampling_protocol",
                                 "positive_mode", "dynamics_backend",
                                 "horizon", "top_k")
                }
                nan_summary.update({
                    "dataset": dataset_id, "method": label, "ik_label": label,
                    "ik_ensemble_size": ensemble, "ik_subsample_size": subsample,
                    "ik_temperature": temp, "stage": stage_label,
                    "num_anchors": int(context.anchor_indices.shape[0]),
                    "num_candidates": int(context.candidate_indices.shape[0]),
                    "ground_truth_type": gt_type,
                    "sampling_protocol": str(cfg.query_pool_mode),
                    "positive_mode": positive_mode,
                    "positive_cutoff": positive_cutoff,
                    "dynamics_backend": str(cfg.dynamics_backend),
                    "horizon": cfg.horizon, "top_k": cfg.top_k,
                    "error": tag,
                })
                ik_rows.append(nan_summary)
                continue

        # Free CUDA memory after each IK computation to avoid fragmentation
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Apply filters
        if cfg.min_time_gap > 0:
            ik_scores = apply_min_time_gap_filter(
                scores=ik_scores,
                anchor_global_indices=context.anchor_indices,
                candidate_global_indices=context.candidate_indices,
                episode_ids=parsed.episode_ids,
                timesteps=parsed.timesteps,
                min_time_gap=cfg.min_time_gap,
            )
        if cfg.min_goal_dist > 0.0:
            ik_scores = apply_min_goal_dist_filter(ik_scores, anchor_xy, candidate_xy, cfg.min_goal_dist)

        _, summary = evaluate_relabel_metrics(
            ground_truth_matrix=gt_matrix,
            similarity_scores=ik_scores,
            method_name=label,
            **_eval_kw,
        )
        summary["stage"] = stage_label
        summary["ik_label"] = label
        summary["ik_ensemble_size"] = ensemble
        summary["ik_subsample_size"] = subsample
        summary["ik_temperature"] = temp
        summary["num_anchors"] = int(context.anchor_indices.shape[0])
        summary["num_candidates"] = int(context.candidate_indices.shape[0])
        ik_rows.append(summary)

    print()  # newline after \r progress
    return {
        "dataset_id": dataset_id,
        "ik_rows": ik_rows,
        "baseline_rows": baseline_rows,
        "context": context,
        "parsed": parsed,
        "gt_type": gt_type,
        "selection_metric": selection_metric,
    }


def select_top_ik_configs(
    ik_rows: list[dict[str, Any]],
    top_n: int,
    selection_metric: str = "spearman_mean",
) -> list[str]:
    """Return the top-N IK labels sorted by selection_metric (descending)."""
    import math
    sorted_rows = sorted(
        (r for r in ik_rows if not math.isnan(float(r.get(selection_metric, float("nan"))))),
        key=lambda r: float(r.get(selection_metric, 0.0)),
        reverse=True,
    )
    return [str(r["ik_label"]) for r in sorted_rows[:top_n]]


def plot_ik_sweep_heatmap(
    ik_rows: list[dict[str, Any]],
    dataset_id: str,
    figures_dir: str,
    ensemble_size: int,
    metric: str = "spearman_mean",
    stage_label: str = "search",
) -> str:
    """
    Heatmap of (subsample_size × temperature) for one ensemble_size.
    """
    rows = [r for r in ik_rows if int(r["ik_ensemble_size"]) == ensemble_size]
    if not rows:
        return ""

    subsamples = sorted({int(r["ik_subsample_size"]) for r in rows})
    temps = sorted({float(r["ik_temperature"]) for r in rows})
    sub_idx = {v: i for i, v in enumerate(subsamples)}
    temp_idx = {v: i for i, v in enumerate(temps)}

    matrix = np.full((len(subsamples), len(temps)), np.nan, dtype=np.float64)
    for r in rows:
        i = sub_idx[int(r["ik_subsample_size"])]
        j = temp_idx[float(r["ik_temperature"])]
        matrix[i, j] = float(r.get(metric, np.nan))

    fig, ax = plt.subplots(figsize=(max(10, len(temps) * 0.8), max(6, len(subsamples) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(np.arange(len(temps)))
    ax.set_xticklabels([str(t) for t in temps], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(subsamples)))
    ax.set_yticklabels([str(s) for s in subsamples], fontsize=8)
    ax.set_xlabel("Temperature", fontsize=10)
    ax.set_ylabel("Subsample Size", fontsize=10)
    ds_slug_str = dataset_slug(dataset_id)
    ax.set_title(
        f"{dataset_id}\nIK e={ensemble_size} – {metric} ({stage_label})", fontsize=10
    )
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()

    ensure_dir(figures_dir)
    out_path = os.path.join(
        figures_dir,
        f"{ds_slug_str}_ik_heatmap_e{ensemble_size}_{stage_label}.png",
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ik_sweep_comparison_bars(
    final_ik_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    dataset_id: str,
    figures_dir: str,
    top_n: int = 5,
    metric: str = "spearman_mean",
) -> str:
    """
    Bar chart comparing top-N IK configs vs. all baselines on a single metric.
    """
    top_ik = sorted(final_ik_rows, key=lambda r: float(r.get(metric, 0.0)), reverse=True)[:top_n]

    all_rows = list(top_ik) + list(baseline_rows)
    labels = [str(r.get("ik_label", r.get("method", "?"))) for r in all_rows]
    values = [float(r.get(metric, 0.0)) for r in all_rows]
    colors = (
        ["#9467BD"] * len(top_ik) +
        [_METHOD_COLORS.get(str(r.get("method", "")), "#888888") for r in baseline_rows]
    )

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_title(f"{dataset_id}\nTop-{top_n} IK vs Baselines – {metric}", fontsize=10)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()

    ensure_dir(figures_dir)
    out_path = os.path.join(
        figures_dir,
        f"{dataset_slug(dataset_id)}_ik_comparison_bars.png",
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_ik_sweep_report(
    cfg: KNNRelabelConfig,
    stage1_results: list[dict[str, Any]],
    stage2_results: list[dict[str, Any]],
    report_path: str,
    top_n: int = 5,
    selection_metric: str = "spearman_mean",
) -> None:
    """Write a markdown report summarising the two-stage IK sweep."""
    lines = [
        "# IK Hyperparameter Sweep Report – kNN Relabeling Benchmark",
        "",
        "## Protocol",
        "",
        f"- Stage 1 (search): {cfg.num_anchors} anchors, {cfg.num_candidates} candidates — all configs",
        f"- Stage 2 (final): full budget — top-{top_n} from Stage 1",
        f"- Selection metric: `{selection_metric}` on `reach_prob` GT",
        f"- min_time_gap: {cfg.min_time_gap}",
        f"- Horizon: {cfg.horizon}",
        f"- top_k: {cfg.top_k}",
        "",
        "## Search Space",
        "",
    ]

    if stage1_results:
        ex = stage1_results[0]
        ik_rows = ex.get("ik_rows", [])
        if ik_rows:
            ensembles = sorted({int(r["ik_ensemble_size"]) for r in ik_rows})
            subsamples = sorted({int(r["ik_subsample_size"]) for r in ik_rows})
            temps = sorted({float(r["ik_temperature"]) for r in ik_rows})
            lines += [
                f"- ensemble_size: {ensembles}",
                f"- subsample_size: {subsamples}",
                f"- temperature: {temps}",
                f"- Total IK configs: {len(ik_rows)}",
                "",
            ]

    for stage_idx, (stage_label, results) in enumerate(
        [("Stage 1 – Search", stage1_results), ("Stage 2 – Final", stage2_results)]
    ):
        if not results:
            continue
        lines += [f"## {stage_label}", ""]
        for res in results:
            ds = res["dataset_id"]
            ik_rows = res["ik_rows"]
            baseline_rows = res["baseline_rows"]
            if not ik_rows:
                continue
            top_rows = sorted(ik_rows, key=lambda r: float(r.get(selection_metric, 0.0)), reverse=True)[:top_n]
            lines += [f"### {ds}", ""]
            # IK top-N table
            lines += [
                f"**Top-{top_n} IK configs (sorted by {selection_metric})**",
                "",
                "| Rank | IK Label | Ensemble | Subsample | Temperature"
                " | Spearman | NDCG@k | Prec@k | MeanGT |",
                "|------|----------|----------|-----------|-------------|"
                "----------|--------|--------|--------|",
            ]
            for rank, row in enumerate(top_rows, 1):
                lines.append(
                    f"| {rank} | {row.get('ik_label','')} "
                    f"| {row.get('ik_ensemble_size','')} "
                    f"| {row.get('ik_subsample_size','')} "
                    f"| {row.get('ik_temperature','')} "
                    f"| {float(row.get('spearman_mean',0)):.4f} "
                    f"| {float(row.get('ndcg_at_k_mean',0)):.4f} "
                    f"| {float(row.get('goal_precision_at_k_mean',0)):.4f} "
                    f"| {float(row.get('mean_gt_reachability_mean',0)):.4f} |"
                )
            lines.append("")
            # Baseline comparison
            if baseline_rows:
                lines += [
                    "**Baseline methods (for comparison)**",
                    "",
                    "| Method | Spearman | NDCG@k | Prec@k | MeanGT | MeanGeo | Diversity | Unique@k |",
                    "|--------|----------|--------|--------|--------|---------|-----------|----------|",
                ]
                for row in sorted(baseline_rows,
                                  key=lambda r: float(r.get("spearman_mean", 0)), reverse=True):
                    geo = float(row.get("mean_geodesic_dist_mean", float("inf")))
                    geo_str = f"{geo:.3f}" if geo < 1e18 else "inf"
                    lines.append(
                        f"| {row.get('method',row.get('dataset','?'))} "
                        f"| {float(row.get('spearman_mean',0)):.4f} "
                        f"| {float(row.get('ndcg_at_k_mean',0)):.4f} "
                        f"| {float(row.get('goal_precision_at_k_mean',0)):.4f} "
                        f"| {float(row.get('mean_gt_reachability_mean',0)):.4f} "
                        f"| {geo_str} "
                        f"| {float(row.get('diversity_mean',0)):.4f} "
                        f"| {float(row.get('unique_goal_ratio_mean',0)):.4f} |"
                    )
                lines.append("")

    ensure_dir(os.path.dirname(report_path))
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[ik-sweep] Report written: {report_path}")


def compute_overall_summary(all_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average summary rows across all datasets for each (method, gt_type) pair."""
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_summary_rows:
        groups[(str(row["method"]), str(row["ground_truth_type"]))].append(row)

    overall: list[dict[str, Any]] = []
    num_cols = [
        "spearman_mean", "pearson_mean", "ndcg_at_k_mean",
        "goal_precision_at_k_mean", "recall_at_k_mean",
        "mean_gt_reachability_mean", "diversity_mean", "unique_goal_ratio_mean",
        "auc_mean", "topk_overlap_mean",
    ]
    for (method, gt_type), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "dataset": "overall",
            "method": method,
            "ground_truth_type": gt_type,
            "sampling_protocol": group[0].get("sampling_protocol", "mixed"),
            "positive_mode": group[0].get("positive_mode", "mixed"),
            "dynamics_backend": group[0].get("dynamics_backend", "mixed"),
            "horizon": group[0]["horizon"],
            "top_k": group[0]["top_k"],
            "num_anchors": sum(int(r["num_anchors"]) for r in group),
            "num_datasets": len(group),
        }
        for col in num_cols:
            vals = [float(r[col]) for r in group if np.isfinite(float(r[col]))]
            row[col] = float(np.mean(vals)) if vals else 0.0
        # geodesic may be inf
        geo_vals = [float(r["mean_geodesic_dist_mean"]) for r in group if float(r["mean_geodesic_dist_mean"]) < 1e18]
        row["mean_geodesic_dist_mean"] = float(np.mean(geo_vals)) if geo_vals else float("inf")
        cutoff_vals = [float(r.get("positive_cutoff", float("nan"))) for r in group if np.isfinite(float(r.get("positive_cutoff", float("nan"))))]
        row["positive_cutoff"] = float(np.mean(cutoff_vals)) if cutoff_vals else float("nan")
        overall.append(row)

    return overall


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_relabel_report(
    cfg: KNNRelabelConfig,
    dataset_results: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    figure_paths: dict[str, list[str]],
    report_path: str,
) -> None:
    lines = [
        "# kNN-based Relabeling Benchmark Report",
        "",
        "## 1. Protocol",
        "",
        "**Task:** For each query state `s_t` from the offline buffer, retrieve the top-k",
        "candidate goals using each of 7 similarity functions, then measure whether those",
        "retrieved goals are truly reachable using an *independent* ground truth.",
        "",
        "**Formal definition:**",
        "- Query states (anchors): sampled from a planning-aligned stride node pool by default",
        "- Candidate goal pool: stride-sampled nodes from the same offline buffer by default",
        "- Similarity functions: Euclidean, Gaussian, Mahalanobis, Adaptive Gaussian,",
        "  Isolation Kernel (IK), One-step Dynamics, Strict Temporal Distance",
        "",
        "## 2. Ground Truth",
        "",
        "**Primary – H-step empirical reachability (`reach_prob`):**",
        "For anchor `s_t` and candidate `y`, find all occurrences of `s_t` in the buffer",
        "and check whether `y` appears within the next H steps.  The reach probability is",
        "the fraction of occurrences that hit `y` within H steps.",
        "",
        "**Secondary – Geodesic-distance threshold (`geodesic`):**",
        "Binary label `1{d_geo(s_t, y) ≤ H_goal}` based on maze shortest-path distance.",
        "",
        "## 3. Positive Label Protocol",
        "",
        f"- `reachability_positive_mode = {cfg.reachability_positive_mode}`",
        f"- `reachability_positive_percentile = {cfg.reachability_positive_percentile}`",
        f"- `reachability_positive_threshold = {cfg.reachability_positive_threshold}`",
        "- Ranking metrics always use continuous `reach_prob`; binary metrics use the configured cutoff.",
        "",
        "## 4. Trivial-Goal Filters",
        "",
        f"- `min_time_gap = {cfg.min_time_gap}`: exclude candidates from the same episode",
        "  whose timestep differs by less than this value.",
        f"- `min_goal_dist = {cfg.min_goal_dist}`: exclude candidates within this Euclidean",
        "  distance from the anchor (0 = disabled).",
        "",
        "## 5. Datasets & Hyperparameters",
        "",
        f"- Datasets: {', '.join(cfg.datasets)}",
        f"- H (horizon): {cfg.horizon}",
        f"- num_anchors: {cfg.num_anchors}",
        f"- num_candidates: {cfg.num_candidates}",
        f"- top_k: {cfg.top_k}",
        f"- candidate_pool_mode/query_pool_mode: {cfg.candidate_pool_mode} / {cfg.query_pool_mode}",
        f"- node_stride_pointmaze/antmaze: {cfg.node_stride_pointmaze} / {cfg.node_stride_antmaze}",
        f"- metric_state_variant: {cfg.metric_state_variant}",
        f"- dynamics_backend: {cfg.dynamics_backend}",
        f"- dynamics_local_knn_m: {cfg.dynamics_local_knn_m}",
        f"- dynamics_local_distance_metric: {cfg.dynamics_local_distance_metric}",
        f"- dynamics_state_variant: {cfg.dynamics_state_variant or cfg.metric_state_variant}",
        f"- IK ensemble/subsample/temperature: {cfg.ik_ensemble_size} / {cfg.ik_subsample_size} / {cfg.ik_temperature}",
        f"- min_time_gap: {cfg.min_time_gap}",
        f"- geodesic H_goal (pointmaze/antmaze): {cfg.geodesic_h_goal_pointmaze} / {cfg.geodesic_h_goal_antmaze}",
        "",
        "## 6. Results",
        "",
    ]

    # Overall table (reach_prob GT only, for brevity)
    reach_overall = [r for r in overall_rows if r["ground_truth_type"] == "reach_prob"]
    if reach_overall:
        lines += ["### 6.1 Overall Summary (reach_prob GT, averaged across datasets)", ""]
        header = "| Method | Spearman | NDCG@k | Prec@k | Recall@k | MeanGT | MeanGeo | Diversity | Unique@k |"
        sep    = "|--------|----------|--------|--------|----------|--------|---------|-----------|----------|"
        lines += [header, sep]
        for row in sorted(reach_overall, key=lambda r: -float(r["spearman_mean"])):
            lines.append(
                f"| {row['method']} "
                f"| {row['spearman_mean']:.4f} "
                f"| {row['ndcg_at_k_mean']:.4f} "
                f"| {row['goal_precision_at_k_mean']:.4f} "
                f"| {row['recall_at_k_mean']:.4f} "
                f"| {row['mean_gt_reachability_mean']:.4f} "
                f"| {row['mean_geodesic_dist_mean']:.2f} "
                f"| {row['diversity_mean']:.4f} "
                f"| {row['unique_goal_ratio_mean']:.4f} |"
            )
        lines.append("")

    # Per-dataset tables
    for result in dataset_results:
        ds = result["dataset"]
        lines += [f"### 6.2 {ds}", ""]
        rows_reach = [r for r in result["summary_rows"] if r["ground_truth_type"] == "reach_prob"]
        if rows_reach:
            lines += ["**GT = reach_prob**", ""]
            lines += [
                "| Method | Spearman | NDCG@k | Prec@k | Recall@k | MeanGT | MeanGeo | Diversity | Unique@k |",
                "|--------|----------|--------|--------|----------|--------|---------|-----------|----------|",
            ]
            for row in sorted(rows_reach, key=lambda r: -float(r["spearman_mean"])):
                lines.append(
                    f"| {row['method']} "
                    f"| {row['spearman_mean']:.4f} "
                    f"| {row['ndcg_at_k_mean']:.4f} "
                    f"| {row['goal_precision_at_k_mean']:.4f} "
                    f"| {row['recall_at_k_mean']:.4f} "
                    f"| {row['mean_gt_reachability_mean']:.4f} "
                    f"| {row['mean_geodesic_dist_mean']:.2f} "
                    f"| {row['diversity_mean']:.4f} "
                    f"| {row['unique_goal_ratio_mean']:.4f} |"
                )
            lines.append("")

    lines += [
        "## 7. Limitations",
        "",
        "- Ground truth is based on offline data distribution; rarely-visited regions have noisy GT.",
        "- IK hyperparameters are fixed per dataset based on prior planning sweep; may not be optimal for relabeling.",
        "- This benchmark is benchmark-only by design and does not include relabel-then-learn training.",
        "- The geodesic GT ignores walls for antmaze (uses cell-graph BFS), which is an approximation.",
        "- `temporal_distance` is intentionally strict: only same-trajectory future states are valid candidates, so its coverage is conservative by design.",
        "",
    ]

    ensure_dir(os.path.dirname(report_path))
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[relabel] Report written to {report_path}")
