from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from core.isolation_kernel import SoftIsolationKernel
from .fitted_baselines import (
    AdaptiveGaussianMetric,
    MahalanobisMetric,
    OneStepDynamicsMetric,
)


@dataclass
class SimilarityBundle:
    euclidean: np.ndarray | None = None
    gaussian: np.ndarray | None = None
    mahalanobis: np.ndarray | None = None
    adaptive_gaussian: np.ndarray | None = None
    temporal_distance: np.ndarray | None = None
    first_hit_temporal: np.ndarray | None = None
    one_step_dynamics: np.ndarray | None = None
    replay_temporal: np.ndarray | None = None
    oracle_temporal: np.ndarray | None = None
    ik: np.ndarray | None = None


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    value = float(stats.pearsonr(x, y)[0])
    if not np.isfinite(value):
        return 0.0
    return value


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    value = float(stats.spearmanr(x, y)[0])
    if not np.isfinite(value):
        return 0.0
    return value


def compute_euclidean_scores(anchor_positions: np.ndarray, candidate_positions: np.ndarray) -> np.ndarray:
    distances = cdist(anchor_positions, candidate_positions, metric="euclidean")
    return (-distances).astype(np.float32)


def _resolve_sigma(
    distances: np.ndarray,
    sigma_mode: str,
    sigma_value: float | None,
    fallback_sigma: float,
) -> np.ndarray:
    if sigma_mode == "fixed":
        if sigma_value is None or sigma_value <= 0:
            raise ValueError("fixed Gaussian sigma requires --gk_sigma > 0")
        return np.full(distances.shape[0], float(sigma_value), dtype=np.float64)

    sigmas = np.zeros(distances.shape[0], dtype=np.float64)
    for row_id in range(distances.shape[0]):
        row = distances[row_id]
        positive = row[row > 1e-12]
        if positive.size == 0:
            sigmas[row_id] = fallback_sigma
        else:
            sigma = float(np.median(positive))
            sigmas[row_id] = sigma if sigma > 0 else fallback_sigma
    return sigmas


def compute_gaussian_scores(
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    sigma_mode: str,
    sigma_value: float | None,
    fallback_sigma: float,
) -> np.ndarray:
    distances = cdist(anchor_positions, candidate_positions, metric="euclidean")
    sigmas = _resolve_sigma(
        distances=distances,
        sigma_mode=sigma_mode,
        sigma_value=sigma_value,
        fallback_sigma=max(float(fallback_sigma), 1e-6),
    )
    variances = np.maximum(sigmas[:, None] ** 2, 1e-12)
    scores = np.exp(-(distances ** 2) / (2.0 * variances))
    return scores.astype(np.float32)


def compute_mahalanobis_distances(
    fit_positions: np.ndarray,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    covariance_estimator: str = "ledoitwolf",
    implementation: str = "whitening",
    eps: float = 1e-6,
) -> np.ndarray:
    metric = MahalanobisMetric.fit(
        fit_positions,
        covariance_estimator=covariance_estimator,
        eps=eps,
    )
    return metric.pairwise_distance(
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        implementation=implementation,
    )


def compute_mahalanobis_scores(
    fit_positions: np.ndarray,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    covariance_estimator: str = "ledoitwolf",
    implementation: str = "whitening",
    eps: float = 1e-6,
) -> np.ndarray:
    metric = MahalanobisMetric.fit(
        fit_positions,
        covariance_estimator=covariance_estimator,
        eps=eps,
    )
    return metric.pairwise_score(
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        implementation=implementation,
    )


def compute_adaptive_gaussian_kernel(
    fit_positions: np.ndarray,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    k: int = 10,
    eps: float = 1e-6,
    anchor_train_indices: np.ndarray | None = None,
    candidate_train_indices: np.ndarray | None = None,
) -> np.ndarray:
    metric = AdaptiveGaussianMetric.fit(fit_positions, k=k, eps=eps)
    return metric.pairwise_kernel(
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        anchor_train_indices=anchor_train_indices,
        candidate_train_indices=candidate_train_indices,
    )


def compute_adaptive_gaussian_distances(
    fit_positions: np.ndarray,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    k: int = 10,
    eps: float = 1e-6,
    anchor_train_indices: np.ndarray | None = None,
    candidate_train_indices: np.ndarray | None = None,
    mode: str = "one_minus_kernel",
) -> np.ndarray:
    metric = AdaptiveGaussianMetric.fit(fit_positions, k=k, eps=eps)
    return metric.pairwise_distance(
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        anchor_train_indices=anchor_train_indices,
        candidate_train_indices=candidate_train_indices,
        mode=mode,
    )


def compute_adaptive_gaussian_scores(
    fit_positions: np.ndarray,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    k: int = 10,
    eps: float = 1e-6,
    anchor_train_indices: np.ndarray | None = None,
    candidate_train_indices: np.ndarray | None = None,
    output: str = "kernel",
) -> np.ndarray:
    metric = AdaptiveGaussianMetric.fit(fit_positions, k=k, eps=eps)
    output_key = str(output).strip().lower()
    if output_key == "distance":
        return (1.0 - metric.pairwise_kernel(
            anchor_positions=anchor_positions,
            candidate_positions=candidate_positions,
            anchor_train_indices=anchor_train_indices,
            candidate_train_indices=candidate_train_indices,
        )).astype(np.float32)
    return metric.pairwise_kernel(
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        anchor_train_indices=anchor_train_indices,
        candidate_train_indices=candidate_train_indices,
    )


def compute_one_step_dynamics_distances(
    fit_states: np.ndarray,
    fit_next_states: np.ndarray,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    backend: str = "local_knn_nextstate",
    num_bins: int = 64,
    distance_metric: str = "jsd",
    alpha: float = 1e-3,
    min_count: int = 5,
    seed: int = 0,
    eps: float = 1e-6,
    local_knn_m: int = 20,
    local_distance_metric: str = "euclidean",
) -> np.ndarray:
    backend_key = str(backend).strip().lower()
    if backend_key == "local_knn_nextstate":
        return _compute_local_knn_one_step_distances(
            fit_states=fit_states,
            fit_next_states=fit_next_states,
            anchor_positions=anchor_positions,
            candidate_positions=candidate_positions,
            local_knn_m=local_knn_m,
            local_distance_metric=local_distance_metric,
        )
    metric = OneStepDynamicsMetric.fit(
        fit_states,
        fit_next_states,
        backend=backend,
        num_bins=num_bins,
        distance_metric=distance_metric,
        alpha=alpha,
        min_count=min_count,
        seed=seed,
        eps=eps,
    )
    return metric.pairwise_distance(
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        distance_metric=distance_metric,
    )


def compute_one_step_dynamics_scores(
    fit_states: np.ndarray,
    fit_next_states: np.ndarray,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    backend: str = "local_knn_nextstate",
    num_bins: int = 64,
    distance_metric: str = "jsd",
    alpha: float = 1e-3,
    min_count: int = 5,
    seed: int = 0,
    eps: float = 1e-6,
    local_knn_m: int = 20,
    local_distance_metric: str = "euclidean",
) -> np.ndarray:
    backend_key = str(backend).strip().lower()
    if backend_key == "local_knn_nextstate":
        distances = _compute_local_knn_one_step_distances(
            fit_states=fit_states,
            fit_next_states=fit_next_states,
            anchor_positions=anchor_positions,
            candidate_positions=candidate_positions,
            local_knn_m=local_knn_m,
            local_distance_metric=local_distance_metric,
        )
        return (-np.asarray(distances, dtype=np.float64)).astype(np.float32)
    metric = OneStepDynamicsMetric.fit(
        fit_states,
        fit_next_states,
        backend=backend,
        num_bins=num_bins,
        distance_metric=distance_metric,
        alpha=alpha,
        min_count=min_count,
        seed=seed,
        eps=eps,
    )
    return metric.pairwise_score(
        anchor_positions=anchor_positions,
        candidate_positions=candidate_positions,
        distance_metric=distance_metric,
    )


def _compute_local_knn_one_step_distances(
    fit_states: np.ndarray,
    fit_next_states: np.ndarray,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    local_knn_m: int = 20,
    local_distance_metric: str = "euclidean",
) -> np.ndarray:
    fit_state_values = np.asarray(fit_states, dtype=np.float32)
    fit_next_values = np.asarray(fit_next_states, dtype=np.float32)
    anchor_values = np.asarray(anchor_positions, dtype=np.float32)
    candidate_values = np.asarray(candidate_positions, dtype=np.float32)

    if fit_state_values.ndim != 2 or fit_next_values.ndim != 2:
        raise ValueError("One-step dynamics expects 2D fit state arrays.")
    if fit_state_values.shape != fit_next_values.shape:
        raise ValueError(
            f"fit_states and fit_next_states must match, got {fit_state_values.shape} vs {fit_next_values.shape}"
        )
    if fit_state_values.shape[0] == 0:
        return np.full(
            (anchor_values.shape[0], candidate_values.shape[0]),
            np.inf,
            dtype=np.float32,
        )

    knn_k = min(max(int(local_knn_m), 1), int(fit_state_values.shape[0]))
    tree = cKDTree(fit_state_values)
    _, nn_indices = tree.query(anchor_values, k=knn_k)
    if np.ndim(nn_indices) == 1:
        nn_indices = np.asarray(nn_indices, dtype=np.int64)[:, None]
    successor_clouds = fit_next_values[np.asarray(nn_indices, dtype=np.int64)]

    metric_name = str(local_distance_metric).strip().lower()
    if metric_name == "euclidean":
        scipy_metric = "euclidean"
    elif metric_name == "sqeuclidean":
        scipy_metric = "sqeuclidean"
    else:
        raise ValueError(
            f"Unsupported local one-step distance metric: {local_distance_metric}"
        )

    distances = np.empty(
        (anchor_values.shape[0], candidate_values.shape[0]),
        dtype=np.float32,
    )
    for anchor_row in range(anchor_values.shape[0]):
        pairwise = cdist(
            successor_clouds[anchor_row],
            candidate_values,
            metric=scipy_metric,
        )
        distances[anchor_row] = np.min(pairwise, axis=0).astype(np.float32)
    return distances


def compute_temporal_distance_scores(
    anchor_global_indices: np.ndarray,
    candidate_global_indices: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
) -> np.ndarray:
    """Strict temporal-distance score for relabeling.

    A candidate is valid only when it appears later in the same trajectory as
    the anchor. The temporal distance is ``t_y - t_x``. Invalid pairs receive a
    score of ``0``; valid pairs are mapped with ``1 / (1 + distance)`` so that
    larger scores always mean "closer / better".
    """
    anchor_indices = np.asarray(anchor_global_indices, dtype=np.int64)
    candidate_indices = np.asarray(candidate_global_indices, dtype=np.int64)
    anchor_ep = np.asarray(episode_ids[anchor_indices], dtype=np.int64)
    candidate_ep = np.asarray(episode_ids[candidate_indices], dtype=np.int64)
    anchor_ts = np.asarray(timesteps[anchor_indices], dtype=np.int64)
    candidate_ts = np.asarray(timesteps[candidate_indices], dtype=np.int64)

    same_episode = anchor_ep[:, None] == candidate_ep[None, :]
    delta_t = candidate_ts[None, :] - anchor_ts[:, None]
    valid = same_episode & (delta_t > 0)

    scores = np.zeros((anchor_indices.shape[0], candidate_indices.shape[0]), dtype=np.float32)
    if np.any(valid):
        scores[valid] = (1.0 / (1.0 + delta_t[valid].astype(np.float64))).astype(np.float32)
    return scores


def fit_soft_isolation_kernel(
    fit_positions: np.ndarray,
    *,
    ensemble_size: int,
    subsample_size: int,
    temperature: float,
    device: str,
) -> tuple[SoftIsolationKernel, torch.device]:
    """Fit the repository SoftIsolationKernel on training positions only."""
    torch_device = torch.device(device)
    with torch.no_grad():
        fit_tensor = torch.as_tensor(fit_positions, dtype=torch.float32, device=torch_device)
        kernel = SoftIsolationKernel(
            input_dim=fit_tensor.shape[1],
            ensemble_size=ensemble_size,
            subsample_size=subsample_size,
            temperature=temperature,
            device=str(torch_device),
        ).to(torch_device)
        kernel.fit(fit_tensor)
    return kernel, torch_device


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return ("out of memory" in text) or ("cublas_status_alloc_failed" in text) or ("OutOfMemoryError" in type(exc).__name__)


def _encode_in_batches(
    kernel: SoftIsolationKernel,
    data: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, data.shape[0], batch_size):
        end = min(start + batch_size, data.shape[0])
        chunk = kernel(data[start:end]) / math.sqrt(kernel.ensemble_size)
        chunks.append(chunk)
    return torch.cat(chunks, dim=0)


def encode_ik_features(
    kernel: SoftIsolationKernel,
    values: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 4096,
    output: Literal["numpy", "torch"] = "numpy",
) -> np.ndarray | torch.Tensor:
    """Encode values with a fitted SoftIsolationKernel.

    Returned features are normalized by ``sqrt(ensemble_size)`` so their dot
    product matches the pointwise IK similarity used throughout the repository.
    """
    if isinstance(values, torch.Tensor):
        value_tensor = values.to(kernel.anchors.device, dtype=torch.float32)
    else:
        value_tensor = torch.as_tensor(values, dtype=torch.float32, device=kernel.anchors.device)
    features = _encode_in_batches(kernel, value_tensor, batch_size=batch_size)
    if output == "torch":
        return features
    return features.detach().cpu().numpy().astype(np.float32)


def compute_ik_scores(
    fit_positions: np.ndarray,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    ensemble_size: int,
    subsample_size: int,
    temperature: float,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    effective_batch_size = max(int(batch_size), 1)
    while True:
        try:
            return _compute_ik_scores_once(
                fit_positions=fit_positions,
                anchor_positions=anchor_positions,
                candidate_positions=candidate_positions,
                ensemble_size=ensemble_size,
                subsample_size=subsample_size,
                temperature=temperature,
                device=device,
                batch_size=effective_batch_size,
            )
        except Exception as exc:
            if effective_batch_size <= 1 or not _is_oom_error(exc):
                raise
            effective_batch_size = max(1, effective_batch_size // 2)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _compute_ik_scores_once(
    fit_positions: np.ndarray,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    ensemble_size: int,
    subsample_size: int,
    temperature: float,
    device: str,
    batch_size: int,
) -> np.ndarray:
    kernel, torch_device = fit_soft_isolation_kernel(
        fit_positions,
        ensemble_size=ensemble_size,
        subsample_size=subsample_size,
        temperature=temperature,
        device=device,
    )
    with torch.no_grad():
        anchor_tensor = torch.as_tensor(anchor_positions, dtype=torch.float32, device=torch_device)
        candidate_tensor = torch.as_tensor(candidate_positions, dtype=torch.float32, device=torch_device)
        num_anchors = int(anchor_tensor.shape[0])
        num_candidates = int(candidate_tensor.shape[0])
        scores = np.empty((num_anchors, num_candidates), dtype=np.float32)

        for a_start in range(0, num_anchors, batch_size):
            a_end = min(a_start + batch_size, num_anchors)
            anchor_features = _encode_in_batches(kernel, anchor_tensor[a_start:a_end], batch_size=batch_size)
            for c_start in range(0, num_candidates, batch_size):
                c_end = min(c_start + batch_size, num_candidates)
                candidate_features = _encode_in_batches(kernel, candidate_tensor[c_start:c_end], batch_size=batch_size)
                block = torch.matmul(anchor_features, candidate_features.T)
                scores[a_start:a_end, c_start:c_end] = block.detach().cpu().numpy().astype(np.float32)
                del candidate_features, block
            del anchor_features

        return scores


def compute_replay_temporal_scores(
    anchor_occurrence_lists: list[np.ndarray],
    candidate_positions: np.ndarray,
    positions: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
    episode_offsets: np.ndarray,
    episode_lengths: np.ndarray,
    match_radius: float,
    temporal_window: int,
) -> np.ndarray:
    candidate_tree = cKDTree(candidate_positions)
    replay_tree = cKDTree(positions)
    candidate_visit_counts = np.asarray(
        [len(replay_tree.query_ball_point(point, r=match_radius)) for point in candidate_positions],
        dtype=np.float64,
    )
    candidate_visit_counts = np.maximum(candidate_visit_counts, 1.0)

    rows = []
    for occurrence_indices in anchor_occurrence_lists:
        if occurrence_indices.size == 0:
            rows.append(np.zeros(candidate_positions.shape[0], dtype=np.float32))
            continue

        row = np.zeros(candidate_positions.shape[0], dtype=np.float64)
        anchor_visits = float(max(occurrence_indices.size, 1))
        for global_index in occurrence_indices:
            episode_id = int(episode_ids[global_index])
            timestep = int(timesteps[global_index])
            episode_start = int(episode_offsets[episode_id])
            remaining_steps = int(episode_lengths[episode_id] - timestep - 1)
            max_tau = min(temporal_window, remaining_steps)

            for tau in range(1, max_tau + 1):
                future_global_index = episode_start + timestep + tau
                future_position = positions[future_global_index]
                hit_candidate_ids = candidate_tree.query_ball_point(future_position, r=match_radius)
                if not hit_candidate_ids:
                    continue
                weight = 1.0 / float(tau)
                for candidate_id in hit_candidate_ids:
                    row[candidate_id] += weight

        row = row / np.sqrt(anchor_visits * candidate_visit_counts)
        max_value = float(np.max(row))
        if max_value > 0:
            row = row / max_value
        rows.append(row.astype(np.float32))

    return np.stack(rows, axis=0)


def compute_first_hit_temporal_distances(
    anchor_occurrence_lists: list[np.ndarray],
    candidate_positions: np.ndarray,
    positions: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
    episode_offsets: np.ndarray,
    episode_lengths: np.ndarray,
    match_radius: float,
    temporal_window: int,
) -> np.ndarray:
    candidate_tree = cKDTree(candidate_positions)
    rows = []
    num_candidates = int(candidate_positions.shape[0])

    for occurrence_indices in anchor_occurrence_lists:
        if occurrence_indices.size == 0:
            rows.append(np.full(num_candidates, np.inf, dtype=np.float32))
            continue

        best_case_tau = np.full(num_candidates, np.inf, dtype=np.float64)
        for global_index in occurrence_indices:
            episode_id = int(episode_ids[global_index])
            timestep = int(timesteps[global_index])
            episode_start = int(episode_offsets[episode_id])
            remaining_steps = int(episode_lengths[episode_id] - timestep - 1)
            max_tau = min(temporal_window, remaining_steps)
            if max_tau <= 0:
                continue

            earliest_hit = np.full(num_candidates, np.inf, dtype=np.float64)
            for tau in range(1, max_tau + 1):
                future_global_index = episode_start + timestep + tau
                future_position = positions[future_global_index]
                hit_candidate_ids = candidate_tree.query_ball_point(future_position, r=match_radius)
                for candidate_id in hit_candidate_ids:
                    if earliest_hit[candidate_id] == np.inf:
                        earliest_hit[candidate_id] = float(tau)
            best_case_tau = np.minimum(best_case_tau, earliest_hit)

        rows.append(best_case_tau.astype(np.float32))

    return np.stack(rows, axis=0)


def distances_to_scores(distances: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    scores = np.zeros_like(distances, dtype=np.float64)
    finite_mask = np.isfinite(distances)
    scores[finite_mask] = 1.0 / (1.0 + np.maximum(distances[finite_mask], 0.0))
    return scores.astype(np.float32)


def topk_overlap(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true.size == 0:
        return 0.0
    top_k = min(k, y_true.size)
    truth_idx = np.argsort(y_true)[::-1][:top_k]
    pred_idx = np.argsort(y_score)[::-1][:top_k]
    return float(len(set(int(x) for x in truth_idx) & set(int(x) for x in pred_idx)) / float(top_k))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    positives = np.flatnonzero(y_true > 0)
    if positives.size == 0:
        return 0.0
    top_k = min(k, y_true.size)
    pred_idx = np.argsort(y_score)[::-1][:top_k]
    hits = len(set(int(x) for x in positives) & set(int(x) for x in pred_idx))
    return float(hits / float(positives.size))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true.size == 0:
        return 0.0
    top_k = min(k, y_true.size)
    order = np.argsort(y_score)[::-1][:top_k]
    gains = np.maximum(y_true[order], 0.0)
    discounts = 1.0 / np.log2(np.arange(2, top_k + 2))
    dcg = float(np.sum(gains * discounts))

    ideal_order = np.argsort(y_true)[::-1][:top_k]
    ideal_gains = np.maximum(y_true[ideal_order], 0.0)
    ideal_dcg = float(np.sum(ideal_gains * discounts))
    if ideal_dcg <= 1e-12:
        return 0.0
    return dcg / ideal_dcg


def auc_from_binary_labels(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    labels = np.asarray(y_true_binary, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    pos_count = int(np.sum(labels == 1))
    neg_count = int(np.sum(labels == 0))
    if pos_count == 0 or neg_count == 0:
        return 0.5

    ranks = stats.rankdata(scores, method="average")
    pos_ranks = np.sum(ranks[labels == 1])
    auc = (pos_ranks - (pos_count * (pos_count + 1) / 2.0)) / float(pos_count * neg_count)
    if not np.isfinite(auc):
        return 0.5
    return float(auc)


def average_precision_from_binary_labels(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision without depending on sklearn."""
    labels = np.asarray(y_true_binary, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    pos_count = int(np.sum(labels == 1))
    if pos_count == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels == 1, dtype=np.float64)
    false_positive = np.cumsum(sorted_labels == 0, dtype=np.float64)
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / float(pos_count)

    precision = np.concatenate([np.asarray([1.0], dtype=np.float64), precision])
    recall = np.concatenate([np.asarray([0.0], dtype=np.float64), recall])
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    if not np.isfinite(ap):
        return 0.0
    return ap


def evaluate_alignment(
    ground_truth: np.ndarray,
    similarity_scores: np.ndarray,
    top_k: int,
    anchor_global_indices: np.ndarray,
    candidate_global_indices: np.ndarray,
    dataset_name: str,
    method_name: str,
    ground_truth_type: str,
    occurrence_counts: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    per_anchor_rows: list[dict[str, float | int | str]] = []

    for anchor_row_id in range(ground_truth.shape[0]):
        gt_row = np.asarray(ground_truth[anchor_row_id], dtype=np.float64)
        score_row = np.asarray(similarity_scores[anchor_row_id], dtype=np.float64)

        valid_mask = np.ones_like(gt_row, dtype=bool)
        valid_mask[candidate_global_indices == anchor_global_indices[anchor_row_id]] = False
        gt_valid = gt_row[valid_mask]
        score_valid = score_row[valid_mask]
        if gt_valid.size == 0:
            continue

        binary_labels = (gt_valid > 0).astype(np.int64)
        row = {
            "dataset": dataset_name,
            "ground_truth_type": ground_truth_type,
            "method": method_name,
            "anchor_row": int(anchor_row_id),
            "anchor_index": int(anchor_global_indices[anchor_row_id]),
            "occurrence_count": float(occurrence_counts[anchor_row_id]),
            "spearman": safe_spearman(score_valid, gt_valid),
            "pearson": safe_pearson(score_valid, gt_valid),
            "recall_at_k": recall_at_k(gt_valid, score_valid, top_k),
            "topk_overlap": topk_overlap(gt_valid, score_valid, top_k),
            "ndcg_at_k": ndcg_at_k(gt_valid, score_valid, top_k),
            "auc": auc_from_binary_labels(binary_labels, score_valid),
        }
        per_anchor_rows.append(row)

    if not per_anchor_rows:
        summary = {
            "dataset": dataset_name,
            "ground_truth_type": ground_truth_type,
            "method": method_name,
            "num_anchors": 0,
            "spearman_mean": 0.0,
            "pearson_mean": 0.0,
            "recall_at_k_mean": 0.0,
            "topk_overlap_mean": 0.0,
            "ndcg_at_k_mean": 0.0,
            "auc_mean": 0.5,
        }
        return per_anchor_rows, summary

    def mean_of(key: str) -> float:
        return float(np.mean([float(row[key]) for row in per_anchor_rows]))

    summary = {
        "dataset": dataset_name,
        "ground_truth_type": ground_truth_type,
        "method": method_name,
        "num_anchors": len(per_anchor_rows),
        "spearman_mean": mean_of("spearman"),
        "pearson_mean": mean_of("pearson"),
        "recall_at_k_mean": mean_of("recall_at_k"),
        "topk_overlap_mean": mean_of("topk_overlap"),
        "ndcg_at_k_mean": mean_of("ndcg_at_k"),
        "auc_mean": mean_of("auc"),
    }
    return per_anchor_rows, summary
