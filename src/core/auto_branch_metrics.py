"""Utilities for M-policy based automatic branching decisions."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _as_2d_points(points) -> np.ndarray:
    """Convert input points to a float32 array with shape ``(N, D)``."""
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 0)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected points with shape (N, D), got {tuple(arr.shape)}")
    return arr


def _pairwise_distances(query_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distances for two point sets."""
    if query_points.size == 0 or reference_points.size == 0:
        return np.zeros((query_points.shape[0], reference_points.shape[0]), dtype=np.float32)

    query_sq = np.sum(np.square(query_points), axis=1, keepdims=True)
    ref_sq = np.sum(np.square(reference_points), axis=1, keepdims=True).T
    sq_dist = np.maximum(query_sq + ref_sq - 2.0 * np.matmul(query_points, reference_points.T), 0.0)
    return np.sqrt(sq_dist).astype(np.float32)


def compute_knn_distances(
        query_points,
        reference_points,
        *,
        k: int,
        mode: str = "knn_mean",
        exclude_self: bool = False,
) -> Dict[str, Optional[np.ndarray]]:
    """Compute kNN distances from each query point to a reference set.

    Args:
        query_points: Array-like point set with shape ``(N, D)``.
        reference_points: Array-like point set with shape ``(M, D)``.
        k: Number of neighbors to aggregate.
        mode: ``"knn_mean"`` or ``"knn_kth"``.
        exclude_self: When ``True`` and query/reference correspond to the same
            set, the diagonal is ignored so each point only considers other
            points in the reference set.

    Returns:
        A dict containing:
        - ``distances``: aggregated distance per query point, or ``None``.
        - ``effective_k``: the neighbor count actually used.
        - ``valid``: whether the computation had enough samples.
        - ``reason``: explanatory text when ``valid`` is ``False``.
    """
    query_points = _as_2d_points(query_points)
    reference_points = _as_2d_points(reference_points)

    if mode not in ("knn_mean", "knn_kth"):
        raise ValueError(f"Unsupported distance mode: {mode!r}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if query_points.shape[0] == 0:
        return {
            "distances": None,
            "effective_k": 0,
            "valid": False,
            "reason": "empty_query_points",
        }
    if reference_points.shape[0] == 0:
        return {
            "distances": None,
            "effective_k": 0,
            "valid": False,
            "reason": "empty_reference_points",
        }

    distance_matrix = _pairwise_distances(query_points, reference_points)
    available_neighbors = reference_points.shape[0]
    if exclude_self:
        if query_points.shape[0] != reference_points.shape[0]:
            raise ValueError("exclude_self=True requires query_points and reference_points to align")
        np.fill_diagonal(distance_matrix, np.inf)
        available_neighbors -= 1

    effective_k = min(int(k), int(available_neighbors))
    if effective_k < 1:
        return {
            "distances": None,
            "effective_k": 0,
            "valid": False,
            "reason": "insufficient_neighbors",
        }

    partition = np.partition(distance_matrix, kth=effective_k - 1, axis=1)[:, :effective_k]
    if mode == "knn_mean":
        distances = np.mean(partition, axis=1)
    else:
        distances = np.max(partition, axis=1)
    return {
        "distances": distances.astype(np.float32),
        "effective_k": effective_k,
        "valid": True,
        "reason": None,
    }


def compute_recent_threshold(recent_points, *, k: int, mode: str = "knn_mean") -> Dict[str, Optional[float]]:
    """Estimate the adaptive recent-buffer distance threshold ``r_t``."""
    recent_points = _as_2d_points(recent_points)
    if recent_points.shape[0] < 2:
        return {
            "threshold": None,
            "effective_k": 0,
            "valid": False,
            "reason": "recent_buffer_too_small",
        }

    knn = compute_knn_distances(
        recent_points,
        recent_points,
        k=k,
        mode=mode,
        exclude_self=True,
    )
    if not knn["valid"]:
        return {
            "threshold": None,
            "effective_k": int(knn["effective_k"]),
            "valid": False,
            "reason": knn["reason"],
        }
    return {
        "threshold": float(np.median(knn["distances"])),
        "effective_k": int(knn["effective_k"]),
        "valid": True,
        "reason": None,
    }


def compute_m_policy(
        fresh_points,
        recent_points,
        *,
        k: int,
        mode: str = "knn_mean",
) -> Dict[str, Optional[object]]:
    """Compute the frontier set and ``M_policy`` statistic.

    The implementation follows:
    - ``d_k(e, R)``: kNN mean or kth-neighbor distance from fresh point ``e`` to recent set ``R``.
    - ``r_t``: median kNN distance inside ``R`` with self-exclusion.
    - ``F_policy``: fresh points whose ``d_k`` exceeds ``r_t``.
    - ``M_policy = |F_policy| / |E|``.
    """
    fresh_points = _as_2d_points(fresh_points)
    recent_points = _as_2d_points(recent_points)

    if fresh_points.shape[0] == 0:
        return {
            "valid": False,
            "reason": "empty_fresh_points",
            "m_policy": None,
            "frontier_mask": None,
            "frontier_points": None,
            "fresh_knn_distances": None,
            "recent_threshold": None,
            "effective_k": 0,
        }

    threshold_info = compute_recent_threshold(recent_points, k=k, mode=mode)
    if not threshold_info["valid"]:
        return {
            "valid": False,
            "reason": threshold_info["reason"],
            "m_policy": None,
            "frontier_mask": None,
            "frontier_points": None,
            "fresh_knn_distances": None,
            "recent_threshold": None,
            "effective_k": int(threshold_info["effective_k"]),
        }

    fresh_knn = compute_knn_distances(
        fresh_points,
        recent_points,
        k=k,
        mode=mode,
        exclude_self=False,
    )
    if not fresh_knn["valid"]:
        return {
            "valid": False,
            "reason": fresh_knn["reason"],
            "m_policy": None,
            "frontier_mask": None,
            "frontier_points": None,
            "fresh_knn_distances": None,
            "recent_threshold": float(threshold_info["threshold"]),
            "effective_k": int(fresh_knn["effective_k"]),
        }

    threshold = float(threshold_info["threshold"])
    frontier_mask = fresh_knn["distances"] > threshold
    frontier_points = fresh_points[frontier_mask]
    m_policy = float(frontier_mask.mean())
    return {
        "valid": True,
        "reason": None,
        "m_policy": m_policy,
        "frontier_mask": frontier_mask,
        "frontier_points": frontier_points,
        "fresh_knn_distances": fresh_knn["distances"],
        "recent_threshold": threshold,
        "effective_k": int(min(threshold_info["effective_k"], fresh_knn["effective_k"])),
    }
