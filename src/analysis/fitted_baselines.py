from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2

try:
    from sklearn.cluster import KMeans
    from sklearn.covariance import EmpiricalCovariance, LedoitWolf
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight environments.
    KMeans = None  # type: ignore[assignment]
    EmpiricalCovariance = None  # type: ignore[assignment]
    LedoitWolf = None  # type: ignore[assignment]
    NearestNeighbors = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False


def _as_2d_float64(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {values.shape}")
    return values


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (values + values.T)


def _stable_eigh_psd(matrix: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    sym = _symmetrize(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    clipped = np.clip(eigenvalues, max(float(eps), 1e-12), None)
    return clipped, eigenvectors


def _manual_empirical_covariance(positions: np.ndarray, eps: float) -> np.ndarray:
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    denominator = float(max(positions.shape[0] - 1, 1))
    covariance = (centered.T @ centered) / denominator
    return _symmetrize(covariance) + (max(float(eps), 1e-12) * np.eye(positions.shape[1], dtype=np.float64))


def _fallback_ledoitwolf_covariance(positions: np.ndarray, eps: float) -> np.ndarray:
    empirical = _manual_empirical_covariance(positions, eps)
    dim = empirical.shape[0]
    trace_mean = float(np.trace(empirical)) / float(max(dim, 1))
    identity = np.eye(dim, dtype=np.float64) * trace_mean
    n = float(max(positions.shape[0], 1))
    shrinkage = dim / float(dim + n)
    return _symmetrize(((1.0 - shrinkage) * empirical) + (shrinkage * identity))


def _estimate_knn_sigmas_fallback(
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    *,
    neighbor_rank: int,
    eps: float,
) -> np.ndarray:
    if train_positions.shape[0] == 0:
        return np.full(query_positions.shape[0], 1.0, dtype=np.float64)
    distances = cdist(query_positions, train_positions, metric="euclidean")
    if query_positions.shape[0] == train_positions.shape[0] and np.allclose(query_positions, train_positions):
        np.fill_diagonal(distances, np.inf)
    rank = min(max(int(neighbor_rank), 1), int(train_positions.shape[0]))
    partition = np.partition(distances, kth=rank - 1, axis=1)
    sigmas = partition[:, rank - 1]
    finite = np.isfinite(sigmas)
    if not np.all(finite):
        replacement = np.nanmedian(sigmas[finite]) if np.any(finite) else 1.0
        sigmas = np.where(finite, sigmas, replacement)
    return np.maximum(sigmas, max(float(eps), 1e-12))


def _fit_kmeans_fallback(
    positions: np.ndarray,
    *,
    num_bins: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if positions.shape[0] == 0:
        return np.zeros((1, positions.shape[1]), dtype=np.float64), np.zeros(0, dtype=np.int64)
    clusters = min(max(int(num_bins), 1), int(positions.shape[0]))
    if clusters == 1:
        return np.mean(positions, axis=0, keepdims=True), np.zeros(positions.shape[0], dtype=np.int64)
    centers, labels = kmeans2(
        np.asarray(positions, dtype=np.float64),
        clusters,
        minit="points",
        iter=20,
        seed=int(seed),
    )
    centers = np.asarray(centers, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    return centers, labels


def _assign_to_centers(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = cdist(points, centers, metric="sqeuclidean")
    return np.argmin(distances, axis=1).astype(np.int64)


@dataclass(frozen=True)
class MahalanobisMetric:
    """Global linear whitening / Mahalanobis distance fitted on training states only.

    This metric applies a single global covariance normalization to state
    differences, so it is useful for ruling out whether Euclidean failures are
    caused by mismatched axis scales alone. It does not model nonlinear manifold
    structure or density-dependent geometry.
    """

    precision_matrix: np.ndarray
    whitening_matrix: np.ndarray
    covariance_matrix: np.ndarray
    estimator_name: str
    eps: float

    @classmethod
    def fit(
        cls,
        train_positions: np.ndarray,
        *,
        covariance_estimator: str = "ledoitwolf",
        eps: float = 1e-6,
    ) -> "MahalanobisMetric":
        positions = _as_2d_float64(train_positions)
        dim = int(positions.shape[1])
        eps_value = max(float(eps), 1e-12)

        covariance = None
        estimator_name = "identity_fallback"
        estimator_sequence: list[tuple[str, object]]
        estimator_key = str(covariance_estimator).strip().lower()
        if estimator_key in {"ledoitwolf", "ledoit_wolf", "lw"}:
            estimator_sequence = []
            if SKLEARN_AVAILABLE:
                estimator_sequence.extend([
                    ("ledoitwolf", LedoitWolf(store_precision=True)),
                    ("empirical", EmpiricalCovariance(store_precision=True)),
                ])
            else:
                estimator_sequence.extend([
                    ("fallback_ledoitwolf", "fallback_ledoitwolf"),
                    ("manual_empirical", "manual_empirical"),
                ])
        elif estimator_key in {"empirical", "empiricalcovariance"}:
            estimator_sequence = []
            if SKLEARN_AVAILABLE:
                estimator_sequence.extend([
                    ("empirical", EmpiricalCovariance(store_precision=True)),
                    ("ledoitwolf", LedoitWolf(store_precision=True)),
                ])
            else:
                estimator_sequence.extend([
                    ("manual_empirical", "manual_empirical"),
                    ("fallback_ledoitwolf", "fallback_ledoitwolf"),
                ])
        else:
            estimator_sequence = []
            if SKLEARN_AVAILABLE:
                estimator_sequence.extend([
                    ("ledoitwolf", LedoitWolf(store_precision=True)),
                    ("empirical", EmpiricalCovariance(store_precision=True)),
                ])
            else:
                estimator_sequence.extend([
                    ("fallback_ledoitwolf", "fallback_ledoitwolf"),
                    ("manual_empirical", "manual_empirical"),
                ])

        if positions.shape[0] >= 2:
            for name, estimator in estimator_sequence:
                try:
                    if estimator == "fallback_ledoitwolf":
                        candidate = _fallback_ledoitwolf_covariance(positions, eps_value)
                    elif estimator == "manual_empirical":
                        candidate = _manual_empirical_covariance(positions, eps_value)
                    else:
                        estimator.fit(positions)
                        candidate = np.asarray(estimator.covariance_, dtype=np.float64)
                    if candidate.shape == (dim, dim) and np.all(np.isfinite(candidate)):
                        covariance = candidate
                        estimator_name = name
                        break
                except Exception:
                    continue

        if covariance is None:
            if positions.shape[0] >= 2:
                centered = positions - np.mean(positions, axis=0, keepdims=True)
                covariance = centered.T @ centered / float(max(positions.shape[0] - 1, 1))
                estimator_name = "manual_empirical"
            else:
                covariance = np.eye(dim, dtype=np.float64)

        covariance = _symmetrize(covariance) + eps_value * np.eye(dim, dtype=np.float64)
        eigenvalues, eigenvectors = _stable_eigh_psd(covariance, eps_value)
        inv_eigenvalues = 1.0 / eigenvalues
        inv_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues)
        precision = (eigenvectors * inv_eigenvalues[None, :]) @ eigenvectors.T
        whitening = (eigenvectors * inv_sqrt_eigenvalues[None, :]) @ eigenvectors.T
        return cls(
            precision_matrix=_symmetrize(precision),
            whitening_matrix=_symmetrize(whitening),
            covariance_matrix=covariance,
            estimator_name=estimator_name,
            eps=eps_value,
        )

    def pairwise_distance(
        self,
        anchor_positions: np.ndarray,
        candidate_positions: np.ndarray,
        *,
        implementation: str = "whitening",
    ) -> np.ndarray:
        anchors = _as_2d_float64(anchor_positions)
        candidates = _as_2d_float64(candidate_positions)
        implementation_key = str(implementation).strip().lower()
        if implementation_key == "precision":
            anchor_proj = anchors @ self.precision_matrix
            candidate_proj = candidates @ self.precision_matrix
            anchor_sq = np.sum(anchor_proj * anchors, axis=1, keepdims=True)
            candidate_sq = np.sum(candidate_proj * candidates, axis=1, keepdims=True).T
            cross = anchor_proj @ candidates.T
            distances_sq = np.maximum(anchor_sq + candidate_sq - (2.0 * cross), 0.0)
            return np.sqrt(distances_sq).astype(np.float32)

        whitened_anchors = anchors @ self.whitening_matrix.T
        whitened_candidates = candidates @ self.whitening_matrix.T
        distances = cdist(whitened_anchors, whitened_candidates, metric="euclidean")
        return np.asarray(distances, dtype=np.float32)

    def pairwise_score(
        self,
        anchor_positions: np.ndarray,
        candidate_positions: np.ndarray,
        *,
        implementation: str = "whitening",
    ) -> np.ndarray:
        distances = self.pairwise_distance(
            anchor_positions=anchor_positions,
            candidate_positions=candidate_positions,
            implementation=implementation,
        )
        return (-np.asarray(distances, dtype=np.float64)).astype(np.float32)


@dataclass(frozen=True)
class AdaptiveGaussianMetric:
    """Local-bandwidth Gaussian fitted on training states only.

    Each point receives its own bandwidth from local k-NN density, making the
    kernel fairer than a single fixed-width Gaussian when replay density varies.
    It is still based on Euclidean geometry with local rescaling, so it is not
    equivalent to Isolation Kernel or a general nonlinear manifold metric.
    """

    train_positions: np.ndarray
    train_sigmas: np.ndarray
    knn: NearestNeighbors | None
    k: int
    effective_k: int
    eps: float

    @classmethod
    def fit(
        cls,
        train_positions: np.ndarray,
        *,
        k: int = 10,
        eps: float = 1e-6,
    ) -> "AdaptiveGaussianMetric":
        positions = _as_2d_float64(train_positions)
        if positions.shape[0] == 0:
            raise ValueError("AdaptiveGaussianMetric requires at least one training point.")

        eps_value = max(float(eps), 1e-12)
        if positions.shape[0] == 1:
            knn = None
            if SKLEARN_AVAILABLE:
                knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
                knn.fit(positions)
            sigmas = np.full(1, 1.0, dtype=np.float64)
            return cls(
                train_positions=positions,
                train_sigmas=sigmas,
                knn=knn,
                k=1,
                effective_k=1,
                eps=eps_value,
            )

        effective_k = min(max(int(k), 1), max(int(positions.shape[0]) - 1, 1))
        knn = None
        if SKLEARN_AVAILABLE:
            knn = NearestNeighbors(n_neighbors=min(effective_k + 1, int(positions.shape[0])), metric="euclidean")
            knn.fit(positions)
            train_distances, _ = knn.kneighbors(positions, return_distance=True)
            train_sigmas = np.maximum(train_distances[:, -1], eps_value)
        else:
            train_sigmas = _estimate_knn_sigmas_fallback(
                positions,
                positions,
                neighbor_rank=effective_k,
                eps=eps_value,
            )
        return cls(
            train_positions=positions,
            train_sigmas=np.asarray(train_sigmas, dtype=np.float64),
            knn=knn,
            k=max(int(k), 1),
            effective_k=effective_k,
            eps=eps_value,
        )

    def estimate_query_sigmas(
        self,
        query_positions: np.ndarray,
        *,
        query_train_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        queries = _as_2d_float64(query_positions)
        result = np.full(queries.shape[0], np.nan, dtype=np.float64)

        if query_train_indices is not None:
            lookup = np.asarray(query_train_indices, dtype=np.int64)
            valid = (lookup >= 0) & (lookup < self.train_sigmas.shape[0])
            result[valid] = self.train_sigmas[lookup[valid]]

        missing = ~np.isfinite(result)
        if np.any(missing):
            if self.knn is not None:
                neighbors = min(max(self.effective_k, 1), int(self.train_positions.shape[0]))
                query_distances, _ = self.knn.kneighbors(queries[missing], n_neighbors=neighbors, return_distance=True)
                result[missing] = np.maximum(query_distances[:, -1], self.eps)
            else:
                result[missing] = _estimate_knn_sigmas_fallback(
                    self.train_positions,
                    queries[missing],
                    neighbor_rank=self.effective_k,
                    eps=self.eps,
                )

        return np.maximum(result, self.eps)

    def pairwise_kernel(
        self,
        anchor_positions: np.ndarray,
        candidate_positions: np.ndarray,
        *,
        anchor_train_indices: np.ndarray | None = None,
        candidate_train_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        anchors = _as_2d_float64(anchor_positions)
        candidates = _as_2d_float64(candidate_positions)
        anchor_sigmas = self.estimate_query_sigmas(anchors, query_train_indices=anchor_train_indices)
        candidate_sigmas = self.estimate_query_sigmas(candidates, query_train_indices=candidate_train_indices)
        sq_distances = cdist(anchors, candidates, metric="sqeuclidean")
        denominators = np.maximum(anchor_sigmas[:, None] * candidate_sigmas[None, :], self.eps)
        kernel = np.exp(-sq_distances / denominators)
        return np.asarray(kernel, dtype=np.float32)

    def pairwise_distance(
        self,
        anchor_positions: np.ndarray,
        candidate_positions: np.ndarray,
        *,
        anchor_train_indices: np.ndarray | None = None,
        candidate_train_indices: np.ndarray | None = None,
        mode: str = "one_minus_kernel",
    ) -> np.ndarray:
        kernel = self.pairwise_kernel(
            anchor_positions=anchor_positions,
            candidate_positions=candidate_positions,
            anchor_train_indices=anchor_train_indices,
            candidate_train_indices=candidate_train_indices,
        )
        mode_key = str(mode).strip().lower()
        if mode_key in {"rkhs", "rkhs_distance", "sqrt_2_minus_2k"}:
            distances = np.sqrt(np.maximum(2.0 - (2.0 * np.asarray(kernel, dtype=np.float64)), 0.0))
            return np.asarray(distances, dtype=np.float32)
        return (1.0 - np.asarray(kernel, dtype=np.float64)).astype(np.float32)


@dataclass(frozen=True)
class OneStepDynamicsMetric:
    """Reward-free one-step dynamics distance fitted on training transitions only.

    The metric compares empirical one-step next-state distributions after
    discretizing the continuous state space. It emphasizes "where the state goes
    next" more than raw geometry, but it is still only a one-step proxy and does
    not encode long-horizon future equivalence.
    """

    backend_requested: str
    backend_used: str
    distance_metric: str
    num_bins_requested: int
    num_bins_used: int
    alpha: float
    min_count: int
    transition_probabilities: np.ndarray
    global_next_distribution: np.ndarray
    row_counts: np.ndarray
    kmeans_model: KMeans | None
    kmeans_centers: np.ndarray | None
    grid_edges: tuple[np.ndarray, np.ndarray] | None
    eps: float

    @classmethod
    def fit(
        cls,
        train_states: np.ndarray,
        train_next_states: np.ndarray,
        *,
        backend: str = "kmeans",
        num_bins: int = 64,
        distance_metric: str = "jsd",
        alpha: float = 1e-3,
        min_count: int = 5,
        seed: int = 0,
        eps: float = 1e-6,
    ) -> "OneStepDynamicsMetric":
        states = _as_2d_float64(train_states)
        next_states = _as_2d_float64(train_next_states)
        if states.shape != next_states.shape:
            raise ValueError(
                f"train_states and train_next_states must match, got {states.shape} vs {next_states.shape}"
            )

        backend_requested = str(backend).strip().lower()
        backend_used = backend_requested
        if backend_used == "grid" and states.shape[1] != 2:
            warnings.warn(
                "OneStepDynamicsMetric grid backend only supports 2D states; falling back to kmeans.",
                RuntimeWarning,
            )
            backend_used = "kmeans"

        num_bins_requested = max(int(num_bins), 1)
        alpha_value = max(float(alpha), 1e-12)
        eps_value = max(float(eps), 1e-12)
        min_count_value = max(int(min_count), 0)
        kmeans_model: KMeans | None = None
        kmeans_centers: np.ndarray | None = None
        grid_edges: tuple[np.ndarray, np.ndarray] | None = None

        if states.shape[0] == 0:
            transition_probabilities = np.ones((1, 1), dtype=np.float64)
            return cls(
                backend_requested=backend_requested,
                backend_used="identity",
                distance_metric=str(distance_metric),
                num_bins_requested=num_bins_requested,
                num_bins_used=1,
                alpha=alpha_value,
                min_count=min_count_value,
                transition_probabilities=transition_probabilities,
                global_next_distribution=np.ones(1, dtype=np.float64),
                row_counts=np.zeros(1, dtype=np.float64),
                kmeans_model=None,
                kmeans_centers=None,
                grid_edges=None,
                eps=eps_value,
            )

        if backend_used == "grid":
            grid_size = max(int(math.ceil(math.sqrt(float(num_bins_requested)))), 1)
            x_min, y_min = np.min(states, axis=0)
            x_max, y_max = np.max(states, axis=0)
            if math.isclose(x_min, x_max):
                x_max = x_min + 1.0
            if math.isclose(y_min, y_max):
                y_max = y_min + 1.0
            x_edges = np.linspace(x_min, x_max, grid_size + 1, dtype=np.float64)
            y_edges = np.linspace(y_min, y_max, grid_size + 1, dtype=np.float64)
            grid_edges = (x_edges, y_edges)
            current_bins = _assign_grid_bins(states, x_edges, y_edges)
            next_bins = _assign_grid_bins(next_states, x_edges, y_edges)
            num_bins_used = int(grid_size * grid_size)
        else:
            num_bins_used = min(num_bins_requested, int(states.shape[0]))
            if SKLEARN_AVAILABLE:
                kmeans_model = KMeans(n_clusters=num_bins_used, random_state=int(seed), n_init=10)
                kmeans_model.fit(states)
                current_bins = np.asarray(kmeans_model.predict(states), dtype=np.int64)
                next_bins = np.asarray(kmeans_model.predict(next_states), dtype=np.int64)
            else:
                kmeans_centers, current_bins = _fit_kmeans_fallback(
                    states,
                    num_bins=num_bins_used,
                    seed=int(seed),
                )
                next_bins = _assign_to_centers(next_states, kmeans_centers)

        counts = np.zeros((num_bins_used, num_bins_used), dtype=np.float64)
        np.add.at(counts, (current_bins, next_bins), 1.0)
        row_counts = np.sum(counts, axis=1)
        global_counts = np.sum(counts, axis=0)
        global_distribution = (global_counts + alpha_value) / float(np.sum(global_counts) + (alpha_value * num_bins_used))
        denominators = row_counts[:, None] + (alpha_value * num_bins_used)
        transition_probabilities = (counts + alpha_value) / np.maximum(denominators, eps_value)
        low_count_rows = row_counts < float(min_count_value)
        if np.any(low_count_rows):
            transition_probabilities[low_count_rows] = global_distribution[None, :]

        return cls(
            backend_requested=backend_requested,
            backend_used=backend_used,
            distance_metric=str(distance_metric).strip().lower(),
            num_bins_requested=num_bins_requested,
            num_bins_used=num_bins_used,
            alpha=alpha_value,
            min_count=min_count_value,
            transition_probabilities=np.asarray(transition_probabilities, dtype=np.float64),
            global_next_distribution=np.asarray(global_distribution, dtype=np.float64),
            row_counts=np.asarray(row_counts, dtype=np.float64),
            kmeans_model=kmeans_model,
            kmeans_centers=(None if kmeans_centers is None else np.asarray(kmeans_centers, dtype=np.float64)),
            grid_edges=grid_edges,
            eps=eps_value,
        )

    def assign_bins(self, positions: np.ndarray) -> np.ndarray:
        queries = _as_2d_float64(positions)
        if self.backend_used == "grid":
            if self.grid_edges is None:
                return np.zeros(queries.shape[0], dtype=np.int64)
            return _assign_grid_bins(queries, self.grid_edges[0], self.grid_edges[1])
        if self.kmeans_model is not None:
            return np.asarray(self.kmeans_model.predict(queries), dtype=np.int64)
        if self.kmeans_centers is not None:
            return _assign_to_centers(queries, self.kmeans_centers)
        return np.zeros(queries.shape[0], dtype=np.int64)

    def pairwise_distance(
        self,
        anchor_positions: np.ndarray,
        candidate_positions: np.ndarray,
        *,
        distance_metric: str | None = None,
    ) -> np.ndarray:
        metric_key = str(distance_metric or self.distance_metric).strip().lower()
        anchor_bins = self.assign_bins(anchor_positions)
        candidate_bins = self.assign_bins(candidate_positions)
        unique_anchor_bins, anchor_inverse = np.unique(anchor_bins, return_inverse=True)
        unique_candidate_bins, candidate_inverse = np.unique(candidate_bins, return_inverse=True)
        anchor_distributions = self.transition_probabilities[unique_anchor_bins]
        candidate_distributions = self.transition_probabilities[unique_candidate_bins]
        lookup = _pairwise_distribution_distance(anchor_distributions, candidate_distributions, metric_key, self.eps)
        distances = lookup[anchor_inverse[:, None], candidate_inverse[None, :]]
        return np.asarray(distances, dtype=np.float32)

    def pairwise_score(
        self,
        anchor_positions: np.ndarray,
        candidate_positions: np.ndarray,
        *,
        distance_metric: str | None = None,
    ) -> np.ndarray:
        distances = self.pairwise_distance(
            anchor_positions=anchor_positions,
            candidate_positions=candidate_positions,
            distance_metric=distance_metric,
        )
        return (-np.asarray(distances, dtype=np.float64)).astype(np.float32)


def _assign_grid_bins(points: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    values = _as_2d_float64(points)
    x_bin = np.clip(np.digitize(values[:, 0], x_edges[1:-1], right=False), 0, len(x_edges) - 2)
    y_bin = np.clip(np.digitize(values[:, 1], y_edges[1:-1], right=False), 0, len(y_edges) - 2)
    return (x_bin * (len(y_edges) - 1) + y_bin).astype(np.int64)


def _pairwise_distribution_distance(
    anchor_distributions: np.ndarray,
    candidate_distributions: np.ndarray,
    distance_metric: str,
    eps: float,
) -> np.ndarray:
    anchors = _as_2d_float64(anchor_distributions)
    candidates = _as_2d_float64(candidate_distributions)
    if distance_metric in {"l1", "manhattan"}:
        return np.sum(np.abs(anchors[:, None, :] - candidates[None, :, :]), axis=2, dtype=np.float64)

    if distance_metric not in {"jsd", "jensen-shannon", "jensen_shannon"}:
        raise ValueError(f"Unsupported one-step dynamics distance metric: {distance_metric}")

    anchors_safe = np.clip(anchors, eps, None)
    candidates_safe = np.clip(candidates, eps, None)
    mean_distribution = 0.5 * (anchors_safe[:, None, :] + candidates_safe[None, :, :])
    kl_anchor = np.sum(anchors_safe[:, None, :] * np.log(anchors_safe[:, None, :] / mean_distribution), axis=2)
    kl_candidate = np.sum(candidates_safe[None, :, :] * np.log(candidates_safe[None, :, :] / mean_distribution), axis=2)
    jsd = np.maximum(0.5 * (kl_anchor + kl_candidate), 0.0)
    return np.sqrt(jsd)


def sample_transition_pairs(
    positions: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
    episode_lengths: np.ndarray,
    *,
    max_pairs: int | None = None,
    seed: int = 0,
    exclude_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    all_positions = _as_2d_float64(positions)
    episode_ids_array = np.asarray(episode_ids, dtype=np.int64)
    timesteps_array = np.asarray(timesteps, dtype=np.int64)
    episode_lengths_array = np.asarray(episode_lengths, dtype=np.int64)

    valid_indices = np.flatnonzero(timesteps_array < (episode_lengths_array[episode_ids_array] - 1))
    if exclude_indices is not None:
        excluded = np.zeros(all_positions.shape[0], dtype=bool)
        excluded[np.asarray(exclude_indices, dtype=np.int64)] = True
        next_indices = valid_indices + 1
        keep_mask = ~excluded[valid_indices] & ~excluded[next_indices]
        valid_indices = valid_indices[keep_mask]

    if valid_indices.size == 0:
        dim = int(all_positions.shape[1])
        empty = np.empty((0, dim), dtype=np.float64)
        return empty, empty.copy()

    if max_pairs is not None and valid_indices.size > int(max_pairs):
        rng = np.random.default_rng(int(seed))
        valid_indices = np.sort(rng.choice(valid_indices, size=int(max_pairs), replace=False))

    next_indices = valid_indices + 1
    return all_positions[valid_indices], all_positions[next_indices]
