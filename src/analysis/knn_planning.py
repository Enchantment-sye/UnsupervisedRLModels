from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from core.isolation_kernel import SoftIsolationKernel
from .fitted_baselines import AdaptiveGaussianMetric, MahalanobisMetric
from .maze_geodesic import MazeSpec, dataset_slug, ensure_dir, load_maze_spec


@dataclass
class KNNPlanningEvalConfig:
    datasets: list[str]
    output_dir: str
    cache_dir: str
    seed: int = 0
    minari_datasets_path: str = "/home/shangyy/.minari/datasets"
    overwrite_cache: bool = False

    base_stride_pointmaze: int = 5
    base_stride_antmaze: int = 5
    max_nodes_pointmaze_umaze: int = 12000
    max_nodes_pointmaze_large: int = 15000
    max_nodes_antmaze_umaze_diverse: int = 12000

    retrieval_top_k: int = 20
    lambda_bridge: float = 1.0
    alpha: float = 1.5
    pointmaze_h_bridge: float = 10.0
    antmaze_h_bridge: float = 15.0
    pointmaze_eps_start: float = 0.5
    pointmaze_eps_goal: float = 0.5
    antmaze_eps_start: float = 1.0
    antmaze_eps_goal: float = 1.0

    num_queries: int = 200
    min_query_geodesic_pointmaze: float = 2.0
    min_query_geodesic_antmaze: float = 4.0
    max_query_attempts: int = 200000

    state_repr: str = "full"
    fit_pool_size: int = 50000
    gk_sigma_mode: str = "median_heuristic"
    gk_sigma: float | None = None
    mahalanobis_covariance_estimator: str = "ledoitwolf"
    mahalanobis_implementation: str = "whitening"
    mahalanobis_eps: float = 1e-6
    adaptive_k_scale: int = 10
    adaptive_eps: float = 1e-6
    ik_ensemble_size: int = 100
    ik_subsample_size: int = 32
    ik_temperature: float = 0.01
    ik_batch_size: int = 1024
    ik_feature_block_mb: int = 64
    ik_device: str = "auto"
    one_step_m: int = 20
    one_step_row_block_size: int = 64
    ik_temporal_bridge_k: int = 0

    pairwise_row_block_size: int = 256
    plot_num_queries: int = 1
    cache_scope: str = "default"
    query_bank_id: str = "default"
    query_bank_size: int | None = None
    methods: tuple[str, ...] | None = None
    task_preset: str = "default"
    state_variant: str = "raw"
    cross_episode_only: bool = False
    same_episode_quota: int | None = None
    query_source: str = "node_sample"
    query_difficulty_filter: str = "all"
    query_limit: int | None = None


@dataclass(frozen=True)
class ParsedPlanningDataset:
    dataset_id: str
    dataset_slug: str
    maze_spec: MazeSpec
    state_full: np.ndarray
    xy: np.ndarray
    qpos_xy: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray
    episode_offsets: np.ndarray
    episode_lengths: np.ndarray
    transition_state_full: np.ndarray
    transition_next_state_full: np.ndarray
    transition_xy: np.ndarray
    transition_next_xy: np.ndarray


@dataclass(frozen=True)
class NodeSet:
    dataset_id: str
    effective_stride: int
    node_global_indices: np.ndarray
    node_episode_ids: np.ndarray
    node_timesteps: np.ndarray
    node_state_full: np.ndarray
    node_xy: np.ndarray
    temporal_src: np.ndarray
    temporal_dst: np.ndarray
    temporal_cost: np.ndarray
    temporal_exclusions: tuple[np.ndarray, ...]

    @property
    def num_nodes(self) -> int:
        return int(self.node_global_indices.shape[0])


@dataclass(frozen=True)
class QuerySet:
    start_node_ids: np.ndarray
    goal_node_ids: np.ndarray
    query_geodesic: np.ndarray
    difficulty: np.ndarray
    start_xy: np.ndarray | None = None
    goal_xy: np.ndarray | None = None


@dataclass(frozen=True)
class MethodTopK:
    method: str
    indices: np.ndarray
    scores: np.ndarray


AVAILABLE_METHODS = (
    "ik",
    "ik_temporal_bridge",
    "gaussian",
    "euclidean",
    "temporal_distance",
    "mahalanobis",
    "adaptive_gaussian",
    "one_step_dynamics",
)

DEFAULT_METHODS = (
    "ik",
    "gaussian",
    "euclidean",
    "temporal_distance",
    "mahalanobis",
    "adaptive_gaussian",
    "one_step_dynamics",
)

AVAILABLE_TASK_PRESETS = (
    "default",
    "large_v2_ik_favoring",
    "large_v2_ik_soft_local_stitching",
    "antmaze_umaze_detour_focus",
)

AVAILABLE_STATE_VARIANTS = (
    "raw",
    "nuisance_v1",
    "nuisance_v2",
)

AVAILABLE_QUERY_SOURCES = (
    "node_sample",
    "shared_bank",
)

AVAILABLE_QUERY_DIFFICULTY_FILTERS = (
    "all",
    "easy",
    "medium",
    "hard",
    "easy_medium",
    "medium_hard",
)

_NUISANCE_V1_SEED = 0
_NUISANCE_V1_DENSITY_K = 16
_NUISANCE_V2_SEED = 17
_NUISANCE_V2_DENSITY_K = 32
_DEFAULT_ALPHA = 1.5


def slice_query_set(queries: QuerySet, limit: int) -> QuerySet:
    count = min(int(limit), int(queries.start_node_ids.shape[0]))
    return QuerySet(
        start_node_ids=np.asarray(queries.start_node_ids[:count], dtype=np.int64),
        goal_node_ids=np.asarray(queries.goal_node_ids[:count], dtype=np.int64),
        query_geodesic=np.asarray(queries.query_geodesic[:count], dtype=np.float32),
        difficulty=np.asarray(queries.difficulty[:count]).astype(str),
        start_xy=None if queries.start_xy is None else np.asarray(queries.start_xy[:count], dtype=np.float32),
        goal_xy=None if queries.goal_xy is None else np.asarray(queries.goal_xy[:count], dtype=np.float32),
    )


def _round_robin_indices(index_groups: list[list[int]]) -> list[int]:
    ordered: list[int] = []
    group_lists = [list(group) for group in index_groups if group]
    while group_lists:
        next_groups: list[list[int]] = []
        for group in group_lists:
            if not group:
                continue
            ordered.append(int(group.pop(0)))
            if group:
                next_groups.append(group)
        group_lists = next_groups
    return ordered


def _merge_topk_with_temporal_bridges(
    ik_topk: MethodTopK,
    temporal_topk: MethodTopK,
    *,
    bridge_k: int,
    top_k: int,
) -> MethodTopK:
    bridge_k = max(0, min(int(bridge_k), int(top_k)))
    merged_indices = np.full((ik_topk.indices.shape[0], int(top_k)), -1, dtype=np.int64)
    merged_scores = np.full((ik_topk.scores.shape[0], int(top_k)), -np.inf, dtype=np.float32)
    for node_id in range(ik_topk.indices.shape[0]):
        chosen: list[int] = []
        chosen_set: set[int] = set()
        if bridge_k > 0:
            for cand in temporal_topk.indices[node_id].tolist():
                cand = int(cand)
                if cand < 0 or cand in chosen_set:
                    continue
                chosen.append(cand)
                chosen_set.add(cand)
                if len(chosen) >= bridge_k:
                    break
        for cand in ik_topk.indices[node_id].tolist():
            cand = int(cand)
            if cand < 0 or cand in chosen_set:
                continue
            chosen.append(cand)
            chosen_set.add(cand)
            if len(chosen) >= int(top_k):
                break
        if chosen:
            merged_indices[node_id, : len(chosen)] = np.asarray(chosen, dtype=np.int64)
            merged_scores[node_id, : len(chosen)] = np.linspace(
                1.0,
                0.0,
                num=len(chosen),
                endpoint=False,
                dtype=np.float32,
            )
    return MethodTopK(method="ik_temporal_bridge", indices=merged_indices, scores=merged_scores)


def _query_start_xy(queries: QuerySet, nodes: NodeSet, query_id: int) -> np.ndarray:
    if queries.start_xy is not None:
        return np.asarray(queries.start_xy[query_id], dtype=np.float32)
    return np.asarray(nodes.node_xy[int(queries.start_node_ids[query_id])], dtype=np.float32)


def _query_goal_xy(queries: QuerySet, nodes: NodeSet, query_id: int) -> np.ndarray:
    if queries.goal_xy is not None:
        return np.asarray(queries.goal_xy[query_id], dtype=np.float32)
    return np.asarray(nodes.node_xy[int(queries.goal_node_ids[query_id])], dtype=np.float32)


def save_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _npz_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def _safe_load_npz(path: str) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=True) as payload:
            return {key: payload[key] for key in payload.files}
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _save_npz(path: str, **kwargs: Any) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **kwargs)


def _save_npz_fast(path: str, **kwargs: Any) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez(path, **kwargs)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _coerce_float_array(values: list[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = values.astype(np.float64, copy=False)
    else:
        array = np.asarray(list(values), dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _safe_nanmean(values: list[Any] | np.ndarray) -> float:
    array = _coerce_float_array(values)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_mean(values: list[Any] | np.ndarray) -> float:
    array = _coerce_float_array(values)
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def _payload_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:12]


def _resolve_torch_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _is_torch_oom(error: Exception) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda oom" in text


def _ensure_matplotlib_cache_dir() -> str:
    cache_dir = os.path.join("/tmp", "metra_matplotlib")
    ensure_dir(cache_dir)
    os.environ["MPLCONFIGDIR"] = cache_dir
    return cache_dir


def apply_task_preset(cfg: KNNPlanningEvalConfig) -> KNNPlanningEvalConfig:
    preset = str(cfg.task_preset or "default")
    if preset == "default":
        return cfg
    if preset == "antmaze_umaze_detour_focus":
        cfg.datasets = ["D4RL/antmaze/umaze-diverse-v1"]
        cfg.max_nodes_antmaze_umaze_diverse = 12000
        cfg.query_source = "node_sample"
        cfg.query_difficulty_filter = "all"
        cfg.query_limit = 200
        cfg.num_queries = 200
        cfg.retrieval_top_k = 20
        cfg.antmaze_h_bridge = 15.0
        if math.isclose(float(cfg.alpha), _DEFAULT_ALPHA):
            cfg.alpha = 0.88
        cfg.lambda_bridge = 1.0
        cfg.antmaze_eps_start = 1.0
        cfg.antmaze_eps_goal = 1.0
        cfg.state_variant = "raw"
        cfg.cross_episode_only = False
        cfg.same_episode_quota = None
        return cfg

    if preset not in {"large_v2_ik_favoring", "large_v2_ik_soft_local_stitching"}:
        raise ValueError(f"Unsupported task preset: {preset}")

    cfg.datasets = ["D4RL/pointmaze/large-v2"]
    cfg.max_nodes_pointmaze_large = 15000
    cfg.query_bank_id = "ik_shared_bank_v2"
    cfg.query_bank_size = 300
    cfg.query_source = "shared_bank"
    cfg.query_difficulty_filter = "easy"
    cfg.query_limit = 30
    cfg.num_queries = 30
    cfg.retrieval_top_k = 8
    cfg.pointmaze_h_bridge = 3.0
    if math.isclose(float(cfg.alpha), _DEFAULT_ALPHA):
        cfg.alpha = 0.8
    cfg.lambda_bridge = 1.0
    cfg.pointmaze_eps_start = 0.5
    cfg.pointmaze_eps_goal = 0.5
    cfg.state_variant = "nuisance_v1"
    if preset == "large_v2_ik_favoring":
        cfg.cross_episode_only = True
        cfg.same_episode_quota = 0
    else:
        cfg.cross_episode_only = False
        cfg.same_episode_quota = 1
    return cfg


def _effective_same_episode_quota(cfg: KNNPlanningEvalConfig) -> int | None:
    if bool(cfg.cross_episode_only):
        return 0
    if cfg.same_episode_quota is None:
        return None
    quota = int(cfg.same_episode_quota)
    if quota < 0:
        raise ValueError(f"same_episode_quota must be non-negative, got {quota}")
    return quota


def _matches_query_difficulty(label: str, difficulty_filter: str) -> bool:
    difficulty = str(label)
    filter_name = str(difficulty_filter or "all")
    if filter_name == "all":
        return True
    if filter_name == "easy":
        return difficulty == "easy"
    if filter_name == "medium":
        return difficulty == "medium"
    if filter_name == "hard":
        return difficulty == "hard"
    if filter_name == "easy_medium":
        return difficulty in {"easy", "medium"}
    if filter_name == "medium_hard":
        return difficulty in {"medium", "hard"}
    raise ValueError(f"Unsupported query difficulty filter: {difficulty_filter}")


def _resolve_query_limit(cfg: KNNPlanningEvalConfig) -> int:
    if cfg.query_limit is not None:
        return int(cfg.query_limit)
    return int(cfg.num_queries)


def _resolve_query_bank_size(cfg: KNNPlanningEvalConfig) -> int:
    if cfg.query_bank_size is not None:
        return int(cfg.query_bank_size)
    return max(_resolve_query_limit(cfg), int(cfg.num_queries))


def _resolve_repr(parsed: ParsedPlanningDataset, cfg: KNNPlanningEvalConfig, *, transitions: bool = False) -> np.ndarray:
    if cfg.state_repr == "xy":
        return parsed.transition_xy if transitions else parsed.xy
    return parsed.transition_state_full if transitions else parsed.state_full


def _is_antmaze(dataset_id: str) -> bool:
    return "antmaze" in dataset_id.lower()


def _resolve_base_stride(dataset_id: str, cfg: KNNPlanningEvalConfig) -> int:
    return int(cfg.base_stride_antmaze if _is_antmaze(dataset_id) else cfg.base_stride_pointmaze)


def _resolve_max_nodes(dataset_id: str, cfg: KNNPlanningEvalConfig) -> int:
    lowered = dataset_id.lower()
    if "pointmaze/large" in lowered:
        return int(cfg.max_nodes_pointmaze_large)
    if "pointmaze/umaze" in lowered:
        return int(cfg.max_nodes_pointmaze_umaze)
    return int(cfg.max_nodes_antmaze_umaze_diverse)


def _resolve_h_bridge(dataset_id: str, cfg: KNNPlanningEvalConfig) -> float:
    return float(cfg.antmaze_h_bridge if _is_antmaze(dataset_id) else cfg.pointmaze_h_bridge)


def _resolve_eps(dataset_id: str, cfg: KNNPlanningEvalConfig) -> tuple[float, float]:
    if _is_antmaze(dataset_id):
        return float(cfg.antmaze_eps_start), float(cfg.antmaze_eps_goal)
    return float(cfg.pointmaze_eps_start), float(cfg.pointmaze_eps_goal)


def _resolve_min_query_geodesic(dataset_id: str, cfg: KNNPlanningEvalConfig) -> float:
    return float(cfg.min_query_geodesic_antmaze if _is_antmaze(dataset_id) else cfg.min_query_geodesic_pointmaze)


def _nuisance_v1_features(
    xy: np.ndarray,
    states: np.ndarray,
    transition_xy: np.ndarray,
    transition_states: np.ndarray,
    transition_next_xy: np.ndarray,
    transition_next_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if states.shape[1] < 4 or transition_states.shape[1] < 4 or transition_next_states.shape[1] < 4:
        raise ValueError("nuisance_v1 expects pointmaze states with at least [x, y, vx, vy].")

    state_xy = np.asarray(xy, dtype=np.float32)
    state_values = np.asarray(states, dtype=np.float32)
    transition_xy = np.asarray(transition_xy, dtype=np.float32)
    transition_next_xy = np.asarray(transition_next_xy, dtype=np.float32)
    transition_values = np.asarray(transition_states, dtype=np.float32)
    transition_next_values = np.asarray(transition_next_states, dtype=np.float32)

    tree = cKDTree(state_xy)
    neighbor_k = min(int(_NUISANCE_V1_DENSITY_K) + 1, int(state_xy.shape[0]))
    distances, _ = tree.query(state_xy, k=neighbor_k)
    if distances.ndim == 1:
        distances = distances[:, None]
    local_scale = distances[:, -1].astype(np.float32)
    low, high = np.percentile(local_scale, [5.0, 95.0])
    normalized = np.clip((local_scale - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    local_scale = (0.5 + 1.5 * normalized).astype(np.float32)

    _, transition_nn = tree.query(transition_xy, k=1)
    _, transition_next_nn = tree.query(transition_next_xy, k=1)
    transition_scale = local_scale[np.asarray(transition_nn, dtype=np.int64)]
    transition_next_scale = local_scale[np.asarray(transition_next_nn, dtype=np.int64)]

    rng = _rng(_NUISANCE_V1_SEED)
    weight = rng.normal(size=(2, 16)).astype(np.float32)
    bias = rng.uniform(0.0, 2.0 * np.pi, size=(16,)).astype(np.float32)

    def _augment(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
        velocity = values[:, 2:4].astype(np.float32)
        projection = velocity @ weight + bias[None, :]
        fourier = np.concatenate([np.sin(projection), np.cos(projection)], axis=1).astype(np.float32)
        repeated = np.repeat(velocity * scales[:, None], 16, axis=1).astype(np.float32)
        return np.concatenate([fourier * scales[:, None], repeated], axis=1).astype(np.float32)

    state_aug = np.concatenate([state_values, _augment(state_values, local_scale)], axis=1).astype(np.float32)
    transition_aug = np.concatenate([transition_values, _augment(transition_values, transition_scale)], axis=1).astype(np.float32)
    transition_next_aug = np.concatenate(
        [transition_next_values, _augment(transition_next_values, transition_next_scale)],
        axis=1,
    ).astype(np.float32)
    return state_aug, transition_aug, transition_next_aug


def _nuisance_v2_features(
    xy: np.ndarray,
    states: np.ndarray,
    transition_xy: np.ndarray,
    transition_states: np.ndarray,
    transition_next_xy: np.ndarray,
    transition_next_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if states.shape[1] < 4 or transition_states.shape[1] < 4 or transition_next_states.shape[1] < 4:
        raise ValueError("nuisance_v2 expects pointmaze states with at least [x, y, vx, vy].")

    state_xy = np.asarray(xy, dtype=np.float32)
    state_values = np.asarray(states, dtype=np.float32)
    transition_xy = np.asarray(transition_xy, dtype=np.float32)
    transition_next_xy = np.asarray(transition_next_xy, dtype=np.float32)
    transition_values = np.asarray(transition_states, dtype=np.float32)
    transition_next_values = np.asarray(transition_next_states, dtype=np.float32)

    tree = cKDTree(state_xy)
    neighbor_k = min(int(_NUISANCE_V2_DENSITY_K) + 1, int(state_xy.shape[0]))
    distances, _ = tree.query(state_xy, k=neighbor_k)
    if distances.ndim == 1:
        distances = distances[:, None]
    local_scale = distances[:, -1].astype(np.float32)
    low, high = np.percentile(local_scale, [2.5, 97.5])
    normalized = np.clip((local_scale - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    direct_scale = (0.25 + 3.75 * normalized).astype(np.float32)
    inverse_scale = (0.25 + 3.75 * (1.0 - normalized)).astype(np.float32)

    _, transition_nn = tree.query(transition_xy, k=1)
    _, transition_next_nn = tree.query(transition_next_xy, k=1)
    transition_direct = direct_scale[np.asarray(transition_nn, dtype=np.int64)]
    transition_inverse = inverse_scale[np.asarray(transition_nn, dtype=np.int64)]
    transition_next_direct = direct_scale[np.asarray(transition_next_nn, dtype=np.int64)]
    transition_next_inverse = inverse_scale[np.asarray(transition_next_nn, dtype=np.int64)]

    rng = _rng(_NUISANCE_V2_SEED)
    weight_primary = rng.normal(scale=1.5, size=(7, 32)).astype(np.float32)
    bias_primary = rng.uniform(0.0, 2.0 * np.pi, size=(32,)).astype(np.float32)
    weight_secondary = rng.normal(scale=0.75, size=(4, 16)).astype(np.float32)
    bias_secondary = rng.uniform(0.0, 2.0 * np.pi, size=(16,)).astype(np.float32)
    sign_pattern = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(32,)).astype(np.float32)

    def _augment(values: np.ndarray, direct: np.ndarray, inverse: np.ndarray) -> np.ndarray:
        velocity = values[:, 2:4].astype(np.float32)
        vnorm = np.linalg.norm(velocity, axis=1, keepdims=True).astype(np.float32)
        xy_local = values[:, :2].astype(np.float32)
        interaction = np.concatenate(
            [
                xy_local * velocity,
                xy_local + velocity,
                xy_local - velocity,
            ],
            axis=1,
        ).astype(np.float32)
        base_primary = np.concatenate([velocity, xy_local, vnorm, direct[:, None], inverse[:, None]], axis=1).astype(np.float32)
        base_secondary = np.concatenate([velocity * direct[:, None], velocity * inverse[:, None]], axis=1).astype(np.float32)

        proj_primary = base_primary @ weight_primary + bias_primary[None, :]
        fourier_primary = np.concatenate([np.sin(proj_primary), np.cos(proj_primary)], axis=1).astype(np.float32)

        proj_secondary = base_secondary @ weight_secondary + bias_secondary[None, :]
        fourier_secondary = np.concatenate([np.sin(proj_secondary), np.cos(proj_secondary)], axis=1).astype(np.float32)

        repeated_velocity = np.repeat(velocity * direct[:, None], 24, axis=1).astype(np.float32)
        repeated_inverse = np.repeat(velocity * inverse[:, None], 24, axis=1).astype(np.float32)
        signed_interaction = np.repeat(interaction[:, :1], 32, axis=1).astype(np.float32) * sign_pattern[None, :]

        return np.concatenate(
            [
                fourier_primary * direct[:, None],
                fourier_secondary * inverse[:, None],
                repeated_velocity,
                repeated_inverse,
                signed_interaction,
            ],
            axis=1,
        ).astype(np.float32)

    state_aug = np.concatenate([state_values, _augment(state_values, direct_scale, inverse_scale)], axis=1).astype(np.float32)
    transition_aug = np.concatenate(
        [transition_values, _augment(transition_values, transition_direct, transition_inverse)],
        axis=1,
    ).astype(np.float32)
    transition_next_aug = np.concatenate(
        [transition_next_values, _augment(transition_next_values, transition_next_direct, transition_next_inverse)],
        axis=1,
    ).astype(np.float32)
    return state_aug, transition_aug, transition_next_aug


def _apply_state_variant(parsed: ParsedPlanningDataset, cfg: KNNPlanningEvalConfig) -> ParsedPlanningDataset:
    variant = str(cfg.state_variant or "raw")
    if variant == "raw":
        return parsed
    if variant not in {"nuisance_v1", "nuisance_v2"}:
        raise ValueError(f"Unsupported state variant: {variant}")

    payload = {
        "dataset": parsed.dataset_id,
        "state_variant": variant,
        "variant_seed": _NUISANCE_V1_SEED if variant == "nuisance_v1" else _NUISANCE_V2_SEED,
        "density_k": _NUISANCE_V1_DENSITY_K if variant == "nuisance_v1" else _NUISANCE_V2_DENSITY_K,
    }
    cache_path = os.path.join(
        cfg.cache_dir,
        f"dataset_variant_{dataset_slug(parsed.dataset_id)}_{_payload_hash(payload)}.npz",
    )
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return ParsedPlanningDataset(
                dataset_id=parsed.dataset_id,
                dataset_slug=parsed.dataset_slug,
                maze_spec=parsed.maze_spec,
                state_full=np.asarray(cached["state_full"], dtype=np.float32),
                xy=parsed.xy,
                qpos_xy=parsed.qpos_xy,
                episode_ids=parsed.episode_ids,
                timesteps=parsed.timesteps,
                episode_offsets=parsed.episode_offsets,
                episode_lengths=parsed.episode_lengths,
                transition_state_full=np.asarray(cached["transition_state_full"], dtype=np.float32),
                transition_next_state_full=np.asarray(cached["transition_next_state_full"], dtype=np.float32),
                transition_xy=parsed.transition_xy,
                transition_next_xy=parsed.transition_next_xy,
            )

    if variant == "nuisance_v1":
        state_aug, transition_aug, transition_next_aug = _nuisance_v1_features(
            xy=parsed.xy,
            states=parsed.state_full,
            transition_xy=parsed.transition_xy,
            transition_states=parsed.transition_state_full,
            transition_next_xy=parsed.transition_next_xy,
            transition_next_states=parsed.transition_next_state_full,
        )
    else:
        state_aug, transition_aug, transition_next_aug = _nuisance_v2_features(
            xy=parsed.xy,
            states=parsed.state_full,
            transition_xy=parsed.transition_xy,
            transition_states=parsed.transition_state_full,
            transition_next_xy=parsed.transition_next_xy,
            transition_next_states=parsed.transition_next_state_full,
        )
    _save_npz_fast(
        cache_path,
        state_full=state_aug,
        transition_state_full=transition_aug,
        transition_next_state_full=transition_next_aug,
        payload_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )
    return ParsedPlanningDataset(
        dataset_id=parsed.dataset_id,
        dataset_slug=parsed.dataset_slug,
        maze_spec=parsed.maze_spec,
        state_full=state_aug,
        xy=parsed.xy,
        qpos_xy=parsed.qpos_xy,
        episode_ids=parsed.episode_ids,
        timesteps=parsed.timesteps,
        episode_offsets=parsed.episode_offsets,
        episode_lengths=parsed.episode_lengths,
        transition_state_full=transition_aug,
        transition_next_state_full=transition_next_aug,
        transition_xy=parsed.transition_xy,
        transition_next_xy=parsed.transition_next_xy,
    )


def _sorted_episode_names(handle: Any) -> list[str]:
    def _episode_key(name: str) -> int:
        if name.startswith("episode_"):
            return int(name.split("_", 1)[1])
        return 0

    return sorted(handle.keys(), key=_episode_key)


def _extract_minari_observations(episode: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = episode.observations
    state_full = np.asarray(observations["observation"], dtype=np.float32)
    xy = np.asarray(observations["achieved_goal"], dtype=np.float32)
    infos = episode.infos if hasattr(episode, "infos") else {}
    qpos_xy = xy.copy()
    if infos and "qpos" in infos:
        qpos = np.asarray(infos["qpos"], dtype=np.float32)
        if qpos.ndim == 2 and qpos.shape[1] >= 2:
            qpos_xy = qpos[:, :2].astype(np.float32)
    return state_full, xy, qpos_xy


def _load_dataset_from_minari(
    dataset_id: str,
    minari_datasets_path: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    import minari

    os.environ.setdefault("MINARI_DATASETS_PATH", minari_datasets_path)
    dataset = minari.load_dataset(dataset_id)

    state_blocks = []
    xy_blocks = []
    qpos_blocks = []
    episode_ids = []
    timesteps = []
    episode_offsets = [0]
    episode_lengths = []

    transition_state_blocks = []
    transition_next_state_blocks = []
    transition_xy_blocks = []
    transition_next_xy_blocks = []

    for episode_id in range(dataset.total_episodes):
        episode = dataset[episode_id]
        state_full, xy, qpos_xy = _extract_minari_observations(episode)
        state_blocks.append(state_full)
        xy_blocks.append(xy)
        qpos_blocks.append(qpos_xy)
        episode_lengths.append(int(state_full.shape[0]))
        episode_ids.append(np.full(state_full.shape[0], episode_id, dtype=np.int32))
        timesteps.append(np.arange(state_full.shape[0], dtype=np.int32))
        episode_offsets.append(episode_offsets[-1] + int(state_full.shape[0]))

        num_transitions = int(min(len(episode.actions), state_full.shape[0] - 1))
        if num_transitions > 0:
            transition_state_blocks.append(state_full[:num_transitions])
            transition_next_state_blocks.append(state_full[1 : num_transitions + 1])
            transition_xy_blocks.append(xy[:num_transitions])
            transition_next_xy_blocks.append(xy[1 : num_transitions + 1])

    return (
        np.concatenate(state_blocks, axis=0).astype(np.float32),
        np.concatenate(xy_blocks, axis=0).astype(np.float32),
        np.concatenate(qpos_blocks, axis=0).astype(np.float32),
        np.concatenate(episode_ids, axis=0).astype(np.int32),
        np.concatenate(timesteps, axis=0).astype(np.int32),
        np.asarray(episode_offsets, dtype=np.int64),
        np.asarray(episode_lengths, dtype=np.int32),
        np.concatenate(transition_state_blocks, axis=0).astype(np.float32),
        np.concatenate(transition_next_state_blocks, axis=0).astype(np.float32),
        np.concatenate(transition_xy_blocks, axis=0).astype(np.float32),
        np.concatenate(transition_next_xy_blocks, axis=0).astype(np.float32),
    )


def _load_dataset_from_hdf5(
    dataset_id: str,
    minari_datasets_path: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    import h5py

    normalized = dataset_id.replace("D4RL/", "")
    namespace, dataset_name = normalized.split("/")
    hdf5_path = os.path.join(minari_datasets_path, "D4RL", namespace, dataset_name, "data", "main_data.hdf5")

    state_blocks = []
    xy_blocks = []
    qpos_blocks = []
    episode_ids = []
    timesteps = []
    episode_offsets = [0]
    episode_lengths = []
    transition_state_blocks = []
    transition_next_state_blocks = []
    transition_xy_blocks = []
    transition_next_xy_blocks = []

    with h5py.File(hdf5_path, "r") as handle:
        for episode_id, episode_name in enumerate(_sorted_episode_names(handle)):
            episode = handle[episode_name]
            state_full = np.asarray(episode["observations"]["observation"], dtype=np.float32)
            xy = np.asarray(episode["observations"]["achieved_goal"], dtype=np.float32)
            qpos_xy = xy.copy()
            if "qpos" in episode["infos"]:
                qpos = np.asarray(episode["infos"]["qpos"], dtype=np.float32)
                if qpos.ndim == 2 and qpos.shape[1] >= 2:
                    qpos_xy = qpos[:, :2].astype(np.float32)

            state_blocks.append(state_full)
            xy_blocks.append(xy)
            qpos_blocks.append(qpos_xy)
            episode_lengths.append(int(state_full.shape[0]))
            episode_ids.append(np.full(state_full.shape[0], episode_id, dtype=np.int32))
            timesteps.append(np.arange(state_full.shape[0], dtype=np.int32))
            episode_offsets.append(episode_offsets[-1] + int(state_full.shape[0]))

            num_transitions = min(int(episode["actions"].shape[0]), int(state_full.shape[0] - 1))
            if num_transitions > 0:
                transition_state_blocks.append(state_full[:num_transitions])
                transition_next_state_blocks.append(state_full[1 : num_transitions + 1])
                transition_xy_blocks.append(xy[:num_transitions])
                transition_next_xy_blocks.append(xy[1 : num_transitions + 1])

    return (
        np.concatenate(state_blocks, axis=0).astype(np.float32),
        np.concatenate(xy_blocks, axis=0).astype(np.float32),
        np.concatenate(qpos_blocks, axis=0).astype(np.float32),
        np.concatenate(episode_ids, axis=0).astype(np.int32),
        np.concatenate(timesteps, axis=0).astype(np.int32),
        np.asarray(episode_offsets, dtype=np.int64),
        np.asarray(episode_lengths, dtype=np.int32),
        np.concatenate(transition_state_blocks, axis=0).astype(np.float32),
        np.concatenate(transition_next_state_blocks, axis=0).astype(np.float32),
        np.concatenate(transition_xy_blocks, axis=0).astype(np.float32),
        np.concatenate(transition_next_xy_blocks, axis=0).astype(np.float32),
    )


def load_or_parse_dataset(dataset_id: str, cfg: KNNPlanningEvalConfig) -> ParsedPlanningDataset:
    cache_path = os.path.join(cfg.cache_dir, f"dataset_parse_{dataset_slug(dataset_id)}.npz")
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            maze_spec = load_maze_spec(dataset_id, minari_root=cfg.minari_datasets_path, cache_dir=cfg.cache_dir)
            parsed = ParsedPlanningDataset(
                dataset_id=dataset_id,
                dataset_slug=dataset_slug(dataset_id),
                maze_spec=maze_spec,
                state_full=np.asarray(cached["state_full"], dtype=np.float32),
                xy=np.asarray(cached["xy"], dtype=np.float32),
                qpos_xy=np.asarray(cached["qpos_xy"], dtype=np.float32),
                episode_ids=np.asarray(cached["episode_ids"], dtype=np.int32),
                timesteps=np.asarray(cached["timesteps"], dtype=np.int32),
                episode_offsets=np.asarray(cached["episode_offsets"], dtype=np.int64),
                episode_lengths=np.asarray(cached["episode_lengths"], dtype=np.int32),
                transition_state_full=np.asarray(cached["transition_state_full"], dtype=np.float32),
                transition_next_state_full=np.asarray(cached["transition_next_state_full"], dtype=np.float32),
                transition_xy=np.asarray(cached["transition_xy"], dtype=np.float32),
                transition_next_xy=np.asarray(cached["transition_next_xy"], dtype=np.float32),
            )
            return _apply_state_variant(parsed, cfg)

    try:
        payload = _load_dataset_from_hdf5(dataset_id, cfg.minari_datasets_path)
    except Exception:
        payload = _load_dataset_from_minari(dataset_id, cfg.minari_datasets_path)

    (
        state_full,
        xy,
        qpos_xy,
        episode_ids,
        timesteps,
        episode_offsets,
        episode_lengths,
        transition_state_full,
        transition_next_state_full,
        transition_xy,
        transition_next_xy,
    ) = payload

    _save_npz(
        cache_path,
        state_full=state_full,
        xy=xy,
        qpos_xy=qpos_xy,
        episode_ids=episode_ids,
        timesteps=timesteps,
        episode_offsets=episode_offsets,
        episode_lengths=episode_lengths,
        transition_state_full=transition_state_full,
        transition_next_state_full=transition_next_state_full,
        transition_xy=transition_xy,
        transition_next_xy=transition_next_xy,
    )
    maze_spec = load_maze_spec(dataset_id, minari_root=cfg.minari_datasets_path, cache_dir=cfg.cache_dir)
    parsed = ParsedPlanningDataset(
        dataset_id=dataset_id,
        dataset_slug=dataset_slug(dataset_id),
        maze_spec=maze_spec,
        state_full=state_full,
        xy=xy,
        qpos_xy=qpos_xy,
        episode_ids=episode_ids,
        timesteps=timesteps,
        episode_offsets=episode_offsets,
        episode_lengths=episode_lengths,
        transition_state_full=transition_state_full,
        transition_next_state_full=transition_next_state_full,
        transition_xy=transition_xy,
        transition_next_xy=transition_next_xy,
    )
    return _apply_state_variant(parsed, cfg)


def _sample_episode_indices(length: int, stride: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    selected = list(range(0, length, int(stride)))
    if selected[-1] != length - 1:
        selected.append(length - 1)
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def load_or_build_nodes(parsed: ParsedPlanningDataset, cfg: KNNPlanningEvalConfig) -> NodeSet:
    payload = {
        "dataset": parsed.dataset_id,
        "base_stride": _resolve_base_stride(parsed.dataset_id, cfg),
        "max_nodes": _resolve_max_nodes(parsed.dataset_id, cfg),
        "seed": cfg.seed,
        "cache_scope": cfg.cache_scope,
        "state_variant": cfg.state_variant,
    }
    cache_path = os.path.join(cfg.cache_dir, f"node_set_{dataset_slug(parsed.dataset_id)}_{_payload_hash(payload)}.npz")
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            temporal_exclusions = tuple(np.asarray(x, dtype=np.int64) for x in cached["temporal_exclusions"])
            return NodeSet(
                dataset_id=parsed.dataset_id,
                effective_stride=int(cached["effective_stride"].item()),
                node_global_indices=np.asarray(cached["node_global_indices"], dtype=np.int64),
                node_episode_ids=np.asarray(cached["node_episode_ids"], dtype=np.int32),
                node_timesteps=np.asarray(cached["node_timesteps"], dtype=np.int32),
                node_state_full=np.asarray(cached["node_state_full"], dtype=np.float32),
                node_xy=np.asarray(cached["node_xy"], dtype=np.float32),
                temporal_src=np.asarray(cached["temporal_src"], dtype=np.int64),
                temporal_dst=np.asarray(cached["temporal_dst"], dtype=np.int64),
                temporal_cost=np.asarray(cached["temporal_cost"], dtype=np.float32),
                temporal_exclusions=temporal_exclusions,
            )

    stride = max(_resolve_base_stride(parsed.dataset_id, cfg), 1)
    max_nodes = _resolve_max_nodes(parsed.dataset_id, cfg)
    max_episode_length = int(np.max(parsed.episode_lengths))
    if int(parsed.episode_lengths.shape[0]) > max_nodes:
        stride = max_episode_length
    episode_sampled: list[np.ndarray] = []
    counts = []
    prev_total = None
    while True:
        episode_sampled = []
        counts = []
        for episode_id, length in enumerate(parsed.episode_lengths.tolist()):
            local = _sample_episode_indices(int(length), stride)
            global_indices = parsed.episode_offsets[episode_id] + local
            episode_sampled.append(global_indices.astype(np.int64))
            counts.append(int(global_indices.shape[0]))
        total_candidates = int(sum(counts))
        if total_candidates <= max_nodes:
            break
        if stride >= max_episode_length:
            break
        if prev_total is not None and total_candidates >= prev_total:
            break
        prev_total = total_candidates
        stride += 1

    total_candidates = int(sum(int(arr.shape[0]) for arr in episode_sampled))
    if total_candidates > max_nodes:
        flat_candidates = np.concatenate(episode_sampled, axis=0).astype(np.int64)
        chosen = np.sort(_rng(cfg.seed + 211).choice(flat_candidates, size=max_nodes, replace=False))
        grouped: list[np.ndarray] = []
        for episode_id in range(parsed.episode_lengths.shape[0]):
            mask = parsed.episode_ids[chosen] == episode_id
            grouped.append(chosen[mask].astype(np.int64))
        episode_sampled = grouped

    node_global_indices = np.concatenate(episode_sampled, axis=0).astype(np.int64)
    node_episode_ids = parsed.episode_ids[node_global_indices].astype(np.int32)
    node_timesteps = parsed.timesteps[node_global_indices].astype(np.int32)
    node_state_full = parsed.state_full[node_global_indices].astype(np.float32)
    node_xy = parsed.xy[node_global_indices].astype(np.float32)

    temporal_src = []
    temporal_dst = []
    temporal_cost = []
    temporal_exclusions: list[list[int]] = [[] for _ in range(node_global_indices.shape[0])]
    node_cursor = 0
    for sample_indices in episode_sampled:
        local_node_ids = np.arange(node_cursor, node_cursor + sample_indices.shape[0], dtype=np.int64)
        node_cursor += sample_indices.shape[0]
        if local_node_ids.shape[0] <= 1:
            continue
        local_timesteps = parsed.timesteps[sample_indices]
        for idx in range(local_node_ids.shape[0] - 1):
            src = int(local_node_ids[idx])
            dst = int(local_node_ids[idx + 1])
            temporal_src.append(src)
            temporal_dst.append(dst)
            temporal_cost.append(float(local_timesteps[idx + 1] - local_timesteps[idx]))
            temporal_exclusions[src].append(dst)
            temporal_exclusions[dst].append(src)

    exclusion_object = np.asarray([np.asarray(sorted(set(values)), dtype=np.int64) for values in temporal_exclusions], dtype=object)
    _save_npz(
        cache_path,
        effective_stride=np.asarray(stride),
        node_global_indices=node_global_indices,
        node_episode_ids=node_episode_ids,
        node_timesteps=node_timesteps,
        node_state_full=node_state_full,
        node_xy=node_xy,
        temporal_src=np.asarray(temporal_src, dtype=np.int64),
        temporal_dst=np.asarray(temporal_dst, dtype=np.int64),
        temporal_cost=np.asarray(temporal_cost, dtype=np.float32),
        temporal_exclusions=exclusion_object,
        payload_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )
    return NodeSet(
        dataset_id=parsed.dataset_id,
        effective_stride=stride,
        node_global_indices=node_global_indices,
        node_episode_ids=node_episode_ids,
        node_timesteps=node_timesteps,
        node_state_full=node_state_full,
        node_xy=node_xy,
        temporal_src=np.asarray(temporal_src, dtype=np.int64),
        temporal_dst=np.asarray(temporal_dst, dtype=np.int64),
        temporal_cost=np.asarray(temporal_cost, dtype=np.float32),
        temporal_exclusions=tuple(np.asarray(sorted(set(values)), dtype=np.int64) for values in temporal_exclusions),
    )


def _fit_state_pool(parsed: ParsedPlanningDataset, cfg: KNNPlanningEvalConfig) -> np.ndarray:
    state_values = _resolve_repr(parsed, cfg, transitions=False)
    count = min(int(cfg.fit_pool_size), int(state_values.shape[0]))
    indices = _rng(cfg.seed + 17).choice(state_values.shape[0], size=count, replace=False)
    return state_values[indices].astype(np.float32)


def _fit_transition_pool(parsed: ParsedPlanningDataset, cfg: KNNPlanningEvalConfig) -> tuple[np.ndarray, np.ndarray]:
    state_values = _resolve_repr(parsed, cfg, transitions=True)
    next_values = parsed.transition_xy if cfg.state_repr == "xy" else parsed.transition_next_state_full
    count = min(int(cfg.fit_pool_size), int(state_values.shape[0]))
    indices = _rng(cfg.seed + 29).choice(state_values.shape[0], size=count, replace=False)
    return state_values[indices].astype(np.float32), next_values[indices].astype(np.float32)


def _topk_from_score_block(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape[1] <= top_k:
        order = np.argsort(scores, axis=1)[:, ::-1]
        return order.astype(np.int64), np.take_along_axis(scores, order, axis=1).astype(np.float32)
    partition = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    partition_scores = np.take_along_axis(scores, partition, axis=1)
    order = np.argsort(partition_scores, axis=1)[:, ::-1]
    top_indices = np.take_along_axis(partition, order, axis=1)
    top_scores = np.take_along_axis(partition_scores, order, axis=1)
    return top_indices.astype(np.int64), top_scores.astype(np.float32)


def _apply_exclusions(scores: np.ndarray, row_start: int, exclusions: tuple[np.ndarray, ...]) -> None:
    for local_row in range(scores.shape[0]):
        global_row = row_start + local_row
        scores[local_row, global_row] = -np.inf
        banned = exclusions[global_row]
        if banned.size > 0:
            scores[local_row, banned] = -np.inf


def _apply_episode_quota_filter(topk: MethodTopK, nodes: NodeSet, same_episode_quota: int | None) -> MethodTopK:
    if same_episode_quota is None:
        return MethodTopK(
            method=topk.method,
            indices=np.asarray(topk.indices, dtype=np.int64).copy(),
            scores=np.asarray(topk.scores, dtype=np.float32).copy(),
        )
    filtered_indices = np.full_like(topk.indices, -1)
    filtered_scores = np.full_like(topk.scores, -np.inf)
    for node_id in range(topk.indices.shape[0]):
        episode_id = int(nodes.node_episode_ids[node_id])
        same_episode_kept = 0
        kept_indices: list[int] = []
        kept_scores: list[float] = []
        for cand, score in zip(topk.indices[node_id].tolist(), topk.scores[node_id].tolist()):
            cand = int(cand)
            if cand < 0:
                continue
            if int(nodes.node_episode_ids[cand]) == episode_id:
                if same_episode_kept >= int(same_episode_quota):
                    continue
                same_episode_kept += 1
            kept_indices.append(cand)
            kept_scores.append(float(score))
            if len(kept_indices) >= topk.indices.shape[1]:
                break
        if kept_indices:
            filtered_indices[node_id, : len(kept_indices)] = np.asarray(kept_indices, dtype=np.int64)
            filtered_scores[node_id, : len(kept_scores)] = np.asarray(kept_scores, dtype=np.float32)
    return MethodTopK(method=topk.method, indices=filtered_indices, scores=filtered_scores)


def _apply_cross_episode_only_filter(topk: MethodTopK, nodes: NodeSet) -> MethodTopK:
    return _apply_episode_quota_filter(topk, nodes, same_episode_quota=0)


def _chunked_topk_from_score_fn(
    num_rows: int,
    num_candidates: int,
    top_k: int,
    exclusions: tuple[np.ndarray, ...],
    row_block_size: int,
    score_fn: Any,
) -> MethodTopK:
    indices = np.full((num_rows, top_k), -1, dtype=np.int64)
    scores = np.full((num_rows, top_k), -np.inf, dtype=np.float32)
    for row_start in range(0, num_rows, row_block_size):
        row_end = min(row_start + row_block_size, num_rows)
        score_block = np.asarray(score_fn(row_start, row_end), dtype=np.float32)
        if score_block.shape != (row_end - row_start, num_candidates):
            raise ValueError(
                f"score_fn returned shape {score_block.shape}, expected {(row_end - row_start, num_candidates)}"
            )
        _apply_exclusions(score_block, row_start=row_start, exclusions=exclusions)
        block_idx, block_scores = _topk_from_score_block(score_block, top_k=top_k)
        indices[row_start:row_end] = block_idx
        scores[row_start:row_end] = block_scores
    return MethodTopK(method="", indices=indices, scores=scores)


def _resolve_gaussian_sigma(fit_values: np.ndarray, cfg: KNNPlanningEvalConfig) -> float:
    if cfg.gk_sigma_mode == "fixed":
        if cfg.gk_sigma is None or cfg.gk_sigma <= 0:
            raise ValueError("Fixed Gaussian sigma requires --gk_sigma > 0")
        return float(cfg.gk_sigma)

    sample = np.asarray(fit_values, dtype=np.float32)
    subset = sample[: min(sample.shape[0], 2048)]
    if subset.shape[0] < 2:
        return 1.0
    distances = cdist(subset, subset, metric="euclidean")
    np.fill_diagonal(distances, np.nan)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return 1.0
    sigma = float(np.median(finite))
    return sigma if sigma > 1e-6 else 1.0


def _compute_euclidean_or_gaussian_topk(
    nodes_repr: np.ndarray,
    exclusions: tuple[np.ndarray, ...],
    cfg: KNNPlanningEvalConfig,
    *,
    gaussian_sigma: float | None = None,
) -> MethodTopK:
    num_nodes = int(nodes_repr.shape[0])

    def _score_fn(row_start: int, row_end: int) -> np.ndarray:
        distances = cdist(nodes_repr[row_start:row_end], nodes_repr, metric="euclidean").astype(np.float32)
        if gaussian_sigma is None:
            return -distances
        scores = np.exp(-((distances ** 2) / (2.0 * max(float(gaussian_sigma), 1e-6) ** 2)))
        return scores.astype(np.float32)

    return _chunked_topk_from_score_fn(
        num_rows=num_nodes,
        num_candidates=num_nodes,
        top_k=cfg.retrieval_top_k,
        exclusions=exclusions,
        row_block_size=cfg.pairwise_row_block_size,
        score_fn=_score_fn,
    )


def _compute_mahalanobis_topk(
    fit_values: np.ndarray,
    nodes_repr: np.ndarray,
    exclusions: tuple[np.ndarray, ...],
    cfg: KNNPlanningEvalConfig,
) -> MethodTopK:
    metric = MahalanobisMetric.fit(
        fit_values,
        covariance_estimator=cfg.mahalanobis_covariance_estimator,
        eps=cfg.mahalanobis_eps,
    )
    if cfg.mahalanobis_implementation == "precision":
        transformed = nodes_repr @ metric.precision_matrix.T
    else:
        transformed = nodes_repr @ metric.whitening_matrix.T
    transformed = np.asarray(transformed, dtype=np.float32)
    num_nodes = int(nodes_repr.shape[0])

    def _score_fn(row_start: int, row_end: int) -> np.ndarray:
        distances = cdist(transformed[row_start:row_end], transformed, metric="euclidean").astype(np.float32)
        return -distances

    return _chunked_topk_from_score_fn(
        num_rows=num_nodes,
        num_candidates=num_nodes,
        top_k=cfg.retrieval_top_k,
        exclusions=exclusions,
        row_block_size=cfg.pairwise_row_block_size,
        score_fn=_score_fn,
    )


def _compute_adaptive_gaussian_topk(
    fit_values: np.ndarray,
    nodes_repr: np.ndarray,
    exclusions: tuple[np.ndarray, ...],
    cfg: KNNPlanningEvalConfig,
) -> MethodTopK:
    metric = AdaptiveGaussianMetric.fit(fit_values, k=cfg.adaptive_k_scale, eps=cfg.adaptive_eps)
    node_sigmas = metric.estimate_query_sigmas(nodes_repr).astype(np.float32)
    num_nodes = int(nodes_repr.shape[0])

    def _score_fn(row_start: int, row_end: int) -> np.ndarray:
        sqdist = cdist(nodes_repr[row_start:row_end], nodes_repr, metric="sqeuclidean").astype(np.float32)
        denom = 2.0 * np.maximum(node_sigmas[row_start:row_end, None] * node_sigmas[None, :], cfg.adaptive_eps)
        scores = np.exp(-(sqdist / denom))
        return scores.astype(np.float32)

    return _chunked_topk_from_score_fn(
        num_rows=num_nodes,
        num_candidates=num_nodes,
        top_k=cfg.retrieval_top_k,
        exclusions=exclusions,
        row_block_size=cfg.pairwise_row_block_size,
        score_fn=_score_fn,
    )


def _encode_ik_features(
    values: np.ndarray,
    fit_values: np.ndarray,
    cfg: KNNPlanningEvalConfig,
) -> Any:
    import torch

    requested_device = _resolve_torch_device(cfg.ik_device)

    def _run_once(device_name: str) -> tuple[Any, Any]:
        device = torch.device(device_name)
        kernel = SoftIsolationKernel(
            input_dim=int(values.shape[1]),
            ensemble_size=cfg.ik_ensemble_size,
            subsample_size=cfg.ik_subsample_size,
            temperature=cfg.ik_temperature,
            device=str(device),
        ).to(device)
        fit_tensor = torch.as_tensor(fit_values, dtype=torch.float32, device=device)
        kernel.fit(fit_tensor)
        outputs = []
        with torch.no_grad():
            value_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
            for start in range(0, value_tensor.shape[0], cfg.ik_batch_size):
                end = min(start + cfg.ik_batch_size, value_tensor.shape[0])
                features = kernel(value_tensor[start:end]) / math.sqrt(float(cfg.ik_ensemble_size))
                outputs.append(features)
        return torch.cat(outputs, dim=0), device

    try:
        return _run_once(requested_device)
    except RuntimeError as error:
        if requested_device != "cuda" or not _is_torch_oom(error):
            raise
        torch.cuda.empty_cache()
        return _run_once("cpu")


def _fit_ik_kernel(
    fit_values: np.ndarray,
    cfg: KNNPlanningEvalConfig,
    *,
    device_name: str | None = None,
) -> tuple[Any, Any]:
    import torch

    resolved_device = device_name or _resolve_torch_device(cfg.ik_device)
    device = torch.device(resolved_device)
    fork_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=fork_devices):
        seed = _ik_random_seed(cfg)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        kernel = SoftIsolationKernel(
            input_dim=int(fit_values.shape[1]),
            ensemble_size=cfg.ik_ensemble_size,
            subsample_size=cfg.ik_subsample_size,
            temperature=cfg.ik_temperature,
            device=str(device),
        ).to(device)
        fit_tensor = torch.as_tensor(fit_values, dtype=torch.float32, device=device)
        kernel.fit(fit_tensor)
    return kernel, device


def _encode_ik_block(
    kernel: Any,
    values_tensor: Any,
    start: int,
    end: int,
    cfg: KNNPlanningEvalConfig,
) -> Any:
    import torch

    outputs = []
    batch_size = max(1, int(cfg.ik_batch_size))
    while True:
        outputs = []
        try:
            with torch.no_grad():
                for block_start in range(int(start), int(end), batch_size):
                    block_end = min(block_start + batch_size, int(end))
                    features = kernel(values_tensor[block_start:block_end]) / math.sqrt(float(cfg.ik_ensemble_size))
                    outputs.append(features)
            return torch.cat(outputs, dim=0)
        except RuntimeError as error:
            if values_tensor.device.type != "cuda" or not _is_torch_oom(error) or batch_size <= 1:
                raise
            outputs = []
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)


def _compute_ik_block_size(cfg: KNNPlanningEvalConfig) -> int:
    total_anchors = max(1, int(cfg.ik_ensemble_size) * int(cfg.ik_subsample_size))
    budget_bytes = max(1, int(cfg.ik_feature_block_mb)) * 1024 * 1024
    budget_rows = max(1, budget_bytes // max(total_anchors * 4, 1))
    return max(1, min(int(cfg.pairwise_row_block_size), int(budget_rows)))


def _ik_random_seed(cfg: KNNPlanningEvalConfig) -> int:
    temp_bits = int(round(float(cfg.ik_temperature) * 1_000_000.0))
    seed = (
        int(cfg.seed)
        + 1009 * int(cfg.ik_ensemble_size)
        + 9176 * int(cfg.ik_subsample_size)
        + 31337 * temp_bits
    )
    return int(seed % (2**31 - 1))


def _compute_ik_topk(
    fit_values: np.ndarray,
    nodes_repr: np.ndarray,
    exclusions: tuple[np.ndarray, ...],
    cfg: KNNPlanningEvalConfig,
) -> MethodTopK:
    import torch

    num_nodes = int(nodes_repr.shape[0])
    indices = np.full((num_nodes, cfg.retrieval_top_k), -1, dtype=np.int64)
    scores = np.full((num_nodes, cfg.retrieval_top_k), -np.inf, dtype=np.float32)
    requested_device = _resolve_torch_device(cfg.ik_device)

    def _run_once(device_name: str) -> tuple[np.ndarray, np.ndarray]:
        kernel, device = _fit_ik_kernel(fit_values, cfg, device_name=device_name)
        node_tensor = torch.as_tensor(nodes_repr, dtype=torch.float32, device=device)
        feature_block_size = _compute_ik_block_size(cfg)

        with torch.no_grad():
            for row_start in range(0, num_nodes, feature_block_size):
                row_end = min(row_start + feature_block_size, num_nodes)
                row_features = _encode_ik_block(kernel, node_tensor, row_start, row_end, cfg)
                row_best_scores = np.full((row_end - row_start, cfg.retrieval_top_k), -np.inf, dtype=np.float32)
                row_best_indices = np.full((row_end - row_start, cfg.retrieval_top_k), -1, dtype=np.int64)
                for col_start in range(0, num_nodes, feature_block_size):
                    col_end = min(col_start + feature_block_size, num_nodes)
                    col_features = _encode_ik_block(kernel, node_tensor, col_start, col_end, cfg)
                    score_block = torch.matmul(row_features, col_features.T).detach().cpu().numpy().astype(np.float32)
                    candidate_indices = np.arange(col_start, col_end, dtype=np.int64)
                    for local_row in range(score_block.shape[0]):
                        global_row = row_start + local_row
                        row_scores = score_block[local_row]
                        if col_start <= global_row < col_end:
                            row_scores[global_row - col_start] = -np.inf
                        banned = exclusions[global_row]
                        if banned.size > 0:
                            mask = (banned >= col_start) & (banned < col_end)
                            if np.any(mask):
                                row_scores[banned[mask] - col_start] = -np.inf
                    repeated_indices = np.broadcast_to(candidate_indices[None, :], score_block.shape)
                    merged_scores = np.concatenate([row_best_scores, score_block], axis=1)
                    merged_indices = np.concatenate([row_best_indices, repeated_indices], axis=1)
                    select = min(cfg.retrieval_top_k, merged_scores.shape[1])
                    partition = np.argpartition(-merged_scores, kth=select - 1, axis=1)[:, :select]
                    partition_scores = np.take_along_axis(merged_scores, partition, axis=1)
                    partition_indices = np.take_along_axis(merged_indices, partition, axis=1)
                    order = np.argsort(partition_scores, axis=1)[:, ::-1]
                    row_best_scores = np.take_along_axis(partition_scores, order, axis=1).astype(np.float32)
                    row_best_indices = np.take_along_axis(partition_indices, order, axis=1).astype(np.int64)
                    del col_features
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                indices[row_start:row_end] = row_best_indices
                scores[row_start:row_end] = row_best_scores
                del row_features
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return indices.copy(), scores.copy()

    try:
        final_indices, final_scores = _run_once(requested_device)
    except RuntimeError as error:
        if requested_device != "cuda" or not _is_torch_oom(error):
            raise
        torch.cuda.empty_cache()
        final_indices, final_scores = _run_once("cpu")
    return MethodTopK(method="ik", indices=final_indices, scores=final_scores)


def _compute_temporal_distance_topk(nodes: NodeSet, cfg: KNNPlanningEvalConfig) -> MethodTopK:
    num_nodes = nodes.num_nodes
    indices = np.full((num_nodes, cfg.retrieval_top_k), -1, dtype=np.int64)
    scores = np.full((num_nodes, cfg.retrieval_top_k), -np.inf, dtype=np.float32)

    for episode_id in np.unique(nodes.node_episode_ids):
        episode_node_ids = np.flatnonzero(nodes.node_episode_ids == episode_id)
        ordered = episode_node_ids[np.argsort(nodes.node_timesteps[episode_node_ids])]
        ordered_timesteps = nodes.node_timesteps[ordered]
        for pos, node_id in enumerate(ordered.tolist()):
            collected: list[tuple[float, int]] = []
            for offset in range(1, ordered.shape[0]):
                left = pos - offset
                right = pos + offset
                candidates = []
                if left >= 0:
                    candidates.append(int(ordered[left]))
                if right < ordered.shape[0]:
                    candidates.append(int(ordered[right]))
                if not candidates:
                    break
                for cand in candidates:
                    if cand in nodes.temporal_exclusions[node_id]:
                        continue
                    delta = abs(int(nodes.node_timesteps[cand]) - int(ordered_timesteps[pos]))
                    collected.append((1.0 / (1.0 + float(delta)), cand))
                if len(collected) >= cfg.retrieval_top_k:
                    break
            if collected:
                collected.sort(key=lambda item: (-item[0], item[1]))
                top = collected[: cfg.retrieval_top_k]
                indices[node_id, : len(top)] = np.asarray([cand for _, cand in top], dtype=np.int64)
                scores[node_id, : len(top)] = np.asarray([score for score, _ in top], dtype=np.float32)
    return MethodTopK(method="temporal_distance", indices=indices, scores=scores)


def _compute_one_step_successor_clouds(
    transition_states: np.ndarray,
    transition_next_states: np.ndarray,
    node_repr: np.ndarray,
    cfg: KNNPlanningEvalConfig,
) -> np.ndarray:
    tree = cKDTree(transition_states)
    _, indices = tree.query(node_repr, k=min(cfg.one_step_m, transition_states.shape[0]))
    if indices.ndim == 1:
        indices = indices[:, None]
    return transition_next_states[np.asarray(indices, dtype=np.int64)].astype(np.float32)


def _compute_one_step_dynamics_topk(
    transition_states: np.ndarray,
    transition_next_states: np.ndarray,
    nodes_repr: np.ndarray,
    exclusions: tuple[np.ndarray, ...],
    cfg: KNNPlanningEvalConfig,
) -> MethodTopK:
    import torch

    successor_clouds = _compute_one_step_successor_clouds(transition_states, transition_next_states, nodes_repr, cfg)
    requested_device = _resolve_torch_device(cfg.ik_device)

    def _run_once(device_name: str) -> tuple[np.ndarray, np.ndarray]:
        device = torch.device(device_name)
        node_tensor = torch.as_tensor(nodes_repr, dtype=torch.float32, device=device)
        successor_tensor = torch.as_tensor(successor_clouds, dtype=torch.float32, device=device)
        num_nodes = int(nodes_repr.shape[0])
        indices = np.full((num_nodes, cfg.retrieval_top_k), -1, dtype=np.int64)
        scores = np.full((num_nodes, cfg.retrieval_top_k), -np.inf, dtype=np.float32)

        with torch.no_grad():
            for row_start in range(0, num_nodes, cfg.one_step_row_block_size):
                row_end = min(row_start + cfg.one_step_row_block_size, num_nodes)
                flat_successors = successor_tensor[row_start:row_end].reshape(-1, successor_tensor.shape[-1])
                distances = torch.cdist(flat_successors, node_tensor)
                distances = distances.view(row_end - row_start, successor_tensor.shape[1], num_nodes)
                min_distances = torch.min(distances, dim=1).values
                score_block = (-min_distances).detach().cpu().numpy().astype(np.float32)
                _apply_exclusions(score_block, row_start=row_start, exclusions=exclusions)
                block_idx, block_score = _topk_from_score_block(score_block, top_k=cfg.retrieval_top_k)
                indices[row_start:row_end] = block_idx
                scores[row_start:row_end] = block_score
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return indices, scores

    try:
        indices, scores = _run_once(requested_device)
    except RuntimeError as error:
        if requested_device != "cuda" or not _is_torch_oom(error):
            raise
        torch.cuda.empty_cache()
        indices, scores = _run_once("cpu")
    return MethodTopK(method="one_step_dynamics", indices=indices, scores=scores)


def load_or_compute_method_topk(
    parsed: ParsedPlanningDataset,
    nodes: NodeSet,
    cfg: KNNPlanningEvalConfig,
    method: str,
) -> MethodTopK:
    effective_same_episode_quota = _effective_same_episode_quota(cfg)
    payload = {
        "dataset": parsed.dataset_id,
        "method": method,
        "cache_scope": cfg.cache_scope,
        "query_bank_id": cfg.query_bank_id,
        "state_repr": cfg.state_repr,
        "state_variant": cfg.state_variant,
        "same_episode_quota": effective_same_episode_quota,
        "effective_stride": nodes.effective_stride,
        "retrieval_top_k": cfg.retrieval_top_k,
        "fit_pool_size": cfg.fit_pool_size,
        "seed": cfg.seed,
        "gk_sigma_mode": cfg.gk_sigma_mode,
        "gk_sigma": cfg.gk_sigma,
        "mahalanobis_covariance_estimator": cfg.mahalanobis_covariance_estimator,
        "mahalanobis_implementation": cfg.mahalanobis_implementation,
        "mahalanobis_eps": cfg.mahalanobis_eps,
        "ik_ensemble_size": cfg.ik_ensemble_size,
        "ik_subsample_size": cfg.ik_subsample_size,
        "ik_temperature": cfg.ik_temperature,
        "adaptive_eps": cfg.adaptive_eps,
        "adaptive_k_scale": cfg.adaptive_k_scale,
        "one_step_m": cfg.one_step_m,
    }
    if method in {"ik", "ik_temporal_bridge"}:
        payload["ik_sampling_version"] = 2
    if method == "ik_temporal_bridge":
        payload["ik_temporal_bridge_k"] = int(cfg.ik_temporal_bridge_k)
    cache_path = os.path.join(
        cfg.cache_dir,
        f"topk_{dataset_slug(parsed.dataset_id)}_{method}_{_payload_hash(payload)}.npz",
    )
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return MethodTopK(
                method=method,
                indices=np.asarray(cached["indices"], dtype=np.int64),
                scores=np.asarray(cached["scores"], dtype=np.float32),
            )

    raw_repr = _resolve_repr(parsed, cfg, transitions=False)
    node_repr = parsed.xy[nodes.node_global_indices].astype(np.float32) if cfg.state_repr == "xy" else parsed.state_full[nodes.node_global_indices].astype(np.float32)
    fit_values = _fit_state_pool(parsed, cfg)
    transition_states, transition_next_states = _fit_transition_pool(parsed, cfg)

    if method == "euclidean":
        result = _compute_euclidean_or_gaussian_topk(node_repr, nodes.temporal_exclusions, cfg, gaussian_sigma=None)
    elif method == "gaussian":
        sigma = _resolve_gaussian_sigma(fit_values, cfg)
        result = _compute_euclidean_or_gaussian_topk(node_repr, nodes.temporal_exclusions, cfg, gaussian_sigma=sigma)
    elif method == "mahalanobis":
        result = _compute_mahalanobis_topk(fit_values, node_repr, nodes.temporal_exclusions, cfg)
    elif method == "adaptive_gaussian":
        result = _compute_adaptive_gaussian_topk(fit_values, node_repr, nodes.temporal_exclusions, cfg)
    elif method == "temporal_distance":
        result = _compute_temporal_distance_topk(nodes, cfg)
    elif method == "ik":
        result = _compute_ik_topk(fit_values, node_repr, nodes.temporal_exclusions, cfg)
    elif method == "ik_temporal_bridge":
        bridge_k = int(cfg.ik_temporal_bridge_k) if int(cfg.ik_temporal_bridge_k) > 0 else min(4, int(cfg.retrieval_top_k))
        ik_topk = load_or_compute_method_topk(parsed, nodes, cfg, "ik")
        temporal_topk = load_or_compute_method_topk(parsed, nodes, cfg, "temporal_distance")
        result = _merge_topk_with_temporal_bridges(
            ik_topk,
            temporal_topk,
            bridge_k=bridge_k,
            top_k=cfg.retrieval_top_k,
        )
    elif method == "one_step_dynamics":
        result = _compute_one_step_dynamics_topk(
            transition_states=transition_states,
            transition_next_states=transition_next_states,
            nodes_repr=node_repr,
            exclusions=nodes.temporal_exclusions,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unsupported retrieval method: {method}")

    if effective_same_episode_quota is not None:
        result = _apply_episode_quota_filter(result, nodes, same_episode_quota=effective_same_episode_quota)

    _save_npz(
        cache_path,
        indices=result.indices,
        scores=result.scores,
        payload_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )
    return MethodTopK(method=method, indices=result.indices, scores=result.scores)


def load_or_sample_queries(parsed: ParsedPlanningDataset, nodes: NodeSet, cfg: KNNPlanningEvalConfig) -> QuerySet:
    payload = {
        "dataset": parsed.dataset_id,
        "cache_scope": cfg.cache_scope,
        "query_bank_id": cfg.query_bank_id,
        "query_bank_size": cfg.query_bank_size,
        "num_queries": cfg.num_queries,
        "min_query_geodesic": _resolve_min_query_geodesic(parsed.dataset_id, cfg),
        "effective_stride": nodes.effective_stride,
        "seed": cfg.seed,
    }
    cache_path = os.path.join(cfg.cache_dir, f"queries_{dataset_slug(parsed.dataset_id)}_{_payload_hash(payload)}.npz")
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return QuerySet(
                start_node_ids=np.asarray(cached["start_node_ids"], dtype=np.int64),
                goal_node_ids=np.asarray(cached["goal_node_ids"], dtype=np.int64),
                query_geodesic=np.asarray(cached["query_geodesic"], dtype=np.float32),
                difficulty=np.asarray(cached["difficulty"]).astype(str),
                start_xy=np.asarray(cached["start_xy"], dtype=np.float32) if "start_xy" in cached else nodes.node_xy[np.asarray(cached["start_node_ids"], dtype=np.int64)].astype(np.float32),
                goal_xy=np.asarray(cached["goal_xy"], dtype=np.float32) if "goal_xy" in cached else nodes.node_xy[np.asarray(cached["goal_node_ids"], dtype=np.int64)].astype(np.float32),
            )

    rng = _rng(cfg.seed + 101)
    min_geodesic = _resolve_min_query_geodesic(parsed.dataset_id, cfg)
    candidates: list[tuple[int, int, float]] = []
    seen_pairs: set[tuple[int, int]] = set()
    attempts = 0
    batch_size = 2048
    while len(candidates) < max(cfg.num_queries * 12, cfg.num_queries + 64) and attempts < cfg.max_query_attempts:
        start_ids = rng.integers(0, nodes.num_nodes, size=batch_size)
        goal_ids = rng.integers(0, nodes.num_nodes, size=batch_size)
        mask = start_ids != goal_ids
        if not np.any(mask):
            attempts += batch_size
            continue
        start_ids = start_ids[mask]
        goal_ids = goal_ids[mask]
        distances = parsed.maze_spec.geodesic_for_pairs(nodes.node_xy[start_ids], nodes.node_xy[goal_ids])
        keep = distances >= min_geodesic
        for start_id, goal_id, distance in zip(start_ids[keep].tolist(), goal_ids[keep].tolist(), distances[keep].tolist()):
            key = (int(start_id), int(goal_id))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidates.append((int(start_id), int(goal_id), float(distance)))
        attempts += batch_size

    if not candidates:
        raise RuntimeError(f"Failed to sample any valid planning queries for {parsed.dataset_id}")

    candidate_array = np.asarray(candidates, dtype=np.float64)
    distances = candidate_array[:, 2].astype(np.float32)
    q1, q2 = np.quantile(distances, [1.0 / 3.0, 2.0 / 3.0])
    bins = {
        "easy": np.flatnonzero(distances <= q1),
        "medium": np.flatnonzero((distances > q1) & (distances <= q2)),
        "hard": np.flatnonzero(distances > q2),
    }
    per_bin = max(cfg.num_queries // 3, 1)
    selected_indices: list[int] = []
    for label in ("easy", "medium", "hard"):
        pool = bins[label]
        if pool.size == 0:
            continue
        chosen = rng.choice(pool, size=min(per_bin, pool.size), replace=False)
        selected_indices.extend(int(x) for x in chosen.tolist())
    if len(selected_indices) < cfg.num_queries:
        remaining = [idx for idx in range(candidate_array.shape[0]) if idx not in set(selected_indices)]
        remaining = remaining[: max(cfg.num_queries - len(selected_indices), 0)]
        selected_indices.extend(remaining)
    selected_indices = selected_indices[: cfg.num_queries]
    selected = candidate_array[selected_indices]
    selected_distances = selected[:, 2].astype(np.float32)
    difficulty = np.full(selected.shape[0], "hard", dtype=object)
    difficulty[selected_distances <= q1] = "easy"
    difficulty[(selected_distances > q1) & (selected_distances <= q2)] = "medium"

    query_set = QuerySet(
        start_node_ids=selected[:, 0].astype(np.int64),
        goal_node_ids=selected[:, 1].astype(np.int64),
        query_geodesic=selected_distances,
        difficulty=difficulty.astype(str),
        start_xy=nodes.node_xy[selected[:, 0].astype(np.int64)].astype(np.float32),
        goal_xy=nodes.node_xy[selected[:, 1].astype(np.int64)].astype(np.float32),
    )
    _save_npz(
        cache_path,
        start_node_ids=query_set.start_node_ids,
        goal_node_ids=query_set.goal_node_ids,
        query_geodesic=query_set.query_geodesic,
        difficulty=np.asarray(query_set.difficulty, dtype=object),
        start_xy=query_set.start_xy,
        goal_xy=query_set.goal_xy,
        payload_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )
    return query_set


def load_or_sample_query_bank(
    parsed: ParsedPlanningDataset,
    cfg: KNNPlanningEvalConfig,
    *,
    bank_size: int,
    cache_dir: str | None = None,
    query_bank_id: str = "shared",
) -> QuerySet:
    resolved_cache_dir = cache_dir or cfg.cache_dir
    payload = {
        "dataset": parsed.dataset_id,
        "query_bank_id": query_bank_id,
        "bank_size": int(bank_size),
        "min_query_geodesic": _resolve_min_query_geodesic(parsed.dataset_id, cfg),
        "seed": cfg.seed,
        "source": "parsed_states",
        "bank_sampling_version": 2,
    }
    cache_path = os.path.join(
        resolved_cache_dir,
        f"query_bank_{dataset_slug(parsed.dataset_id)}_{_payload_hash(payload)}.npz",
    )
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return QuerySet(
                start_node_ids=np.asarray(cached["start_node_ids"], dtype=np.int64),
                goal_node_ids=np.asarray(cached["goal_node_ids"], dtype=np.int64),
                query_geodesic=np.asarray(cached["query_geodesic"], dtype=np.float32),
                difficulty=np.asarray(cached["difficulty"]).astype(str),
                start_xy=np.asarray(cached["start_xy"], dtype=np.float32),
                goal_xy=np.asarray(cached["goal_xy"], dtype=np.float32),
            )

    rng = _rng(cfg.seed + 313)
    min_geodesic = _resolve_min_query_geodesic(parsed.dataset_id, cfg)
    total_states = int(parsed.xy.shape[0])
    candidates: list[tuple[int, int, float]] = []
    seen_pairs: set[tuple[int, int]] = set()
    attempts = 0
    batch_size = 4096
    target_pool_size = max(int(bank_size) * 12, int(bank_size) + 128)
    while len(candidates) < target_pool_size and attempts < cfg.max_query_attempts:
        start_ids = rng.integers(0, total_states, size=batch_size)
        goal_ids = rng.integers(0, total_states, size=batch_size)
        mask = start_ids != goal_ids
        if not np.any(mask):
            attempts += batch_size
            continue
        start_ids = start_ids[mask]
        goal_ids = goal_ids[mask]
        distances = parsed.maze_spec.geodesic_for_pairs(parsed.xy[start_ids], parsed.xy[goal_ids])
        keep = distances >= min_geodesic
        for start_id, goal_id, distance in zip(start_ids[keep].tolist(), goal_ids[keep].tolist(), distances[keep].tolist()):
            key = (int(start_id), int(goal_id))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidates.append((int(start_id), int(goal_id), float(distance)))
        attempts += batch_size

    if len(candidates) < int(bank_size):
        raise RuntimeError(
            f"Failed to sample enough shared planning queries for {parsed.dataset_id}: "
            f"wanted {bank_size}, got {len(candidates)}"
        )

    candidate_array = np.asarray(candidates, dtype=np.float64)
    distances = candidate_array[:, 2].astype(np.float32)
    q1, q2 = np.quantile(distances, [1.0 / 3.0, 2.0 / 3.0])
    bins = {
        "easy": np.flatnonzero(distances <= q1),
        "medium": np.flatnonzero((distances > q1) & (distances <= q2)),
        "hard": np.flatnonzero(distances > q2),
    }
    per_bin = max(int(bank_size) // 3, 1)
    selected_per_label: dict[str, list[int]] = {}
    selected_set: set[int] = set()
    for label in ("easy", "medium", "hard"):
        pool = bins[label]
        if pool.size == 0:
            selected_per_label[label] = []
            continue
        chosen = rng.choice(pool, size=min(per_bin, pool.size), replace=False)
        selected_per_label[label] = [int(idx) for idx in chosen.tolist()]
        selected_set.update(selected_per_label[label])
    selected_indices = _round_robin_indices(
        [
            selected_per_label.get("easy", []),
            selected_per_label.get("medium", []),
            selected_per_label.get("hard", []),
        ]
    )
    if len(selected_indices) < int(bank_size):
        for idx in range(candidate_array.shape[0]):
            if idx in selected_set:
                continue
            selected_indices.append(int(idx))
            selected_set.add(int(idx))
            if len(selected_indices) >= int(bank_size):
                break
    selected_indices = selected_indices[: int(bank_size)]
    selected = candidate_array[selected_indices]
    selected_distances = selected[:, 2].astype(np.float32)
    difficulty = np.full(selected.shape[0], "hard", dtype=object)
    difficulty[selected_distances <= q1] = "easy"
    difficulty[(selected_distances > q1) & (selected_distances <= q2)] = "medium"

    query_bank = QuerySet(
        start_node_ids=selected[:, 0].astype(np.int64),
        goal_node_ids=selected[:, 1].astype(np.int64),
        query_geodesic=selected_distances,
        difficulty=difficulty.astype(str),
        start_xy=parsed.xy[selected[:, 0].astype(np.int64)].astype(np.float32),
        goal_xy=parsed.xy[selected[:, 1].astype(np.int64)].astype(np.float32),
    )
    _save_npz(
        cache_path,
        start_node_ids=query_bank.start_node_ids,
        goal_node_ids=query_bank.goal_node_ids,
        query_geodesic=query_bank.query_geodesic,
        difficulty=np.asarray(query_bank.difficulty, dtype=object),
        start_xy=query_bank.start_xy,
        goal_xy=query_bank.goal_xy,
        payload_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )
    return query_bank


def _select_queries_from_bank(query_bank: QuerySet, cfg: KNNPlanningEvalConfig) -> QuerySet:
    selected_indices = [
        idx
        for idx, label in enumerate(np.asarray(query_bank.difficulty).astype(str).tolist())
        if _matches_query_difficulty(label, cfg.query_difficulty_filter)
    ]
    if not selected_indices:
        raise RuntimeError(
            f"No queries match difficulty filter '{cfg.query_difficulty_filter}' in bank '{cfg.query_bank_id}'."
        )
    limit = min(_resolve_query_limit(cfg), len(selected_indices))
    chosen = np.asarray(selected_indices[:limit], dtype=np.int64)
    return QuerySet(
        start_node_ids=np.asarray(query_bank.start_node_ids[chosen], dtype=np.int64),
        goal_node_ids=np.asarray(query_bank.goal_node_ids[chosen], dtype=np.int64),
        query_geodesic=np.asarray(query_bank.query_geodesic[chosen], dtype=np.float32),
        difficulty=np.asarray(query_bank.difficulty[chosen]).astype(str),
        start_xy=np.asarray(query_bank.start_xy[chosen], dtype=np.float32) if query_bank.start_xy is not None else None,
        goal_xy=np.asarray(query_bank.goal_xy[chosen], dtype=np.float32) if query_bank.goal_xy is not None else None,
    )


def load_or_prepare_queries(parsed: ParsedPlanningDataset, nodes: NodeSet, cfg: KNNPlanningEvalConfig) -> QuerySet:
    if cfg.query_source == "node_sample":
        return load_or_sample_queries(parsed, nodes, cfg)
    if cfg.query_source != "shared_bank":
        raise ValueError(f"Unsupported query source: {cfg.query_source}")

    payload = {
        "dataset": parsed.dataset_id,
        "query_source": cfg.query_source,
        "query_bank_id": cfg.query_bank_id,
        "query_bank_size": _resolve_query_bank_size(cfg),
        "query_difficulty_filter": cfg.query_difficulty_filter,
        "query_limit": _resolve_query_limit(cfg),
        "seed": cfg.seed,
        "cache_scope": cfg.cache_scope,
    }
    cache_path = os.path.join(
        cfg.cache_dir,
        f"prepared_queries_{dataset_slug(parsed.dataset_id)}_{_payload_hash(payload)}.npz",
    )
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return QuerySet(
                start_node_ids=np.asarray(cached["start_node_ids"], dtype=np.int64),
                goal_node_ids=np.asarray(cached["goal_node_ids"], dtype=np.int64),
                query_geodesic=np.asarray(cached["query_geodesic"], dtype=np.float32),
                difficulty=np.asarray(cached["difficulty"]).astype(str),
                start_xy=np.asarray(cached["start_xy"], dtype=np.float32),
                goal_xy=np.asarray(cached["goal_xy"], dtype=np.float32),
            )

    query_bank = load_or_sample_query_bank(
        parsed,
        cfg,
        bank_size=_resolve_query_bank_size(cfg),
        cache_dir=cfg.cache_dir,
        query_bank_id=cfg.query_bank_id,
    )
    selected = _select_queries_from_bank(query_bank, cfg)
    _save_npz(
        cache_path,
        start_node_ids=selected.start_node_ids,
        goal_node_ids=selected.goal_node_ids,
        query_geodesic=selected.query_geodesic,
        difficulty=np.asarray(selected.difficulty, dtype=object),
        start_xy=selected.start_xy,
        goal_xy=selected.goal_xy,
        payload_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )
    return selected


def _build_graph_arrays(
    parsed: ParsedPlanningDataset,
    nodes: NodeSet,
    cfg: KNNPlanningEvalConfig,
    topk: MethodTopK,
) -> dict[str, np.ndarray]:
    h_bridge = _resolve_h_bridge(parsed.dataset_id, cfg)
    flat_rows = np.repeat(np.arange(nodes.num_nodes, dtype=np.int64), cfg.retrieval_top_k)
    flat_cols = topk.indices.reshape(-1)
    valid_candidates = flat_cols >= 0
    flat_rows = flat_rows[valid_candidates]
    flat_cols = flat_cols[valid_candidates]
    retrieval_geodesic = parsed.maze_spec.geodesic_for_pairs(nodes.node_xy[flat_rows], nodes.node_xy[flat_cols])
    valid_edges = retrieval_geodesic <= h_bridge

    retrieval_src = flat_rows[valid_edges].astype(np.int64)
    retrieval_dst = flat_cols[valid_edges].astype(np.int64)
    retrieval_dgt = retrieval_geodesic[valid_edges].astype(np.float32)
    retrieval_cost = (cfg.lambda_bridge * retrieval_dgt).astype(np.float32)

    return {
        "retrieval_src": retrieval_src,
        "retrieval_dst": retrieval_dst,
        "retrieval_dgt": retrieval_dgt,
        "retrieval_cost": retrieval_cost,
    }


def load_or_build_graph(
    parsed: ParsedPlanningDataset,
    nodes: NodeSet,
    cfg: KNNPlanningEvalConfig,
    topk: MethodTopK,
) -> dict[str, Any]:
    effective_same_episode_quota = _effective_same_episode_quota(cfg)
    payload = {
        "dataset": parsed.dataset_id,
        "method": topk.method,
        "cache_scope": cfg.cache_scope,
        "query_bank_id": cfg.query_bank_id,
        "state_variant": cfg.state_variant,
        "same_episode_quota": effective_same_episode_quota,
        "effective_stride": nodes.effective_stride,
        "top_k": cfg.retrieval_top_k,
        "lambda_bridge": cfg.lambda_bridge,
        "h_bridge": _resolve_h_bridge(parsed.dataset_id, cfg),
    }
    if topk.method in {"ik", "ik_temporal_bridge"}:
        payload["ik_sampling_version"] = 2
    if topk.method == "ik_temporal_bridge":
        payload["ik_temporal_bridge_k"] = int(cfg.ik_temporal_bridge_k)
    cache_path = os.path.join(
        cfg.cache_dir,
        f"graph_{dataset_slug(parsed.dataset_id)}_{topk.method}_{_payload_hash(payload)}.npz",
    )
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            edge_targets = tuple(np.asarray(arr, dtype=np.int64) for arr in cached["edge_targets"])
            edge_costs = tuple(np.asarray(arr, dtype=np.float32) for arr in cached["edge_costs"])
            edge_is_retrieval = tuple(np.asarray(arr, dtype=np.int8) for arr in cached["edge_is_retrieval"])
            edge_dgt = tuple(np.asarray(arr, dtype=np.float32) for arr in cached["edge_dgt"])
            return {
                "retrieval_src": np.asarray(cached["retrieval_src"], dtype=np.int64),
                "retrieval_dst": np.asarray(cached["retrieval_dst"], dtype=np.int64),
                "retrieval_dgt": np.asarray(cached["retrieval_dgt"], dtype=np.float32),
                "edge_targets": edge_targets,
                "edge_costs": edge_costs,
                "edge_is_retrieval": edge_is_retrieval,
                "edge_dgt": edge_dgt,
            }

    graph_arrays = _build_graph_arrays(parsed, nodes, cfg, topk)
    adjacency_targets: list[list[int]] = [[] for _ in range(nodes.num_nodes)]
    adjacency_costs: list[list[float]] = [[] for _ in range(nodes.num_nodes)]
    adjacency_is_retrieval: list[list[int]] = [[] for _ in range(nodes.num_nodes)]
    adjacency_dgt: list[list[float]] = [[] for _ in range(nodes.num_nodes)]

    for src, dst, cost in zip(nodes.temporal_src.tolist(), nodes.temporal_dst.tolist(), nodes.temporal_cost.tolist()):
        adjacency_targets[int(src)].append(int(dst))
        adjacency_costs[int(src)].append(float(cost))
        adjacency_is_retrieval[int(src)].append(0)
        adjacency_dgt[int(src)].append(0.0)

    for src, dst, cost, dgt in zip(
        graph_arrays["retrieval_src"].tolist(),
        graph_arrays["retrieval_dst"].tolist(),
        graph_arrays["retrieval_cost"].tolist(),
        graph_arrays["retrieval_dgt"].tolist(),
    ):
        adjacency_targets[int(src)].append(int(dst))
        adjacency_costs[int(src)].append(float(cost))
        adjacency_is_retrieval[int(src)].append(1)
        adjacency_dgt[int(src)].append(float(dgt))

    edge_targets = tuple(np.asarray(values, dtype=np.int64) for values in adjacency_targets)
    edge_costs = tuple(np.asarray(values, dtype=np.float32) for values in adjacency_costs)
    edge_is_retrieval = tuple(np.asarray(values, dtype=np.int8) for values in adjacency_is_retrieval)
    edge_dgt = tuple(np.asarray(values, dtype=np.float32) for values in adjacency_dgt)

    _save_npz(
        cache_path,
        retrieval_src=graph_arrays["retrieval_src"],
        retrieval_dst=graph_arrays["retrieval_dst"],
        retrieval_dgt=graph_arrays["retrieval_dgt"],
        edge_targets=np.asarray(edge_targets, dtype=object),
        edge_costs=np.asarray(edge_costs, dtype=object),
        edge_is_retrieval=np.asarray(edge_is_retrieval, dtype=object),
        edge_dgt=np.asarray(edge_dgt, dtype=object),
    )
    graph_arrays["edge_targets"] = edge_targets
    graph_arrays["edge_costs"] = edge_costs
    graph_arrays["edge_is_retrieval"] = edge_is_retrieval
    graph_arrays["edge_dgt"] = edge_dgt
    return graph_arrays


def multi_source_dijkstra(
    graph: dict[str, Any],
    num_nodes: int,
    source_ids: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, Any]:
    distances = np.full(num_nodes, np.inf, dtype=np.float64)
    predecessors = np.full(num_nodes, -1, dtype=np.int64)
    predecessor_is_retrieval = np.zeros(num_nodes, dtype=np.int8)
    predecessor_dgt = np.zeros(num_nodes, dtype=np.float32)
    expanded = 0
    heap: list[tuple[float, int]] = []

    for source in np.asarray(source_ids, dtype=np.int64):
        if source < 0 or source >= num_nodes:
            continue
        if distances[source] > 0.0:
            distances[source] = 0.0
            heapq.heappush(heap, (0.0, int(source)))

    if not heap:
        return {"found": False, "expanded_nodes": 0}

    while heap:
        current_dist, node_id = heapq.heappop(heap)
        if current_dist > distances[node_id]:
            continue
        expanded += 1
        if target_mask[node_id]:
            path_nodes = [int(node_id)]
            retrieval_edge_count = 0
            retrieval_edge_dgt: list[float] = []
            while predecessors[path_nodes[-1]] >= 0:
                current = path_nodes[-1]
                if predecessor_is_retrieval[current]:
                    retrieval_edge_count += 1
                    retrieval_edge_dgt.append(float(predecessor_dgt[current]))
                path_nodes.append(int(predecessors[current]))
            path_nodes.reverse()
            retrieval_edge_dgt.reverse()
            return {
                "found": True,
                "path_nodes": np.asarray(path_nodes, dtype=np.int64),
                "path_cost": float(current_dist),
                "expanded_nodes": int(expanded),
                "retrieval_edge_count": int(retrieval_edge_count),
                "retrieval_edge_dgt": np.asarray(retrieval_edge_dgt, dtype=np.float32),
            }

        targets = graph["edge_targets"][node_id]
        costs = graph["edge_costs"][node_id]
        is_retrieval = graph["edge_is_retrieval"][node_id]
        dgt_values = graph["edge_dgt"][node_id]
        for edge_idx in range(targets.shape[0]):
            neigh = int(targets[edge_idx])
            next_dist = float(current_dist + float(costs[edge_idx]))
            if next_dist >= distances[neigh]:
                continue
            distances[neigh] = next_dist
            predecessors[neigh] = int(node_id)
            predecessor_is_retrieval[neigh] = int(is_retrieval[edge_idx])
            predecessor_dgt[neigh] = float(dgt_values[edge_idx])
            heapq.heappush(heap, (next_dist, neigh))

    return {"found": False, "expanded_nodes": int(expanded)}


def _dcg(gains: np.ndarray) -> float:
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    return float(np.sum(gains.astype(np.float64) * discounts))


def evaluate_retrieval_metrics(
    parsed: ParsedPlanningDataset,
    nodes: NodeSet,
    cfg: KNNPlanningEvalConfig,
    topk: MethodTopK,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    h_bridge = _resolve_h_bridge(parsed.dataset_id, cfg)
    effective_same_episode_quota = _effective_same_episode_quota(cfg)
    for row_start in range(0, nodes.num_nodes, cfg.pairwise_row_block_size):
        row_end = min(row_start + cfg.pairwise_row_block_size, nodes.num_nodes)
        geodesic_block = parsed.maze_spec.geodesic_distances(nodes.node_xy[row_start:row_end], nodes.node_xy)
        for local_row in range(row_end - row_start):
            node_id = row_start + local_row
            geodesic_row = geodesic_block[local_row].astype(np.float32)
            geodesic_row[node_id] = np.inf
            banned = nodes.temporal_exclusions[node_id]
            if banned.size > 0:
                geodesic_row[banned] = np.inf
            if effective_same_episode_quota == 0:
                same_episode = np.flatnonzero(nodes.node_episode_ids == nodes.node_episode_ids[node_id])
                if same_episode.size > 0:
                    geodesic_row[same_episode] = np.inf

            positives = geodesic_row <= h_bridge
            positive_count = int(np.sum(positives))
            retrieved_idx = topk.indices[node_id]
            valid_retrieved = retrieved_idx >= 0
            retrieved_dgt = geodesic_row[retrieved_idx[valid_retrieved]] if np.any(valid_retrieved) else np.zeros(0, dtype=np.float32)
            hits = int(np.sum(retrieved_dgt <= h_bridge))
            precision = float(hits / max(int(np.sum(valid_retrieved)), 1))
            recall = float(hits / positive_count) if positive_count > 0 else 0.0
            mean_retrieved_dgt = float(np.mean(retrieved_dgt)) if retrieved_dgt.size > 0 else float("nan")
            pred_gains = (1.0 / (1.0 + retrieved_dgt.astype(np.float64))) if retrieved_dgt.size > 0 else np.zeros(0, dtype=np.float64)
            ideal = np.sort((1.0 / (1.0 + geodesic_row[positives].astype(np.float64))))[::-1][: cfg.retrieval_top_k]
            ideal_dcg = _dcg(ideal)
            ndcg = (_dcg(pred_gains[: cfg.retrieval_top_k]) / ideal_dcg) if ideal_dcg > 1e-12 else 0.0
            rows.append(
                {
                    "dataset": parsed.dataset_id,
                    "method": topk.method,
                    "node_id": int(node_id),
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "mean_retrieved_geodesic": mean_retrieved_dgt,
                    "ndcg_at_k": ndcg,
                    "positive_count": positive_count,
                }
            )

    summary = {
        "dataset": parsed.dataset_id,
        "method": topk.method,
        "precision_at_k": _safe_nanmean([row["precision_at_k"] for row in rows]),
        "recall_at_k": _safe_nanmean([row["recall_at_k"] for row in rows]),
        "mean_retrieved_geodesic": _safe_nanmean([row["mean_retrieved_geodesic"] for row in rows]),
        "ndcg_at_k": _safe_nanmean([row["ndcg_at_k"] for row in rows]),
    }
    return rows, summary


def evaluate_planning_queries(
    parsed: ParsedPlanningDataset,
    nodes: NodeSet,
    cfg: KNNPlanningEvalConfig,
    graph: dict[str, Any],
    topk: MethodTopK,
    queries: QuerySet,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eps_start, eps_goal = _resolve_eps(parsed.dataset_id, cfg)
    tree = cKDTree(nodes.node_xy)
    target_cache: dict[int, np.ndarray] = {}
    query_rows: list[dict[str, Any]] = []
    successful_suboptimality = []
    successful_retrieval_edges = []
    successful_path_cost = []
    expanded_nodes = []

    for query_id in range(queries.start_node_ids.shape[0]):
        start_xy = _query_start_xy(queries, nodes, query_id)
        goal_xy = _query_goal_xy(queries, nodes, query_id)
        source_ids = np.asarray(tree.query_ball_point(start_xy, r=eps_start), dtype=np.int64)
        target_cache_key = int(query_id)
        if target_cache_key in target_cache:
            target_ids = target_cache[target_cache_key]
        else:
            target_ids = np.asarray(tree.query_ball_point(goal_xy, r=eps_goal), dtype=np.int64)
            target_cache[target_cache_key] = target_ids

        if source_ids.size == 0:
            query_rows.append(
                {
                    "dataset": parsed.dataset_id,
                    "method": topk.method,
                    "query_id": int(query_id),
                    "difficulty": str(queries.difficulty[query_id]),
                    "success": 0,
                    "path_found": 0,
                    "failure_reason": "no_source",
                    "query_geodesic": float(queries.query_geodesic[query_id]),
                    "path_cost": float("nan"),
                    "path_suboptimality": float("nan"),
                    "retrieval_edge_count": float("nan"),
                    "expanded_nodes": 0,
                }
            )
            continue
        if target_ids.size == 0:
            query_rows.append(
                {
                    "dataset": parsed.dataset_id,
                    "method": topk.method,
                    "query_id": int(query_id),
                    "difficulty": str(queries.difficulty[query_id]),
                    "success": 0,
                    "path_found": 0,
                    "failure_reason": "no_target",
                    "query_geodesic": float(queries.query_geodesic[query_id]),
                    "path_cost": float("nan"),
                    "path_suboptimality": float("nan"),
                    "retrieval_edge_count": float("nan"),
                    "expanded_nodes": 0,
                }
            )
            continue

        target_mask = np.zeros(nodes.num_nodes, dtype=bool)
        target_mask[target_ids] = True
        result = multi_source_dijkstra(graph, nodes.num_nodes, source_ids, target_mask)
        if not result["found"]:
            query_rows.append(
                {
                    "dataset": parsed.dataset_id,
                    "method": topk.method,
                    "query_id": int(query_id),
                    "difficulty": str(queries.difficulty[query_id]),
                    "success": 0,
                    "path_found": 0,
                    "failure_reason": "no_path",
                    "query_geodesic": float(queries.query_geodesic[query_id]),
                    "path_cost": float("nan"),
                    "path_suboptimality": float("nan"),
                    "retrieval_edge_count": float("nan"),
                    "expanded_nodes": int(result["expanded_nodes"]),
                }
            )
            expanded_nodes.append(int(result["expanded_nodes"]))
            continue

        path_nodes = np.asarray(result["path_nodes"], dtype=np.int64)
        path_cost = float(result["path_cost"])
        query_geodesic = float(queries.query_geodesic[query_id])
        suboptimality = path_cost / max(query_geodesic, 1e-6)
        start_match = float(np.linalg.norm(nodes.node_xy[path_nodes[0]] - start_xy)) <= eps_start
        goal_match = float(np.linalg.norm(nodes.node_xy[path_nodes[-1]] - goal_xy)) <= eps_goal
        retrieval_valid = bool(np.all(np.asarray(result["retrieval_edge_dgt"], dtype=np.float32) <= _resolve_h_bridge(parsed.dataset_id, cfg) + 1e-6))
        detour_ok = path_cost <= (cfg.alpha * query_geodesic)
        success = int(start_match and goal_match and retrieval_valid and detour_ok)
        failure_reason = ""
        if not success:
            failure_reason = "detour" if not detour_ok else "constraint"

        query_rows.append(
            {
                "dataset": parsed.dataset_id,
                "method": topk.method,
                "query_id": int(query_id),
                "difficulty": str(queries.difficulty[query_id]),
                "success": success,
                "path_found": 1,
                "failure_reason": failure_reason,
                "query_geodesic": query_geodesic,
                "path_cost": path_cost,
                "path_suboptimality": suboptimality,
                "retrieval_edge_count": int(result["retrieval_edge_count"]),
                "expanded_nodes": int(result["expanded_nodes"]),
                "path_nodes": " ".join(str(int(x)) for x in path_nodes.tolist()),
            }
        )
        expanded_nodes.append(int(result["expanded_nodes"]))
        if success:
            successful_suboptimality.append(suboptimality)
            successful_retrieval_edges.append(int(result["retrieval_edge_count"]))
            successful_path_cost.append(path_cost)

    summary = {
        "dataset": parsed.dataset_id,
        "method": topk.method,
        "planning_success_rate": _safe_mean([float(row["success"]) for row in query_rows]),
        "path_suboptimality": _safe_mean(successful_suboptimality),
        "mean_num_retrieval_edges": _safe_mean(successful_retrieval_edges),
        "mean_path_cost": _safe_mean(successful_path_cost),
        "mean_expanded_nodes": _safe_mean(expanded_nodes),
    }
    return query_rows, summary


def _merge_metric_summaries(
    retrieval_summary: dict[str, Any],
    planning_summary: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(retrieval_summary)
    merged.update(planning_summary)
    return merged


def evaluate_single_method(
    parsed: ParsedPlanningDataset,
    nodes: NodeSet,
    cfg: KNNPlanningEvalConfig,
    method: str,
    queries: QuerySet,
) -> tuple[MethodTopK, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    topk = load_or_compute_method_topk(parsed, nodes, cfg, method)
    retrieval_rows, retrieval_summary = evaluate_retrieval_metrics(parsed, nodes, cfg, topk)
    graph = load_or_build_graph(parsed, nodes, cfg, topk)
    query_rows, planning_summary = evaluate_planning_queries(parsed, nodes, cfg, graph, topk, queries)
    merged = _merge_metric_summaries(retrieval_summary, planning_summary)
    return topk, merged, retrieval_rows, query_rows


def _overall_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    result = []
    for method, method_rows in sorted(grouped.items()):
        result.append(
            {
                "dataset": "overall",
                "method": method,
                "precision_at_k": _safe_nanmean([row["precision_at_k"] for row in method_rows]),
                "recall_at_k": _safe_nanmean([row["recall_at_k"] for row in method_rows]),
                "mean_retrieved_geodesic": _safe_nanmean([row["mean_retrieved_geodesic"] for row in method_rows]),
                "ndcg_at_k": _safe_nanmean([row["ndcg_at_k"] for row in method_rows]),
                "planning_success_rate": _safe_nanmean([row["planning_success_rate"] for row in method_rows]),
                "path_suboptimality": _safe_nanmean([row["path_suboptimality"] for row in method_rows]),
                "mean_num_retrieval_edges": _safe_nanmean([row["mean_num_retrieval_edges"] for row in method_rows]),
                "mean_path_cost": _safe_nanmean([row["mean_path_cost"] for row in method_rows]),
                "mean_expanded_nodes": _safe_nanmean([row["mean_expanded_nodes"] for row in method_rows]),
            }
        )
    return result


def _plot_dataset_bars(dataset_id: str, rows: list[dict[str, Any]], figure_path: str) -> None:
    _ensure_matplotlib_cache_dir()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [row["method"] for row in rows]
    success = [float(row["planning_success_rate"]) for row in rows]
    precision = [float(row["precision_at_k"]) for row in rows]
    suboptimality = [float(row["path_suboptimality"]) if np.isfinite(row["path_suboptimality"]) else 0.0 for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(methods, success, color="#4C78A8")
    axes[0].set_title("Planning Success")
    axes[1].bar(methods, precision, color="#F58518")
    axes[1].set_title("Precision@k")
    axes[2].bar(methods, suboptimality, color="#54A24B")
    axes[2].set_title("Path Suboptimality")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle(dataset_id)
    fig.tight_layout()
    ensure_dir(os.path.dirname(figure_path))
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_maze_background(ax: Any, maze_spec: MazeSpec) -> None:
    from matplotlib.patches import Rectangle

    width = maze_spec.width
    height = maze_spec.height
    half_scale = 0.5 * float(maze_spec.maze_size_scaling)
    for row in range(height):
        for col in range(width):
            if maze_spec.maze_map[row, col] != 1:
                continue
            center = maze_spec.rowcol_to_xy((row, col))
            rect = (
                center[0] - half_scale,
                center[1] - half_scale,
                2.0 * half_scale,
                2.0 * half_scale,
            )
            ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], color="#D9D9D9"))


def _plot_query_paths(
    parsed: ParsedPlanningDataset,
    nodes: NodeSet,
    queries: QuerySet,
    per_query_rows: list[dict[str, Any]],
    methods: list[str],
    figure_path: str,
) -> None:
    _ensure_matplotlib_cache_dir()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    for row in per_query_rows:
        rows_by_method.setdefault(str(row["method"]), []).append(row)

    selected_query_id = None
    for query_id in sorted({int(row["query_id"]) for row in per_query_rows}):
        if any(int(row["query_id"]) == query_id and int(row["path_found"]) == 1 for row in per_query_rows):
            selected_query_id = query_id
            break
    if selected_query_id is None:
        if per_query_rows:
            selected_query_id = min(int(row["query_id"]) for row in per_query_rows)
        else:
            return

    start_xy = _query_start_xy(queries, nodes, selected_query_id)
    goal_xy = _query_goal_xy(queries, nodes, selected_query_id)

    fig, axes = plt.subplots(1, len(methods), figsize=(4.4 * len(methods), 4.4), squeeze=False)
    for axis, method in zip(axes[0], methods):
        _draw_maze_background(axis, parsed.maze_spec)
        axis.scatter(start_xy[0], start_xy[1], color="#2ca02c", s=60, marker="o")
        axis.scatter(goal_xy[0], goal_xy[1], color="#d62728", s=60, marker="X")
        method_rows = [row for row in rows_by_method.get(method, []) if int(row["query_id"]) == selected_query_id]
        if method_rows:
            row = method_rows[0]
            if "path_nodes" in row and row["path_nodes"]:
                path_nodes = np.asarray([int(x) for x in str(row["path_nodes"]).split()], dtype=np.int64)
                path_xy = nodes.node_xy[path_nodes]
                axis.plot(path_xy[:, 0], path_xy[:, 1], color="#1f77b4", linewidth=2.2)
        axis.set_title(method)
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    fig.suptitle(f"{parsed.dataset_id} query {selected_query_id}")
    fig.tight_layout()
    ensure_dir(os.path.dirname(figure_path))
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_report(
    cfg: KNNPlanningEvalConfig,
    per_dataset_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    datasets: list[ParsedPlanningDataset],
) -> str:
    effective_same_episode_quota = _effective_same_episode_quota(cfg)
    best_success = max(overall_rows, key=lambda row: float(row["planning_success_rate"])) if overall_rows else None
    best_precision = max(overall_rows, key=lambda row: float(row["precision_at_k"])) if overall_rows else None
    temporal_row = next((row for row in overall_rows if str(row["method"]) == "temporal_distance"), None)
    lines = [
        "# 离线 kNN-based Planning 评估总结",
        "",
        "## 1. 任务定义",
        "",
        "- 节点来自离线轨迹中的下采样状态。",
        "- 时间边连接同一条轨迹中相邻采样节点，作为真实转移骨架。",
        "- 检索边先由相似度提出，再由独立的 geodesic ground truth 做可达性验证。",
        "- 规划阶段在统一构造的有向图上运行 Dijkstra，比较不同相似度对 retrieval edge 质量和最终规划成效的影响。",
        "",
    ]
    if cfg.task_preset == "large_v2_ik_favoring":
        lines.extend(
            [
                "### large_v2_ik_favoring preset",
                "",
                "- 该任务是 `large-v2` 上的 IK-favoring stress test，用于考察 cross-trajectory local stitching。",
                "- retrieval candidates 只允许跨 episode，主动弱化 temporal long-jump。",
                "- 查询固定为 `ik_shared_bank_v2` 中的 `easy30` 前缀。",
                "- 状态表示使用 `nuisance_v1`：在原始 pointmaze 状态上附加 velocity Fourier features 与 density-scaled repeated velocity nuisance dims。",
                "",
            ]
        )
    elif cfg.task_preset == "large_v2_ik_soft_local_stitching":
        lines.extend(
            [
                "### large_v2_ik_soft_local_stitching preset",
                "",
                "- 该任务是 `large-v2` 上的 softer IK-friendly local stitching task，用于在保持 IK 排名第一的同时，让 `temporal_distance` 不再退化到 0。",
                "- retrieval candidates 采用 same-episode quota 策略：每个节点最多保留 1 个 same-episode retrieval candidate，其余候选仍然优先考察 cross-trajectory stitching。",
                "- 查询固定为 `ik_shared_bank_v2` 中的 `easy30` 前缀，状态表示使用 `nuisance_v1`。",
                "- 这是当前已验证的最宽松折中：若把 `alpha` 放宽到 `1.0`，IK 就会失去按 `success -> suboptimality` 的领先位置。",
                "",
            ]
        )
    elif cfg.task_preset == "antmaze_umaze_detour_focus":
        lines.extend(
            [
                "### antmaze_umaze_detour_focus preset",
                "",
                "- 该任务固定在 `D4RL/antmaze/umaze-diverse-v1`，保留原始 full-state 表示与默认 graph construction。",
                "- 设计重点是不改 retrieval graph，只收紧 detour 预算，把 `alpha` 从 `1.5` 收紧到 `0.88`。",
                "- 这样可以在不压垮所有方法的前提下，让 success rate 从接近 `1.0` 的饱和区间回落到有区分度的 regime。",
                "- 当前验证结果表明：在这版任务上，IK 与 Mahalanobis 的 success rate 打平，但 IK 的成功路径平均 suboptimality 更小，因此按 `success -> suboptimality` 排名第一。",
                "",
            ]
        )
    lines.extend(
        [
        "## 2. Ground Truth 构造",
        "",
        "- PointMaze 与 AntMaze 都从 `metadata.json -> env_spec.kwargs.maze_map` 恢复迷宫布局。",
        "- `d_gt(u, v)` 由三部分组成：`u` 到最近可通行 cell center 的欧氏距离、cell graph 最短路距离、以及 cell center 到 `v` 的欧氏距离。",
        "- PointMaze 的位置坐标使用 `achieved_goal`，它与 `qpos[:2]`、`observation[:2]` 一致。",
        "- AntMaze 的位置坐标明确使用 `achieved_goal / qpos[:2]`，不使用 `observation[:2]`。",
        "",
        "## 3. 相似度实现",
        "",
        "- IK：复用仓库中的 `SoftIsolationKernel`，先编码节点，再用特征内积做批量 top-k 检索。",
        "- Gaussian Kernel：默认使用 median heuristic 估计全局 `sigma`。",
        "- Euclidean distance：默认使用全状态欧氏距离，保留 `xy` 表示切换开关。",
        "- Temporal distance：仅在同轨迹内按时间差定义，相似度取 `1 / (1 + |Δt|)`，跨轨迹得分为 0。",
        "- Mahalanobis / whitening：使用 Ledoit-Wolf 协方差估计与 `eps` 对角正则，默认通过 whitening 后计算欧氏距离。",
        "- Adaptive Gaussian：每个点用第 `k_scale` 近邻距离估计局部尺度，再构造自适应高斯核。",
        "- One-step dynamics：对每个节点先找最近的离线 transition 状态，再把其 one-step successor cloud 与候选节点做最小欧氏距离匹配。",
        "",
        "## 4. 关键参数",
        "",
        f"- retrieval_top_k: `{cfg.retrieval_top_k}`",
        f"- lambda_bridge: `{cfg.lambda_bridge}`",
        f"- alpha: `{cfg.alpha}`",
        f"- pointmaze H_bridge: `{cfg.pointmaze_h_bridge}`",
        f"- antmaze H_bridge: `{cfg.antmaze_h_bridge}`",
        f"- pointmaze eps_start / eps_goal: `{cfg.pointmaze_eps_start}` / `{cfg.pointmaze_eps_goal}`",
        f"- antmaze eps_start / eps_goal: `{cfg.antmaze_eps_start}` / `{cfg.antmaze_eps_goal}`",
        f"- num_queries: `{cfg.num_queries}`",
        f"- state_repr: `{cfg.state_repr}`",
        f"- state_variant: `{cfg.state_variant}`",
        f"- cross_episode_only: `{cfg.cross_episode_only}`",
        f"- same_episode_quota (effective): `{effective_same_episode_quota}`",
        f"- query_source: `{cfg.query_source}`",
        f"- query_difficulty_filter: `{cfg.query_difficulty_filter}`",
        f"- query_limit: `{_resolve_query_limit(cfg)}`",
        f"- adaptive_k_scale: `{cfg.adaptive_k_scale}`",
        f"- one_step_m: `{cfg.one_step_m}`",
        "",
        "## 5. 数据集说明",
        "",
    ])
    for dataset in datasets:
        lines.extend(
            [
                f"- `{dataset.dataset_id}`：maze kind 为 `{dataset.maze_spec.maze_kind}`，cell scaling 为 `{dataset.maze_spec.maze_size_scaling}`。",
            ]
        )
    lines.extend(
        [
            "",
            "## 6. 总体结果",
            "",
        ]
    )
    for row in overall_rows:
        lines.append(
            f"- `{row['method']}`：success `{row['planning_success_rate']:.3f}`，precision@k `{row['precision_at_k']:.3f}`，suboptimality `{row['path_suboptimality']:.3f}`。"
        )
    lines.extend(["", "## 7. 结果解读", ""])
    if best_success is not None:
        lines.append(
            f"- 当前汇总里规划成功率最高的方法是 `{best_success['method']}`，success rate 为 `{best_success['planning_success_rate']:.3f}`。"
        )
    if best_precision is not None:
        lines.append(
            f"- 当前汇总里 retrieval precision@k 最高的方法是 `{best_precision['method']}`，precision@k 为 `{best_precision['precision_at_k']:.3f}`。"
        )
    if temporal_row is not None and cfg.task_preset == "large_v2_ik_favoring":
        lines.append(
            f"- sanity check：`temporal_distance` 的 success rate 为 `{temporal_row['planning_success_rate']:.3f}`，说明 strict cross-episode-only 版本确实显著弱化了 temporal long-jump。"
        )
    if temporal_row is not None and cfg.task_preset == "large_v2_ik_soft_local_stitching":
        lines.append(
            f"- sanity check：`temporal_distance` 的 success rate 为 `{temporal_row['planning_success_rate']:.3f}`，说明 soft quota 版本仍然显著弱化了 temporal long-jump，但不再把 same-episode retrieval 完全压到 0。"
        )
    lines.extend(
        [
            "- temporal distance 只在同轨迹内工作，因此通常 retrieval coverage 很弱，更适合作为保守基线而不是跨轨迹拼接方法。",
            "- IK、Adaptive Gaussian 与 Mahalanobis 在局部邻域建模上更灵活，通常更容易给出 geodesic 更短的候选邻居。",
            "",
            "## 8. 已知局限",
            "",
            "- one-step dynamics 为了控制计算量，使用了采样后的 transition pool，而不是全量逐步转移。",
            "- query 由采样节点图构造，因此最终难度分布会受到节点预算和有效 stride 的影响。",
            "- 正式大规模运行会显著比 dry-run 更耗时，因为 7 种方法都需要对同一节点集做精确 top-k 检索与 geodesic 验证。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_knn_planning_eval(cfg: KNNPlanningEvalConfig) -> dict[str, Any]:
    cfg = apply_task_preset(cfg)
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir)
    ensure_dir(os.path.join(cfg.output_dir, "tables"))
    ensure_dir(os.path.join(cfg.output_dir, "figures"))
    ensure_dir(os.path.join(cfg.output_dir, "logs"))

    with open(os.path.join(cfg.output_dir, "logs", "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, ensure_ascii=False)

    datasets = [load_or_parse_dataset(dataset_id, cfg) for dataset_id in cfg.datasets]
    methods = list(cfg.methods) if cfg.methods else list(DEFAULT_METHODS)

    per_dataset_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    per_node_rows: list[dict[str, Any]] = []

    for parsed in datasets:
        nodes = load_or_build_nodes(parsed, cfg)
        queries = load_or_prepare_queries(parsed, nodes, cfg)
        dataset_method_rows: list[dict[str, Any]] = []
        dataset_query_rows: list[dict[str, Any]] = []
        for method in methods:
            _, merged, retrieval_rows, query_rows = evaluate_single_method(parsed, nodes, cfg, method, queries)
            per_node_rows.extend(retrieval_rows)
            per_query_rows.extend(query_rows)
            dataset_method_rows.append(merged)
            per_dataset_rows.append(merged)
            dataset_query_rows.extend(query_rows)

        _plot_dataset_bars(
            parsed.dataset_id,
            dataset_method_rows,
            os.path.join(cfg.output_dir, "figures", f"{parsed.dataset_slug}_bars.png"),
        )
        _plot_query_paths(
            parsed,
            nodes,
            queries,
            dataset_query_rows,
            methods=methods,
            figure_path=os.path.join(cfg.output_dir, "figures", f"{parsed.dataset_slug}_query_paths.png"),
        )

    overall_rows = _overall_summary(per_dataset_rows)
    per_dataset_table = os.path.join(cfg.output_dir, "tables", "per_dataset_metrics.csv")
    per_query_table = os.path.join(cfg.output_dir, "tables", "per_query_metrics.csv")
    per_node_table = os.path.join(cfg.output_dir, "tables", "per_node_retrieval_metrics.csv")
    overall_table = os.path.join(cfg.output_dir, "tables", "overall_summary.csv")
    report_path = os.path.join(cfg.output_dir, "report_cn.md")

    save_csv(per_dataset_table, per_dataset_rows, fieldnames=list(per_dataset_rows[0].keys()))
    save_csv(per_query_table, per_query_rows, fieldnames=sorted({key for row in per_query_rows for key in row.keys()}))
    save_csv(per_node_table, per_node_rows, fieldnames=list(per_node_rows[0].keys()))
    save_csv(overall_table, overall_rows, fieldnames=list(overall_rows[0].keys()))

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(_build_report(cfg, per_dataset_rows, overall_rows, datasets))

    return {
        "per_dataset_table": per_dataset_table,
        "per_query_table": per_query_table,
        "per_node_table": per_node_table,
        "overall_table": overall_table,
        "report_path": report_path,
    }
