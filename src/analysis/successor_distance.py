from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .fitted_baselines import AdaptiveGaussianMetric
from .maze_geodesic import dataset_slug, ensure_dir
from .reachability_alignment import ParsedDataset, load_or_parse_dataset
from .similarity_metrics import (
    auc_from_binary_labels,
    average_precision_from_binary_labels,
    fit_soft_isolation_kernel,
    recall_at_k,
)


DEFAULT_DATASETS = [
    "D4RL/pointmaze/umaze-v2",
    "D4RL/pointmaze/large-v2",
    "D4RL/antmaze/umaze-diverse-v1",
]
DEFAULT_HORIZONS = [10, 20, 50]
DEFAULT_IK_ENSEMBLE_GRID = (100, 200, 400)
DEFAULT_IK_SUBSAMPLE_GRID = tuple(2**power for power in range(1, 16))
DEFAULT_IK_TEMPERATURE_GRID = (0.0001, 0.001, 0.002, 0.004, 0.008, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0)
METHOD_ORDER = ["raw", "idk", "gdk", "wasserstein_w2", "adaptive_gdk"]
METHOD_LABELS = {
    "raw": "Raw",
    "idk": "IDK",
    "gdk": "GDK",
    "wasserstein_w2": "W2",
    "adaptive_gdk": "Adaptive-GDK",
}
METHOD_COLORS = {
    "raw": "#4C78A8",
    "idk": "#237A57",
    "gdk": "#F58518",
    "wasserstein_w2": "#E45756",
    "adaptive_gdk": "#72B7B2",
}
SUCCESSOR_CACHE_VERSION = 1
IDK_SEARCH_FIELDS = [
    "dataset",
    "dataset_slug",
    "horizon",
    "split",
    "ik_ensemble_size",
    "ik_subsample_size",
    "ik_temperature",
    "num_pairs",
    "positive_fraction",
    "auroc",
    "auprc",
]
PARTIAL_SEARCH_SAVE_EVERY = 1
IDK_PROGRESS_LOG_EVERY = 1
IDK_MAX_CHUNK_ANCHORS = 65_536
IDK_MAX_CDIST_VALUES = 16_000_000


@dataclass(frozen=True)
class EpisodeSplit:
    train_episode_ids: np.ndarray
    val_episode_ids: np.ndarray
    test_episode_ids: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray

    def episode_ids_for(self, split_name: str) -> np.ndarray:
        key = str(split_name).strip().lower()
        if key == "train":
            return self.train_episode_ids
        if key == "val":
            return self.val_episode_ids
        if key == "test":
            return self.test_episode_ids
        raise ValueError(f"Unknown split name: {split_name}")

    def indices_for(self, split_name: str) -> np.ndarray:
        key = str(split_name).strip().lower()
        if key == "train":
            return self.train_indices
        if key == "val":
            return self.val_indices
        if key == "test":
            return self.test_indices
        raise ValueError(f"Unknown split name: {split_name}")


@dataclass(frozen=True)
class GridSpec:
    x_edges: np.ndarray
    y_edges: np.ndarray


@dataclass(frozen=True)
class FutureWindowBundle:
    split_name: str
    horizon: int
    valid_global_indices: np.ndarray
    future_windows: np.ndarray
    future_endpoints: np.ndarray
    future_region_ids: np.ndarray


@dataclass(frozen=True)
class RetrievalBank:
    query_local_indices: np.ndarray
    candidate_local_indices: np.ndarray


@dataclass(frozen=True)
class PairSample:
    first_local_indices: np.ndarray
    second_local_indices: np.ndarray
    labels: np.ndarray
    positive_fraction: float


@dataclass(frozen=True)
class SuccessorEvalResult:
    dataset: str
    horizon: int
    best_ik_row: dict[str, Any]
    summary_rows: list[dict[str, Any]]
    recall_rows: list[dict[str, Any]]
    figure_paths: dict[str, str]
    sigma: float


@dataclass
class SuccessorDistanceConfig:
    datasets: list[str]
    output_dir: str
    cache_dir: str
    seed: int = 0
    horizon_values: list[int] = field(default_factory=lambda: list(DEFAULT_HORIZONS))

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    grid_nx: int = 20
    grid_ny: int = 20
    search_num_pairs: int = 20000
    eval_num_pairs: int = 50000
    search_pair_batch_size: int = 256
    eval_pair_batch_size: int = 512

    num_queries: int = 128
    num_candidates: int = 256
    recall_k_values: tuple[int, ...] = (5, 10, 20)
    plot_top_k: int = 20
    query_matrix_batch_size: int = 8

    raw_gamma: float | None = None
    gdk_sigma_num_pairs: int = 20000
    adaptive_gaussian_k: int = 10
    adaptive_gaussian_eps: float = 1e-6
    fit_pool_size: int = 50000

    ik_ensemble_sizes: tuple[int, ...] = field(default_factory=lambda: DEFAULT_IK_ENSEMBLE_GRID)
    ik_subsample_sizes: tuple[int, ...] = field(default_factory=lambda: DEFAULT_IK_SUBSAMPLE_GRID)
    ik_temperatures: tuple[float, ...] = field(default_factory=lambda: DEFAULT_IK_TEMPERATURE_GRID)
    ik_batch_size: int = 4096
    ik_device: str = "auto"
    ik_explicit_max_feature_values: int = 50_000_000
    ik_chunk_ensemble_size: int = 8

    selection_metric: str = "auprc"
    overwrite_cache: bool = False
    minari_datasets_path: str = "/home/shangyy/.minari/datasets"

    @property
    def search_dir(self) -> str:
        return os.path.join(self.output_dir, "search")

    @property
    def figures_dir(self) -> str:
        return os.path.join(self.output_dir, "figures")

    @property
    def tables_dir(self) -> str:
        return os.path.join(self.output_dir, "tables")


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:12]


def _npz_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def _save_npz(path: str, **kwargs: Any) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **kwargs)


def _load_npz(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _safe_load_npz(path: str) -> dict[str, Any] | None:
    try:
        return _load_npz(path)
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _save_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_csv_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _parsed_from_payload(payload: dict[str, Any]) -> ParsedDataset:
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


def load_or_parse_dataset_with_fallback(
    dataset_id: str,
    cfg: SuccessorDistanceConfig,
) -> ParsedDataset:
    slug = dataset_slug(dataset_id)
    local_cache_path = os.path.join(cfg.cache_dir, f"dataset_parse_{slug}.npz")
    if _npz_exists(local_cache_path):
        payload = _safe_load_npz(local_cache_path)
        if payload is not None:
            return _parsed_from_payload(payload)

    fallback_pattern = os.path.join(_repo_root(), "outputs", "**", "cache", f"dataset_parse_{slug}.npz")
    for cache_path in sorted(glob.glob(fallback_pattern, recursive=True)):
        payload = _safe_load_npz(cache_path)
        if payload is None:
            continue
        if str(payload["dataset_id"].item()) != str(dataset_id):
            continue
        _save_npz(local_cache_path, **payload)
        return _parsed_from_payload(payload)

    try:
        return load_or_parse_dataset(
            dataset_id=dataset_id,
            cache_dir=cfg.cache_dir,
            overwrite_cache=cfg.overwrite_cache,
            minari_datasets_path=cfg.minari_datasets_path,
            seed=cfg.seed,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"Could not parse dataset {dataset_id}: neither local/fallback parse cache nor Minari is available."
        ) from error


def _split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.asarray([float(ratios[0]), float(ratios[1]), float(ratios[2])], dtype=np.float64)
    if not np.isclose(np.sum(raw), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")
    values = raw * float(total)
    counts = np.floor(values).astype(np.int64)
    remainder = int(total - int(np.sum(counts)))
    if remainder > 0:
        order = np.argsort(-(values - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _indices_from_episode_ids(parsed: ParsedDataset, episode_ids: np.ndarray) -> np.ndarray:
    indices = []
    for episode_id in np.asarray(episode_ids, dtype=np.int64).tolist():
        start = int(parsed.episode_offsets[int(episode_id)])
        end = int(parsed.episode_offsets[int(episode_id) + 1])
        indices.append(np.arange(start, end, dtype=np.int64))
    if not indices:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(indices, axis=0).astype(np.int64)


def load_or_create_episode_split(parsed: ParsedDataset, cfg: SuccessorDistanceConfig) -> EpisodeSplit:
    payload = {
        "cache_version": SUCCESSOR_CACHE_VERSION,
        "dataset": parsed.dataset_id,
        "seed": cfg.seed,
        "train_ratio": cfg.train_ratio,
        "val_ratio": cfg.val_ratio,
        "test_ratio": cfg.test_ratio,
    }
    cache_path = os.path.join(cfg.cache_dir, f"episode_split_{dataset_slug(parsed.dataset_id)}_{_hash_payload(payload)}.npz")
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return EpisodeSplit(
                train_episode_ids=np.asarray(cached["train_episode_ids"], dtype=np.int64),
                val_episode_ids=np.asarray(cached["val_episode_ids"], dtype=np.int64),
                test_episode_ids=np.asarray(cached["test_episode_ids"], dtype=np.int64),
                train_indices=np.asarray(cached["train_indices"], dtype=np.int64),
                val_indices=np.asarray(cached["val_indices"], dtype=np.int64),
                test_indices=np.asarray(cached["test_indices"], dtype=np.int64),
            )

    rng = np.random.default_rng(cfg.seed)
    all_episode_ids = np.arange(parsed.total_episodes, dtype=np.int64)
    shuffled = rng.permutation(all_episode_ids)
    train_count, val_count, test_count = _split_counts(
        parsed.total_episodes,
        (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio),
    )
    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError(
            f"Episode split produced an empty partition for {parsed.dataset_id}: "
            f"{train_count}/{val_count}/{test_count}"
        )
    train_episode_ids = np.sort(shuffled[:train_count]).astype(np.int64)
    val_episode_ids = np.sort(shuffled[train_count:train_count + val_count]).astype(np.int64)
    test_episode_ids = np.sort(shuffled[train_count + val_count:train_count + val_count + test_count]).astype(np.int64)
    split = EpisodeSplit(
        train_episode_ids=train_episode_ids,
        val_episode_ids=val_episode_ids,
        test_episode_ids=test_episode_ids,
        train_indices=_indices_from_episode_ids(parsed, train_episode_ids),
        val_indices=_indices_from_episode_ids(parsed, val_episode_ids),
        test_indices=_indices_from_episode_ids(parsed, test_episode_ids),
    )
    _save_npz(
        cache_path,
        train_episode_ids=split.train_episode_ids,
        val_episode_ids=split.val_episode_ids,
        test_episode_ids=split.test_episode_ids,
        train_indices=split.train_indices,
        val_indices=split.val_indices,
        test_indices=split.test_indices,
    )
    return split


def build_or_load_grid_spec(
    parsed: ParsedDataset,
    split: EpisodeSplit,
    cfg: SuccessorDistanceConfig,
) -> GridSpec:
    payload = {
        "cache_version": SUCCESSOR_CACHE_VERSION,
        "dataset": parsed.dataset_id,
        "seed": cfg.seed,
        "grid_nx": cfg.grid_nx,
        "grid_ny": cfg.grid_ny,
        "train_episode_ids": split.train_episode_ids.tolist(),
    }
    cache_path = os.path.join(cfg.cache_dir, f"grid_spec_{dataset_slug(parsed.dataset_id)}_{_hash_payload(payload)}.npz")
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return GridSpec(
                x_edges=np.asarray(cached["x_edges"], dtype=np.float32),
                y_edges=np.asarray(cached["y_edges"], dtype=np.float32),
            )

    train_positions = np.asarray(parsed.positions[split.train_indices], dtype=np.float32)
    x_min, y_min = np.min(train_positions, axis=0)
    x_max, y_max = np.max(train_positions, axis=0)
    if math.isclose(float(x_min), float(x_max)):
        x_max = x_min + 1.0
    if math.isclose(float(y_min), float(y_max)):
        y_max = y_min + 1.0
    x_edges = np.linspace(float(x_min), float(x_max), int(cfg.grid_nx) + 1, dtype=np.float32)
    y_edges = np.linspace(float(y_min), float(y_max), int(cfg.grid_ny) + 1, dtype=np.float32)
    spec = GridSpec(x_edges=x_edges, y_edges=y_edges)
    _save_npz(cache_path, x_edges=spec.x_edges, y_edges=spec.y_edges)
    return spec


def assign_region_ids(points: np.ndarray, grid: GridSpec) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    clipped_x = np.clip(values[:, 0], float(grid.x_edges[0]), float(grid.x_edges[-1]))
    clipped_y = np.clip(values[:, 1], float(grid.y_edges[0]), float(grid.y_edges[-1]))
    x_bins = np.clip(np.digitize(clipped_x, grid.x_edges[1:-1], right=False), 0, len(grid.x_edges) - 2)
    y_bins = np.clip(np.digitize(clipped_y, grid.y_edges[1:-1], right=False), 0, len(grid.y_edges) - 2)
    return (x_bins * (len(grid.y_edges) - 1) + y_bins).astype(np.int64)


def build_or_load_future_window_bundle(
    parsed: ParsedDataset,
    split: EpisodeSplit,
    split_name: str,
    horizon: int,
    grid: GridSpec,
    cfg: SuccessorDistanceConfig,
) -> FutureWindowBundle:
    split_episode_ids = split.episode_ids_for(split_name)
    payload = {
        "cache_version": SUCCESSOR_CACHE_VERSION,
        "dataset": parsed.dataset_id,
        "split_name": split_name,
        "horizon": int(horizon),
        "grid_nx": cfg.grid_nx,
        "grid_ny": cfg.grid_ny,
        "split_episode_ids": split_episode_ids.tolist(),
        "grid_x_min": float(grid.x_edges[0]),
        "grid_x_max": float(grid.x_edges[-1]),
        "grid_y_min": float(grid.y_edges[0]),
        "grid_y_max": float(grid.y_edges[-1]),
    }
    cache_path = os.path.join(cfg.cache_dir, f"future_bundle_{dataset_slug(parsed.dataset_id)}_{_hash_payload(payload)}.npz")
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return FutureWindowBundle(
                split_name=str(cached["split_name"].item()),
                horizon=int(cached["horizon"].item()),
                valid_global_indices=np.asarray(cached["valid_global_indices"], dtype=np.int64),
                future_windows=np.asarray(cached["future_windows"], dtype=np.float32),
                future_endpoints=np.asarray(cached["future_endpoints"], dtype=np.float32),
                future_region_ids=np.asarray(cached["future_region_ids"], dtype=np.int64),
            )

    split_mask = np.isin(parsed.episode_ids, split_episode_ids)
    remaining = parsed.episode_lengths[parsed.episode_ids] - parsed.timesteps - 1
    valid_mask = split_mask & (remaining >= int(horizon))
    valid_global_indices = np.flatnonzero(valid_mask).astype(np.int64)
    if valid_global_indices.size == 0:
        raise RuntimeError(f"No valid H={horizon} states found for {parsed.dataset_id} split={split_name}")

    offsets = np.arange(1, int(horizon) + 1, dtype=np.int64)[None, :]
    future_windows = np.asarray(parsed.positions[valid_global_indices[:, None] + offsets], dtype=np.float32)
    future_endpoints = np.asarray(future_windows[:, -1, :], dtype=np.float32)
    future_region_ids = assign_region_ids(future_endpoints, grid)
    bundle = FutureWindowBundle(
        split_name=str(split_name),
        horizon=int(horizon),
        valid_global_indices=valid_global_indices,
        future_windows=future_windows,
        future_endpoints=future_endpoints,
        future_region_ids=future_region_ids.astype(np.int64),
    )
    _save_npz(
        cache_path,
        split_name=np.asarray(bundle.split_name),
        horizon=np.asarray(bundle.horizon),
        valid_global_indices=bundle.valid_global_indices,
        future_windows=bundle.future_windows,
        future_endpoints=bundle.future_endpoints,
        future_region_ids=bundle.future_region_ids,
    )
    return bundle


def _resolve_torch_device(device: str) -> str:
    if device != "auto":
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _torch_seed(seed: int) -> None:
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _sample_pair_indices(
    region_ids: np.ndarray,
    num_pairs: int,
    seed: int,
) -> PairSample:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    regions = np.asarray(region_ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    positives_target = num_pairs // 2
    negatives_target = num_pairs - positives_target

    region_to_members: dict[int, np.ndarray] = {}
    for region_id in np.unique(regions):
        members = np.flatnonzero(regions == int(region_id)).astype(np.int64)
        region_to_members[int(region_id)] = members
    positive_regions = [region_id for region_id, members in region_to_members.items() if members.size >= 2]
    if not positive_regions:
        raise RuntimeError("No positive regions with at least two members were found.")

    first_list: list[int] = []
    second_list: list[int] = []
    labels: list[int] = []

    for _ in range(positives_target):
        region_id = int(positive_regions[rng.integers(0, len(positive_regions))])
        members = region_to_members[region_id]
        chosen = rng.choice(members, size=2, replace=False).astype(np.int64)
        first_list.append(int(chosen[0]))
        second_list.append(int(chosen[1]))
        labels.append(1)

    total = int(regions.shape[0])
    negative_attempts = 0
    while len(labels) < num_pairs:
        pair = rng.integers(0, total, size=2)
        negative_attempts += 1
        if int(pair[0]) == int(pair[1]):
            continue
        if int(regions[int(pair[0])]) == int(regions[int(pair[1])]):
            continue
        first_list.append(int(pair[0]))
        second_list.append(int(pair[1]))
        labels.append(0)
        if negative_attempts > max(10 * negatives_target, 1000) and len(labels) < positives_target + 1:
            raise RuntimeError("Failed to sample enough negative pairs with distinct future regions.")

    labels_array = np.asarray(labels, dtype=np.int64)
    return PairSample(
        first_local_indices=np.asarray(first_list, dtype=np.int64),
        second_local_indices=np.asarray(second_list, dtype=np.int64),
        labels=labels_array,
        positive_fraction=float(np.mean(labels_array)),
    )


def load_or_create_pair_sample(
    parsed: ParsedDataset,
    bundle: FutureWindowBundle,
    split_name: str,
    horizon: int,
    num_pairs: int,
    seed: int,
    cfg: SuccessorDistanceConfig,
) -> PairSample:
    payload = {
        "cache_version": SUCCESSOR_CACHE_VERSION,
        "dataset": parsed.dataset_id,
        "split_name": split_name,
        "horizon": int(horizon),
        "num_pairs": int(num_pairs),
        "seed": int(seed),
    }
    cache_path = os.path.join(cfg.cache_dir, f"pair_sample_{dataset_slug(parsed.dataset_id)}_{_hash_payload(payload)}.npz")
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return PairSample(
                first_local_indices=np.asarray(cached["first_local_indices"], dtype=np.int64),
                second_local_indices=np.asarray(cached["second_local_indices"], dtype=np.int64),
                labels=np.asarray(cached["labels"], dtype=np.int64),
                positive_fraction=float(cached["positive_fraction"].item()),
            )

    sample = _sample_pair_indices(bundle.future_region_ids, num_pairs=num_pairs, seed=seed)
    _save_npz(
        cache_path,
        first_local_indices=sample.first_local_indices,
        second_local_indices=sample.second_local_indices,
        labels=sample.labels,
        positive_fraction=np.asarray(sample.positive_fraction),
    )
    return sample


def load_or_create_retrieval_bank(
    parsed: ParsedDataset,
    bundle: FutureWindowBundle,
    split_name: str,
    horizon: int,
    cfg: SuccessorDistanceConfig,
) -> RetrievalBank:
    payload = {
        "cache_version": SUCCESSOR_CACHE_VERSION,
        "dataset": parsed.dataset_id,
        "split_name": split_name,
        "horizon": int(horizon),
        "seed": cfg.seed,
        "num_queries": cfg.num_queries,
        "num_candidates": cfg.num_candidates,
    }
    cache_path = os.path.join(cfg.cache_dir, f"retrieval_bank_{dataset_slug(parsed.dataset_id)}_{_hash_payload(payload)}.npz")
    if _npz_exists(cache_path) and not cfg.overwrite_cache:
        cached = _safe_load_npz(cache_path)
        if cached is not None:
            return RetrievalBank(
                query_local_indices=np.asarray(cached["query_local_indices"], dtype=np.int64),
                candidate_local_indices=np.asarray(cached["candidate_local_indices"], dtype=np.int64),
            )

    rng = np.random.default_rng(cfg.seed + 991 + int(horizon))
    total = int(bundle.valid_global_indices.shape[0])
    if total < 2:
        raise RuntimeError("Need at least two valid states to build a retrieval bank.")
    query_count = min(int(cfg.num_queries), total)
    candidate_count = min(int(cfg.num_candidates), total)
    query_local_indices = np.sort(rng.choice(total, size=query_count, replace=False).astype(np.int64))
    candidate_local_indices = np.sort(rng.choice(total, size=candidate_count, replace=False).astype(np.int64))
    bank = RetrievalBank(query_local_indices=query_local_indices, candidate_local_indices=candidate_local_indices)
    _save_npz(
        cache_path,
        query_local_indices=bank.query_local_indices,
        candidate_local_indices=bank.candidate_local_indices,
    )
    return bank


def _raw_weights(horizon: int, gamma: float | None) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if gamma is None:
        return np.full(horizon, 1.0 / float(horizon), dtype=np.float32)
    gamma_value = float(gamma)
    if gamma_value <= 0:
        raise ValueError("raw_gamma must be positive when provided")
    exponents = np.arange(horizon, dtype=np.float32)
    weights = np.power(np.float32(gamma_value), exponents, dtype=np.float32)
    return (weights / np.sum(weights)).astype(np.float32)


def compute_raw_successor_paired_distances(
    windows_a: np.ndarray,
    windows_b: np.ndarray,
    *,
    gamma: float | None = None,
) -> np.ndarray:
    first = np.asarray(windows_a, dtype=np.float32)
    second = np.asarray(windows_b, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(f"Paired windows must match, got {first.shape} vs {second.shape}")
    weights = _raw_weights(first.shape[1], gamma)
    sq = np.sum((first - second) ** 2, axis=2, dtype=np.float32)
    return np.sqrt(np.maximum(np.sum(sq * weights[None, :], axis=1), 0.0)).astype(np.float32)


def compute_raw_successor_distance_matrix(
    query_windows: np.ndarray,
    candidate_windows: np.ndarray,
    *,
    gamma: float | None = None,
) -> np.ndarray:
    queries = np.asarray(query_windows, dtype=np.float32)
    candidates = np.asarray(candidate_windows, dtype=np.float32)
    weights = _raw_weights(queries.shape[1], gamma)
    diff = queries[:, None, :, :] - candidates[None, :, :, :]
    sq = np.sum(diff * diff, axis=3, dtype=np.float32)
    return np.sqrt(np.maximum(np.sum(sq * weights[None, None, :], axis=2), 0.0)).astype(np.float32)


def _window_self_rbf_kernel_batch(
    windows: np.ndarray,
    sigma: float,
    *,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    outputs = np.zeros(values.shape[0], dtype=np.float32)
    denominator = max(2.0 * float(sigma) * float(sigma), 1e-12)
    for start in range(0, values.shape[0], batch_size):
        end = min(start + batch_size, values.shape[0])
        block = values[start:end]
        diff = block[:, :, None, :] - block[:, None, :, :]
        sqdist = np.sum(diff * diff, axis=3, dtype=np.float32)
        kernel = np.exp(-(sqdist / denominator), dtype=np.float32)
        outputs[start:end] = np.mean(kernel, axis=(1, 2), dtype=np.float32)
    return outputs


def compute_gdk_paired_distances(
    windows_a: np.ndarray,
    windows_b: np.ndarray,
    *,
    sigma: float,
    batch_size: int = 512,
) -> np.ndarray:
    first = np.asarray(windows_a, dtype=np.float32)
    second = np.asarray(windows_b, dtype=np.float32)
    outputs = np.zeros(first.shape[0], dtype=np.float32)
    k_aa = _window_self_rbf_kernel_batch(first, sigma, batch_size=batch_size)
    k_bb = _window_self_rbf_kernel_batch(second, sigma, batch_size=batch_size)
    denominator = max(2.0 * float(sigma) * float(sigma), 1e-12)
    for start in range(0, first.shape[0], batch_size):
        end = min(start + batch_size, first.shape[0])
        block_a = first[start:end]
        block_b = second[start:end]
        diff = block_a[:, :, None, :] - block_b[:, None, :, :]
        sqdist = np.sum(diff * diff, axis=3, dtype=np.float32)
        k_ab = np.mean(np.exp(-(sqdist / denominator), dtype=np.float32), axis=(1, 2), dtype=np.float32)
        outputs[start:end] = np.sqrt(np.maximum(k_aa[start:end] + k_bb[start:end] - (2.0 * k_ab), 0.0))
    return outputs.astype(np.float32)


def compute_gdk_distance_matrix(
    query_windows: np.ndarray,
    candidate_windows: np.ndarray,
    *,
    sigma: float,
    query_batch_size: int = 8,
) -> np.ndarray:
    queries = np.asarray(query_windows, dtype=np.float32)
    candidates = np.asarray(candidate_windows, dtype=np.float32)
    denominator = max(2.0 * float(sigma) * float(sigma), 1e-12)
    k_qq = _window_self_rbf_kernel_batch(queries, sigma, batch_size=max(1, query_batch_size))
    k_cc = _window_self_rbf_kernel_batch(candidates, sigma, batch_size=max(1, query_batch_size))
    matrix = np.zeros((queries.shape[0], candidates.shape[0]), dtype=np.float32)
    for start in range(0, queries.shape[0], query_batch_size):
        end = min(start + query_batch_size, queries.shape[0])
        block = queries[start:end]
        diff = block[:, None, :, None, :] - candidates[None, :, None, :, :]
        sqdist = np.sum(diff * diff, axis=4, dtype=np.float32)
        k_qc = np.mean(np.exp(-(sqdist / denominator), dtype=np.float32), axis=(2, 3), dtype=np.float32)
        matrix[start:end] = np.sqrt(np.maximum(k_qq[start:end, None] + k_cc[None, :] - (2.0 * k_qc), 0.0))
    return matrix.astype(np.float32)


def compute_adaptive_gdk_paired_distances(
    metric: AdaptiveGaussianMetric,
    windows_a: np.ndarray,
    windows_b: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    first = np.asarray(windows_a, dtype=np.float32)
    second = np.asarray(windows_b, dtype=np.float32)
    first_sigmas = metric.estimate_query_sigmas(first.reshape(-1, first.shape[-1])).reshape(first.shape[0], first.shape[1])
    second_sigmas = metric.estimate_query_sigmas(second.reshape(-1, second.shape[-1])).reshape(second.shape[0], second.shape[1])
    outputs = np.zeros(first.shape[0], dtype=np.float32)
    for start in range(0, first.shape[0], batch_size):
        end = min(start + batch_size, first.shape[0])
        block_a = first[start:end]
        block_b = second[start:end]
        sigma_a = first_sigmas[start:end]
        sigma_b = second_sigmas[start:end]

        diff_ab = block_a[:, :, None, :] - block_b[:, None, :, :]
        sqdist_ab = np.sum(diff_ab * diff_ab, axis=3, dtype=np.float32)
        denom_ab = np.maximum(sigma_a[:, :, None] * sigma_b[:, None, :], metric.eps)
        k_ab = np.mean(np.exp(-(sqdist_ab / denom_ab), dtype=np.float32), axis=(1, 2), dtype=np.float32)

        diff_aa = block_a[:, :, None, :] - block_a[:, None, :, :]
        sqdist_aa = np.sum(diff_aa * diff_aa, axis=3, dtype=np.float32)
        denom_aa = np.maximum(sigma_a[:, :, None] * sigma_a[:, None, :], metric.eps)
        k_aa = np.mean(np.exp(-(sqdist_aa / denom_aa), dtype=np.float32), axis=(1, 2), dtype=np.float32)

        diff_bb = block_b[:, :, None, :] - block_b[:, None, :, :]
        sqdist_bb = np.sum(diff_bb * diff_bb, axis=3, dtype=np.float32)
        denom_bb = np.maximum(sigma_b[:, :, None] * sigma_b[:, None, :], metric.eps)
        k_bb = np.mean(np.exp(-(sqdist_bb / denom_bb), dtype=np.float32), axis=(1, 2), dtype=np.float32)
        outputs[start:end] = np.sqrt(np.maximum(k_aa + k_bb - (2.0 * k_ab), 0.0))
    return outputs.astype(np.float32)


def compute_adaptive_gdk_distance_matrix(
    metric: AdaptiveGaussianMetric,
    query_windows: np.ndarray,
    candidate_windows: np.ndarray,
    *,
    query_batch_size: int = 8,
) -> np.ndarray:
    queries = np.asarray(query_windows, dtype=np.float32)
    candidates = np.asarray(candidate_windows, dtype=np.float32)
    q_sigmas = metric.estimate_query_sigmas(queries.reshape(-1, queries.shape[-1])).reshape(queries.shape[0], queries.shape[1])
    c_sigmas = metric.estimate_query_sigmas(candidates.reshape(-1, candidates.shape[-1])).reshape(candidates.shape[0], candidates.shape[1])

    k_qq = np.zeros(queries.shape[0], dtype=np.float32)
    for start in range(0, queries.shape[0], query_batch_size):
        end = min(start + query_batch_size, queries.shape[0])
        block = queries[start:end]
        sigma = q_sigmas[start:end]
        diff = block[:, :, None, :] - block[:, None, :, :]
        sqdist = np.sum(diff * diff, axis=3, dtype=np.float32)
        denom = np.maximum(sigma[:, :, None] * sigma[:, None, :], metric.eps)
        k_qq[start:end] = np.mean(np.exp(-(sqdist / denom), dtype=np.float32), axis=(1, 2), dtype=np.float32)

    k_cc = np.zeros(candidates.shape[0], dtype=np.float32)
    for start in range(0, candidates.shape[0], query_batch_size):
        end = min(start + query_batch_size, candidates.shape[0])
        block = candidates[start:end]
        sigma = c_sigmas[start:end]
        diff = block[:, :, None, :] - block[:, None, :, :]
        sqdist = np.sum(diff * diff, axis=3, dtype=np.float32)
        denom = np.maximum(sigma[:, :, None] * sigma[:, None, :], metric.eps)
        k_cc[start:end] = np.mean(np.exp(-(sqdist / denom), dtype=np.float32), axis=(1, 2), dtype=np.float32)

    matrix = np.zeros((queries.shape[0], candidates.shape[0]), dtype=np.float32)
    for start in range(0, queries.shape[0], query_batch_size):
        end = min(start + query_batch_size, queries.shape[0])
        block = queries[start:end]
        block_sigmas = q_sigmas[start:end]
        diff = block[:, None, :, None, :] - candidates[None, :, None, :, :]
        sqdist = np.sum(diff * diff, axis=4, dtype=np.float32)
        denom = np.maximum(block_sigmas[:, None, :, None] * c_sigmas[None, :, None, :], metric.eps)
        k_qc = np.mean(np.exp(-(sqdist / denom), dtype=np.float32), axis=(2, 3), dtype=np.float32)
        matrix[start:end] = np.sqrt(np.maximum(k_qq[start:end, None] + k_cc[None, :] - (2.0 * k_qc), 0.0))
    return matrix.astype(np.float32)


def compute_wasserstein_paired_distances(
    windows_a: np.ndarray,
    windows_b: np.ndarray,
) -> np.ndarray:
    first = np.asarray(windows_a, dtype=np.float32)
    second = np.asarray(windows_b, dtype=np.float32)
    outputs = np.zeros(first.shape[0], dtype=np.float32)
    for idx in range(first.shape[0]):
        diff = first[idx][:, None, :] - second[idx][None, :, :]
        cost = np.sum(diff * diff, axis=2, dtype=np.float32)
        row_ind, col_ind = linear_sum_assignment(cost)
        outputs[idx] = float(np.sqrt(np.mean(cost[row_ind, col_ind], dtype=np.float32)))
    return outputs


def compute_wasserstein_distance_matrix(
    query_windows: np.ndarray,
    candidate_windows: np.ndarray,
) -> np.ndarray:
    queries = np.asarray(query_windows, dtype=np.float32)
    candidates = np.asarray(candidate_windows, dtype=np.float32)
    matrix = np.zeros((queries.shape[0], candidates.shape[0]), dtype=np.float32)
    for q_idx in range(queries.shape[0]):
        for c_idx in range(candidates.shape[0]):
            diff = queries[q_idx][:, None, :] - candidates[c_idx][None, :, :]
            cost = np.sum(diff * diff, axis=2, dtype=np.float32)
            row_ind, col_ind = linear_sum_assignment(cost)
            matrix[q_idx, c_idx] = float(np.sqrt(np.mean(cost[row_ind, col_ind], dtype=np.float32)))
    return matrix


def _window_mean_ik_features_explicit(
    kernel: Any,
    windows: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    import torch

    total = int(windows.shape[0])
    horizon = int(windows.shape[1])
    flattened = np.asarray(windows, dtype=np.float32).reshape(total * horizon, windows.shape[2])
    tensor = torch.as_tensor(flattened, dtype=torch.float32, device=kernel.anchors.device)
    features = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            end = min(start + batch_size, tensor.shape[0])
            features.append(kernel(tensor[start:end]) / math.sqrt(float(kernel.ensemble_size)))
    stacked = torch.cat(features, dim=0).reshape(total, horizon, -1).mean(dim=1)
    return stacked.detach().cpu().numpy().astype(np.float32)


def _effective_ensemble_chunk_size(subsample_size: int, requested_chunk_size: int) -> int:
    max_by_anchor_budget = max(1, int(IDK_MAX_CHUNK_ANCHORS) // max(int(subsample_size), 1))
    return max(1, min(int(requested_chunk_size), max_by_anchor_budget))


def _window_batch_size_for_chunk(
    total_windows: int,
    horizon: int,
    anchors_per_chunk: int,
    point_batch_size_hint: int,
) -> int:
    requested_windows = max(1, int(point_batch_size_hint) // max(int(horizon), 1))
    budget_windows = max(1, int(IDK_MAX_CDIST_VALUES) // max(int(horizon) * int(anchors_per_chunk), 1))
    return max(1, min(int(total_windows), requested_windows, budget_windows))


def _chunk_mean_assignments_torch(
    windows: np.ndarray | torch.Tensor,
    anchor_chunk: torch.Tensor,
    temperature: float,
    *,
    point_batch_size_hint: int,
) -> torch.Tensor:
    if isinstance(windows, torch.Tensor):
        values = windows.to(anchor_chunk.device, dtype=torch.float32)
    else:
        values = torch.as_tensor(windows, dtype=torch.float32, device=anchor_chunk.device)
    total, horizon, dim = values.shape
    ensemble_chunk = int(anchor_chunk.shape[0])
    subsample_size = int(anchor_chunk.shape[1])
    anchor_flat = anchor_chunk.reshape(ensemble_chunk * subsample_size, dim)
    window_batch_size = _window_batch_size_for_chunk(
        total_windows=total,
        horizon=horizon,
        anchors_per_chunk=ensemble_chunk * subsample_size,
        point_batch_size_hint=point_batch_size_hint,
    )
    outputs = []
    temp = max(float(temperature), 1e-8)
    for start in range(0, total, window_batch_size):
        end = min(start + window_batch_size, total)
        batch = values[start:end].reshape(-1, dim)
        dist = torch.cdist(batch, anchor_flat, p=2).view(end - start, horizon, ensemble_chunk, subsample_size)
        assign = torch.softmax(-dist / temp, dim=-1)
        outputs.append(assign.mean(dim=1))
    return torch.cat(outputs, dim=0)


def _compute_idk_pair_distances_chunked_torch(
    anchors: torch.Tensor,
    windows_a: np.ndarray,
    windows_b: np.ndarray,
    *,
    temperature: float,
    ensemble_size: int,
    subsample_size: int,
    batch_size: int,
    pair_batch_size: int,
    ensemble_chunk_size: int,
) -> np.ndarray:
    values_a = torch.as_tensor(windows_a, dtype=torch.float32, device=anchors.device)
    values_b = torch.as_tensor(windows_b, dtype=torch.float32, device=anchors.device)
    outputs = torch.zeros(values_a.shape[0], dtype=torch.float32, device=anchors.device)
    total_ensembles = int(ensemble_size)
    effective_chunk = _effective_ensemble_chunk_size(subsample_size, ensemble_chunk_size)
    for start in range(0, windows_a.shape[0], pair_batch_size):
        end = min(start + pair_batch_size, windows_a.shape[0])
        block_a = values_a[start:end]
        block_b = values_b[start:end]
        k_aa = torch.zeros(block_a.shape[0], dtype=torch.float32, device=anchors.device)
        k_bb = torch.zeros(block_b.shape[0], dtype=torch.float32, device=anchors.device)
        k_ab = torch.zeros(block_a.shape[0], dtype=torch.float32, device=anchors.device)
        for ensemble_start in range(0, total_ensembles, effective_chunk):
            ensemble_end = min(ensemble_start + effective_chunk, total_ensembles)
            anchor_chunk = anchors[ensemble_start:ensemble_end]
            mean_a = _chunk_mean_assignments_torch(
                block_a,
                anchor_chunk,
                temperature,
                point_batch_size_hint=batch_size,
            )
            mean_b = _chunk_mean_assignments_torch(
                block_b,
                anchor_chunk,
                temperature,
                point_batch_size_hint=batch_size,
            )
            normalizer = float(total_ensembles)
            k_aa += torch.sum(mean_a * mean_a, dim=(1, 2)) / normalizer
            k_bb += torch.sum(mean_b * mean_b, dim=(1, 2)) / normalizer
            k_ab += torch.sum(mean_a * mean_b, dim=(1, 2)) / normalizer
        outputs[start:end] = torch.sqrt(torch.clamp(k_aa + k_bb - (2.0 * k_ab), min=0.0))
    return outputs.detach().cpu().numpy().astype(np.float32)


def _compute_idk_distance_matrix_chunked_torch(
    anchors: torch.Tensor,
    query_windows: np.ndarray,
    candidate_windows: np.ndarray,
    *,
    temperature: float,
    ensemble_size: int,
    subsample_size: int,
    batch_size: int,
    query_batch_size: int,
    ensemble_chunk_size: int,
) -> np.ndarray:
    queries = torch.as_tensor(query_windows, dtype=torch.float32, device=anchors.device)
    candidates = torch.as_tensor(candidate_windows, dtype=torch.float32, device=anchors.device)
    total_ensembles = int(ensemble_size)
    effective_chunk = _effective_ensemble_chunk_size(subsample_size, ensemble_chunk_size)
    k_qq = torch.zeros(queries.shape[0], dtype=torch.float32, device=anchors.device)
    k_cc = torch.zeros(candidates.shape[0], dtype=torch.float32, device=anchors.device)
    cross = torch.zeros((queries.shape[0], candidates.shape[0]), dtype=torch.float32, device=anchors.device)
    for ensemble_start in range(0, total_ensembles, effective_chunk):
        ensemble_end = min(ensemble_start + effective_chunk, total_ensembles)
        anchor_chunk = anchors[ensemble_start:ensemble_end]
        mean_c = _chunk_mean_assignments_torch(
            candidates,
            anchor_chunk,
            temperature,
            point_batch_size_hint=batch_size,
        )
        normalizer = float(total_ensembles)
        k_cc += torch.sum(mean_c * mean_c, dim=(1, 2)) / normalizer
        for q_start in range(0, queries.shape[0], query_batch_size):
            q_end = min(q_start + query_batch_size, queries.shape[0])
            mean_q = _chunk_mean_assignments_torch(
                queries[q_start:q_end],
                anchor_chunk,
                temperature,
                point_batch_size_hint=batch_size,
            )
            k_qq[q_start:q_end] += torch.sum(mean_q * mean_q, dim=(1, 2)) / normalizer
            cross[q_start:q_end] += torch.einsum("qes,ces->qc", mean_q, mean_c) / normalizer
    matrix = torch.sqrt(torch.clamp(k_qq[:, None] + k_cc[None, :] - (2.0 * cross), min=0.0))
    return matrix.detach().cpu().numpy().astype(np.float32)


def compute_idk_paired_distances(
    train_positions: np.ndarray,
    windows_a: np.ndarray,
    windows_b: np.ndarray,
    *,
    seed: int,
    ensemble_size: int,
    subsample_size: int,
    temperature: float,
    batch_size: int,
    pair_batch_size: int,
    device: str,
    explicit_max_feature_values: int,
    ensemble_chunk_size: int,
) -> np.ndarray:
    _torch_seed(seed)
    kernel, _ = fit_soft_isolation_kernel(
        train_positions,
        ensemble_size=ensemble_size,
        subsample_size=subsample_size,
        temperature=temperature,
        device=device,
    )
    total_windows = int(windows_a.shape[0] + windows_b.shape[0])
    total_dim = int(ensemble_size) * int(subsample_size)
    if (total_windows * total_dim) <= int(explicit_max_feature_values):
        mean_a = _window_mean_ik_features_explicit(kernel, windows_a, batch_size=batch_size)
        mean_b = _window_mean_ik_features_explicit(kernel, windows_b, batch_size=batch_size)
        return np.sqrt(np.maximum(np.sum((mean_a - mean_b) ** 2, axis=1, dtype=np.float32), 0.0)).astype(np.float32)

    anchors = kernel.anchors.detach().reshape(int(ensemble_size), int(subsample_size), -1).contiguous()
    return _compute_idk_pair_distances_chunked_torch(
        anchors,
        windows_a,
        windows_b,
        temperature=temperature,
        ensemble_size=ensemble_size,
        subsample_size=subsample_size,
        batch_size=batch_size,
        pair_batch_size=pair_batch_size,
        ensemble_chunk_size=ensemble_chunk_size,
    )


def compute_idk_distance_matrix(
    train_positions: np.ndarray,
    query_windows: np.ndarray,
    candidate_windows: np.ndarray,
    *,
    seed: int,
    ensemble_size: int,
    subsample_size: int,
    temperature: float,
    batch_size: int,
    query_batch_size: int,
    device: str,
    explicit_max_feature_values: int,
    ensemble_chunk_size: int,
) -> np.ndarray:
    _torch_seed(seed)
    kernel, _ = fit_soft_isolation_kernel(
        train_positions,
        ensemble_size=ensemble_size,
        subsample_size=subsample_size,
        temperature=temperature,
        device=device,
    )
    total_windows = int(query_windows.shape[0] + candidate_windows.shape[0])
    total_dim = int(ensemble_size) * int(subsample_size)
    if (total_windows * total_dim) <= int(explicit_max_feature_values):
        mean_q = _window_mean_ik_features_explicit(kernel, query_windows, batch_size=batch_size)
        mean_c = _window_mean_ik_features_explicit(kernel, candidate_windows, batch_size=batch_size)
        q_sq = np.sum(mean_q * mean_q, axis=1, dtype=np.float32)
        c_sq = np.sum(mean_c * mean_c, axis=1, dtype=np.float32)
        cross = np.matmul(mean_q, mean_c.T).astype(np.float32)
        return np.sqrt(np.maximum(q_sq[:, None] + c_sq[None, :] - (2.0 * cross), 0.0)).astype(np.float32)

    anchors = kernel.anchors.detach().reshape(int(ensemble_size), int(subsample_size), -1).contiguous()
    return _compute_idk_distance_matrix_chunked_torch(
        anchors,
        query_windows,
        candidate_windows,
        temperature=temperature,
        ensemble_size=ensemble_size,
        subsample_size=subsample_size,
        batch_size=batch_size,
        query_batch_size=query_batch_size,
        ensemble_chunk_size=ensemble_chunk_size,
    )


def _fit_sigma_median_heuristic(
    train_positions: np.ndarray,
    *,
    num_pairs: int,
    seed: int,
) -> float:
    values = np.asarray(train_positions, dtype=np.float32)
    rng = np.random.default_rng(seed)
    total = int(values.shape[0])
    if total < 2:
        return 1.0
    pair_count = min(int(num_pairs), total * 2)
    first = rng.integers(0, total, size=pair_count, endpoint=False)
    second = rng.integers(0, total, size=pair_count, endpoint=False)
    mask = first != second
    if not np.any(mask):
        return max(float(np.linalg.norm(values[1] - values[0])), 1e-6)
    distances = np.linalg.norm(values[first[mask]] - values[second[mask]], axis=1)
    distances = distances[np.isfinite(distances) & (distances > 1e-12)]
    if distances.size == 0:
        return 1.0
    return float(np.median(distances))


def _subsample_fit_positions(
    positions: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float32)
    if values.shape[0] <= int(max_points):
        return values
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(values.shape[0], size=int(max_points), replace=False))
    return np.asarray(values[keep], dtype=np.float32)


def _pair_windows(bundle: FutureWindowBundle, sample: PairSample) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(bundle.future_windows[sample.first_local_indices], dtype=np.float32),
        np.asarray(bundle.future_windows[sample.second_local_indices], dtype=np.float32),
    )


def _binary_pair_metrics(labels: np.ndarray, distances: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = -np.asarray(distances, dtype=np.float64)
    return {
        "auroc": auc_from_binary_labels(y_true, y_score),
        "auprc": average_precision_from_binary_labels(y_true, y_score),
    }


def _evaluate_recall_rows(
    bundle: FutureWindowBundle,
    bank: RetrievalBank,
    distance_matrix: np.ndarray,
    recall_k_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    query_regions = bundle.future_region_ids[bank.query_local_indices]
    candidate_regions = bundle.future_region_ids[bank.candidate_local_indices]
    query_globals = bundle.valid_global_indices[bank.query_local_indices]
    candidate_globals = bundle.valid_global_indices[bank.candidate_local_indices]
    rows = []
    for row_idx, query_global_index in enumerate(query_globals.tolist()):
        labels = (candidate_regions == query_regions[row_idx]).astype(np.int64)
        distances = np.asarray(distance_matrix[row_idx], dtype=np.float32).copy()
        self_mask = candidate_globals == int(query_global_index)
        if np.any(self_mask):
            distances[self_mask] = np.inf
            labels[self_mask] = 0
        row: dict[str, Any] = {
            "query_row": int(row_idx),
            "query_global_index": int(query_global_index),
            "query_future_region_id": int(query_regions[row_idx]),
            "num_positives": int(np.sum(labels)),
        }
        for k in recall_k_values:
            row[f"recall_at_{int(k)}"] = float(recall_at_k(labels.astype(np.float32), -distances.astype(np.float32), int(k)))
        rows.append(row)
    return rows


def _summarize_recall_rows(recall_rows: list[dict[str, Any]], recall_k_values: tuple[int, ...]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not recall_rows:
        for k in recall_k_values:
            summary[f"recall_at_{int(k)}"] = 0.0
        return summary
    for k in recall_k_values:
        summary[f"recall_at_{int(k)}"] = float(np.mean([float(row[f"recall_at_{int(k)}"]) for row in recall_rows]))
    return summary


def _plot_method_topk_map(
    dataset_id: str,
    horizon: int,
    method_name: str,
    parsed: ParsedDataset,
    bundle: FutureWindowBundle,
    bank: RetrievalBank,
    distance_matrix: np.ndarray,
    top_k: int,
    figure_path: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    query_row = 0
    query_local = int(bank.query_local_indices[query_row])
    query_global = int(bundle.valid_global_indices[query_local])
    query_position = np.asarray(parsed.positions[query_global], dtype=np.float32)
    query_region = int(bundle.future_region_ids[query_local])

    candidate_locals = np.asarray(bank.candidate_local_indices, dtype=np.int64)
    candidate_globals = np.asarray(bundle.valid_global_indices[candidate_locals], dtype=np.int64)
    candidate_positions = np.asarray(parsed.positions[candidate_globals], dtype=np.float32)
    candidate_regions = np.asarray(bundle.future_region_ids[candidate_locals], dtype=np.int64)

    distances = np.asarray(distance_matrix[query_row], dtype=np.float32).copy()
    self_mask = candidate_globals == query_global
    distances[self_mask] = np.inf
    order = np.argsort(distances)[: min(int(top_k), int(distances.shape[0]))]

    rng = np.random.default_rng(0)
    background = np.asarray(parsed.positions, dtype=np.float32)
    if background.shape[0] > 6000:
        keep = np.sort(rng.choice(background.shape[0], size=6000, replace=False))
        background = background[keep]

    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    ax.scatter(background[:, 0], background[:, 1], s=4, color="#C7CED8", alpha=0.12, linewidths=0.0)
    positive_mask = candidate_regions[order] == query_region
    if np.any(~positive_mask):
        ax.scatter(
            candidate_positions[order][~positive_mask, 0],
            candidate_positions[order][~positive_mask, 1],
            s=58,
            color="#C44E52",
            marker="X",
            alpha=0.95,
            label="Top-k negative",
        )
    if np.any(positive_mask):
        ax.scatter(
            candidate_positions[order][positive_mask, 0],
            candidate_positions[order][positive_mask, 1],
            s=64,
            color="#2D8A5D",
            marker="o",
            alpha=0.98,
            label="Top-k positive",
        )
    ax.scatter(query_position[0], query_position[1], s=180, color="#111827", marker="*", edgecolors="white", linewidths=0.9, label="Query")
    ax.set_title(f"{dataset_id}\nH={horizon} | {METHOD_LABELS[method_name]} top-{min(top_k, len(order))}", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ensure_dir(os.path.dirname(figure_path))
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _selection_key(row: dict[str, Any], metric_name: str) -> tuple[float, float, float, float, float]:
    primary = float(row[metric_name])
    secondary = float(row["auroc"])
    smaller_subsample = -float(row["ik_subsample_size"])
    smaller_temperature = -float(row["ik_temperature"])
    smaller_ensemble = -float(row["ik_ensemble_size"])
    return (primary, secondary, smaller_ensemble, smaller_subsample, smaller_temperature)


def _idk_partial_search_path(cfg: SuccessorDistanceConfig, dataset_id: str, horizon: int) -> str:
    return os.path.join(cfg.search_dir, f"idk_search_{dataset_slug(dataset_id)}_h{int(horizon)}.csv")


def _canonicalize_search_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": str(row["dataset"]),
        "dataset_slug": str(row["dataset_slug"]),
        "horizon": int(row["horizon"]),
        "split": str(row["split"]),
        "ik_ensemble_size": int(row["ik_ensemble_size"]),
        "ik_subsample_size": int(row["ik_subsample_size"]),
        "ik_temperature": float(row["ik_temperature"]),
        "num_pairs": int(row["num_pairs"]),
        "positive_fraction": float(row["positive_fraction"]),
        "auroc": float(row["auroc"]),
        "auprc": float(row["auprc"]),
    }


def _sort_search_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [_canonicalize_search_row(row) for row in rows],
        key=lambda row: (
            str(row["dataset_slug"]),
            int(row["horizon"]),
            int(row["ik_ensemble_size"]),
            int(row["ik_subsample_size"]),
            float(row["ik_temperature"]),
        ),
    )


def _summarize_overall(summary_rows: list[dict[str, Any]], recall_k_values: tuple[int, ...]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    overall_rows = []
    for method_name in METHOD_ORDER:
        group = grouped.get(method_name, [])
        if not group:
            continue
        row: dict[str, Any] = {
            "method": method_name,
            "num_dataset_h_pairs": int(len(group)),
            "auroc_mean": float(np.mean([float(item["auroc"]) for item in group])),
            "auprc_mean": float(np.mean([float(item["auprc"]) for item in group])),
        }
        for k in recall_k_values:
            row[f"recall_at_{int(k)}_mean"] = float(np.mean([float(item[f"recall_at_{int(k)}"]) for item in group]))
        overall_rows.append(row)
    return overall_rows


def _report_lines(
    cfg: SuccessorDistanceConfig,
    search_rows: list[dict[str, Any]],
    best_ik_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    figure_paths: dict[str, str],
) -> list[str]:
    lines = [
        "# Successor Distance Report",
        "",
        "## Setup",
        "",
        f"- Datasets: {', '.join(cfg.datasets)}",
        f"- Horizons: {', '.join(str(int(h)) for h in cfg.horizon_values)}",
        f"- Grid: {cfg.grid_nx} x {cfg.grid_ny}",
        f"- Episode split: {cfg.train_ratio:.2f}/{cfg.val_ratio:.2f}/{cfg.test_ratio:.2f}",
        f"- Pair sampling (search / eval): {cfg.search_num_pairs} / {cfg.eval_num_pairs}",
        f"- Retrieval bank (queries / candidates): {cfg.num_queries} / {cfg.num_candidates}",
        f"- IDK selection metric: {cfg.selection_metric}",
        "",
        "## Methods",
        "",
        "- `raw`: ordered future-window weighted L2 baseline.",
        "- `idk`: future empirical distribution compared by the repository Isolation Kernel.",
        "- `gdk`: Gaussian distributional kernel with train-only median-heuristic sigma.",
        "- `wasserstein_w2`: exact 2-Wasserstein over uniformly weighted future point sets.",
        "- `adaptive_gdk`: distributional kernel built from the repository adaptive Gaussian point kernel.",
        "",
        "## Best IDK Configs",
        "",
        "| Dataset | H | Ensemble | Subsample | Temperature | Val AUROC | Val AUPRC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best_ik_rows:
        lines.append(
            f"| {row['dataset']} | {row['horizon']} | {row['ik_ensemble_size']} | {row['ik_subsample_size']} | "
            f"{row['ik_temperature']} | {row['auroc']:.4f} | {row['auprc']:.4f} |"
        )
    lines.extend(["", "## Final Summary", "", "| Dataset | H | Method | AUROC | AUPRC | " + " | ".join(
        [f"Recall@{int(k)}" for k in cfg.recall_k_values]
    ) + " |", "| --- | ---: | --- | ---: | ---: | " + " | ".join(["---:"] * len(cfg.recall_k_values)) + " |"])
    for row in summary_rows:
        recall_values = " | ".join([f"{float(row[f'recall_at_{int(k)}']):.4f}" for k in cfg.recall_k_values])
        lines.append(
            f"| {row['dataset']} | {row['horizon']} | {row['method']} | {row['auroc']:.4f} | {row['auprc']:.4f} | {recall_values} |"
        )
    lines.extend(["", "## Overall", "", "| Method | AUROC | AUPRC | " + " | ".join(
        [f"Recall@{int(k)}" for k in cfg.recall_k_values]
    ) + " |", "| --- | ---: | ---: | " + " | ".join(["---:"] * len(cfg.recall_k_values)) + " |"])
    for row in overall_rows:
        recall_values = " | ".join([f"{float(row[f'recall_at_{int(k)}_mean']):.4f}" for k in cfg.recall_k_values])
        lines.append(
            f"| {row['method']} | {row['auroc_mean']:.4f} | {row['auprc_mean']:.4f} | {recall_values} |"
        )
    lines.extend(["", "## Figures", ""])
    for figure_key, figure_path in sorted(figure_paths.items()):
        lines.append(f"- `{figure_key}`: `{figure_path}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- All state labels are defined by whether the H-step endpoints land in the same uniform spatial grid cell.",
            "- Train split is used for IK fitting, Gaussian sigma estimation, adaptive Gaussian fitting, and grid boundary estimation only.",
            "- Search and final evaluation use different pair samples to keep model selection on validation only.",
        ]
    )
    return lines


def _fit_train_artifacts(
    parsed: ParsedDataset,
    split: EpisodeSplit,
    cfg: SuccessorDistanceConfig,
    horizon: int,
) -> tuple[np.ndarray, float, AdaptiveGaussianMetric]:
    train_positions_full = np.asarray(parsed.positions[split.train_indices], dtype=np.float32)
    train_positions = _subsample_fit_positions(
        train_positions_full,
        max_points=cfg.fit_pool_size,
        seed=cfg.seed + 17 + int(horizon),
    )
    sigma = _fit_sigma_median_heuristic(
        train_positions,
        num_pairs=cfg.gdk_sigma_num_pairs,
        seed=cfg.seed + 29 + int(horizon),
    )
    adaptive_metric = AdaptiveGaussianMetric.fit(
        train_positions,
        k=cfg.adaptive_gaussian_k,
        eps=cfg.adaptive_gaussian_eps,
    )
    return train_positions, float(max(sigma, 1e-6)), adaptive_metric


def _compute_method_pair_distances(
    method_name: str,
    windows_a: np.ndarray,
    windows_b: np.ndarray,
    *,
    cfg: SuccessorDistanceConfig,
    seed: int,
    train_positions: np.ndarray,
    sigma: float,
    adaptive_metric: AdaptiveGaussianMetric | None,
    ik_ensemble_size: int | None = None,
    ik_subsample_size: int | None = None,
    ik_temperature: float | None = None,
    pair_batch_size: int,
) -> np.ndarray:
    if method_name == "raw":
        return compute_raw_successor_paired_distances(windows_a, windows_b, gamma=cfg.raw_gamma)
    if method_name == "gdk":
        return compute_gdk_paired_distances(windows_a, windows_b, sigma=sigma, batch_size=pair_batch_size)
    if method_name == "wasserstein_w2":
        return compute_wasserstein_paired_distances(windows_a, windows_b)
    if method_name == "adaptive_gdk":
        if adaptive_metric is None:
            raise ValueError("Adaptive-GDK requires a fitted AdaptiveGaussianMetric")
        return compute_adaptive_gdk_paired_distances(adaptive_metric, windows_a, windows_b, batch_size=pair_batch_size)
    if method_name == "idk":
        if ik_ensemble_size is None or ik_subsample_size is None or ik_temperature is None:
            raise ValueError("IDK distances require explicit IK hyperparameters")
        return compute_idk_paired_distances(
            train_positions=train_positions,
            windows_a=windows_a,
            windows_b=windows_b,
            seed=seed,
            ensemble_size=int(ik_ensemble_size),
            subsample_size=int(ik_subsample_size),
            temperature=float(ik_temperature),
            batch_size=cfg.ik_batch_size,
            pair_batch_size=pair_batch_size,
            device=_resolve_torch_device(cfg.ik_device),
            explicit_max_feature_values=cfg.ik_explicit_max_feature_values,
            ensemble_chunk_size=cfg.ik_chunk_ensemble_size,
        )
    raise ValueError(f"Unsupported method: {method_name}")


def _compute_method_distance_matrix(
    method_name: str,
    query_windows: np.ndarray,
    candidate_windows: np.ndarray,
    *,
    cfg: SuccessorDistanceConfig,
    seed: int,
    train_positions: np.ndarray,
    sigma: float,
    adaptive_metric: AdaptiveGaussianMetric | None,
    ik_ensemble_size: int | None = None,
    ik_subsample_size: int | None = None,
    ik_temperature: float | None = None,
) -> np.ndarray:
    if method_name == "raw":
        return compute_raw_successor_distance_matrix(query_windows, candidate_windows, gamma=cfg.raw_gamma)
    if method_name == "gdk":
        return compute_gdk_distance_matrix(query_windows, candidate_windows, sigma=sigma, query_batch_size=cfg.query_matrix_batch_size)
    if method_name == "wasserstein_w2":
        return compute_wasserstein_distance_matrix(query_windows, candidate_windows)
    if method_name == "adaptive_gdk":
        if adaptive_metric is None:
            raise ValueError("Adaptive-GDK requires a fitted AdaptiveGaussianMetric")
        return compute_adaptive_gdk_distance_matrix(adaptive_metric, query_windows, candidate_windows, query_batch_size=cfg.query_matrix_batch_size)
    if method_name == "idk":
        if ik_ensemble_size is None or ik_subsample_size is None or ik_temperature is None:
            raise ValueError("IDK matrix requires explicit IK hyperparameters")
        return compute_idk_distance_matrix(
            train_positions=train_positions,
            query_windows=query_windows,
            candidate_windows=candidate_windows,
            seed=seed,
            ensemble_size=int(ik_ensemble_size),
            subsample_size=int(ik_subsample_size),
            temperature=float(ik_temperature),
            batch_size=cfg.ik_batch_size,
            query_batch_size=cfg.query_matrix_batch_size,
            device=_resolve_torch_device(cfg.ik_device),
            explicit_max_feature_values=cfg.ik_explicit_max_feature_values,
            ensemble_chunk_size=cfg.ik_chunk_ensemble_size,
        )
    raise ValueError(f"Unsupported method: {method_name}")


def _idk_search_rows(
    dataset_id: str,
    horizon: int,
    val_bundle: FutureWindowBundle,
    val_pair_sample: PairSample,
    cfg: SuccessorDistanceConfig,
    train_positions: np.ndarray,
) -> list[dict[str, Any]]:
    windows_a, windows_b = _pair_windows(val_bundle, val_pair_sample)
    search_path = _idk_partial_search_path(cfg, dataset_id, horizon)
    rows = []
    completed: set[tuple[int, int, float]] = set()
    if os.path.exists(search_path) and not cfg.overwrite_cache:
        rows = _sort_search_rows(_load_csv_rows(search_path))
        for row in rows:
            completed.add(
                (
                    int(row["ik_ensemble_size"]),
                    int(row["ik_subsample_size"]),
                    float(row["ik_temperature"]),
                )
            )

    configs = list(product(
        cfg.ik_ensemble_sizes,
        cfg.ik_subsample_sizes,
        cfg.ik_temperatures,
    ))
    total_configs = len(configs)
    started_at = time.time()
    print(
        f"[successor-distance] IDK search start dataset={dataset_id} horizon={int(horizon)} "
        f"completed={len(completed)}/{total_configs}",
        flush=True,
    )
    completed_count = len(completed)
    for config_index, (ensemble_size, subsample_size, temperature) in enumerate(configs, start=1):
        key = (int(ensemble_size), int(subsample_size), float(temperature))
        if key in completed:
            continue
        distances = _compute_method_pair_distances(
            "idk",
            windows_a,
            windows_b,
            cfg=cfg,
            seed=cfg.seed + 103 * int(horizon) + int(ensemble_size) + int(subsample_size),
            train_positions=train_positions,
            sigma=1.0,
            adaptive_metric=None,
            ik_ensemble_size=int(ensemble_size),
            ik_subsample_size=int(subsample_size),
            ik_temperature=float(temperature),
            pair_batch_size=cfg.search_pair_batch_size,
        )
        metrics = _binary_pair_metrics(val_pair_sample.labels, distances)
        row = {
            "dataset": dataset_id,
            "dataset_slug": dataset_slug(dataset_id),
            "horizon": int(horizon),
            "split": "val",
            "ik_ensemble_size": int(ensemble_size),
            "ik_subsample_size": int(subsample_size),
            "ik_temperature": float(temperature),
            "num_pairs": int(val_pair_sample.labels.shape[0]),
            "positive_fraction": float(val_pair_sample.positive_fraction),
            "auroc": float(metrics["auroc"]),
            "auprc": float(metrics["auprc"]),
        }
        rows.append(row)
        completed.add(key)
        completed_count += 1
        if (
            completed_count % int(PARTIAL_SEARCH_SAVE_EVERY) == 0
            or completed_count == total_configs
        ):
            rows = _sort_search_rows(rows)
            _save_csv(search_path, rows, IDK_SEARCH_FIELDS)
        if (
            completed_count % int(IDK_PROGRESS_LOG_EVERY) == 0
            or completed_count == total_configs
        ):
            elapsed = max(time.time() - started_at, 1e-6)
            rate = completed_count / elapsed
            remaining = max(total_configs - completed_count, 0)
            eta_seconds = remaining / max(rate, 1e-6)
            print(
                f"[successor-distance] IDK search dataset={dataset_id} horizon={int(horizon)} "
                f"{completed_count}/{total_configs} "
                f"ensemble={int(ensemble_size)} subsample={int(subsample_size)} temp={float(temperature):g} "
                f"auprc={float(metrics['auprc']):.4f} elapsed={elapsed/60.0:.1f}m eta={eta_seconds/3600.0:.1f}h",
                flush=True,
            )
    rows = _sort_search_rows(rows)
    _save_csv(search_path, rows, IDK_SEARCH_FIELDS)
    return rows


def _best_idk_row(search_rows: list[dict[str, Any]], cfg: SuccessorDistanceConfig) -> dict[str, Any]:
    if not search_rows:
        raise RuntimeError("IDK search produced no rows.")
    return max(search_rows, key=lambda row: _selection_key(row, cfg.selection_metric))


def analyze_successor_distance_dataset(
    parsed: ParsedDataset,
    dataset_id: str,
    cfg: SuccessorDistanceConfig,
    result_callback: Callable[[SuccessorEvalResult], None] | None = None,
) -> list[SuccessorEvalResult]:
    split = load_or_create_episode_split(parsed, cfg)
    grid = build_or_load_grid_spec(parsed, split, cfg)
    results: list[SuccessorEvalResult] = []

    for horizon in cfg.horizon_values:
        print(
            f"[successor-distance] dataset={dataset_id} horizon={int(horizon)} fitting train artifacts",
            flush=True,
        )
        train_positions, sigma, adaptive_metric = _fit_train_artifacts(parsed, split, cfg, horizon)
        val_bundle = build_or_load_future_window_bundle(parsed, split, "val", horizon, grid, cfg)
        test_bundle = build_or_load_future_window_bundle(parsed, split, "test", horizon, grid, cfg)

        val_pair_sample = load_or_create_pair_sample(
            parsed,
            val_bundle,
            "val",
            horizon,
            num_pairs=cfg.search_num_pairs,
            seed=cfg.seed + 101 + int(horizon),
            cfg=cfg,
        )
        search_rows = _idk_search_rows(
            dataset_id=dataset_id,
            horizon=int(horizon),
            val_bundle=val_bundle,
            val_pair_sample=val_pair_sample,
            cfg=cfg,
            train_positions=train_positions,
        )
        best_ik = _best_idk_row(search_rows, cfg)
        print(
            f"[successor-distance] dataset={dataset_id} horizon={int(horizon)} best_idk "
            f"ensemble={int(best_ik['ik_ensemble_size'])} subsample={int(best_ik['ik_subsample_size'])} "
            f"temp={float(best_ik['ik_temperature']):g} val_{cfg.selection_metric}={float(best_ik[cfg.selection_metric]):.4f}",
            flush=True,
        )

        test_pair_sample = load_or_create_pair_sample(
            parsed,
            test_bundle,
            "test",
            horizon,
            num_pairs=cfg.eval_num_pairs,
            seed=cfg.seed + 307 + int(horizon),
            cfg=cfg,
        )
        retrieval_bank = load_or_create_retrieval_bank(parsed, test_bundle, "test", horizon, cfg)
        pair_a, pair_b = _pair_windows(test_bundle, test_pair_sample)
        query_windows = np.asarray(test_bundle.future_windows[retrieval_bank.query_local_indices], dtype=np.float32)
        candidate_windows = np.asarray(test_bundle.future_windows[retrieval_bank.candidate_local_indices], dtype=np.float32)

        summary_rows: list[dict[str, Any]] = []
        recall_rows_all: list[dict[str, Any]] = []
        figure_paths: dict[str, str] = {}

        for method_name in METHOD_ORDER:
            print(
                f"[successor-distance] dataset={dataset_id} horizon={int(horizon)} evaluating method={method_name}",
                flush=True,
            )
            distances = _compute_method_pair_distances(
                method_name,
                pair_a,
                pair_b,
                cfg=cfg,
                seed=cfg.seed + 509 + int(horizon),
                train_positions=train_positions,
                sigma=sigma,
                adaptive_metric=adaptive_metric,
                ik_ensemble_size=int(best_ik["ik_ensemble_size"]),
                ik_subsample_size=int(best_ik["ik_subsample_size"]),
                ik_temperature=float(best_ik["ik_temperature"]),
                pair_batch_size=cfg.eval_pair_batch_size,
            )
            pair_metrics = _binary_pair_metrics(test_pair_sample.labels, distances)
            distance_matrix = _compute_method_distance_matrix(
                method_name,
                query_windows,
                candidate_windows,
                cfg=cfg,
                seed=cfg.seed + 677 + int(horizon),
                train_positions=train_positions,
                sigma=sigma,
                adaptive_metric=adaptive_metric,
                ik_ensemble_size=int(best_ik["ik_ensemble_size"]),
                ik_subsample_size=int(best_ik["ik_subsample_size"]),
                ik_temperature=float(best_ik["ik_temperature"]),
            )
            recall_rows = _evaluate_recall_rows(test_bundle, retrieval_bank, distance_matrix, cfg.recall_k_values)
            recall_summary = _summarize_recall_rows(recall_rows, cfg.recall_k_values)
            summary_row: dict[str, Any] = {
                "dataset": dataset_id,
                "dataset_slug": dataset_slug(dataset_id),
                "horizon": int(horizon),
                "method": method_name,
                "num_pairs": int(test_pair_sample.labels.shape[0]),
                "positive_fraction": float(test_pair_sample.positive_fraction),
                "auroc": float(pair_metrics["auroc"]),
                "auprc": float(pair_metrics["auprc"]),
                "gdk_sigma": float(sigma),
                "best_ik_ensemble_size": int(best_ik["ik_ensemble_size"]),
                "best_ik_subsample_size": int(best_ik["ik_subsample_size"]),
                "best_ik_temperature": float(best_ik["ik_temperature"]),
            }
            summary_row.update(recall_summary)
            summary_rows.append(summary_row)

            for row in recall_rows:
                recall_rows_all.append(
                    {
                        "dataset": dataset_id,
                        "dataset_slug": dataset_slug(dataset_id),
                        "horizon": int(horizon),
                        "method": method_name,
                        **row,
                    }
                )

            figure_key = f"{dataset_slug(dataset_id)}_h{int(horizon)}_{method_name}_topk_query"
            figure_path = os.path.join(cfg.figures_dir, f"{figure_key}.png")
            _plot_method_topk_map(
                dataset_id=dataset_id,
                horizon=int(horizon),
                method_name=method_name,
                parsed=parsed,
                bundle=test_bundle,
                bank=retrieval_bank,
                distance_matrix=distance_matrix,
                top_k=cfg.plot_top_k,
                figure_path=figure_path,
            )
            figure_paths[figure_key] = figure_path

        best_ik_with_rows = dict(best_ik)
        best_ik_with_rows["search_rows"] = search_rows
        results.append(
            SuccessorEvalResult(
                dataset=dataset_id,
                horizon=int(horizon),
                best_ik_row=best_ik_with_rows,
                summary_rows=summary_rows,
                recall_rows=recall_rows_all,
                figure_paths=figure_paths,
                sigma=float(sigma),
            )
        )
        if result_callback is not None:
            result_callback(results[-1])
        print(
            f"[successor-distance] dataset={dataset_id} horizon={int(horizon)} complete",
            flush=True,
        )

    return results


def _write_successor_outputs(
    cfg: SuccessorDistanceConfig,
    search_rows: list[dict[str, Any]],
    best_ik_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    recall_rows: list[dict[str, Any]],
    figure_paths: dict[str, str],
) -> dict[str, Any]:
    overall_rows = _summarize_overall(summary_rows, cfg.recall_k_values)
    search_full_path = os.path.join(cfg.search_dir, "idk_search_full.csv")
    search_best_path = os.path.join(cfg.search_dir, "idk_best_config.csv")
    per_dataset_path = os.path.join(cfg.tables_dir, "per_dataset_metrics.csv")
    overall_path = os.path.join(cfg.tables_dir, "overall_summary.csv")
    recall_path = os.path.join(cfg.tables_dir, "per_query_recall.csv")
    report_path = os.path.join(cfg.output_dir, "report.md")

    _save_csv(search_full_path, _sort_search_rows(search_rows), IDK_SEARCH_FIELDS)
    _save_csv(search_best_path, best_ik_rows, IDK_SEARCH_FIELDS)

    per_dataset_fields = [
        "dataset",
        "dataset_slug",
        "horizon",
        "method",
        "num_pairs",
        "positive_fraction",
        "auroc",
        "auprc",
        "gdk_sigma",
        "best_ik_ensemble_size",
        "best_ik_subsample_size",
        "best_ik_temperature",
    ] + [f"recall_at_{int(k)}" for k in cfg.recall_k_values]
    _save_csv(per_dataset_path, summary_rows, per_dataset_fields)

    overall_fields = ["method", "num_dataset_h_pairs", "auroc_mean", "auprc_mean"] + [
        f"recall_at_{int(k)}_mean" for k in cfg.recall_k_values
    ]
    _save_csv(overall_path, overall_rows, overall_fields)

    recall_fields = [
        "dataset",
        "dataset_slug",
        "horizon",
        "method",
        "query_row",
        "query_global_index",
        "query_future_region_id",
        "num_positives",
    ] + [f"recall_at_{int(k)}" for k in cfg.recall_k_values]
    _save_csv(recall_path, recall_rows, recall_fields)

    lines = _report_lines(cfg, search_rows, best_ik_rows, summary_rows, overall_rows, figure_paths)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return {
        "overall_rows": overall_rows,
        "search_full_path": search_full_path,
        "search_best_path": search_best_path,
        "per_dataset_path": per_dataset_path,
        "overall_path": overall_path,
        "recall_path": recall_path,
        "report_path": report_path,
    }


def run_successor_distance(cfg: SuccessorDistanceConfig) -> dict[str, Any]:
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir)
    ensure_dir(cfg.search_dir)
    ensure_dir(cfg.tables_dir)
    ensure_dir(cfg.figures_dir)

    all_results: list[SuccessorEvalResult] = []
    search_rows: list[dict[str, Any]] = []
    best_ik_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    figure_paths: dict[str, str] = {}
    output_paths: dict[str, Any] | None = None

    def _consume_result(result: SuccessorEvalResult) -> None:
        search_rows.extend(list(result.best_ik_row.pop("search_rows")))
        best_ik_rows.append(dict(result.best_ik_row))
        summary_rows.extend(result.summary_rows)
        recall_rows.extend(result.recall_rows)
        figure_paths.update(result.figure_paths)

        nonlocal output_paths
        output_paths = _write_successor_outputs(
            cfg=cfg,
            search_rows=search_rows,
            best_ik_rows=best_ik_rows,
            summary_rows=summary_rows,
            recall_rows=recall_rows,
            figure_paths=figure_paths,
        )
        print(
            f"[successor-distance] wrote partial outputs after dataset={result.dataset} horizon={int(result.horizon)}",
            flush=True,
        )

    for dataset_id in cfg.datasets:
        print(f"[successor-distance] loading dataset={dataset_id}", flush=True)
        parsed = load_or_parse_dataset_with_fallback(dataset_id, cfg)
        dataset_results = analyze_successor_distance_dataset(
            parsed,
            dataset_id,
            cfg,
            result_callback=_consume_result,
        )
        all_results.extend(dataset_results)

    if output_paths is None:
        output_paths = _write_successor_outputs(
            cfg=cfg,
            search_rows=search_rows,
            best_ik_rows=best_ik_rows,
            summary_rows=summary_rows,
            recall_rows=recall_rows,
            figure_paths=figure_paths,
        )

    return {
        "search_rows": search_rows,
        "best_ik_rows": best_ik_rows,
        "summary_rows": summary_rows,
        "recall_rows": recall_rows,
        "overall_rows": output_paths["overall_rows"],
        "search_full_path": output_paths["search_full_path"],
        "search_best_path": output_paths["search_best_path"],
        "per_dataset_path": output_paths["per_dataset_path"],
        "overall_path": output_paths["overall_path"],
        "recall_path": output_paths["recall_path"],
        "report_path": output_paths["report_path"],
        "figure_paths": figure_paths,
        "config": asdict(cfg),
    }
