from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import heapq
import math
import time
from typing import Iterable

import numpy as np
import torch

from core.isolation_kernel import SoftIsolationKernel


class SkipReason(IntEnum):
    OK = 0
    DISABLED = 1
    INTERVAL_NOT_DUE = 2
    MISSING_EVAL_PATHS = 3
    MISSING_SKILL_LABELS = 4
    MISSING_STATE_POINTS_OR_COORDINATES = 5
    TOO_FEW_CLUSTERS = 6
    TOO_FEW_TRAJS_PER_CLUSTER = 7
    DETERMINISTIC_REPEATS_DEGENERATE = 8
    NONFINITE_DISTANCE_MATRIX = 9
    TIMEOUT = 10
    BACKEND_EXCEPTION = 11
    UNSUPPORTED_BACKEND_FOR_ENV = 12
    OPTIONAL_DEPENDENCY_MISSING = 13
    VIDEO_TRAJECTORIES_NOT_SUITABLE = 14


class StateSpaceCode(IntEnum):
    XY = 1
    NORMALIZED_RAW_STATE = 2
    UNSUPPORTED = 3


BACKEND_MASKS = {
    "temporal": 1,
    "ikse": 2,
    "euclidean": 4,
}


@dataclass
class StructureOptions:
    options: np.ndarray
    skill_ids: np.ndarray
    anchor_options: np.ndarray
    anchor_skill_ids: np.ndarray
    subsampled: bool


@dataclass
class StateExtractionResult:
    traj_state_sequences: list[np.ndarray]
    point_states: np.ndarray
    state_space_name: str
    state_space_code: int
    state_tag_suffix: str
    skipped: bool
    skip_reason_code: int
    entropy_num_points: int
    subsampled: bool


@dataclass
class GroupingResult:
    labels: np.ndarray
    groups: dict[int, list[int]]
    valid_groups: dict[int, list[int]]
    skipped: bool
    skip_reason_code: int
    dbi_num_trajs: int
    num_skills: int
    num_trajs_per_skill_mean: float


def parse_structure_backends(backends) -> list[str]:
    if backends is None:
        return ["temporal", "ikse"]
    if isinstance(backends, str):
        values = [item.strip() for item in backends.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in backends if str(item).strip()]
    ordered = []
    for value in values:
        if value not in ("temporal", "ikse"):
            raise ValueError(f"Unsupported structure metrics backend: {value}")
        if value not in ordered:
            ordered.append(value)
    if not ordered:
        raise ValueError("At least one structure metrics backend is required")
    return ordered


def build_structure_eval_options(
        *,
        discrete: bool,
        dim_skill: int,
        unit_length: bool,
        rollouts_per_skill: int,
        num_skills: int,
        max_trajs: int,
        anchor_seed: int,
        num_random_trajectories: int = 16,
        use_hierarchical_skill: bool = False,
        num_skill_levels: int = 1) -> StructureOptions:
    rollouts_per_skill = int(rollouts_per_skill)
    if rollouts_per_skill < 1:
        raise ValueError("rollouts_per_skill must be >= 1")
    if int(max_trajs) < 1:
        raise ValueError("max_trajs must be >= 1")
    if int(dim_skill) < 1:
        return StructureOptions(
            options=np.zeros((0, 0), dtype=np.float32),
            skill_ids=np.asarray([], dtype=np.int64),
            anchor_options=np.zeros((0, 0), dtype=np.float32),
            anchor_skill_ids=np.asarray([], dtype=np.int64),
            subsampled=False,
        )

    max_skills_by_trajs = max(1, int(max_trajs) // rollouts_per_skill)
    if num_skills is None or int(num_skills) < 0:
        requested_skills = int(dim_skill) if discrete else min(16, max(1, int(num_random_trajectories)))
    else:
        requested_skills = int(num_skills)
    requested_skills = max(1, requested_skills)
    rng = np.random.RandomState(int(anchor_seed))

    if discrete:
        available_ids = np.arange(int(dim_skill), dtype=np.int64)
        if requested_skills < available_ids.size:
            available_ids = available_ids[:requested_skills]
        subsampled = available_ids.size > max_skills_by_trajs
        if subsampled:
            selected = np.sort(rng.choice(available_ids, size=max_skills_by_trajs, replace=False))
        else:
            selected = available_ids
        base = np.eye(int(dim_skill), dtype=np.float32)[selected]
        anchor_ids = selected.astype(np.int64)
    else:
        base_count = requested_skills
        subsampled = base_count > max_skills_by_trajs
        if int(dim_skill) == 2:
            angles = np.linspace(0.0, 2.0 * np.pi, num=base_count, endpoint=False)
            radius = 1.0 if bool(unit_length) else 1.5
            all_base = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1).astype(np.float32)
        else:
            all_base = rng.randn(base_count, int(dim_skill)).astype(np.float32)
            if bool(unit_length):
                all_base = all_base / (np.linalg.norm(all_base, axis=1, keepdims=True) + 1e-8)
        all_ids = np.arange(base_count, dtype=np.int64)
        if subsampled:
            selected = np.sort(rng.choice(all_ids, size=max_skills_by_trajs, replace=False))
            base = all_base[selected]
            anchor_ids = selected.astype(np.int64)
        else:
            base = all_base
            anchor_ids = all_ids

    if bool(use_hierarchical_skill):
        depth = max(1, int(num_skill_levels))
        hierarchical = np.zeros((base.shape[0], depth, int(dim_skill)), dtype=np.float32)
        hierarchical[:, 0, :] = base
        base = hierarchical

    options = np.repeat(base, rollouts_per_skill, axis=0).astype(np.float32)
    skill_ids = np.repeat(anchor_ids, rollouts_per_skill).astype(np.int64)
    return StructureOptions(
        options=options,
        skill_ids=skill_ids,
        anchor_options=base.astype(np.float32),
        anchor_skill_ids=anchor_ids.astype(np.int64),
        subsampled=bool(subsampled),
    )


def extract_structure_state_sequences(
        trajectories,
        env_name: str,
        state_space_mode: str = "auto",
        *,
        states_per_traj: int = 10,
        max_points: int = 1000,
        anchor_seed: int = 0) -> StateExtractionResult:
    del state_space_mode
    trajectories = list(trajectories or [])
    if not trajectories:
        return _state_skip(SkipReason.MISSING_EVAL_PATHS)

    use_xy = _env_uses_xy(env_name)
    sequences = []
    skip_reason = SkipReason.MISSING_STATE_POINTS_OR_COORDINATES
    for trajectory in trajectories:
        sequence = _extract_xy_sequence(trajectory) if use_xy else _extract_raw_state_sequence(trajectory)
        if sequence is None:
            continue
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[0] < 1 or sequence.shape[1] < 1:
            continue
        if _looks_like_pixel_state(sequence):
            skip_reason = SkipReason.MISSING_STATE_POINTS_OR_COORDINATES
            continue
        sequences.append(sequence)

    if not sequences:
        return _state_skip(skip_reason)

    if use_xy:
        normalized_sequences = [sequence[:, :2].astype(np.float32) for sequence in sequences]
        state_space_name = "XY"
        state_code = int(StateSpaceCode.XY)
        suffix = "XY"
    else:
        stacked = np.concatenate(sequences, axis=0).astype(np.float32)
        mean = stacked.mean(axis=0, keepdims=True)
        std = stacked.std(axis=0, keepdims=True) + 1e-6
        normalized_sequences = [((sequence - mean) / std).astype(np.float32) for sequence in sequences]
        state_space_name = "NormalizedRawState"
        state_code = int(StateSpaceCode.NORMALIZED_RAW_STATE)
        suffix = "NormalizedRawState"

    sampled_points, sampled_sequences, subsampled = _sample_state_points(
        normalized_sequences,
        states_per_traj=states_per_traj,
        max_points=max_points,
        anchor_seed=anchor_seed,
    )
    if sampled_points.shape[0] < 2:
        return _state_skip(SkipReason.MISSING_STATE_POINTS_OR_COORDINATES)

    return StateExtractionResult(
        traj_state_sequences=sampled_sequences,
        point_states=sampled_points,
        state_space_name=state_space_name,
        state_space_code=state_code,
        state_tag_suffix=suffix,
        skipped=False,
        skip_reason_code=int(SkipReason.OK),
        entropy_num_points=int(sampled_points.shape[0]),
        subsampled=bool(subsampled),
    )


def group_trajectories_by_skill(
        trajectories,
        options=None,
        skill_ids=None,
        *,
        min_trajs_per_cluster: int = 3) -> GroupingResult:
    trajectories = list(trajectories or [])
    labels = _coerce_skill_labels(trajectories, options=options, skill_ids=skill_ids)
    if labels is None or len(labels) != len(trajectories):
        return GroupingResult(
            labels=np.asarray([], dtype=np.int64),
            groups={},
            valid_groups={},
            skipped=True,
            skip_reason_code=int(SkipReason.MISSING_SKILL_LABELS),
            dbi_num_trajs=0,
            num_skills=0,
            num_trajs_per_skill_mean=0.0,
        )

    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(np.asarray(labels, dtype=np.int64)):
        groups.setdefault(int(label), []).append(int(idx))
    valid_groups = {
        label: indices
        for label, indices in groups.items()
        if len(indices) >= int(min_trajs_per_cluster)
    }
    counts = [len(indices) for indices in groups.values()]
    if len(valid_groups) < 2:
        reason = SkipReason.TOO_FEW_CLUSTERS if len(groups) < 2 else SkipReason.TOO_FEW_TRAJS_PER_CLUSTER
        return GroupingResult(
            labels=np.asarray(labels, dtype=np.int64),
            groups=groups,
            valid_groups=valid_groups,
            skipped=True,
            skip_reason_code=int(reason),
            dbi_num_trajs=sum(len(indices) for indices in valid_groups.values()),
            num_skills=len(groups),
            num_trajs_per_skill_mean=float(np.mean(counts)) if counts else 0.0,
        )

    return GroupingResult(
        labels=np.asarray(labels, dtype=np.int64),
        groups=groups,
        valid_groups=valid_groups,
        skipped=False,
        skip_reason_code=int(SkipReason.OK),
        dbi_num_trajs=sum(len(indices) for indices in valid_groups.values()),
        num_skills=len(groups),
        num_trajs_per_skill_mean=float(np.mean(counts)) if counts else 0.0,
    )


def compute_temporal_structure_metrics(
        traj_state_sequences,
        grouping: GroupingResult,
        *,
        state_suffix: str,
        device="cpu",
        knn_k: int = 8,
        soft_dtw_gamma: float = 1.0,
        deterministic_policy: bool = True,
        degenerate_policy: str = "skip"):
    del device
    result = {
        "StructureMetricsTemporalSkipped": 1.0,
        "StructureMetricsTemporalSkipReasonCode": float(SkipReason.OK),
        "StructureMetricsTemporalDegenerateWarnCompute": 0.0,
    }
    if len(traj_state_sequences) < 2:
        result["StructureMetricsTemporalSkipReasonCode"] = float(SkipReason.MISSING_EVAL_PATHS)
        return result

    point_distances, trajectory_indices = _build_temporal_point_distances(traj_state_sequences, knn_k=knn_k)
    if point_distances is None or not np.all(np.isfinite(point_distances)):
        result["StructureMetricsTemporalSkipReasonCode"] = float(SkipReason.NONFINITE_DISTANCE_MATRIX)
        return result

    entropy = _knn_entropy_from_distances(point_distances)
    if entropy is not None:
        result[f"Entropy_TemporalParticle_{state_suffix}"] = float(entropy)
        result["Entropy_TemporalParticle"] = float(entropy)

    dbi = None
    dbi_reason = int(grouping.skip_reason_code)
    degenerate = _has_degenerate_repeats(traj_state_sequences, grouping.valid_groups)
    allow_degenerate_compute = str(degenerate_policy) == "warn_compute"
    if deterministic_policy and degenerate and allow_degenerate_compute:
        result["StructureMetricsTemporalDegenerateWarnCompute"] = 1.0
    if deterministic_policy and degenerate and not allow_degenerate_compute:
        dbi_reason = int(SkipReason.DETERMINISTIC_REPEATS_DEGENERATE)
    elif not grouping.skipped:
        trajectory_distances = _trajectory_soft_dtw_distance_matrix(
            point_distances,
            trajectory_indices,
            gamma=soft_dtw_gamma,
        )
        if np.all(np.isfinite(trajectory_distances)):
            valid_indices, valid_labels = _valid_group_indices_and_labels(grouping)
            dbi = _calculate_medoid_dbi(
                trajectory_distances[np.ix_(valid_indices, valid_indices)],
                valid_labels,
            )
            dbi_reason = int(SkipReason.OK)
        else:
            dbi_reason = int(SkipReason.NONFINITE_DISTANCE_MATRIX)

    if dbi is not None:
        result[f"DBI_TemporalMedoid_{state_suffix}"] = float(dbi)
        result["DBI_TemporalMedoid"] = float(dbi)
    if entropy is not None and dbi is not None:
        result["Score_TemporalParticle_DBI"] = float(entropy) / (float(dbi) + 1e-8)
        result["StructureMetricsTemporalSkipped"] = 0.0
        result["StructureMetricsTemporalSkipReasonCode"] = float(SkipReason.OK)
    else:
        result["StructureMetricsTemporalSkipReasonCode"] = float(dbi_reason)
    return result


def compute_ikse_structure_metrics(
        point_states,
        traj_state_sequences,
        grouping: GroupingResult,
        *,
        state_suffix: str,
        device="cpu",
        anchor_seed: int = 0,
        deterministic_policy: bool = True,
        degenerate_policy: str = "skip",
        ensemble_size: int = 100,
        subsample_size: int | None = None):
    del device
    result = {
        "StructureMetricsIKSkipped": 1.0,
        "StructureMetricsIKSkipReasonCode": float(SkipReason.OK),
        "StructureMetricsIKDegenerateWarnCompute": 0.0,
    }
    point_states = np.asarray(point_states, dtype=np.float32)
    if point_states.ndim != 2 or point_states.shape[0] < 2:
        result["StructureMetricsIKSkipReasonCode"] = float(SkipReason.MISSING_STATE_POINTS_OR_COORDINATES)
        return result

    try:
        kernel, maps = _fit_ik_maps(
            point_states,
            anchor_seed=anchor_seed,
            ensemble_size=ensemble_size,
            subsample_size=subsample_size,
        )
        entropy = _calculate_ik_entropy(maps, ensemble_size=ensemble_size)
        result[f"Entropy_IKDE_{state_suffix}"] = float(entropy)
        result["Entropy_IKDE"] = float(entropy)
    except Exception:
        result["StructureMetricsIKSkipReasonCode"] = float(SkipReason.BACKEND_EXCEPTION)
        return result

    dbi = None
    dbi_reason = int(grouping.skip_reason_code)
    degenerate = _has_degenerate_repeats(traj_state_sequences, grouping.valid_groups)
    allow_degenerate_compute = str(degenerate_policy) == "warn_compute"
    if deterministic_policy and degenerate and allow_degenerate_compute:
        result["StructureMetricsIKDegenerateWarnCompute"] = 1.0
    if deterministic_policy and degenerate and not allow_degenerate_compute:
        dbi_reason = int(SkipReason.DETERMINISTIC_REPEATS_DEGENERATE)
    elif not grouping.skipped:
        try:
            traj_maps = _trajectory_ik_mean_maps(
                kernel,
                traj_state_sequences,
                ensemble_size=ensemble_size,
            )
            valid_indices, valid_labels = _valid_group_indices_and_labels(grouping)
            dbi = _calculate_ik_legacy_dbi(
                traj_maps[torch.as_tensor(valid_indices, dtype=torch.long)],
                valid_labels,
                ensemble_size=ensemble_size,
            )
            dbi_reason = int(SkipReason.OK)
        except Exception:
            dbi_reason = int(SkipReason.BACKEND_EXCEPTION)

    if dbi is not None:
        result[f"DBI_IKMeanRatio_Legacy_{state_suffix}"] = float(dbi)
        result["DBI_IKMeanRatio_Legacy"] = float(dbi)
    if entropy is not None and dbi is not None:
        result["IKSE_LegacyDBI"] = float(entropy) / (float(dbi) + 1e-8)
        result["StructureMetricsIKSkipped"] = 0.0
        result["StructureMetricsIKSkipReasonCode"] = float(SkipReason.OK)
    else:
        result["StructureMetricsIKSkipReasonCode"] = float(dbi_reason)
    return result


def compute_training_eval_structure_metrics(
        trajectories,
        *,
        options=None,
        skill_ids=None,
        cfg=None,
        env_name: str = "",
        device="cpu",
        backends=None,
        used_video_trajectories: bool = False,
        used_extra_rollouts: bool = False,
        options_subsampled: bool = False):
    start_time = time.perf_counter()
    requested_backends = parse_structure_backends(backends)
    metrics = _base_metric_dict(requested_backends)
    metrics["StructureMetricsEnabled"] = 1.0
    metrics["StructureMetricsOriginIsTrainingHook"] = 1.0
    metrics["StructureMetricsUsedVideoTrajectories"] = float(bool(used_video_trajectories))
    metrics["StructureMetricsUsedExtraRollouts"] = float(bool(used_extra_rollouts))

    trajectories = list(trajectories or [])
    metrics["StructureMetricsNumTrajs"] = float(len(trajectories))
    if not trajectories:
        _mark_all_skipped(metrics, requested_backends, SkipReason.MISSING_EVAL_PATHS)
        _finish_metrics(metrics, start_time)
        return metrics

    states_per_traj = int(getattr(cfg, "eval_structure_metrics_states_per_traj", 10))
    max_points = int(getattr(cfg, "eval_structure_metrics_max_points", 1000))
    anchor_seed = int(getattr(cfg, "eval_structure_metrics_anchor_seed", 0))
    rollouts_per_skill = int(getattr(cfg, "eval_structure_metrics_rollouts_per_skill", 3))
    deterministic_policy = str(getattr(cfg, "eval_structure_metrics_policy_mode", "deterministic")) == "deterministic"
    degenerate_policy = str(getattr(cfg, "eval_structure_metrics_degenerate_policy", "skip"))

    state_result = extract_structure_state_sequences(
        trajectories,
        env_name,
        states_per_traj=states_per_traj,
        max_points=max_points,
        anchor_seed=anchor_seed,
    )
    metrics["StructureMetricsStateSpaceCode"] = float(state_result.state_space_code)
    metrics["StructureMetricsEntropyNumPoints"] = float(state_result.entropy_num_points)
    metrics["StructureMetricsSubsampled"] = float(bool(options_subsampled or state_result.subsampled))
    if state_result.skipped:
        _mark_all_skipped(metrics, requested_backends, SkipReason(state_result.skip_reason_code))
        _finish_metrics(metrics, start_time)
        return metrics

    grouping = group_trajectories_by_skill(
        trajectories,
        options=options,
        skill_ids=skill_ids,
        min_trajs_per_cluster=rollouts_per_skill,
    )
    metrics["StructureMetricsNumSkills"] = float(grouping.num_skills)
    metrics["StructureMetricsNumTrajsPerSkillMean"] = float(grouping.num_trajs_per_skill_mean)
    metrics["StructureMetricsDBINumTrajs"] = float(grouping.dbi_num_trajs)
    degenerate_repeats = _has_degenerate_repeats(state_result.traj_state_sequences, grouping.valid_groups)
    metrics["StructureMetricsDegenerateRepeats"] = float(degenerate_repeats)
    metrics["StructureMetricsDegeneratePolicyWarnCompute"] = float(
        deterministic_policy and degenerate_repeats and degenerate_policy == "warn_compute"
    )

    if "temporal" in requested_backends:
        temporal_metrics = compute_temporal_structure_metrics(
            state_result.traj_state_sequences,
            grouping,
            state_suffix=state_result.state_tag_suffix,
            device=device,
            knn_k=int(getattr(cfg, "temporal_graph_knn_k", 8)),
            soft_dtw_gamma=float(getattr(cfg, "soft_dtw_gamma", 1.0)),
            deterministic_policy=deterministic_policy,
            degenerate_policy=degenerate_policy,
        )
        metrics.update(temporal_metrics)
    if "ikse" in requested_backends:
        ik_metrics = compute_ikse_structure_metrics(
            state_result.point_states,
            state_result.traj_state_sequences,
            grouping,
            state_suffix=state_result.state_tag_suffix,
            device="cpu",
            anchor_seed=anchor_seed,
            deterministic_policy=deterministic_policy,
            degenerate_policy=degenerate_policy,
        )
        metrics.update(ik_metrics)

    backend_skips = []
    if "temporal" in requested_backends:
        backend_skips.append(bool(metrics.get("StructureMetricsTemporalSkipped", 1.0)))
    if "ikse" in requested_backends:
        backend_skips.append(bool(metrics.get("StructureMetricsIKSkipped", 1.0)))
    all_skipped = bool(backend_skips) and all(backend_skips)
    metrics["StructureMetricsSkipped"] = float(all_skipped)
    metrics["StructureMetricsSkipReasonCode"] = _combined_skip_reason(metrics, requested_backends, all_skipped)
    _finish_metrics(metrics, start_time)
    return metrics


def infer_skill_ids_from_trajectories(trajectories):
    return _coerce_skill_labels(list(trajectories or []), options=None, skill_ids=None)


def video_trajectories_are_suitable(trajectories, *, min_trajs_per_cluster: int, policy_mode: str, video_policy_mode: str | None):
    if not trajectories:
        return False, int(SkipReason.VIDEO_TRAJECTORIES_NOT_SUITABLE), None
    if video_policy_mode is not None and str(video_policy_mode) != str(policy_mode):
        return False, int(SkipReason.VIDEO_TRAJECTORIES_NOT_SUITABLE), None
    labels = infer_skill_ids_from_trajectories(trajectories)
    if labels is None:
        return False, int(SkipReason.MISSING_SKILL_LABELS), None
    grouping = group_trajectories_by_skill(
        trajectories,
        skill_ids=labels,
        min_trajs_per_cluster=min_trajs_per_cluster,
    )
    if grouping.skipped:
        return False, int(SkipReason.VIDEO_TRAJECTORIES_NOT_SUITABLE), labels
    return True, int(SkipReason.OK), labels


def interval_skip_metrics(backends) -> dict[str, float]:
    requested_backends = parse_structure_backends(backends)
    metrics = _base_metric_dict(requested_backends)
    metrics["StructureMetricsEnabled"] = 1.0
    metrics["StructureMetricsOriginIsTrainingHook"] = 1.0
    metrics["StructureMetricsSkipped"] = 1.0
    metrics["StructureMetricsSkipReasonCode"] = float(SkipReason.INTERVAL_NOT_DUE)
    for backend in requested_backends:
        if backend == "temporal":
            metrics["StructureMetricsTemporalSkipped"] = 1.0
            metrics["StructureMetricsTemporalSkipReasonCode"] = float(SkipReason.INTERVAL_NOT_DUE)
        elif backend == "ikse":
            metrics["StructureMetricsIKSkipped"] = 1.0
            metrics["StructureMetricsIKSkipReasonCode"] = float(SkipReason.INTERVAL_NOT_DUE)
    return metrics


def exception_skip_metrics(backends, reason: SkipReason = SkipReason.BACKEND_EXCEPTION) -> dict[str, float]:
    requested_backends = parse_structure_backends(backends)
    metrics = _base_metric_dict(requested_backends)
    metrics["StructureMetricsEnabled"] = 1.0
    metrics["StructureMetricsOriginIsTrainingHook"] = 1.0
    _mark_all_skipped(metrics, requested_backends, reason)
    metrics["StructureMetricsElapsedSec"] = 0.0
    return metrics


def _base_metric_dict(backends: Iterable[str]) -> dict[str, float]:
    requested = list(backends)
    mask = sum(BACKEND_MASKS[backend] for backend in requested)
    return {
        "MetricBackend_IsIKSE": float("ikse" in requested),
        "MetricBackend_UsesTemporalDistance": float("temporal" in requested),
        "MetricBackend_UsesIsolationKernel": float("ikse" in requested),
        "MetricBackend_UsesEuclideanDistance": 0.0,
        "StructureMetricsBackendMask": float(mask),
        "StructureMetricsNumBackends": float(len(requested)),
        "StructureMetricsEnabled": 1.0,
        "StructureMetricsOriginIsTrainingHook": 1.0,
        "StructureMetricsElapsedSec": 0.0,
        "StructureMetricsNumTrajs": 0.0,
        "StructureMetricsNumSkills": 0.0,
        "StructureMetricsNumTrajsPerSkillMean": 0.0,
        "StructureMetricsDBINumTrajs": 0.0,
        "StructureMetricsEntropyNumPoints": 0.0,
        "StructureMetricsStateSpaceCode": float(StateSpaceCode.UNSUPPORTED),
        "StructureMetricsSubsampled": 0.0,
        "StructureMetricsUsedVideoTrajectories": 0.0,
        "StructureMetricsUsedExtraRollouts": 0.0,
        "StructureMetricsSkipped": 0.0,
        "StructureMetricsSkipReasonCode": float(SkipReason.OK),
        "StructureMetricsDegenerateRepeats": 0.0,
        "StructureMetricsDegeneratePolicyWarnCompute": 0.0,
    }


def _mark_all_skipped(metrics, backends, reason: SkipReason):
    metrics["StructureMetricsSkipped"] = 1.0
    metrics["StructureMetricsSkipReasonCode"] = float(reason)
    if "temporal" in backends:
        metrics["StructureMetricsTemporalSkipped"] = 1.0
        metrics["StructureMetricsTemporalSkipReasonCode"] = float(reason)
    if "ikse" in backends:
        metrics["StructureMetricsIKSkipped"] = 1.0
        metrics["StructureMetricsIKSkipReasonCode"] = float(reason)


def _finish_metrics(metrics, start_time):
    metrics["StructureMetricsElapsedSec"] = float(time.perf_counter() - start_time)


def _combined_skip_reason(metrics, backends, all_skipped):
    if not all_skipped:
        return float(SkipReason.OK)
    reasons = []
    if "temporal" in backends:
        reasons.append(int(metrics.get("StructureMetricsTemporalSkipReasonCode", SkipReason.OK)))
    if "ikse" in backends:
        reasons.append(int(metrics.get("StructureMetricsIKSkipReasonCode", SkipReason.OK)))
    for reason in reasons:
        if reason != int(SkipReason.OK):
            return float(reason)
    return float(SkipReason.OK)


def _state_skip(reason: SkipReason) -> StateExtractionResult:
    return StateExtractionResult(
        traj_state_sequences=[],
        point_states=np.zeros((0, 0), dtype=np.float32),
        state_space_name="Unsupported",
        state_space_code=int(StateSpaceCode.UNSUPPORTED),
        state_tag_suffix="Unsupported",
        skipped=True,
        skip_reason_code=int(reason),
        entropy_num_points=0,
        subsampled=False,
    )


def _env_uses_xy(env_name: str) -> bool:
    env_name = str(env_name or "").lower()
    if "ant" in env_name:
        return True
    if not env_name.startswith("dmc_"):
        return False
    return any(name in env_name for name in ("quadruped", "humanoid", "walker", "cheetah"))


def _extract_xy_sequence(trajectory):
    env_infos = trajectory.get("env_infos", {}) or {}
    coordinates = env_infos.get("coordinates")
    next_coordinates = env_infos.get("next_coordinates")
    if coordinates is None or next_coordinates is None:
        return None
    coordinates = np.asarray(coordinates, dtype=np.float32)
    next_coordinates = np.asarray(next_coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0 or coordinates.shape[1] < 2:
        return None
    points = [coordinates[:, :2]]
    if next_coordinates.ndim == 2 and next_coordinates.shape[0] > 0 and next_coordinates.shape[1] >= 2:
        points.append(next_coordinates[-1:, :2])
    return np.concatenate(points, axis=0)


def _extract_raw_state_sequence(trajectory):
    env_infos = trajectory.get("env_infos", {}) or {}
    sequence = _sequence_from_pair(env_infos.get("ori_obs"), env_infos.get("next_ori_obs"))
    if sequence is not None:
        return sequence
    sequence = _sequence_from_pair(trajectory.get("observations"), trajectory.get("next_observations"))
    if sequence is not None:
        return sequence
    sequence = _sequence_from_pair(trajectory.get("observations"), trajectory.get("last_observations"))
    if sequence is not None:
        return sequence
    observations = trajectory.get("observations")
    if observations is None:
        return None
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2:
        return None
    return observations


def _sequence_from_pair(current, next_values):
    if current is None:
        return None
    current = np.asarray(current, dtype=np.float32)
    if current.ndim != 2 or current.shape[0] == 0:
        return None
    if next_values is None:
        return current
    next_values = np.asarray(next_values, dtype=np.float32)
    if next_values.ndim == 2 and next_values.shape[0] > 0 and next_values.shape[1] == current.shape[1]:
        return np.concatenate([current, next_values[-1:, :]], axis=0)
    return current


def _looks_like_pixel_state(sequence):
    sequence = np.asarray(sequence)
    if sequence.ndim > 2:
        return True
    if sequence.ndim != 2 or sequence.shape[1] < 3:
        return False
    dim = int(sequence.shape[1])
    channels, rem = divmod(dim, 3)
    if rem != 0:
        return False
    side = int(math.sqrt(max(channels, 1)))
    return side * side * 3 == dim and side >= 16


def _sample_state_points(sequences, *, states_per_traj: int, max_points: int, anchor_seed: int):
    sampled_sequences = []
    sampled_points = []
    subsampled = False
    for sequence in sequences:
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.shape[0] > int(states_per_traj):
            indices = np.linspace(0, sequence.shape[0] - 1, num=int(states_per_traj), dtype=np.int64)
            sequence = sequence[indices]
            subsampled = True
        sampled_sequences.append(sequence)
        sampled_points.append(sequence)
    if not sampled_points:
        return np.zeros((0, 0), dtype=np.float32), [], subsampled
    points = np.concatenate(sampled_points, axis=0).astype(np.float32)
    if points.shape[0] > int(max_points):
        rng = np.random.RandomState(int(anchor_seed))
        selected = np.sort(rng.choice(points.shape[0], size=int(max_points), replace=False))
        points = points[selected]
        subsampled = True
    return points, sampled_sequences, subsampled


def _coerce_skill_labels(trajectories, *, options=None, skill_ids=None):
    if skill_ids is not None:
        labels = np.asarray(skill_ids, dtype=np.int64)
        if labels.shape[0] == len(trajectories):
            return labels
    inferred = []
    for trajectory in trajectories:
        agent_infos = trajectory.get("agent_infos", {}) or {}
        if "skill" not in agent_infos:
            return None
        skill = np.asarray(agent_infos["skill"])
        if skill.size == 0:
            return None
        if skill.ndim >= 2:
            skill = skill.reshape(skill.shape[0], -1)[0]
        skill = skill.reshape(-1)
        if skill.size == 0:
            return None
        if np.isclose(skill.sum(), 1.0) and np.isclose(skill.max(), 1.0):
            inferred.append(int(np.argmax(skill)))
        else:
            inferred.append(_stable_continuous_label(skill))
    if len(inferred) == len(trajectories):
        return np.asarray(inferred, dtype=np.int64)
    if options is not None and len(options) == len(trajectories):
        labels = []
        for option in np.asarray(options):
            option = np.asarray(option, dtype=np.float32).reshape(-1)
            if option.size == 0:
                return None
            if np.isclose(option.sum(), 1.0) and np.isclose(option.max(), 1.0):
                labels.append(int(np.argmax(option)))
            else:
                labels.append(_stable_continuous_label(option))
        return np.asarray(labels, dtype=np.int64)
    return None


def _stable_continuous_label(skill):
    rounded = tuple(np.round(np.asarray(skill, dtype=np.float32), decimals=6).tolist())
    return abs(hash(rounded)) % (2 ** 31)


def _build_temporal_point_distances(sequences, *, knn_k: int):
    point_chunks = []
    trajectory_indices = []
    cursor = 0
    for sequence in sequences:
        sequence = np.asarray(sequence, dtype=np.float32)
        point_chunks.append(sequence)
        indices = np.arange(cursor, cursor + sequence.shape[0], dtype=np.int64)
        trajectory_indices.append(indices)
        cursor += sequence.shape[0]
    if not point_chunks:
        return None, []
    points = np.concatenate(point_chunks, axis=0).astype(np.float32)
    if points.shape[0] < 2:
        return None, []
    euclidean = _pairwise_euclidean(points)
    if not np.all(np.isfinite(euclidean)):
        return euclidean, trajectory_indices
    adjacency = [[] for _ in range(points.shape[0])]
    for indices in trajectory_indices:
        for left, right in zip(indices[:-1], indices[1:]):
            weight = float(euclidean[left, right])
            adjacency[int(left)].append((int(right), weight))
            adjacency[int(right)].append((int(left), weight))
    k = min(max(1, int(knn_k)), points.shape[0] - 1)
    neighbors = np.argpartition(euclidean, kth=k, axis=1)[:, :k + 1]
    for idx in range(points.shape[0]):
        for neighbor in neighbors[idx]:
            neighbor = int(neighbor)
            if neighbor == idx:
                continue
            weight = float(euclidean[idx, neighbor])
            adjacency[idx].append((neighbor, weight))
            adjacency[neighbor].append((idx, weight))
    distances = _all_pairs_shortest_paths(adjacency)
    disconnected = ~np.isfinite(distances)
    if np.any(disconnected):
        distances[disconnected] = euclidean[disconnected]
    return distances.astype(np.float32), trajectory_indices


def _pairwise_euclidean(points):
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32))
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32)


def _dijkstra(adjacency, start):
    distances = np.full(len(adjacency), np.inf, dtype=np.float32)
    distances[int(start)] = 0.0
    heap = [(0.0, int(start))]
    while heap:
        current_distance, node = heapq.heappop(heap)
        if current_distance > float(distances[node]):
            continue
        for neighbor, weight in adjacency[node]:
            candidate = current_distance + float(weight)
            if candidate < float(distances[neighbor]):
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, int(neighbor)))
    return distances


def _all_pairs_shortest_paths(adjacency):
    try:
        from scipy.sparse.csgraph import shortest_path
    except ImportError:
        return np.stack([_dijkstra(adjacency, start) for start in range(len(adjacency))], axis=0)

    graph = np.full((len(adjacency), len(adjacency)), np.inf, dtype=np.float32)
    np.fill_diagonal(graph, 0.0)
    for node, edges in enumerate(adjacency):
        for neighbor, weight in edges:
            neighbor = int(neighbor)
            weight = float(weight)
            if weight < float(graph[node, neighbor]):
                graph[node, neighbor] = weight
    return shortest_path(graph, directed=False, unweighted=False).astype(np.float32)


def _knn_entropy_from_distances(distances, *, k: int = 3, eps: float = 1e-9):
    distances = np.asarray(distances, dtype=np.float32).copy()
    if distances.ndim != 2 or distances.shape[0] < 2:
        return None
    if not np.all(np.isfinite(distances)):
        return None
    np.fill_diagonal(distances, np.inf)
    effective_k = min(int(k), distances.shape[0] - 1)
    kth = np.partition(distances, kth=effective_k - 1, axis=1)[:, effective_k - 1]
    if not np.all(np.isfinite(kth)):
        return None
    return float(math.lgamma(distances.shape[0]) - math.lgamma(effective_k) + np.mean(np.log(kth + eps)))


def _trajectory_soft_dtw_distance_matrix(point_distances, trajectory_indices, *, gamma: float):
    num_trajs = len(trajectory_indices)
    result = np.zeros((num_trajs, num_trajs), dtype=np.float32)
    for i in range(num_trajs):
        for j in range(i + 1, num_trajs):
            local = point_distances[np.ix_(trajectory_indices[i], trajectory_indices[j])]
            distance = max(0.0, _soft_dtw(local, gamma=gamma))
            result[i, j] = distance
            result[j, i] = distance
    return result


def _soft_dtw(cost, *, gamma: float):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return 0.0
    gamma = max(float(gamma), 1e-6)
    rows, cols = cost.shape
    dp = np.full((rows + 1, cols + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            vals = np.asarray([dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                soft_min = np.inf
            else:
                min_val = finite.min()
                soft_min = min_val - gamma * np.log(np.exp(-(finite - min_val) / gamma).sum())
            dp[i, j] = cost[i - 1, j - 1] + soft_min
    return float(dp[rows, cols])


def _calculate_medoid_dbi(distance_matrix, labels, *, eps: float = 1e-8):
    distance_matrix = np.asarray(distance_matrix, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return None
    medoid_indices = []
    spreads = []
    valid_labels = []
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        if cluster_indices.size < 1:
            continue
        cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        medoid_local = int(np.argmin(cluster_distances.mean(axis=1)))
        medoid_idx = int(cluster_indices[medoid_local])
        medoid_indices.append(medoid_idx)
        spreads.append(float(distance_matrix[cluster_indices, medoid_idx].mean()))
        valid_labels.append(int(label))
    if len(valid_labels) < 2:
        return None
    spreads = np.asarray(spreads, dtype=np.float32)
    medoid_distances = distance_matrix[np.ix_(medoid_indices, medoid_indices)]
    ratios = np.zeros_like(medoid_distances, dtype=np.float32)
    for i in range(len(valid_labels)):
        for j in range(len(valid_labels)):
            if i == j:
                continue
            ratios[i, j] = (spreads[i] + spreads[j]) / (medoid_distances[i, j] + eps)
    return float(ratios.max(axis=1).mean())


def _fit_ik_maps(points, *, anchor_seed: int, ensemble_size: int, subsample_size: int | None):
    points = np.asarray(points, dtype=np.float32)
    effective_subsample = int(subsample_size or min(1024, max(1, points.shape[0])))
    tensor_points = torch.from_numpy(points).float()
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(anchor_seed))
        kernel = SoftIsolationKernel(
            points.shape[1],
            ensemble_size=int(ensemble_size),
            subsample_size=effective_subsample,
            device="cpu",
        ).to("cpu")
        with torch.no_grad():
            kernel.fit(tensor_points)
            maps = kernel(tensor_points)
    finally:
        torch.random.set_rng_state(rng_state)
    return kernel, maps


def _calculate_ik_entropy(maps, *, ensemble_size: int):
    with torch.no_grad():
        phi_hat = maps.mean(dim=0)
        p_s = torch.sum(maps * phi_hat, dim=1) / float(ensemble_size)
        entropy = -torch.mean(torch.log(p_s + 1e-9))
    return float(entropy.item())


def _trajectory_ik_mean_maps(kernel, sequences, *, ensemble_size: int):
    del ensemble_size
    mean_maps = []
    with torch.no_grad():
        for sequence in sequences:
            tensor = torch.from_numpy(np.asarray(sequence, dtype=np.float32)).float()
            maps = kernel(tensor)
            mean_maps.append(maps.mean(dim=0))
    return torch.stack(mean_maps, dim=0)


def _calculate_ik_legacy_dbi(mappings, labels, *, ensemble_size: int):
    labels = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)
    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2:
        return None
    centroids = torch.stack([mappings[labels == label].mean(0) for label in unique_labels])
    distances = 1.0 - torch.matmul(centroids, centroids.T) / float(ensemble_size)
    spreads = torch.diag(distances)
    ratios = (spreads.view(-1, 1) + spreads.view(1, -1)) / torch.clamp(distances, min=1e-8)
    ratios.fill_diagonal_(0)
    mean_ratios = torch.sum(ratios, dim=1) / float(len(unique_labels) - 1)
    return float(torch.mean(mean_ratios).item())


def _has_degenerate_repeats(sequences, valid_groups):
    for indices in valid_groups.values():
        if len(indices) < 2:
            continue
        first = np.asarray(sequences[indices[0]], dtype=np.float32)
        all_same = True
        for index in indices[1:]:
            other = np.asarray(sequences[index], dtype=np.float32)
            if first.shape != other.shape or not np.allclose(first, other, atol=1e-8, rtol=1e-8):
                all_same = False
                break
        if all_same:
            return True
    return False


def _valid_group_indices_and_labels(grouping: GroupingResult):
    indices = []
    labels = []
    for new_label, original_label in enumerate(sorted(grouping.valid_groups)):
        for index in grouping.valid_groups[original_label]:
            indices.append(int(index))
            labels.append(int(new_label))
    return np.asarray(indices, dtype=np.int64), np.asarray(labels, dtype=np.int64)
