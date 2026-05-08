from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class GroundTruthBundle:
    reach_prob: np.ndarray
    temporal_reachability: np.ndarray
    oracle_temporal: np.ndarray
    occurrence_counts: np.ndarray


def filter_occurrences_with_future(
    occurrence_indices: np.ndarray,
    timesteps: np.ndarray,
    episode_lengths: np.ndarray,
    episode_ids: np.ndarray,
) -> np.ndarray:
    if occurrence_indices.size == 0:
        return occurrence_indices
    keep = timesteps[occurrence_indices] < (episode_lengths[episode_ids[occurrence_indices]] - 1)
    return occurrence_indices[keep]


def compute_ground_truth_for_anchor(
    occurrence_indices: np.ndarray,
    candidate_tree: cKDTree,
    positions: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
    episode_offsets: np.ndarray,
    episode_lengths: np.ndarray,
    horizon: int,
    match_radius: float,
    num_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    valid_occurrences = filter_occurrences_with_future(
        occurrence_indices=occurrence_indices,
        timesteps=timesteps,
        episode_lengths=episode_lengths,
        episode_ids=episode_ids,
    )

    if valid_occurrences.size == 0:
        zeros = np.zeros(num_candidates, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy(), 0

    hit_counts = np.zeros(num_candidates, dtype=np.float64)
    temporal_reachability = np.zeros(num_candidates, dtype=np.float64)
    oracle_temporal = np.zeros(num_candidates, dtype=np.float64)

    for global_index in valid_occurrences:
        episode_id = int(episode_ids[global_index])
        timestep = int(timesteps[global_index])
        episode_start = int(episode_offsets[episode_id])
        remaining_steps = int(episode_lengths[episode_id] - timestep - 1)
        max_tau = min(horizon, remaining_steps)

        earliest_hit = np.full(num_candidates, np.inf, dtype=np.float64)
        for tau in range(1, max_tau + 1):
            future_global_index = episode_start + timestep + tau
            future_position = positions[future_global_index]
            hit_candidate_ids = candidate_tree.query_ball_point(future_position, r=match_radius)
            for candidate_id in hit_candidate_ids:
                if earliest_hit[candidate_id] == np.inf:
                    earliest_hit[candidate_id] = float(tau)

        reached_mask = np.isfinite(earliest_hit)
        hit_counts[reached_mask] += 1.0
        temporal_reachability[reached_mask] += 1.0 / earliest_hit[reached_mask]
        oracle_temporal[reached_mask] += 1.0 / (1.0 + earliest_hit[reached_mask])

    denominator = float(valid_occurrences.size)
    reach_prob = (hit_counts / denominator).astype(np.float32)
    temporal_reachability = (temporal_reachability / denominator).astype(np.float32)
    oracle_temporal = (oracle_temporal / denominator).astype(np.float32)
    return reach_prob, temporal_reachability, oracle_temporal, int(valid_occurrences.size)


def compute_ground_truth_bundle(
    anchor_occurrence_lists: list[np.ndarray],
    candidate_positions: np.ndarray,
    positions: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
    episode_offsets: np.ndarray,
    episode_lengths: np.ndarray,
    horizon: int,
    match_radius: float,
) -> GroundTruthBundle:
    candidate_tree = cKDTree(candidate_positions)
    num_candidates = int(candidate_positions.shape[0])

    reach_rows = []
    temporal_rows = []
    oracle_rows = []
    occurrence_counts = []

    for occurrence_indices in anchor_occurrence_lists:
        reach_prob, temporal_reachability, oracle_temporal, used_occurrences = compute_ground_truth_for_anchor(
            occurrence_indices=occurrence_indices,
            candidate_tree=candidate_tree,
            positions=positions,
            episode_ids=episode_ids,
            timesteps=timesteps,
            episode_offsets=episode_offsets,
            episode_lengths=episode_lengths,
            horizon=horizon,
            match_radius=match_radius,
            num_candidates=num_candidates,
        )
        reach_rows.append(reach_prob)
        temporal_rows.append(temporal_reachability)
        oracle_rows.append(oracle_temporal)
        occurrence_counts.append(float(used_occurrences))

    return GroundTruthBundle(
        reach_prob=np.stack(reach_rows, axis=0),
        temporal_reachability=np.stack(temporal_rows, axis=0),
        oracle_temporal=np.stack(oracle_rows, axis=0),
        occurrence_counts=np.asarray(occurrence_counts, dtype=np.float32),
    )
