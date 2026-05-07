import numpy as np


def _select_coord_dims(coords, coord_dims):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    if isinstance(coord_dims, int):
        return coords[:, :coord_dims]
    return coords[:, list(coord_dims)]


def _trajectory_coordinates(trajectory, coord_dims):
    env_infos = trajectory.get("env_infos", {}) or {}
    coordinates = env_infos.get("coordinates")
    next_coordinates = env_infos.get("next_coordinates")
    if coordinates is None or next_coordinates is None:
        return None

    coordinates = _select_coord_dims(coordinates, coord_dims)
    next_coordinates = _select_coord_dims(next_coordinates, coord_dims)
    if len(coordinates) == 0 or len(next_coordinates) == 0:
        return None
    return np.concatenate([coordinates, next_coordinates[-1:]], axis=0)


def compute_locomotion_coverage_metrics(trajectories, coord_dims, prefix="Policy", bin_size=1.0):
    coord_trajectories = []
    for trajectory in trajectories:
        coords = _trajectory_coordinates(trajectory, coord_dims)
        if coords is not None:
            coord_trajectories.append(coords)

    if not coord_trajectories:
        return {}

    coords = np.concatenate(coord_trajectories, axis=0)
    bins = np.floor(coords / float(bin_size))
    unique_bins = np.unique(bins, axis=0)

    traj_lengths = [len(traj_coords) - 1 for traj_coords in coord_trajectories]
    final_displacements = np.array(
        [np.linalg.norm(traj_coords[-1] - traj_coords[0]) for traj_coords in coord_trajectories],
        dtype=np.float64,
    )
    step_speeds = []
    for traj_coords in coord_trajectories:
        if len(traj_coords) > 1:
            step_speeds.extend(np.linalg.norm(np.diff(traj_coords, axis=0), axis=1))

    y_range = 0.0
    if coords.shape[1] >= 2:
        y_range = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))

    return {
        "MjNumTrajs": len(trajectories),
        "MjAvgTrajLen": float(np.mean(traj_lengths)),
        "MjNumCoords": len(coords),
        "MjNumUniqueCoords": len(unique_bins),
        f"{prefix}StateCoverageXYBins": len(unique_bins),
        f"{prefix}FinalXYDispMean": float(np.mean(final_displacements)),
        f"{prefix}FinalXYDispMax": float(np.max(final_displacements)),
        f"{prefix}XRange": float(np.max(coords[:, 0]) - np.min(coords[:, 0])),
        f"{prefix}YRange": y_range,
        f"{prefix}MeanSpeed": float(np.mean(step_speeds)) if step_speeds else 0.0,
    }
