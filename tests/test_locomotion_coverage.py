import numpy as np

from envs.locomotion_coverage import compute_locomotion_coverage_metrics


def _trajectory(coordinates, next_coordinates=None):
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if next_coordinates is None:
        next_coordinates = coordinates.copy()
    return {
        "env_infos": {
            "coordinates": coordinates,
            "next_coordinates": np.asarray(next_coordinates, dtype=np.float32),
        },
    }


def test_xy_coverage_counts_three_bins():
    trajectory = _trajectory(
        coordinates=[[0.2, 0.2], [1.2, 0.2]],
        next_coordinates=[[1.2, 0.2], [2.2, 0.2]],
    )

    metrics = compute_locomotion_coverage_metrics([trajectory], coord_dims=2)

    assert metrics["PolicyStateCoverageXYBins"] == 3
    assert metrics["MjNumUniqueCoords"] == 3


def test_ant_final_displacement_uses_xy():
    trajectory = _trajectory(
        coordinates=[[0.0, 0.0], [1.0, 0.0]],
        next_coordinates=[[1.0, 0.0], [3.0, 4.0]],
    )

    metrics = compute_locomotion_coverage_metrics([trajectory], coord_dims=2)

    assert metrics["PolicyFinalXYDispMean"] == 5.0
    assert metrics["PolicyFinalXYDispMax"] == 5.0


def test_cheetah_coverage_uses_x_only():
    trajectory = _trajectory(
        coordinates=[[0.2, 0.2], [0.2, 5.2]],
        next_coordinates=[[0.2, 5.2], [1.2, 9.2]],
    )

    metrics = compute_locomotion_coverage_metrics([trajectory], coord_dims=1)

    assert metrics["PolicyStateCoverageXYBins"] == 2
    assert np.isclose(metrics["PolicyXRange"], 1.0)
    assert metrics["PolicyYRange"] == 0.0


def test_last_next_coordinate_is_counted():
    trajectory = _trajectory(
        coordinates=[[0.2, 0.2], [0.4, 0.4]],
        next_coordinates=[[0.4, 0.4], [4.2, 4.2]],
    )

    metrics = compute_locomotion_coverage_metrics([trajectory], coord_dims=2)

    assert metrics["MjNumCoords"] == 3
    assert metrics["PolicyStateCoverageXYBins"] == 2
