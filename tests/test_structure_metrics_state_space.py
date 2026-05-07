import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from core.metrics.trajectory_structure import (
    SkipReason,
    StateSpaceCode,
    extract_structure_state_sequences,
)


def test_ant_uses_xy_coordinates():
    # Selector coverage only; the current training-hook smoke uses the DMC path.
    coords = np.asarray([[1.0, 2.0], [1.5, 2.5]], dtype=np.float32)
    result = extract_structure_state_sequences(
        [
            {
                "env_infos": {
                    "coordinates": coords,
                    "next_coordinates": coords + 0.25,
                },
                "observations": np.ones((2, 4), dtype=np.float32),
            }
        ],
        "ant",
    )

    assert result.skipped is False
    assert result.state_space_code == int(StateSpaceCode.XY)
    assert result.state_tag_suffix == "XY"
    assert result.traj_state_sequences[0].shape[1] == 2


def test_dmc_quadruped_run_forward_color_uses_xy_coordinates():
    coords = np.asarray([[0.0, 0.0], [0.5, 0.2]], dtype=np.float32)
    result = extract_structure_state_sequences(
        [{"env_infos": {"coordinates": coords, "next_coordinates": coords + 0.1}}],
        "dmc_quadruped_run_forward_color",
    )

    assert result.state_space_code == int(StateSpaceCode.XY)
    assert result.state_tag_suffix == "XY"


def test_dmc_humanoid_run_color_uses_xy_coordinates():
    coords = np.asarray([[2.0, -1.0], [2.5, -0.5]], dtype=np.float32)
    result = extract_structure_state_sequences(
        [{"env_infos": {"coordinates": coords, "next_coordinates": coords + 0.2}}],
        "dmc_humanoid_run_color",
    )

    assert result.skipped is False
    assert result.state_space_code == int(StateSpaceCode.XY)
    assert result.state_tag_suffix == "XY"


def test_other_env_uses_normalized_raw_state_from_ori_obs():
    result = extract_structure_state_sequences(
        [
            {
                "env_infos": {
                    "ori_obs": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    "next_ori_obs": np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                }
            }
        ],
        "metaworld_door",
    )

    assert result.skipped is False
    assert result.state_space_code == int(StateSpaceCode.NORMALIZED_RAW_STATE)
    assert result.state_tag_suffix == "NormalizedRawState"
    assert np.allclose(np.concatenate(result.traj_state_sequences).mean(axis=0), 0.0, atol=1e-6)


def test_kitchen_uses_normalized_raw_state():
    result = extract_structure_state_sequences(
        [
            {
                "env_infos": {
                    "ori_obs": np.asarray([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32),
                    "next_ori_obs": np.asarray([[2.0, 4.0, 6.0], [3.0, 5.0, 7.0]], dtype=np.float32),
                }
            }
        ],
        "kitchen",
    )

    assert result.skipped is False
    assert result.state_space_code == int(StateSpaceCode.NORMALIZED_RAW_STATE)
    assert result.state_tag_suffix == "NormalizedRawState"


def test_other_env_falls_back_to_observations():
    result = extract_structure_state_sequences(
        [{"observations": np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)}],
        "custom_state_env",
    )

    assert result.skipped is False
    assert result.state_space_code == int(StateSpaceCode.NORMALIZED_RAW_STATE)


def test_pixel_only_raw_state_is_skipped():
    result = extract_structure_state_sequences(
        [{"observations": np.zeros((2, 64 * 64 * 3), dtype=np.float32)}],
        "custom_pixel_env",
    )

    assert result.skipped is True
    assert result.state_space_code == int(StateSpaceCode.UNSUPPORTED)
    assert result.skip_reason_code == int(SkipReason.MISSING_STATE_POINTS_OR_COORDINATES)
