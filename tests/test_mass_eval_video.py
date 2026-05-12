import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.mass.trainer import make_video_grid, obs_to_video_frame


def test_obs_to_video_frame_uses_last_rgb_frame_from_flat_stack():
    obs = np.zeros((64, 64, 9), dtype=np.uint8)
    obs[..., :3] = 10
    obs[..., 3:6] = 20
    obs[..., 6:9] = [30, 40, 50]

    frame = obs_to_video_frame(obs.reshape(-1), (64, 64, 9))

    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame[0, 0] == np.array([30, 40, 50], dtype=np.uint8))


def test_make_video_grid_builds_3x3_and_pads_short_rollouts():
    trajectories = []
    for idx in range(9):
        frame = np.full((4, 5, 3), idx, dtype=np.uint8)
        trajectories.append([frame, frame + 1])

    grid = make_video_grid(trajectories, rows=3, cols=3, skip_frames=1)

    assert len(grid) == 2
    assert grid[0].shape == (12, 15, 3)
    assert np.all(grid[0][:4, :5] == 0)
    assert np.all(grid[0][8:12, 10:15] == 8)
    assert np.all(grid[1][:4, :5] == 1)
