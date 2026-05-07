import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(REPO_ROOT, 'src')
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from config.base import MotionAnalysisConfig
from utils import utils, video_motion


def make_cfg():
    return MotionAnalysisConfig(
        enabled=1,
        resize_h=32,
        resize_w=32,
        blur_kernel=3,
        frame_gap=1,
        pixel_threshold_mode='adaptive',
        fixed_tau_p=0.04,
        smooth_window=3,
        large_motion_threshold=2.0,
        eps=1e-8,
    )


def make_static_video(num_frames=12, size=32):
    return np.zeros((num_frames, size, size, 3), dtype=np.uint8)


def make_flash_video(num_frames=16, size=32):
    frames = np.zeros((num_frames, size, size, 3), dtype=np.uint8)
    frames[num_frames // 2, :, :, :] = 255
    return frames


def test_threshold_and_short_video_are_safe():
    cfg = make_cfg()
    deltas = video_motion.compute_frame_deltas(np.zeros((1, 32, 32), dtype=np.float32), frame_gap=1)
    tau_p = video_motion.compute_motion_pixel_threshold(deltas, mode='adaptive', fixed_tau_p=cfg.fixed_tau_p, eps=cfg.eps)
    assert np.isclose(tau_p, cfg.fixed_tau_p)

    short_result = video_motion.analyze_video_frames(make_static_video(num_frames=1), 'short', cfg)
    assert short_result['status'] == 'insufficient_frames'
    assert np.isclose(short_result['large_motion_ratio'], 0.0)


def test_motion_burst_scores_higher_than_static():
    cfg = make_cfg()
    cfg.blur_kernel = 1
    cfg.smooth_window = 1
    cfg.large_motion_threshold = 1.0

    static_result = video_motion.analyze_video_frames(make_static_video(num_frames=16), 'static', cfg)
    flash_result = video_motion.analyze_video_frames(make_flash_video(), 'flash', cfg)

    assert static_result['status'] == 'ok'
    assert flash_result['status'] == 'ok'
    assert np.isclose(static_result['large_motion_ratio'], 0.0)
    assert flash_result['large_motion_ratio'] > static_result['large_motion_ratio']


def test_split_montage_round_trip():
    num_videos = 18
    videos = np.stack([
        np.full((4, 3, 6, 5), fill_value=float(idx) / 20.0, dtype=np.float32)
        for idx in range(num_videos)
    ], axis=0)
    montage = utils.prepare_video(videos)
    split_videos = video_motion.split_montage_frames(
        montage,
        discrete=1,
        dim_skill=9,
        num_video_repeats=2,
    )

    assert len(split_videos) == num_videos
    for idx, split_video in enumerate(split_videos):
        assert split_video.shape == (4, 6, 5, 3)
        assert np.allclose(split_video[0, 0, 0, 0], videos[idx, 0, 0, 0, 0])


def test_collection_mean_matches_member_average():
    cfg = make_cfg()
    videos = [
        {'video_id': 'static', 'frames': make_static_video()},
        {'video_id': 'flash', 'frames': make_flash_video()},
    ]
    result = video_motion.analyze_video_collection(videos, cfg)
    expected_mean = np.mean([item['large_motion_ratio'] for item in result['video_results']])

    assert result['num_videos'] == 2
    assert np.isclose(result['mean_large_motion_ratio'], expected_mean)


def main():
    test_threshold_and_short_video_are_safe()
    test_motion_burst_scores_higher_than_static()
    test_split_montage_round_trip()
    test_collection_mean_matches_member_average()
    print('video motion tests passed')


if __name__ == '__main__':
    main()
