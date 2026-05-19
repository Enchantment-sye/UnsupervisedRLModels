import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from config.base import MotionAnalysisConfig
from utils import utils, video_motion


def _cfg():
    return MotionAnalysisConfig(
        enabled=0,
        resize_h=16,
        resize_w=16,
        blur_kernel=1,
        frame_gap=1,
        pixel_threshold_mode='adaptive',
        fixed_tau_p=0.04,
        smooth_window=1,
        large_motion_threshold=2.0,
        video_pixel_knn_k=2,
        video_pixel_max_points=128,
        eps=1e-8,
    )


def _moving_square_video(start_y, start_x, *, frames=8, size=24, drift_x=1, drift_y=0):
    video = np.zeros((frames, size, size, 3), dtype=np.uint8)
    for t in range(frames):
        y0 = int(np.clip(start_y + drift_y * t, 0, size - 5))
        x0 = int(np.clip(start_x + drift_x * t, 0, size - 5))
        video[t, y0:y0 + 5, x0:x0 + 5, :] = 255
    return video


def _entries(videos):
    return [{'video_id': f'video_{idx:03d}', 'frames': video} for idx, video in enumerate(videos)]


def test_video_pixel_metrics_separate_moving_skill_clusters():
    cfg = _cfg()
    videos = [
        _moving_square_video(3, 2),
        _moving_square_video(3, 3),
        _moving_square_video(15, 15, drift_x=-1),
        _moving_square_video(14, 15, drift_x=-1),
    ]

    metrics = video_motion.compute_video_pixel_motion_metrics(
        _entries(videos),
        cfg,
        num_video_repeats=2,
    )

    assert metrics['VideoPixelMetricsSkipped'] == 0.0
    assert metrics['VideoPixelDBISkipped'] == 0.0
    assert np.isfinite(metrics['VideoPixelDBI_MotionMSE'])
    assert metrics['VideoPixelDBI_MotionMSE'] < 0.6
    assert metrics['VideoPixelTripletAccuracy_MotionMSE'] == 1.0
    assert np.isfinite(metrics['VideoPixelEntropy_MotionKNN'])
    assert metrics['VideoPixelSameDifferentRatio_MotionMSE'] < 1.0


def test_video_pixel_metrics_outlier_repeat_increases_spread():
    cfg = _cfg()
    good_videos = [
        _moving_square_video(3, 2),
        _moving_square_video(3, 3),
        _moving_square_video(15, 15, drift_x=-1),
        _moving_square_video(14, 15, drift_x=-1),
    ]
    bad_videos = [
        _moving_square_video(3, 2),
        _moving_square_video(15, 15, drift_x=-1),
        _moving_square_video(15, 15, drift_x=-1),
        _moving_square_video(14, 15, drift_x=-1),
    ]

    good = video_motion.compute_video_pixel_motion_metrics(_entries(good_videos), cfg, num_video_repeats=2)
    bad = video_motion.compute_video_pixel_motion_metrics(_entries(bad_videos), cfg, num_video_repeats=2)

    assert bad['VideoPixelSameSkillMotionMSEMax'] > good['VideoPixelSameSkillMotionMSEMax']
    assert bad['VideoPixelDBI_MotionMSE'] > good['VideoPixelDBI_MotionMSE']


def test_video_pixel_metrics_static_videos_fallback_without_nan():
    cfg = _cfg()
    videos = [
        np.zeros((5, 24, 24, 3), dtype=np.uint8),
        np.zeros((5, 24, 24, 3), dtype=np.uint8),
        np.full((5, 24, 24, 3), 64, dtype=np.uint8),
        np.full((5, 24, 24, 3), 64, dtype=np.uint8),
    ]

    metrics = video_motion.compute_video_pixel_motion_metrics(
        _entries(videos),
        cfg,
        num_video_repeats=2,
    )

    assert metrics['VideoPixelMetricsSkipped'] == 0.0
    assert np.isfinite(metrics['VideoPixelDBI_MotionMSE'])
    assert np.isfinite(metrics['VideoPixelEntropy_MotionKNN'])


def test_video_pixel_metrics_repeat_one_skips_dbi_but_keeps_entropy():
    cfg = _cfg()
    videos = [
        _moving_square_video(3, 2),
        _moving_square_video(15, 15, drift_x=-1),
    ]

    metrics = video_motion.compute_video_pixel_motion_metrics(
        _entries(videos),
        cfg,
        num_video_repeats=1,
    )

    assert metrics['VideoPixelMetricsSkipped'] == 0.0
    assert metrics['VideoPixelDBISkipped'] == 1.0
    assert 'VideoPixelDBI_MotionMSE' not in metrics
    assert np.isfinite(metrics['VideoPixelEntropy_MotionKNN'])


def test_video_pixel_metrics_after_montage_split():
    cfg = _cfg()
    videos = np.stack([
        np.transpose(_moving_square_video(3, 2), (0, 3, 1, 2)),
        np.transpose(_moving_square_video(3, 3), (0, 3, 1, 2)),
        np.transpose(_moving_square_video(15, 15, drift_x=-1), (0, 3, 1, 2)),
        np.transpose(_moving_square_video(14, 15, drift_x=-1), (0, 3, 1, 2)),
    ], axis=0)
    montage = utils.prepare_video(videos, n_cols=2)
    split = video_motion.split_montage_frames(
        montage,
        discrete=1,
        dim_skill=2,
        num_video_repeats=2,
        n_cols=2,
    )

    metrics = video_motion.compute_video_pixel_motion_metrics(
        _entries(split),
        cfg,
        num_video_repeats=2,
    )

    assert len(split) == 4
    assert metrics['VideoPixelDBISkipped'] == 0.0
    assert np.isfinite(metrics['VideoPixelDBI_MotionMSE'])


def test_continuous_montage_split_defaults_to_three_skill_groups_per_row():
    tile_size = 12
    videos = []
    for idx in range(18):
        video = np.full((4, tile_size, tile_size, 3), idx, dtype=np.uint8)
        videos.append(np.transpose(video, (0, 3, 1, 2)))
    montage = utils.prepare_video(np.stack(videos, axis=0), n_cols=6)

    split = video_motion.split_montage_frames(
        montage,
        discrete=0,
        dim_skill=4,
        num_video_repeats=2,
    )

    assert video_motion.infer_expected_video_count(0, 4, 2) == 18
    assert video_motion.infer_skill_video_grid_cols(0, 4, 2) == 6
    assert len(split) == 18
    assert split[0].shape[1:3] == (tile_size, tile_size)
    assert np.allclose(split[-1].mean(), 17.0 / 255.0)
