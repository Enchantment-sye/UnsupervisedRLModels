import os

import cv2
import numpy as np

from utils import utils


def _log(logger, level, message, *args):
    if logger is not None:
        getattr(logger, level)(message, *args)
        return
    if args:
        message = message % args
    print(message)


def infer_expected_video_count(discrete, dim_skill, num_video_repeats):
    repeats = max(1, int(num_video_repeats))
    if int(discrete):
        return max(1, int(dim_skill)) * repeats
    base_count = 9 if int(dim_skill) == 2 else 16
    return base_count * repeats




def format_video_id(index, num_video_repeats):
    repeats = max(1, int(num_video_repeats))
    skill_idx = int(index) // repeats
    repeat_idx = int(index) % repeats
    return f'skill_{skill_idx:03d}_repeat_{repeat_idx:02d}'


def load_video_frames(video_path, logger=None):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        _log(logger, 'warning', "[MotionAnalysis] failed to open video: %s", video_path)
        return []

    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        _log(logger, 'warning', "[MotionAnalysis] empty or undecodable video: %s", video_path)
    return frames


def split_montage_frames(frames, discrete, dim_skill, num_video_repeats, n_cols=None, logger=None):
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[0] == 0:
        return []

    expected_videos = infer_expected_video_count(discrete, dim_skill, num_video_repeats)
    n_cols = utils.infer_video_grid_cols(expected_videos, n_cols=n_cols)
    n_rows = int(np.ceil(expected_videos / float(n_cols)))
    frame_h, frame_w = frames.shape[1:3]
    tile_h = frame_h // n_rows
    tile_w = frame_w // n_cols

    if tile_h == 0 or tile_w == 0:
        _log(
            logger,
            'warning',
            "[MotionAnalysis] montage grid is incompatible with frame shape (%d, %d) for %d videos",
            frame_h,
            frame_w,
            expected_videos,
        )
        return []

    cropped_h = tile_h * n_rows
    cropped_w = tile_w * n_cols
    if cropped_h != frame_h or cropped_w != frame_w:
        _log(
            logger,
            'warning',
            "[MotionAnalysis] cropping montage from (%d, %d) to (%d, %d) for inverse tiling",
            frame_h,
            frame_w,
            cropped_h,
            cropped_w,
        )
        frames = frames[:, :cropped_h, :cropped_w]

    videos = []
    for idx in range(expected_videos):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        y0 = row_idx * tile_h
        y1 = y0 + tile_h
        x0 = col_idx * tile_w
        x1 = x0 + tile_w
        videos.append(frames[:, y0:y1, x0:x1, :])
    return videos


def preprocess_frame(frame, resize_h=128, resize_w=128, blur_kernel=3):
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        gray = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        gray = np.squeeze(frame)
        if gray.ndim != 2:
            raise ValueError(f"Unsupported frame shape for preprocessing: {frame.shape}")

    resize_h = max(1, int(resize_h))
    resize_w = max(1, int(resize_w))
    gray = cv2.resize(gray, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

    blur_kernel = max(1, int(blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray /= 255.0
    return np.clip(gray, 0.0, 1.0)


def compute_frame_deltas(processed_frames, frame_gap=1):
    frames = np.asarray(processed_frames, dtype=np.float32)
    frame_gap = max(1, int(frame_gap))
    if frames.shape[0] <= frame_gap:
        return np.zeros((0,) + frames.shape[1:], dtype=np.float32)
    return np.abs(frames[frame_gap:] - frames[:-frame_gap])


def compute_motion_pixel_threshold(deltas, mode='adaptive', fixed_tau_p=0.04, eps=1e-8):
    fixed_tau_p = float(np.clip(fixed_tau_p, 0.02, 0.08))
    deltas = np.asarray(deltas, dtype=np.float32)

    if mode == 'fixed':
        return fixed_tau_p
    if deltas.size == 0:
        return fixed_tau_p

    delta_values = deltas.reshape(-1)
    median_delta = float(np.median(delta_values))
    mad_delta = float(np.median(np.abs(delta_values - median_delta)))
    if not np.isfinite(median_delta) or not np.isfinite(mad_delta):
        return fixed_tau_p
    if mad_delta <= eps:
        return float(np.clip(max(median_delta, fixed_tau_p), 0.02, 0.08))
    return float(np.clip(median_delta + 3.0 * mad_delta, 0.02, 0.08))


def compute_motion_scores(deltas, tau_p, eps=1e-8):
    deltas = np.asarray(deltas, dtype=np.float32)
    if deltas.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {
            'area_ratio': empty,
            'intensity': empty,
            'score': empty,
        }

    active_mask = deltas > float(tau_p)
    active_count = active_mask.sum(axis=(1, 2)).astype(np.float32)
    total_pixels = float(deltas.shape[1] * deltas.shape[2])
    area_ratio = active_count / max(total_pixels, eps)
    active_energy = (deltas * active_mask).sum(axis=(1, 2))
    intensity = active_energy / (active_count + eps)
    return {
        'area_ratio': area_ratio.astype(np.float32),
        'intensity': intensity.astype(np.float32),
        'score': (area_ratio * intensity).astype(np.float32),
    }


def moving_average(scores, window=5):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores

    window = max(1, int(window))
    if window == 1 or scores.size == 1:
        return scores.copy()

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(scores, (pad_left, pad_right), mode='edge')
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode='valid').astype(np.float32)


def classify_large_motion_frames(smoothed_scores, tau_m=2.0, eps=1e-8):
    smoothed_scores = np.asarray(smoothed_scores, dtype=np.float32)
    if smoothed_scores.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {
            'z_scores': empty,
            'large_motion_mask': np.zeros((0,), dtype=bool),
            'median': 0.0,
            'mad': 0.0,
            'std': 0.0,
            'normalizer': 'empty',
        }

    median_s = float(np.median(smoothed_scores))
    mad_s = float(np.median(np.abs(smoothed_scores - median_s)))
    if mad_s > eps:
        z_scores = (smoothed_scores - median_s) / (mad_s + eps)
        normalizer = 'mad'
    else:
        std_s = float(np.std(smoothed_scores))
        if std_s > eps:
            z_scores = (smoothed_scores - float(np.mean(smoothed_scores))) / (std_s + eps)
            normalizer = 'std'
        else:
            z_scores = np.zeros_like(smoothed_scores)
            normalizer = 'flat'

    return {
        'z_scores': z_scores.astype(np.float32),
        'large_motion_mask': z_scores > float(tau_m),
        'median': median_s,
        'mad': mad_s,
        'std': float(np.std(smoothed_scores)),
        'normalizer': normalizer,
    }


def analyze_video_frames(video_frames, video_id, cfg):
    result = {
        'video_id': video_id,
        'status': 'ok',
        'num_frames': int(len(video_frames)) if video_frames is not None else 0,
        'effective_frame_count': 0,
        'tau_p': 0.0,
        'large_motion_count': 0,
        'large_motion_ratio': 0.0,
    }

    if video_frames is None or len(video_frames) == 0:
        result['status'] = 'decode_failed'
        return result

    processed_frames = np.asarray([
        preprocess_frame(
            frame,
            resize_h=cfg.resize_h,
            resize_w=cfg.resize_w,
            blur_kernel=cfg.blur_kernel,
        )
        for frame in video_frames
    ], dtype=np.float32)

    deltas = compute_frame_deltas(processed_frames, frame_gap=cfg.frame_gap)
    result['effective_frame_count'] = int(deltas.shape[0])
    if result['effective_frame_count'] == 0:
        result['status'] = 'insufficient_frames'
        return result

    tau_p = compute_motion_pixel_threshold(
        deltas,
        mode=cfg.pixel_threshold_mode,
        fixed_tau_p=cfg.fixed_tau_p,
        eps=cfg.eps,
    )
    scores = compute_motion_scores(deltas, tau_p, eps=cfg.eps)
    smoothed_scores = moving_average(scores['score'], window=cfg.smooth_window)
    classification = classify_large_motion_frames(
        smoothed_scores,
        tau_m=cfg.large_motion_threshold,
        eps=cfg.eps,
    )

    large_motion_count = int(classification['large_motion_mask'].sum())
    result.update({
        'tau_p': float(tau_p),
        'large_motion_count': large_motion_count,
        'large_motion_ratio': float(
            large_motion_count / max(1, result['effective_frame_count'])
        ),
        'area_ratios': scores['area_ratio'],
        'intensities': scores['intensity'],
        'scores': scores['score'],
        'smoothed_scores': smoothed_scores,
        'z_scores': classification['z_scores'],
    })
    return result


def analyze_video_collection(video_list, cfg):
    video_results = []
    for idx, entry in enumerate(video_list):
        if isinstance(entry, dict):
            frames = entry.get('frames')
            video_id = entry.get('video_id', f'video_{idx:03d}')
        else:
            frames = entry
            video_id = f'video_{idx:03d}'
        video_results.append(analyze_video_frames(frames, video_id, cfg))

    if video_results:
        mean_ratio = float(np.mean([item['large_motion_ratio'] for item in video_results]))
    else:
        mean_ratio = 0.0

    return {
        'video_results': video_results,
        'num_videos': len(video_results),
        'num_valid_videos': sum(1 for item in video_results if item['status'] == 'ok'),
        'mean_large_motion_ratio': mean_ratio,
    }


def analyze_montage_video(video_path, discrete, dim_skill, num_video_repeats, cfg, logger=None):
    expanded_path = os.path.abspath(os.path.expanduser(video_path))
    frames = load_video_frames(expanded_path, logger=logger)
    split_videos = split_montage_frames(
        frames,
        discrete=discrete,
        dim_skill=dim_skill,
        num_video_repeats=num_video_repeats,
        logger=logger,
    )
    result = analyze_video_collection(
        [
            {'video_id': format_video_id(idx, num_video_repeats), 'frames': video_frames}
            for idx, video_frames in enumerate(split_videos)
        ],
        cfg,
    )
    result.update({
        'source_video_path': expanded_path,
        'expected_video_count': infer_expected_video_count(discrete, dim_skill, num_video_repeats),
    })
    return result


def log_motion_analysis(result, logger=None):
    source_video_path = result.get('source_video_path')
    if source_video_path:
        _log(logger, 'info', "[MotionAnalysis] source_video=%s", source_video_path)

    for video_result in result.get('video_results', []):
        _log(
            logger,
            'info',
            "[MotionAnalysis] video_id=%s status=%s effective_frames=%d tau_p=%.4f r_video=%.6f",
            video_result.get('video_id', 'unknown'),
            video_result.get('status', 'unknown'),
            int(video_result.get('effective_frame_count', 0)),
            float(video_result.get('tau_p', 0.0)),
            float(video_result.get('large_motion_ratio', 0.0)),
        )

    _log(
        logger,
        'info',
        "[MotionAnalysis] num_videos=%d num_valid_videos=%d mean_large_motion_ratio=%.6f",
        int(result.get('num_videos', 0)),
        int(result.get('num_valid_videos', 0)),
        float(result.get('mean_large_motion_ratio', 0.0)),
    )
