import math
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
    return 9 * repeats


def infer_skill_video_grid_cols(discrete, dim_skill, num_video_repeats, n_cols=None):
    expected_videos = infer_expected_video_count(discrete, dim_skill, num_video_repeats)
    if n_cols is not None or int(discrete):
        return utils.infer_video_grid_cols(expected_videos, n_cols=n_cols)
    return 3 * max(1, int(num_video_repeats))




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
    n_cols = infer_skill_video_grid_cols(discrete, dim_skill, num_video_repeats, n_cols=n_cols)
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


VIDEO_PIXEL_SKIP_NONE = 0
VIDEO_PIXEL_SKIP_NO_VIDEOS = 1
VIDEO_PIXEL_SKIP_INSUFFICIENT_VALID_VIDEOS = 2
VIDEO_PIXEL_SKIP_TOO_FEW_CLUSTERS = 3
VIDEO_PIXEL_SKIP_NONFINITE_DISTANCE = 4
VIDEO_PIXEL_SKIP_INSUFFICIENT_MOTION_POINTS = 5


def _cfg_value(cfg, name, default):
    return getattr(cfg, name, default)


def _iter_video_entries(video_list):
    if video_list is None:
        return
    for idx, entry in enumerate(video_list):
        if isinstance(entry, dict):
            frames = entry.get('frames')
            video_id = entry.get('video_id', f'video_{idx:03d}')
        else:
            frames = entry
            video_id = f'video_{idx:03d}'
        yield idx, video_id, frames


def _preprocess_video_for_pixel_metrics(frames, cfg):
    if frames is None or len(frames) == 0:
        return None

    processed = np.asarray([
        preprocess_frame(
            frame,
            resize_h=_cfg_value(cfg, 'resize_h', 128),
            resize_w=_cfg_value(cfg, 'resize_w', 128),
            blur_kernel=_cfg_value(cfg, 'blur_kernel', 3),
        )
        for frame in frames
    ], dtype=np.float32)
    frame_gap = max(1, int(_cfg_value(cfg, 'frame_gap', 1)))
    motion = compute_frame_deltas(processed, frame_gap=frame_gap)
    if motion.shape[0] == 0:
        return None
    return {
        'frames': processed[frame_gap:].astype(np.float32, copy=False),
        'motion': motion.astype(np.float32, copy=False),
    }


def _motion_mse_distance(video_i, video_j, eps):
    len_i = min(video_i['frames'].shape[0], video_i['motion'].shape[0])
    len_j = min(video_j['frames'].shape[0], video_j['motion'].shape[0])
    length = min(len_i, len_j)
    if length <= 0:
        return np.nan

    frames_i = video_i['frames'][:length]
    frames_j = video_j['frames'][:length]
    motion_i = video_i['motion'][:length]
    motion_j = video_j['motion'][:length]
    diff_sq = np.square(frames_i - frames_j, dtype=np.float32)
    weights = motion_i + motion_j
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(weight_sum) or weight_sum <= eps:
        return float(np.mean(diff_sq, dtype=np.float64))
    return float(np.sum(weights * diff_sq, dtype=np.float64) / (weight_sum + eps))


def _pairwise_video_motion_mse(processed_videos, eps):
    num_videos = len(processed_videos)
    distances = np.zeros((num_videos, num_videos), dtype=np.float64)
    for i in range(num_videos):
        for j in range(i + 1, num_videos):
            distance = _motion_mse_distance(processed_videos[i], processed_videos[j], eps)
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


def _finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _compute_medoid_dbi(distance_matrix, labels, eps):
    unique_labels = [label for label in np.unique(labels) if np.sum(labels == label) >= 2]
    if len(unique_labels) < 2:
        return None, VIDEO_PIXEL_SKIP_TOO_FEW_CLUSTERS

    medoids = {}
    spreads = {}
    for label in unique_labels:
        indices = np.flatnonzero(labels == label)
        intra = distance_matrix[np.ix_(indices, indices)]
        if not np.all(np.isfinite(intra)):
            return None, VIDEO_PIXEL_SKIP_NONFINITE_DISTANCE
        medoid_local = int(np.argmin(np.mean(intra, axis=1)))
        medoid = int(indices[medoid_local])
        medoids[label] = medoid
        spreads[label] = float(np.mean(distance_matrix[medoid, indices]))

    ratios = []
    for label_i in unique_labels:
        worst = -np.inf
        for label_j in unique_labels:
            if label_i == label_j:
                continue
            center_distance = float(distance_matrix[medoids[label_i], medoids[label_j]])
            if not np.isfinite(center_distance):
                return None, VIDEO_PIXEL_SKIP_NONFINITE_DISTANCE
            ratio = (spreads[label_i] + spreads[label_j]) / max(center_distance, eps)
            worst = max(worst, ratio)
        if np.isfinite(worst):
            ratios.append(worst)

    if not ratios:
        return None, VIDEO_PIXEL_SKIP_TOO_FEW_CLUSTERS
    return float(np.mean(ratios)), VIDEO_PIXEL_SKIP_NONE


def _compute_same_different_stats(distance_matrix, labels, eps):
    same_distances = []
    nearest_different = []
    triplet_correct = 0
    triplet_total = 0

    for idx in range(distance_matrix.shape[0]):
        same = np.flatnonzero(labels == labels[idx])
        same = same[same != idx]
        different = np.flatnonzero(labels != labels[idx])
        if same.size:
            same_distances.extend(distance_matrix[idx, same].tolist())
        if different.size:
            nearest_different.append(float(np.min(distance_matrix[idx, different])))
        for positive_idx in same:
            positive_distance = float(distance_matrix[idx, positive_idx])
            negative_distances = distance_matrix[idx, different]
            finite_negatives = negative_distances[np.isfinite(negative_distances)]
            if finite_negatives.size == 0 or not np.isfinite(positive_distance):
                continue
            triplet_correct += int(np.sum(positive_distance < finite_negatives))
            triplet_total += int(finite_negatives.size)

    same_values = _finite_values(same_distances)
    different_values = _finite_values(nearest_different)
    metrics = {}
    if same_values.size:
        metrics['VideoPixelSameSkillMotionMSEMedian'] = float(np.median(same_values))
        metrics['VideoPixelSameSkillMotionMSEMax'] = float(np.max(same_values))
    if different_values.size:
        metrics['VideoPixelNearestDifferentMotionMSEMedian'] = float(np.median(different_values))
    if same_values.size and different_values.size:
        same_median = float(np.median(same_values))
        different_median = float(np.median(different_values))
        metrics['VideoPixelSameDifferentRatio_MotionMSE'] = float(
            same_median / max(different_median, eps)
        )
    if triplet_total > 0:
        metrics['VideoPixelTripletAccuracy_MotionMSE'] = float(triplet_correct / triplet_total)
    return metrics


def _pairwise_euclidean_distances(points):
    points = np.asarray(points, dtype=np.float32)
    norms = np.sum(points * points, axis=1, dtype=np.float64)
    gram = (points @ points.T).astype(np.float64)
    distances_sq = norms[:, None] + norms[None, :] - 2.0 * gram
    np.maximum(distances_sq, 0.0, out=distances_sq)
    return np.sqrt(distances_sq, out=distances_sq)


def _compute_motion_knn_entropy(processed_videos, cfg, eps):
    motion_frames = []
    for video in processed_videos:
        motion = np.asarray(video['motion'], dtype=np.float32)
        if motion.size:
            motion_frames.append(motion.reshape(motion.shape[0], -1))
    if not motion_frames:
        return None, VIDEO_PIXEL_SKIP_INSUFFICIENT_MOTION_POINTS

    points = np.concatenate(motion_frames, axis=0)
    finite_mask = np.all(np.isfinite(points), axis=1)
    points = points[finite_mask]
    if points.shape[0] < 2:
        return None, VIDEO_PIXEL_SKIP_INSUFFICIENT_MOTION_POINTS

    max_points = int(_cfg_value(cfg, 'video_pixel_max_points', 2048) or 0)
    if max_points > 0 and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]

    num_points = int(points.shape[0])
    k = max(1, int(_cfg_value(cfg, 'video_pixel_knn_k', 8)))
    k = min(k, num_points - 1)
    distances = _pairwise_euclidean_distances(points)
    np.fill_diagonal(distances, np.inf)
    kth_distances = np.partition(distances, kth=k - 1, axis=1)[:, k - 1]
    kth_distances = kth_distances[np.isfinite(kth_distances)]
    if kth_distances.size == 0:
        return None, VIDEO_PIXEL_SKIP_INSUFFICIENT_MOTION_POINTS

    entropy = (
        math.lgamma(num_points)
        - math.lgamma(k)
        + float(np.mean(np.log(kth_distances + eps)))
    )
    return float(entropy), VIDEO_PIXEL_SKIP_NONE


def compute_video_pixel_motion_metrics(video_list, cfg, num_video_repeats):
    eps = float(_cfg_value(cfg, 'eps', 1e-8))
    repeats = max(1, int(num_video_repeats))
    processed_videos = []
    labels = []
    input_video_count = 0 if video_list is None else len(video_list)

    for original_idx, video_id, frames in _iter_video_entries(video_list):
        processed = _preprocess_video_for_pixel_metrics(frames, cfg)
        if processed is None:
            continue
        processed['video_id'] = video_id
        processed['original_index'] = int(original_idx)
        processed_videos.append(processed)
        labels.append(int(original_idx) // repeats)

    labels = np.asarray(labels, dtype=np.int64)
    num_videos = len(processed_videos)
    num_skills = int(len(np.unique(labels))) if labels.size else 0
    metrics = {
        'VideoPixelMetricsNumVideos': float(num_videos),
        'VideoPixelMetricsNumSkills': float(num_skills),
        'VideoPixelMetricsSkipped': 0.0,
        'VideoPixelMetricsSkipReasonCode': float(VIDEO_PIXEL_SKIP_NONE),
        'VideoPixelDBISkipped': 1.0,
        'VideoPixelDBISkipReasonCode': float(VIDEO_PIXEL_SKIP_TOO_FEW_CLUSTERS),
        'VideoPixelEntropySkipped': 1.0,
        'VideoPixelEntropySkipReasonCode': float(VIDEO_PIXEL_SKIP_INSUFFICIENT_MOTION_POINTS),
    }

    if input_video_count == 0:
        metrics['VideoPixelMetricsSkipped'] = 1.0
        metrics['VideoPixelMetricsSkipReasonCode'] = float(VIDEO_PIXEL_SKIP_NO_VIDEOS)
        return metrics
    if num_videos < 2:
        metrics['VideoPixelMetricsSkipped'] = 1.0
        metrics['VideoPixelMetricsSkipReasonCode'] = float(VIDEO_PIXEL_SKIP_INSUFFICIENT_VALID_VIDEOS)
        return metrics

    distance_matrix = _pairwise_video_motion_mse(processed_videos, eps)
    if not np.all(np.isfinite(distance_matrix)):
        metrics['VideoPixelMetricsSkipped'] = 1.0
        metrics['VideoPixelMetricsSkipReasonCode'] = float(VIDEO_PIXEL_SKIP_NONFINITE_DISTANCE)
        return metrics

    stats = _compute_same_different_stats(distance_matrix, labels, eps)
    metrics.update(stats)

    dbi, dbi_reason = _compute_medoid_dbi(distance_matrix, labels, eps)
    if dbi is not None and np.isfinite(dbi):
        metrics['VideoPixelDBI_MotionMSE'] = float(dbi)
        metrics['VideoPixelDBISkipped'] = 0.0
        metrics['VideoPixelDBISkipReasonCode'] = float(VIDEO_PIXEL_SKIP_NONE)
    else:
        metrics['VideoPixelDBISkipReasonCode'] = float(dbi_reason)

    entropy, entropy_reason = _compute_motion_knn_entropy(processed_videos, cfg, eps)
    if entropy is not None and np.isfinite(entropy):
        metrics['VideoPixelEntropy_MotionKNN'] = float(entropy)
        metrics['VideoPixelEntropySkipped'] = 0.0
        metrics['VideoPixelEntropySkipReasonCode'] = float(VIDEO_PIXEL_SKIP_NONE)
    else:
        metrics['VideoPixelEntropySkipReasonCode'] = float(entropy_reason)

    if (
        metrics.get('VideoPixelDBISkipped', 1.0) >= 1.0
        and metrics.get('VideoPixelEntropySkipped', 1.0) >= 1.0
    ):
        metrics['VideoPixelMetricsSkipped'] = 1.0
        metrics['VideoPixelMetricsSkipReasonCode'] = float(
            max(
                metrics.get('VideoPixelDBISkipReasonCode', VIDEO_PIXEL_SKIP_TOO_FEW_CLUSTERS),
                metrics.get('VideoPixelEntropySkipReasonCode', VIDEO_PIXEL_SKIP_INSUFFICIENT_MOTION_POINTS),
            )
        )

    return metrics


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


def analyze_video_collection(video_list, cfg, num_video_repeats=None):
    video_results = []
    for _, video_id, frames in _iter_video_entries(video_list):
        video_results.append(analyze_video_frames(frames, video_id, cfg))

    if video_results:
        mean_ratio = float(np.mean([item['large_motion_ratio'] for item in video_results]))
    else:
        mean_ratio = 0.0

    result = {
        'video_results': video_results,
        'num_videos': len(video_results),
        'num_valid_videos': sum(1 for item in video_results if item['status'] == 'ok'),
        'mean_large_motion_ratio': mean_ratio,
    }
    if num_video_repeats is not None:
        video_pixel_metrics = compute_video_pixel_motion_metrics(
            video_list,
            cfg,
            num_video_repeats=num_video_repeats,
        )
        result['video_pixel_metrics'] = video_pixel_metrics
        result.update(video_pixel_metrics)
    return result


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
        num_video_repeats=num_video_repeats,
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
    pixel_metrics = result.get('video_pixel_metrics') or {}
    if pixel_metrics:
        dbi = pixel_metrics.get('VideoPixelDBI_MotionMSE', float('nan'))
        entropy = pixel_metrics.get('VideoPixelEntropy_MotionKNN', float('nan'))
        _log(
            logger,
            'info',
            "[MotionAnalysis] video_pixel_dbi=%.6f video_pixel_entropy=%.6f skipped=%d reason=%d",
            float(dbi),
            float(entropy),
            int(pixel_metrics.get('VideoPixelMetricsSkipped', 0.0)),
            int(pixel_metrics.get('VideoPixelMetricsSkipReasonCode', 0.0)),
        )
