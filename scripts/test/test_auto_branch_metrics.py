import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.auto_branch_metrics import compute_knn_distances, compute_m_policy, compute_recent_threshold


def test_knn_mean_distance():
    query = np.array([[0.0, 0.0]], dtype=np.float32)
    ref = np.array([[1.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    result = compute_knn_distances(query, ref, k=2, mode='knn_mean')
    assert result['valid'] is True
    assert result['effective_k'] == 2
    assert np.allclose(result['distances'], np.array([1.5], dtype=np.float32))


def test_knn_kth_distance():
    query = np.array([[0.0, 0.0]], dtype=np.float32)
    ref = np.array([[1.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    result = compute_knn_distances(query, ref, k=2, mode='knn_kth')
    assert result['valid'] is True
    assert np.allclose(result['distances'], np.array([2.0], dtype=np.float32))


def test_recent_threshold_excludes_self():
    recent = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    result = compute_recent_threshold(recent, k=1, mode='knn_mean')
    assert result['valid'] is True
    assert result['effective_k'] == 1
    assert np.isclose(result['threshold'], 1.0)


def test_m_policy_frontier_ratio():
    fresh = np.array([[0.1], [0.2], [10.0]], dtype=np.float32)
    recent = np.array([[0.0], [0.3], [0.6], [0.9], [1.2]], dtype=np.float32)
    result = compute_m_policy(fresh, recent, k=2, mode='knn_mean')
    assert result['valid'] is True
    assert result['effective_k'] == 2
    assert result['frontier_points'].shape[0] == 1
    assert np.isclose(result['m_policy'], 1.0 / 3.0)


def test_small_recent_buffer_degrades_safely():
    fresh = np.array([[0.0], [1.0]], dtype=np.float32)
    recent = np.array([[0.0]], dtype=np.float32)
    result = compute_m_policy(fresh, recent, k=5, mode='knn_mean')
    assert result['valid'] is False
    assert result['reason'] == 'recent_buffer_too_small'


def main():
    test_knn_mean_distance()
    test_knn_kth_distance()
    test_recent_threshold_excludes_self()
    test_m_policy_frontier_ratio()
    test_small_recent_buffer_degrades_safely()
    print("auto branch metrics tests passed")


if __name__ == "__main__":
    main()
