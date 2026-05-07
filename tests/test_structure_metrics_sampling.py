import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from core.metrics.trajectory_structure import build_structure_eval_options


def test_discrete_skill_requests_three_rollouts_per_skill():
    options = build_structure_eval_options(
        discrete=True,
        dim_skill=4,
        unit_length=True,
        rollouts_per_skill=3,
        num_skills=-1,
        max_trajs=12,
        anchor_seed=7,
    )

    assert options.options.shape == (12, 4)
    assert options.skill_ids.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert np.allclose(options.options[0], np.asarray([1, 0, 0, 0], dtype=np.float32))


def test_continuous_2d_anchors_are_fixed_on_unit_circle():
    options = build_structure_eval_options(
        discrete=False,
        dim_skill=2,
        unit_length=True,
        rollouts_per_skill=3,
        num_skills=4,
        max_trajs=12,
        anchor_seed=123,
    )

    anchors = options.anchor_options
    assert np.allclose(np.linalg.norm(anchors, axis=1), 1.0)
    assert np.allclose(anchors[0], np.asarray([1.0, 0.0], dtype=np.float32))
    assert np.allclose(anchors[1], np.asarray([0.0, 1.0], dtype=np.float32), atol=1e-6)


def test_continuous_high_dim_anchors_are_seeded_and_reproducible():
    kwargs = dict(
        discrete=False,
        dim_skill=5,
        unit_length=True,
        rollouts_per_skill=3,
        num_skills=6,
        max_trajs=18,
        anchor_seed=42,
    )

    first = build_structure_eval_options(**kwargs)
    second = build_structure_eval_options(**kwargs)

    assert np.allclose(first.anchor_options, second.anchor_options)
    assert np.allclose(np.linalg.norm(first.anchor_options, axis=1), 1.0)


def test_max_trajs_limits_number_of_skill_anchors():
    options = build_structure_eval_options(
        discrete=True,
        dim_skill=8,
        unit_length=True,
        rollouts_per_skill=3,
        num_skills=-1,
        max_trajs=9,
        anchor_seed=5,
    )

    assert options.options.shape[0] == 9
    assert len(np.unique(options.skill_ids)) == 3
    assert options.subsampled is True
