import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import get_parser, make_config_from_args
from envs.galaxea_sim import GalaxeaSimEnv, TRIVIEW_CAMERA_KEYS, TRIVIEW_IMAGE_KEY
from models.encoders import GalaxeaR1LiteTriViewEncoder, WithEncoder


class _CameraEntityStub:
    def __init__(self):
        self.poses = []

    def set_pose(self, pose):
        self.poses.append(pose)


class _CameraStub:
    def __init__(self):
        self.entity = _CameraEntityStub()
        self.fovy = None

    def set_fovy(self, fovy):
        self.fovy = fovy


class _RenderStub:
    def __init__(self):
        self.default_camera = _CameraStub()

    def render(self):
        return np.full((2, 2, 3), 7, dtype=np.uint8)


def _adapter(*, flatten_obs=False):
    env = GalaxeaSimEnv.__new__(GalaxeaSimEnv)
    env._size = (2, 2)
    env.flatten_obs = bool(flatten_obs)
    env.encoder = True
    env._image_key = TRIVIEW_IMAGE_KEY
    env._uses_triview = True
    env._video_view_preset = "default"
    env._env = _RenderStub()
    env._last_image = env._placeholder_image()
    return env


def _obs():
    upper = {}
    for idx, key in enumerate(TRIVIEW_CAMERA_KEYS, start=1):
        upper[key] = np.full((2, 2, 3), idx, dtype=np.uint8)
    return {"upper_body_observations": upper}


def test_galaxea_triview_image_shape_and_channel_order():
    image = _adapter()._extract_image(_obs())

    assert image.shape == (2, 2, 9)
    assert np.all(image[..., 0:3] == 1)
    assert np.all(image[..., 3:6] == 2)
    assert np.all(image[..., 6:9] == 3)


def test_galaxea_triview_flattened_shape():
    image = _adapter(flatten_obs=True)._extract_image(_obs())

    assert image.shape == (2 * 2 * 9,)


def test_galaxea_triview_missing_camera_raises():
    obs = _obs()
    del obs["upper_body_observations"]["rgb_right_hand"]

    with pytest.raises(KeyError, match="rgb_right_hand"):
        _adapter()._extract_image(obs)


def test_galaxea_third_person_video_source_uses_render_camera():
    env = _adapter()
    env._last_image = np.zeros((2, 2, 9), dtype=np.uint8)

    frame = env.capture_video_frame(source="third_person")

    assert frame.shape == (2, 2, 3)
    assert np.all(frame == 7)
    assert env._env.default_camera.entity.poses == []


def test_galaxea_robot_full_body_preset_sets_third_person_camera(monkeypatch):
    env = _adapter()
    env._video_view_preset = "robot_full_body"
    pose = object()
    calls = []
    monkeypatch.setattr(
        env,
        "_make_sapien_look_at_pose",
        lambda position, target: calls.append((np.asarray(position), np.asarray(target))) or pose,
    )

    frame = env.capture_video_frame(source="third_person")

    assert frame.shape == (2, 2, 3)
    assert env._env.default_camera.entity.poses == [pose]
    assert env._env.default_camera.fovy is not None
    assert np.allclose(calls[0][0], [-2.2, -1.8, 1.05])
    assert np.allclose(calls[0][1], [0.0, -0.05, 0.75])


def test_galaxea_observation_video_source_does_not_set_third_person_camera():
    env = _adapter()
    env._video_view_preset = "robot_full_body"

    frame = env.capture_video_frame(source="observation")

    assert frame.shape == (2, 2, 9)
    assert env._env.default_camera.entity.poses == []


def test_galaxea_triview_encoder_fuses_three_views_and_preserves_tail():
    encoder = GalaxeaR1LiteTriViewEncoder(
        pixel_shape=(64, 64, 9),
        fusion_hidden_dim=16,
        fusion_output_dim=32,
    )
    arm_calls = []
    hook = encoder.arm_encoder.register_forward_hook(lambda *_: arm_calls.append(1))
    try:
        obs = torch.randint(0, 255, (2, 64 * 64 * 9 + 5), dtype=torch.float32)
        out = encoder(obs)
    finally:
        hook.remove()

    assert len(arm_calls) == 2
    assert encoder.arm_encoder is not encoder.head_encoder
    assert out.shape == (2, 32 + 5)


def test_galaxea_triview_encoder_works_inside_with_encoder_wrapper():
    encoder = GalaxeaR1LiteTriViewEncoder(
        pixel_shape=(64, 64, 9),
        fusion_hidden_dim=16,
        fusion_output_dim=32,
    )
    wrapped = WithEncoder(encoder=encoder, module=torch.nn.Linear(32 + 4, 3))
    obs = torch.randint(0, 255, (2, 64 * 64 * 9 + 4), dtype=torch.float32)

    assert wrapped(obs).shape == (2, 3)


def test_galaxea_triview_config_auto_selects_multiview_encoder():
    parser = get_parser()
    args = parser.parse_args([
        "--task",
        "galaxea_r1lite_blocks_stack_easy",
        "--encoder",
        "1",
        "--galaxea-sim-image-key",
        TRIVIEW_IMAGE_KEY,
    ])

    cfg = make_config_from_args(args)

    assert cfg.net.encoder_type == "galaxea-r1lite-triview"


def test_galaxea_triview_config_rejects_single_image_hf_encoder():
    parser = get_parser()
    args = parser.parse_args([
        "--task",
        "galaxea_r1lite_blocks_stack_easy",
        "--encoder",
        "1",
        "--encoder_type",
        "resnet-101",
        "--galaxea-sim-image-key",
        TRIVIEW_IMAGE_KEY,
    ])

    with pytest.raises(ValueError, match="rgb_left_right_head"):
        make_config_from_args(args)
