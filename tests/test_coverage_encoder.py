import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.cov_encoder.coverage_encoder import (
    CoverageEncoder,
    DirectCoverageEncoder,
    ResNet101Teacher,
    load_coverage_encoder_checkpoint,
    save_coverage_encoder_checkpoint,
)
from models import encoders as model_encoders


class _FakeTeacher(nn.Module):
    def forward(self, obs):
        return torch.ones(obs.shape[0], 2048, device=obs.device)


class _FakeProcessor:
    image_mean = [0.5, 0.4, 0.3]
    image_std = [0.2, 0.2, 0.2]
    size = {}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class _TinyHFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, pixel_values):
        return SimpleNamespace(pooler_output=pixel_values.mean(dim=(2, 3)))


class _TinyHFEncoder(model_encoders.BaseHuggingFaceEncoder):
    def _load_model(self, model_dir, use_safetensors):
        return _TinyHFModel()

    def _get_feature_dim(self):
        return 3


def test_coverage_encoder_shape_norm_and_losses_backward():
    torch.manual_seed(0)
    encoder = CoverageEncoder(pixel_shape=(64, 64, 9), action_dim=3, latent_dim=32)
    obs = torch.randint(0, 256, (4, 64 * 64 * 9), dtype=torch.uint8).float()
    next_obs = torch.randint(0, 256, (4, 64 * 64 * 9), dtype=torch.uint8).float()
    actions = torch.randn(4, 3)

    z = encoder(obs)
    assert z.shape == (4, 32)
    assert torch.isfinite(z).all()
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)

    losses = encoder.compute_cov_loss(
        {"obs": obs, "next_obs": next_obs, "actions": actions},
        teacher=_FakeTeacher(),
    )
    for key in ("loss_total", "loss_dist", "loss_aug", "loss_var", "loss_cov", "loss_inv"):
        assert torch.isfinite(losses[key])
    losses["loss_total"].backward()
    assert any(param.grad is not None for param in encoder.parameters())


def test_huggingface_encoder_normalization_buffers_follow_device(monkeypatch):
    monkeypatch.setattr(model_encoders, "AutoImageProcessor", _FakeProcessor)
    encoder = _TinyHFEncoder("/tmp/fake-model", device="cpu", pixel_shape=(8, 8, 3), finetune=False)

    assert encoder._img_mean.device == next(encoder.model.parameters()).device
    assert encoder._img_std.device == next(encoder.model.parameters()).device

    obs = torch.randint(0, 256, (2, 8 * 8 * 3), dtype=torch.uint8).float()
    out = encoder(obs)
    assert out.shape == (2, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_huggingface_encoder_cuda_preprocess_buffers_align(monkeypatch):
    monkeypatch.setattr(model_encoders, "AutoImageProcessor", _FakeProcessor)
    encoder = _TinyHFEncoder("/tmp/fake-model", device="cuda", pixel_shape=(8, 8, 3), finetune=False)

    obs = torch.randint(0, 256, (2, 8 * 8 * 3), dtype=torch.uint8, device="cuda").float()
    out = encoder(obs)
    assert out.device.type == "cuda"
    assert out.shape == (2, 3)


def test_teacher_loader_local_resnet_safetensors_or_clear_skip():
    path = "/home/shangyy/models/resnet-101/"
    if not os.path.isdir(path):
        pytest.skip("local ResNet-101 teacher directory is absent")
    try:
        teacher = ResNet101Teacher(path, device="cpu", pixel_shape=(64, 64, 9))
    except ImportError as exc:
        pytest.skip(str(exc))
    obs = torch.randint(0, 256, (1, 64 * 64 * 9), dtype=torch.uint8).float()
    feat = teacher(obs)
    assert feat.shape == (1, 2048)
    assert not feat.requires_grad


def test_coverage_encoder_checkpoint_roundtrip_and_metadata(tmp_path):
    torch.manual_seed(0)
    encoder = CoverageEncoder(pixel_shape=(64, 64, 3), action_dim=2, latent_dim=8)
    path = tmp_path / "coverage_encoder.pt"

    save_coverage_encoder_checkpoint(
        str(path),
        encoder,
        pixel_shape=(64, 64, 3),
        action_dim=2,
        latent_dim=8,
        task="debug_dummy",
        teacher_path="/tmp/teacher",
        config={"distill_train_steps": 1},
        global_step=3,
        distill_steps=1,
    )
    loaded, meta = load_coverage_encoder_checkpoint(
        str(path),
        pixel_shape=(64, 64, 3),
        action_dim=2,
        latent_dim=8,
        device="cpu",
    )

    assert meta["pixel_shape"] == (64, 64, 3)
    assert meta["action_dim"] == 2
    assert meta["latent_dim"] == 8
    assert all(not param.requires_grad for param in loaded.parameters())
    obs = torch.randint(0, 256, (2, 64 * 64 * 3), dtype=torch.uint8).float()
    assert loaded(obs).shape == (2, 8)

    with pytest.raises(RuntimeError, match="incompatible latent_dim"):
        load_coverage_encoder_checkpoint(
            str(path),
            pixel_shape=(64, 64, 3),
            action_dim=2,
            latent_dim=16,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("encoder_type", "path"),
    [
        ("resnet-101", "/home/shangyy/models/resnet-101/"),
        ("dinov3", "/home/shangyy/models/dinov3-vits16-pretrain-lvd1689m/"),
    ],
)
def test_direct_coverage_encoder_local_backbones_or_skip(encoder_type, path):
    if not os.path.isdir(path):
        pytest.skip(f"local {encoder_type} model directory is absent")
    encoder = DirectCoverageEncoder(
        encoder_type=encoder_type,
        model_dir=path,
        pixel_shape=(64, 64, 3),
        action_dim=2,
        device="cpu",
    )
    obs = torch.randint(0, 256, (2, 64 * 64 * 3), dtype=torch.uint8).float()
    z = encoder(obs)
    assert z.shape[0] == 2
    assert z.shape[1] == encoder.latent_dim
    assert torch.isfinite(z).all()
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert all(param.requires_grad is False for param in encoder.parameters())
