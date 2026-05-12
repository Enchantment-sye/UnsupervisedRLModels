from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101

from models.layers import CNN

from .utils import image_to_bchw, last_rgb_frame_bchw, random_shift


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def effective_rank(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if z.dim() != 2:
        z = z.reshape(z.shape[0], -1)
    if z.shape[0] <= 1:
        return z.new_tensor(1.0)
    values = torch.linalg.svdvals(z.float())
    denom = values.sum().clamp_min(eps)
    probs = values / denom
    entropy = -(probs * torch.log(probs + eps)).sum()
    return torch.exp(entropy)


class ResNet101Teacher(nn.Module):
    """Frozen local ResNet-101 feature teacher.

    The loader never downloads weights. It first tries PyTorch checkpoint files
    and then supports the local HuggingFace ResNet safetensors layout used by
    `/home/shangyy/models/resnet-101/model.safetensors`.
    """

    def __init__(self, model_dir: str, device="cpu", pixel_shape=None):
        super().__init__()
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        self.pixel_shape = tuple(pixel_shape) if pixel_shape is not None else None
        self.model = resnet101(weights=None)
        self._load_local_weights()
        self.model.fc = nn.Identity()
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.register_buffer(
            "_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.to(device)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _candidate_files(self):
        root = Path(self.model_dir)
        if not root.exists():
            return []
        suffixes = {".pt", ".pth", ".ckpt", ".safetensors"}
        files = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lowered = path.name.lower()
            if path.suffix.lower() in suffixes or "state_dict" in lowered:
                files.append(path)
        files.sort(key=lambda p: (p.suffix.lower() != ".safetensors", str(p)))
        return files

    def _load_local_weights(self) -> None:
        candidates = self._candidate_files()
        errors = []
        for path in candidates:
            try:
                state = self._load_state_file(path)
                self._load_state_dict_flex(state)
                return
            except Exception as exc:  # noqa: BLE001 - report every local candidate failure.
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
        message = [
            f"Failed to load ResNet-101 teacher weights from {self.model_dir}.",
            "Expected local .pt/.pth/.ckpt/state_dict or .safetensors files.",
            f"Candidates: {[str(p) for p in candidates]}",
        ]
        if errors:
            message.append("Errors:")
            message.extend(errors)
        raise RuntimeError("\n".join(message))

    def _load_state_file(self, path: Path) -> Dict[str, torch.Tensor]:
        if path.suffix.lower() == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "safetensors is required to load model.safetensors. "
                    "Use /home/shangyy/miniconda3/envs/metra_idk/bin/python or install safetensors."
                ) from exc
            state = load_file(str(path), device="cpu")
            if any(key.startswith("resnet.") for key in state):
                return self._map_hf_resnet_to_torchvision(state)
            return dict(state)

        checkpoint = torch.load(str(path), map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model", "teacher", "module"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    checkpoint = value
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint)}")
        return {
            self._strip_prefix(str(key)): value
            for key, value in checkpoint.items()
            if torch.is_tensor(value)
        }

    @staticmethod
    def _strip_prefix(key: str) -> str:
        for prefix in ("module.", "model.", "backbone."):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    @staticmethod
    def _map_hf_resnet_to_torchvision(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mapped: Dict[str, torch.Tensor] = {
            "conv1.weight": state["resnet.embedder.embedder.convolution.weight"],
        }
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            mapped[f"bn1.{suffix}"] = state[f"resnet.embedder.embedder.normalization.{suffix}"]

        depths = (3, 4, 23, 3)
        for stage_idx, depth in enumerate(depths):
            for layer_idx in range(depth):
                out_base = f"layer{stage_idx + 1}.{layer_idx}"
                for conv_idx in range(3):
                    hf_base = f"resnet.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{conv_idx}"
                    mapped[f"{out_base}.conv{conv_idx + 1}.weight"] = state[f"{hf_base}.convolution.weight"]
                    for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                        mapped[f"{out_base}.bn{conv_idx + 1}.{suffix}"] = state[
                            f"{hf_base}.normalization.{suffix}"
                        ]

                shortcut = f"resnet.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut"
                if f"{shortcut}.convolution.weight" in state:
                    mapped[f"{out_base}.downsample.0.weight"] = state[f"{shortcut}.convolution.weight"]
                    for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                        mapped[f"{out_base}.downsample.1.{suffix}"] = state[f"{shortcut}.normalization.{suffix}"]

        if "classifier.1.weight" in state:
            mapped["fc.weight"] = state["classifier.1.weight"]
            mapped["fc.bias"] = state["classifier.1.bias"]
        return mapped

    def _load_state_dict_flex(self, state: Dict[str, torch.Tensor]) -> None:
        target = self.model.state_dict()
        if not set(state).issubset(set(target)):
            stripped = {self._strip_prefix(key): value for key, value in state.items()}
            state = stripped
        shape_bad = [
            key
            for key, value in state.items()
            if key in target and tuple(value.shape) != tuple(target[key].shape)
        ]
        if shape_bad:
            raise RuntimeError(f"Teacher checkpoint has incompatible tensor shapes for keys: {shape_bad[:10]}")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        allowed_missing = {"fc.weight", "fc.bias"}
        bad_missing = sorted(set(missing) - allowed_missing)
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Teacher checkpoint did not match torchvision ResNet-101. "
                f"missing={bad_missing[:10]}, unexpected={list(unexpected)[:10]}"
            )

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = last_rgb_frame_bchw(obs, pixel_shape=self.pixel_shape, device=self.device)
        if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            x = x.float().div_(255.0)
        else:
            x = x.float()
            if x.numel() and float(x.detach().max()) > 1.5:
                x = x.div(255.0)
        x = (x - self._mean) / self._std
        return self.model(x).detach()


class CoverageEncoder(nn.Module):
    def __init__(
        self,
        *,
        pixel_shape,
        action_dim: int,
        latent_dim: int = 32,
        spectral_normalization: bool = False,
    ):
        super().__init__()
        self.pixel_shape = tuple(pixel_shape)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.pixel_depth = int(self.pixel_shape[-1])

        self.encoder = CNN(self.pixel_depth, spectral_normalization=spectral_normalization)
        with torch.no_grad():
            dummy = torch.zeros(1, self.pixel_depth, self.pixel_shape[0], self.pixel_shape[1])
            conv_dim = int(self.encoder(dummy).shape[-1])

        self.projector = nn.Linear(conv_dim, self.latent_dim)
        self.layer_norm = nn.LayerNorm(self.latent_dim)
        self.distill_head = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.action_dim),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = image_to_bchw(obs, pixel_shape=self.pixel_shape, device=self.device)
        if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            x = x.float().div_(255.0)
        else:
            x = x.float()
            if x.numel() and float(x.detach().max()) > 1.5:
                x = x.div(255.0)
        h = self.encoder(x).reshape(x.shape[0], -1)
        z = self.projector(h)
        z = self.layer_norm(z)
        return F.normalize(z, p=2, dim=-1)

    def freeze(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def compute_cov_loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        teacher: Optional[nn.Module],
        lambda_dist: float = 1.0,
        lambda_aug: float = 1.0,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        lambda_inv: float = 0.1,
        var_gamma: float = 1.0,
        aug_pad: int = 4,
    ) -> Dict[str, torch.Tensor]:
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"].reshape(batch["actions"].shape[0], -1).to(self.device).float()

        z = self(obs)
        z_next = self(next_obs)

        if "teacher_features" in batch:
            teacher_feat = batch["teacher_features"].detach().to(device=z.device, dtype=z.dtype)
        elif teacher is None:
            teacher_feat = torch.zeros(z.shape[0], 2048, device=z.device)
        else:
            with torch.no_grad():
                teacher_feat = teacher(obs).detach().to(device=z.device, dtype=z.dtype)
        pred_teacher = self.distill_head(z)
        loss_dist = F.mse_loss(pred_teacher, teacher_feat)

        obs_bchw = image_to_bchw(obs, pixel_shape=self.pixel_shape, device=self.device)
        z1 = self(random_shift(obs_bchw, pad=aug_pad))
        z2 = self(random_shift(obs_bchw, pad=aug_pad))
        loss_aug = F.mse_loss(z1, z2)

        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        loss_var = torch.relu(z.new_tensor(float(var_gamma)) - std).mean()

        if z.shape[0] <= 1:
            loss_cov = z.new_zeros(())
        else:
            centered = z - z.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / (z.shape[0] - 1)
            offdiag = cov - torch.diag(torch.diagonal(cov))
            loss_cov = offdiag.square().sum() / z.shape[1]

        inv_pred = self.inverse_head(torch.cat([z, z_next], dim=-1))
        loss_inv = F.mse_loss(inv_pred, actions)

        loss_total = (
            float(lambda_dist) * loss_dist
            + float(lambda_aug) * loss_aug
            + float(lambda_var) * loss_var
            + float(lambda_cov) * loss_cov
            + float(lambda_inv) * loss_inv
        )
        return {
            "loss_total": loss_total,
            "loss_dist": loss_dist,
            "loss_aug": loss_aug,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
            "loss_inv": loss_inv,
            "embedding_std_mean": std.mean().detach(),
            "effective_rank": effective_rank(z).detach(),
        }


class DirectCoverageEncoder(nn.Module):
    """Frozen local visual backbone used directly as coverage encoder.

    This path is intentionally training-free: ResNet-101 or DINOv3 features are
    computed under no-grad, flattened, and L2-normalized before entering MASS.
    """

    def __init__(
        self,
        *,
        encoder_type: str,
        model_dir: str,
        pixel_shape,
        action_dim: int = 0,
        device="cpu",
    ):
        super().__init__()
        self.encoder_type = "dinov3" if str(encoder_type) == "dino-v3" else str(encoder_type)
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        self.pixel_shape = tuple(pixel_shape)
        self.action_dim = int(action_dim)
        self._device = torch.device(device)

        if self.encoder_type == "resnet-101":
            from models.encoders import ResNetEncoder

            self.backbone = ResNetEncoder(
                model_dir=self.model_dir,
                device=self._device,
                pixel_shape=self.pixel_shape,
                finetune=False,
            )
        elif self.encoder_type == "dinov3":
            from models.encoders import DINOEncoder

            self.backbone = DINOEncoder(
                model_dir=self.model_dir,
                device=self._device,
                pixel_shape=self.pixel_shape,
                finetune=False,
            )
        else:
            raise ValueError(f"Unsupported direct coverage encoder type: {encoder_type!r}")

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.latent_dim = self._infer_latent_dim()

    @property
    def device(self):
        return self._device

    def _infer_latent_dim(self) -> int:
        feature_dim = int(getattr(self.backbone, "feature_dim", 0) or 0)
        if feature_dim > 0:
            return feature_dim
        with torch.no_grad():
            dummy = torch.zeros(1, int(torch.tensor(self.pixel_shape).prod().item()), device=self.device)
            z = self.forward(dummy)
        return int(z.shape[-1])

    def freeze(self) -> None:
        self.eval()
        self.backbone.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, device=self.device)
        else:
            obs = obs.to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        feat = self.backbone(obs)
        feat = feat.reshape(feat.shape[0], -1).float()
        return F.normalize(feat, p=2, dim=-1)


def _as_tuple_or_none(value: Any) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    return tuple(int(v) for v in value)


def _checkpoint_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "coverage_encoder", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise TypeError("Coverage encoder checkpoint must contain a state_dict or coverage_encoder dict")


def _require_metadata(name: str, actual: Any, expected: Any, path: str) -> None:
    if actual is None:
        raise RuntimeError(f"Coverage encoder checkpoint {path} is missing required metadata: {name}")
    if actual != expected:
        raise RuntimeError(
            f"Coverage encoder checkpoint {path} has incompatible {name}: "
            f"checkpoint={actual}, expected={expected}"
        )


def save_coverage_encoder_checkpoint(
    path: str,
    encoder: CoverageEncoder,
    *,
    pixel_shape,
    action_dim: int,
    latent_dim: int,
    task: Optional[str] = None,
    teacher_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    global_step: int = 0,
    distill_steps: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "state_dict": encoder.state_dict(),
        "pixel_shape": tuple(pixel_shape),
        "action_dim": int(action_dim),
        "latent_dim": int(latent_dim),
        "task": task,
        "teacher_path": teacher_path,
        "config": config or {},
        "global_step": int(global_step),
        "distill_steps": int(distill_steps),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_coverage_encoder_checkpoint(
    path: str,
    *,
    pixel_shape,
    action_dim: int,
    latent_dim: int,
    device="cpu",
    strict_metadata: bool = True,
    freeze: bool = True,
) -> tuple[CoverageEncoder, Dict[str, Any]]:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Coverage encoder checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)

    expected_pixel_shape = tuple(pixel_shape)
    expected_action_dim = int(action_dim)
    expected_latent_dim = int(latent_dim)
    if isinstance(checkpoint, dict) and strict_metadata:
        _require_metadata("pixel_shape", _as_tuple_or_none(checkpoint.get("pixel_shape")), expected_pixel_shape, path)
        _require_metadata("action_dim", checkpoint.get("action_dim"), expected_action_dim, path)
        _require_metadata("latent_dim", checkpoint.get("latent_dim"), expected_latent_dim, path)

    encoder = CoverageEncoder(
        pixel_shape=expected_pixel_shape,
        action_dim=expected_action_dim,
        latent_dim=expected_latent_dim,
    ).to(device)
    encoder.load_state_dict(state_dict, strict=True)
    if freeze:
        encoder.freeze()
    else:
        encoder.eval()
    return encoder, checkpoint if isinstance(checkpoint, dict) else {"state_dict": state_dict}
