import torch
import torch.nn as nn
import numpy as np
import abc
try:
    from transformers import AutoImageProcessor, AutoModel, ResNetModel
except ImportError:
    AutoImageProcessor = None
    AutoModel = None
    ResNetModel = None
from core.encoder_factory import EncoderFactory
from .layers import CNN

class BaseHuggingFaceEncoder(torch.nn.Module):
    """
    Base class for HuggingFace model based encoders (DINO, ResNet, etc.)
    Handles common initialization, image preprocessing, and forward pass logic including FrameStack pooling.
    """
    def __init__(self, model_dir, device, pixel_shape, finetune=False, use_safetensors=False, **kwargs):
        super().__init__()
        if AutoImageProcessor is None:
            raise ImportError(
                "transformers is required for HuggingFace encoders. "
                "Install transformers or use encoder_type='original'."
            )
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.pixel_shape = tuple(pixel_shape)
        self.pixel_dim = int(np.prod(self.pixel_shape))

        # HuggingFace processor used only for reading config (mean/std/size), NOT for runtime preprocessing
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)

        # Cache normalization config as torch buffers (so .to(device) works and no python list overhead)
        mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])
        self.register_buffer("_img_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        # Optional resize target (if processor defines it)
        self._resize_h, self._resize_w = None, None
        size = getattr(self.processor, "size", None)
        if isinstance(size, dict):
            if "height" in size and "width" in size:
                self._resize_h, self._resize_w = int(size["height"]), int(size["width"])
            elif "shortest_edge" in size:
                s = int(size["shortest_edge"])
                self._resize_h, self._resize_w = s, s

        self.model = self._load_model(model_dir, use_safetensors).to(self.device)
        # Move this module's non-parameter buffers (normalization mean/std) with
        # the HuggingFace model. Moving only self.model leaves buffers on CPU.
        self.to(self.device)

        # finetune flag must exist (important)
        self.finetune = bool(finetune)

        # Initialize finetune/freeze state deterministically
        self.set_finetune(self.finetune)

        # Feature dim probe (your existing logic)
        self.feature_dim = self._get_feature_dim()
        self.n_frames = self.pixel_shape[-1] // 3 if (self.pixel_shape[-1] % 3 == 0) else 1

    def set_finetune(self, finetune: bool):
        """
        Enable/disable finetuning of the underlying HuggingFace vision backbone.

        Stability-first policy:
          - We ALWAYS keep the HF backbone in eval() to avoid BN/Dropout drift.
          - `finetune` only controls whether gradients/optimizer updates are allowed.
        """
        self.finetune = bool(finetune)

        for p in self.model.parameters():
            p.requires_grad_(self.finetune)

        # Keep HuggingFace model in eval regardless of finetune to avoid stochasticity / BN drift.
        self.model.eval()
        return self

    @abc.abstractmethod
    def _load_model(self, model_dir, use_safetensors):
        """Load and return the specific HuggingFace model."""
        pass

    @abc.abstractmethod
    def _get_feature_dim(self):
        """Return the output feature dimension of the model."""
        pass

    def _ensure_uint8_rgb(self, arr):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] not in (1, 3):
            arr = arr[..., :3]
        return arr

    def _encode_batch_hwcs(self, batch_hwcs):
        from contextlib import nullcontext

        images = [self._ensure_uint8_rgb(img) for img in batch_hwcs]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        # Ensure inputs match model dtype
        model_dtype = next(self.model.parameters()).dtype
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=model_dtype)

        use_amp = (self.device.type == "cuda")
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

        need_grad = bool(getattr(self, "finetune", False))
        if need_grad:
            with amp_ctx:
                out = self.model(**inputs)
        else:
            with torch.inference_mode():
                with amp_ctx:
                    out = self.model(**inputs)

        return out.pooler_output

    def train(self, mode: bool = True):
        """
        Override train() to keep the underlying HuggingFace model in eval() for stability.

        Notes:
          - Autograd/updates are controlled by `self.finetune` (see forward()).
          - Dropout/BN in the HuggingFace backbone are disabled regardless of `mode`.
        """
        super().train(mode)
        self.model.eval()
        return self

    def _to_bchw(self, img: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
          - (H,W,C), (B,H,W,C), (C,H,W), (B,C,H,W), or flattened later reshaped outside
        Returns:
          - (B, C, H, W) float tensor (dtype preserved before normalization)
        """
        import torch

        if img.dim() == 3:
            # HWC or CHW
            if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = img.unsqueeze(0)          # CHW -> BCHW
            else:
                img = img.unsqueeze(0)          # HWC -> BHWC
        if img.dim() != 4:
            raise ValueError(f"Unsupported image tensor shape: {tuple(img.shape)}")

        # BHWC -> BCHW
        if img.shape[-1] in (1, 3) and img.shape[1] not in (1, 3):
            img = img.permute(0, 3, 1, 2).contiguous()

        # If single channel, replicate to 3 (rare, but safe)
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)

        return img

    def _preprocess_pixel_values(self, bchw: torch.Tensor) -> torch.Tensor:
        """
        bchw: (B,3,H,W) uint8/int/float
        Returns:
          pixel_values: (B,3,Ht,Wt) float in model dtype, normalized
        """
        import torch
        import torch.nn.functional as F

        x = bchw

        # cast to float32 first for stable normalize, then to model dtype
        if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            x = x.float().div_(255.0)
        else:
            # float: might be 0..255 or 0..1
            mx = float(x.detach().max()) if x.numel() > 0 else 1.0
            if mx > 1.5:
                x = x.float().div_(255.0)
            else:
                x = x.float()

        # Optional resize to processor size (kept differentiable)
        if self._resize_h is not None and self._resize_w is not None:
            if x.shape[-2] != self._resize_h or x.shape[-1] != self._resize_w:
                x = F.interpolate(x, size=(self._resize_h, self._resize_w), mode="bilinear", align_corners=False)

        # Normalize. Keep buffers defensively aligned with the current input in
        # case a wrapper moved only the HF model or restored buffers on CPU.
        mean = self._img_mean.to(device=x.device, dtype=x.dtype)
        std = self._img_std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / (std + 1e-8)

        # Match model dtype (important for finetune + mixed precision)
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(dtype=model_dtype)

        return x

    def _encode_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B,3,H,W) already normalized
        Returns:
          (B, D) feature
        """
        out = self.model(pixel_values=pixel_values)

        # Most HF vision backbones provide pooler_output; fallback to CLS token
        feat = getattr(out, "pooler_output", None)
        if feat is not None:
            return feat

        last_hidden = getattr(out, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden[:, 0]

        # Final fallback (rare)
        if isinstance(out, (tuple, list)) and len(out) > 0:
            x0 = out[0]
            if x0.dim() == 3:
                return x0[:, 0]
            if x0.dim() == 2:
                return x0

        raise RuntimeError("Cannot extract features from HuggingFace model outputs.")

    def forward(self, x):
        """
        Patched (end-to-end differentiable):
          - full torch path (no numpy/PIL)
          - supports:
              1) flat (B, pixel_dim + tail_dim): encode image part then concat tail
              2) dict: {"image": ..., "state": ...}  (state optional)
              3) tuple/list: (image, state)
              4) image only: returns z
          - framestack: channels=3*k -> encode per-frame then max-pool over frames
        """
        import torch
        import torch.nn.functional as F
        from contextlib import nullcontext

        state = None
        # Unpack dict / tuple
        if isinstance(x, dict):
            state = x.get("state", None)
            x = x.get("image", x.get("pixels", x))
        elif isinstance(x, (tuple, list)) and len(x) == 2:
            x, state = x

        # Ensure torch tensor on device
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.to(self.device)

        # Helper: state -> torch, align batch
        def _state_to_torch(state_like, batch_size: int):
            if state_like is None:
                return None
            if not isinstance(state_like, torch.Tensor):
                s = torch.as_tensor(state_like, dtype=torch.float32, device=self.device)
            else:
                s = state_like.to(device=self.device, dtype=torch.float32)
            if s.dim() == 1:
                s = s.unsqueeze(0)
            if s.shape[0] == 1 and batch_size > 1:
                s = s.repeat(batch_size, 1)
            if s.shape[0] != batch_size:
                raise ValueError(f"State batch mismatch: state batch {s.shape[0]} vs image batch {batch_size}")
            return s

        # Choose grad context
        need_grad = bool(getattr(self, "finetune", False)) and self.training
        ctx = nullcontext() if need_grad else torch.inference_mode()

        # Case 1: flat vector (B, D) where first pixel_dim are image bytes
        if x.dim() == 2:
            B = x.shape[0]
            pixel = x[:, :self.pixel_dim]
            tail = x[:, self.pixel_dim:]

            # reshape to (B, H, W, C) using pixel_shape
            pixel = pixel.reshape((B,) + self.pixel_shape)

            # ensure tensor type
            if not isinstance(pixel, torch.Tensor):
                pixel = torch.as_tensor(pixel, device=self.device)

            # Handle framestack: C = 3*k
            C = pixel.shape[-1]
            if C in (1, 3):
                with ctx:
                    bchw = self._to_bchw(pixel)
                    pv = self._preprocess_pixel_values(bchw)
                    z = self._encode_pixel_values(pv)
            elif C % 3 == 0:
                k = C // 3
                # (B,H,W,3k) -> (B,k,H,W,3) -> (B*k,H,W,3)
                frames = pixel.view(B, pixel.shape[1], pixel.shape[2], k, 3).permute(0, 3, 1, 2, 4).contiguous()
                frames = frames.view(B * k, pixel.shape[1], pixel.shape[2], 3)
                with ctx:
                    bchw = self._to_bchw(frames)
                    pv = self._preprocess_pixel_values(bchw)
                    z_all = self._encode_pixel_values(pv)
                z = z_all.view(B, k, -1).max(dim=1)[0]
            else:
                # fallback: take first 3 channels
                with ctx:
                    bchw = self._to_bchw(pixel[..., :3])
                    pv = self._preprocess_pixel_values(bchw)
                    z = self._encode_pixel_values(pv)

            # concat tail (preferred) else external state
            if tail.shape[1] > 0:
                return torch.cat([z, tail.to(dtype=torch.float32)], dim=-1)

            st = _state_to_torch(state, B)
            return z if st is None or st.numel() == 0 else torch.cat([z, st], dim=-1)

        # Case 2: image tensor/array (HWC/BHWC/CHW/BCHW)
        # Convert to BCHW and optionally framestack
        if x.dim() == 3 or x.dim() == 4:
            # If HWC with C=3*k, framestack max-pool
            if x.dim() == 3:
                x_img = x
                B = 1
            else:
                x_img = x
                B = x_img.shape[0]

            # Convert to BCHW if possible
            # If channels last and C=3*k, we handle framestack similarly
            if x_img.dim() == 3:
                C = x_img.shape[-1] if x_img.shape[-1] not in (1, 3) else x_img.shape[0]
            else:
                C = x_img.shape[-1] if x_img.shape[-1] not in (1, 3) else x_img.shape[1]

            # Normalize path
            if x_img.dim() == 3:
                x_img = x_img.unsqueeze(0)

            # BHWC framestack
            if x_img.shape[-1] % 3 == 0 and x_img.shape[-1] not in (1, 3):
                k = x_img.shape[-1] // 3
                H, W = x_img.shape[1], x_img.shape[2]
                frames = x_img.view(B, H, W, k, 3).permute(0, 3, 1, 2, 4).contiguous()
                frames = frames.view(B * k, H, W, 3)
                with ctx:
                    bchw = self._to_bchw(frames)
                    pv = self._preprocess_pixel_values(bchw)
                    z_all = self._encode_pixel_values(pv)
                z = z_all.view(B, k, -1).max(dim=1)[0]
            else:
                with ctx:
                    bchw = self._to_bchw(x_img)
                    pv = self._preprocess_pixel_values(bchw)
                    z = self._encode_pixel_values(pv)

            st = _state_to_torch(state, B)
            return z if st is None or st.numel() == 0 else torch.cat([z, st], dim=-1)

        raise ValueError(f"Unsupported input type/shape for HuggingFaceEncoder: {type(x)} {getattr(x, 'shape', None)}")


@EncoderFactory.register('dinov3')
class DINOEncoder(BaseHuggingFaceEncoder):
    def __init__(self, model_dir='/home/shangyy/models/dinov3-vits16-pretrain-lvd1689m', device='cuda', pixel_shape=(64,64,3), finetune=False, **kwargs):
        super().__init__(model_dir, device, pixel_shape, finetune, use_safetensors=False, **kwargs)

    def _load_model(self, model_dir, use_safetensors):
        # DINO typically doesn't enforce safetensors by default in this codebase context, or maybe it does
        # Keeping consistent with original code: AutoModel.from_pretrained(..., local_files_only=True)
        return AutoModel.from_pretrained(model_dir, local_files_only=True)

    def _get_feature_dim(self):
        return int(getattr(self.model.config, 'hidden_size', 0)) if hasattr(self.model, 'config') else 0


@EncoderFactory.register('resnet-101')
class ResNetEncoder(BaseHuggingFaceEncoder):
    def __init__(self, model_dir='/home/shangyy/models/resnet-101', device='cuda', pixel_shape=(64,64,3), finetune=False, **kwargs):
        super().__init__(model_dir, device, pixel_shape, finetune, use_safetensors=True, **kwargs)

    def _load_model(self, model_dir, use_safetensors):
        return ResNetModel.from_pretrained(model_dir, use_safetensors=use_safetensors)

    def _get_feature_dim(self):
        base_dim = 2048 # Default for ResNet-101
        if hasattr(self.model.config, 'hidden_sizes'):
            base_dim = self.model.config.hidden_sizes[-1]
        return base_dim


@EncoderFactory.register('original')
class Encoder(nn.Module):
    def __init__(
            self,
            pixel_shape,
            spectral_normalization=False,
            **kwargs,
    ):
        super().__init__()

        self.pixel_shape = tuple(pixel_shape)
        self.pixel_dim = int(np.prod(self.pixel_shape))

        self.pixel_depth = self.pixel_shape[-1]

        self.encoder = CNN(self.pixel_depth, spectral_normalization=spectral_normalization)

    def forward(self, input):
        pixel, state = self._split_pixel_and_state(input)

        pixel = pixel / 255.

        rep = self.encoder(pixel)
        rep = rep.reshape(rep.shape[0], -1)
        output = rep if state is None or state.numel() == 0 else torch.cat([rep, state], dim=-1)

        return output

    def _split_pixel_and_state(self, input):
        state = None
        if isinstance(input, dict):
            state = input.get("state", None)
            input = input.get("image", input.get("pixels", input))
        elif isinstance(input, (tuple, list)) and len(input) == 2:
            input, state = input

        if not torch.is_tensor(input):
            input = torch.as_tensor(input)
        device = next(self.parameters()).device
        input = input.to(device=device)

        if input.dim() == 1:
            input = input.unsqueeze(0)

        if input.dim() == 2:
            batch_size = input.shape[0]
            pixel = input[..., :self.pixel_dim].reshape(batch_size, *self.pixel_shape)
            flat_state = input[..., self.pixel_dim:]
            state = flat_state if flat_state.shape[-1] > 0 else state
        elif input.dim() == 3:
            pixel = input.unsqueeze(0)
            batch_size = 1
        elif input.dim() == 4:
            pixel = input
            batch_size = pixel.shape[0]
        else:
            raise ValueError(f"Unsupported image input shape for Encoder: {tuple(input.shape)}")

        pixel = self._to_bchw(pixel)
        state = self._state_to_torch(state, batch_size, device=device)
        return pixel.float(), state

    def _to_bchw(self, pixel):
        if pixel.dim() != 4:
            raise ValueError(f"Expected rank-4 image tensor, got {tuple(pixel.shape)}")
        if pixel.shape[-1] == self.pixel_depth:
            return pixel.permute(0, 3, 1, 2).contiguous()
        if pixel.shape[1] == self.pixel_depth:
            return pixel.contiguous()
        raise ValueError(
            f"Cannot infer channel axis for image shape {tuple(pixel.shape)} "
            f"with expected depth {self.pixel_depth}."
        )

    @staticmethod
    def _state_to_torch(state, batch_size, *, device):
        if state is None:
            return None
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device=device, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.shape[0] == 1 and batch_size > 1:
            state = state.repeat(batch_size, 1)
        if state.shape[0] != batch_size:
            raise ValueError(f"State batch mismatch: state batch {state.shape[0]} vs image batch {batch_size}")
        return state


@EncoderFactory.register('galaxea-r1lite-triview')
class GalaxeaR1LiteTriViewEncoder(nn.Module):
    """R1 Lite multiview encoder: shared wrist CNN, independent head CNN, MLP fusion."""

    def __init__(
            self,
            pixel_shape,
            spectral_normalization=False,
            fusion_hidden_dim=512,
            fusion_output_dim=None,
            **kwargs,
    ):
        super().__init__()
        self.pixel_shape = tuple(pixel_shape)
        self.pixel_dim = int(np.prod(self.pixel_shape))
        self.pixel_depth = int(self.pixel_shape[-1])
        if self.pixel_depth != 9:
            raise ValueError(
                "galaxea-r1lite-triview expects pixel_shape with 9 channels "
                f"(left/right/head RGB), got {self.pixel_shape}."
            )

        self.arm_encoder = CNN(3, spectral_normalization=spectral_normalization)
        self.head_encoder = CNN(3, spectral_normalization=spectral_normalization)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.pixel_shape[0], self.pixel_shape[1])
            arm_dim = int(self.arm_encoder(dummy).reshape(1, -1).shape[-1])
            head_dim = int(self.head_encoder(dummy).reshape(1, -1).shape[-1])

        fusion_output_dim = int(fusion_output_dim or arm_dim)
        self.feature_dim = fusion_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(arm_dim * 2 + head_dim, int(fusion_hidden_dim)),
            nn.ELU(),
            nn.Linear(int(fusion_hidden_dim), fusion_output_dim),
            nn.ELU(),
        )

    def forward(self, input):
        pixel, state = self._split_pixel_and_state(input)
        pixel = pixel.float() / 255.

        left = pixel[:, 0:3]
        right = pixel[:, 3:6]
        head = pixel[:, 6:9]

        left_feat = self.arm_encoder(left).reshape(pixel.shape[0], -1)
        right_feat = self.arm_encoder(right).reshape(pixel.shape[0], -1)
        head_feat = self.head_encoder(head).reshape(pixel.shape[0], -1)
        fused = self.fusion(torch.cat([left_feat, right_feat, head_feat], dim=-1))
        return fused if state is None or state.numel() == 0 else torch.cat([fused, state], dim=-1)

    def _split_pixel_and_state(self, input):
        state = None
        if isinstance(input, dict):
            state = input.get("state", None)
            input = input.get("image", input.get("pixels", input))
        elif isinstance(input, (tuple, list)) and len(input) == 2:
            input, state = input

        if not torch.is_tensor(input):
            input = torch.as_tensor(input)
        device = next(self.parameters()).device
        input = input.to(device=device)

        if input.dim() == 1:
            input = input.unsqueeze(0)

        if input.dim() == 2:
            batch_size = input.shape[0]
            pixel = input[..., :self.pixel_dim].reshape(batch_size, *self.pixel_shape)
            flat_state = input[..., self.pixel_dim:]
            state = flat_state if flat_state.shape[-1] > 0 else state
        elif input.dim() == 3:
            pixel = input.unsqueeze(0)
            batch_size = 1
        elif input.dim() == 4:
            pixel = input
            batch_size = pixel.shape[0]
        else:
            raise ValueError(f"Unsupported image input shape for GalaxeaR1LiteTriViewEncoder: {tuple(input.shape)}")

        pixel = self._to_bchw(pixel)
        state = Encoder._state_to_torch(state, batch_size, device=device)
        return pixel.float(), state

    def _to_bchw(self, pixel):
        if pixel.dim() != 4:
            raise ValueError(f"Expected rank-4 image tensor, got {tuple(pixel.shape)}")
        if pixel.shape[-1] == 9:
            return pixel.permute(0, 3, 1, 2).contiguous()
        if pixel.shape[1] == 9:
            return pixel.contiguous()
        raise ValueError(
            f"Cannot infer channel axis for Galaxea triview image shape {tuple(pixel.shape)}; "
            "expected 9 channels."
        )


class StopGradEncoder(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        # IMPORTANT: do NOT register the shared encoder as a submodule,
        # otherwise traj_encoder.state_dict()/parameters() will include it.
        # We bypass nn.Module.__setattr__ to avoid submodule registration.
        self.__dict__['_enc'] = enc

    @property
    def enc(self):
        return self.__dict__['_enc']

    def forward(self, x):
        prev_training = self.enc.training
        try:
            # Avoid BN/Dropout state updates from the traj encoder path.
            self.enc.eval()
            with torch.no_grad():
                y = self.enc(x)
        finally:
            self.enc.train(prev_training)
        return y.detach()


class WithEncoder(nn.Module):
    def __init__(
            self,
            encoder,
            module,
    ):
        super().__init__()

        self.encoder = encoder
        self.module = module

    def get_rep(self, input):
        return self.encoder(input)

    def forward(self, *inputs):
        rep = self.get_rep(inputs[0])
        return self.module(rep, *inputs[1:])

    def forward_mode(self, *inputs):
        rep = self.get_rep(inputs[0])
        return self.module.forward_mode(rep, *inputs[1:])
