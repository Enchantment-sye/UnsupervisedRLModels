from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def timestamped_distill_work_dir(root: str, task: str, seed: int) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rel = os.path.join(task, "cov_encoder_distill", "resnet101", f"{stamp}_seed{seed}")
    candidates = [
        os.path.join(root, rel),
        os.path.join(os.getcwd(), "outputs", rel),
        os.path.join("/tmp", "metra", rel),
    ]
    errors = []
    for candidate in candidates:
        try:
            return ensure_dir(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise OSError("Could not create coverage distill work_dir. Tried:\n" + "\n".join(errors))


def save_config_json(path: str, cfg: Any) -> None:
    if is_dataclass(cfg):
        payload = asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        payload = vars(cfg)
    else:
        payload = cfg
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=4, sort_keys=True)


def obs_image(timestep_or_obs):
    if isinstance(timestep_or_obs, tuple) and timestep_or_obs:
        timestep_or_obs = timestep_or_obs[0]
    if isinstance(timestep_or_obs, dict):
        if "image" in timestep_or_obs:
            return timestep_or_obs["image"]
        if "obs" in timestep_or_obs:
            return timestep_or_obs["obs"]
    return timestep_or_obs


def normalize_step_output(step_output, *, use_pixels: bool = True) -> Dict[str, Any]:
    if isinstance(step_output, tuple) and len(step_output) == 5:
        obs, reward, terminated, truncated, info = step_output
        done = bool(terminated or truncated)
        image = obs.get("image") if isinstance(obs, dict) and use_pixels else obs
        return {
            "image": image,
            "reward": reward,
            "is_terminal": done,
            "info": info or {},
        }
    if not isinstance(step_output, dict):
        raise TypeError(f"Unsupported env.step output type: {type(step_output)}")
    return step_output


def sample_action(env):
    action_space = getattr(getattr(env, "spec", None), "action_space", None)
    if action_space is None:
        action_space = getattr(env, "action_space", None)
    if action_space is None:
        act_space = getattr(env, "act_space", None)
        if isinstance(act_space, dict):
            action_space = act_space.get("action")
        else:
            action_space = act_space
    if action_space is None or not hasattr(action_space, "sample"):
        raise AttributeError("Could not find a sample-able action space on the environment")
    action = action_space.sample()
    if isinstance(action, dict) and set(action.keys()) == {"action"}:
        action = action["action"]
    return np.asarray(action, dtype=np.float32)


def env_step(env, action):
    try:
        out = env.step({"action": action})
    except Exception:
        out = env.step(action)
    return normalize_step_output(out)


def infer_pixel_shape(env):
    obs_space = getattr(getattr(env, "spec", None), "observation_space", None)
    if obs_space is not None and hasattr(obs_space, "shape"):
        return tuple(obs_space.shape)
    obs_space = getattr(env, "obs_space", None)
    if isinstance(obs_space, dict) and "image" in obs_space:
        return tuple(obs_space["image"].shape)
    raise AttributeError("Could not infer pixel_shape from env.spec.observation_space or env.obs_space['image']")


def to_torch(value, device, dtype=torch.float32):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def image_to_bchw(obs, *, pixel_shape=None, device=None, dtype=torch.float32):
    x = to_torch(obs, device=device or "cpu", dtype=dtype)
    if x.dim() == 1:
        if pixel_shape is None:
            raise ValueError("pixel_shape is required for flattened image observations")
        x = x.reshape((1,) + tuple(pixel_shape))
    elif x.dim() == 2:
        if pixel_shape is None:
            raise ValueError("pixel_shape is required for flattened image batches")
        x = x.reshape((x.shape[0],) + tuple(pixel_shape))
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected image tensor rank 3/4 or flattened batch, got {tuple(x.shape)}")

    expected_c = int(pixel_shape[-1]) if pixel_shape is not None and len(pixel_shape) == 3 else None
    if expected_c is not None and x.shape[-1] == expected_c:
        x = x.permute(0, 3, 1, 2).contiguous()
    elif x.shape[-1] in (1, 3, 6, 9, 12) and x.shape[1] not in (1, 3, 6, 9, 12):
        x = x.permute(0, 3, 1, 2).contiguous()
    elif x.shape[1] in (1, 3, 6, 9, 12) or expected_c is None:
        x = x.contiguous()
    else:
        raise ValueError(f"Cannot infer channel axis for image shape {tuple(x.shape)}")
    return x.float()


def last_rgb_frame_bchw(obs, *, pixel_shape=None, device=None):
    x = image_to_bchw(obs, pixel_shape=pixel_shape, device=device)
    if x.shape[1] < 3:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        if x.shape[1] % 3 != 0:
            raise ValueError(f"Expected RGB or RGB frame stack channels, got C={x.shape[1]}")
        x = x[:, -3:]
    return x


def random_shift(x, pad: int = 4):
    if pad <= 0:
        return x
    if x.dim() != 4:
        raise ValueError(f"random_shift expects BCHW image tensor, got {tuple(x.shape)}")
    n, _, h, w = x.shape
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    max_y = 2 * pad + 1
    ys = torch.randint(0, max_y, (n,), device=x.device)
    xs = torch.randint(0, max_y, (n,), device=x.device)
    crops = []
    for i in range(n):
        crops.append(padded[i : i + 1, :, ys[i] : ys[i] + h, xs[i] : xs[i] + w])
    return torch.cat(crops, dim=0)
