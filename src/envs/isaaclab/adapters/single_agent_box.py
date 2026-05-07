import time
from typing import Dict, List

import gym
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from envs.wrappers import Box, EnvSpec

from ..launcher import release_isaaclab_app
from ..cfg_builders import get_default_viewer_preset_name
from .vision import ImageAcquisitionError, IsaacLabImageProvider, is_image_array


_GALAXEA_STATE_KEY_ORDER = (
    "joint_pos",
    "joint_vel",
    "left_ee_pose",
    "right_ee_pose",
    "object_pose",
    "goal_pose",
)


class IsaacLabSingleAgentBoxEnv:
    def __init__(self, env, task_spec, request):
        self._env = env
        self._task_spec = task_spec
        self._request = request
        self._image_provider = IsaacLabImageProvider(env, request, task_spec)
        self._seed_applied = False
        self._last_obs_dict = None
        self._last_extras = {}
        self._last_state = np.zeros(self._infer_state_space().shape, dtype=np.float32)
        self._last_image = self._image_provider.placeholder_image()
        self._last_image_tensor = None
        self._default_viewer_preset = get_default_viewer_preset_name(task_spec)
        self._active_viewer_preset = self._default_viewer_preset
        self._timing_totals = {"TimeImagePostprocess": 0.0}
        self._obs_space = self._build_obs_space()
        self._act_space = self._build_act_space()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def obs_space(self):
        return self._copy_space_dict(self._obs_space)

    @property
    def act_space(self):
        return self._copy_space_dict(self._act_space)

    @property
    def spec(self):
        return EnvSpec(obs_space=self.obs_space, act_space=self.act_space)

    def reset(self):
        reset_kwargs = {}
        if not self._seed_applied and self._request.seed is not None:
            reset_kwargs["seed"] = int(self._request.seed)
            self._seed_applied = True

        reset_output = self._env.reset(**reset_kwargs)
        obs_dict, extras = self._normalize_reset_output(reset_output)
        state = self._extract_state(obs_dict)
        image_tensor = self._resolve_image_tensor(
            obs_dict,
            extras,
            allow_placeholder=not self._request.encoder,
        )
        image = self._resolve_image(
            obs_dict,
            extras,
            allow_placeholder=not self._request.encoder,
            image_tensor=image_tensor,
        )
        self._cache_step(obs_dict, extras, state, image, image_tensor=image_tensor)
        return self._build_timestep(
            reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
        )

    def step(self, action):
        action_tensor = self._prepare_action_tensor(action)
        step_output = self._env.step(action_tensor)
        if not isinstance(step_output, tuple) or len(step_output) != 5:
            raise TypeError(
                "Isaac Lab env.step() must return (obs_dict, reward, terminated, truncated, extras)."
            )
        obs_dict, reward, terminated, truncated, extras = step_output
        state = self._extract_state(obs_dict)
        image_tensor = self._resolve_image_tensor(
            obs_dict,
            extras,
            allow_placeholder=not self._request.encoder,
        )
        image = self._resolve_image(
            obs_dict,
            extras,
            allow_placeholder=not self._request.encoder,
            image_tensor=image_tensor,
        )
        done = bool(self._to_scalar(terminated) or self._to_scalar(truncated))
        self._cache_step(obs_dict, extras, state, image, image_tensor=image_tensor)
        return self._build_timestep(
            reward=self._to_scalar(reward),
            is_first=False,
            is_last=done,
            is_terminal=done,
        )

    def render(self, mode="offscreen"):
        if self._request.flatten_obs:
            return self._last_image.reshape(self._request.render_size, self._request.render_size, 3)
        return self._last_image

    def capture_video_frame(self, source=None):
        video_source = (source or self._request.video_source or "observation").lower()
        if video_source == "observation":
            return self._last_image.copy()

        if video_source == "render":
            image = self._image_provider._render_image()
            if image is None:
                return None
            image = self._image_provider._finalize_image(image)
            if self._is_invalid_render_video_frame(image):
                return None
            return image

        raise ValueError(f"Unsupported Isaac Lab video frame source: {video_source!r}")

    def default_viewer_preset_name(self):
        return self._default_viewer_preset

    def get_train_image_tensor(self):
        return self._last_image_tensor

    def get_parallel_train_handles(self):
        action_space = self._single_action_space()
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        return {
            "raw_env": self._env,
            "task_spec": self._task_spec,
            "request": self._request,
            "action_low": low,
            "action_high": high,
        }

    def consume_timing_metrics(self):
        metrics = dict(self._timing_totals)
        for key in self._timing_totals:
            self._timing_totals[key] = 0.0
        return metrics

    def close(self):
        try:
            if hasattr(self._env, "close"):
                self._env.close()
        finally:
            release_isaaclab_app()

    def _single_observation_space(self):
        for candidate in (
            getattr(getattr(self._env, "unwrapped", self._env), "single_observation_space", None),
            getattr(self._env, "single_observation_space", None),
            getattr(self._env, "observation_space", None),
        ):
            if candidate is not None:
                if hasattr(candidate, "spaces") and "policy" in candidate.spaces:
                    return candidate.spaces["policy"]
                return candidate
        return gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

    def _single_action_space(self):
        for candidate in (
            getattr(getattr(self._env, "unwrapped", self._env), "single_action_space", None),
            getattr(self._env, "single_action_space", None),
            getattr(self._env, "action_space", None),
        ):
            if candidate is not None:
                return candidate
        raise AttributeError("Isaac Lab env does not expose an action space.")

    def _infer_state_space(self):
        obs_space = self._single_observation_space()
        if all(hasattr(obs_space, attr) for attr in ("low", "high", "shape")):
            low = np.asarray(obs_space.low, dtype=np.float32)
            high = np.asarray(obs_space.high, dtype=np.float32)
            return Box(low=low, high=high, shape=obs_space.shape, dtype=np.float32)

        if hasattr(obs_space, "spaces"):
            flat_dim = self._flatten_space_dict(obs_space.spaces)
            return Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

        raise TypeError(
            f"Unsupported Isaac Lab policy observation space type: {type(obs_space)!r}"
        )

    def _flatten_space_dict(self, spaces: Dict) -> int:
        dims: List[int] = []
        for key in self._ordered_state_keys(spaces):
            space = spaces[key]
            if self._should_skip_state_key(key) or len(getattr(space, "shape", ())) >= 3:
                continue
            dims.append(int(np.prod(space.shape)))
        if not dims:
            for key in self._ordered_state_keys(spaces):
                space = spaces[key]
                dims.append(int(np.prod(space.shape)))
        return int(sum(dims))

    def _build_obs_space(self):
        state_space = self._infer_state_space()
        image_shape = self._image_provider.image_space_shape
        image_space = Box(
            low=0,
            high=255,
            shape=image_shape,
            dtype=np.uint8,
        )
        return {
            "image": image_space,
            "state": state_space,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "info": {
                "state": state_space,
            },
        }

    def _build_act_space(self):
        action_space = self._single_action_space()
        if not all(hasattr(action_space, attr) for attr in ("low", "high", "shape")):
            raise TypeError(
                f"Unsupported Isaac Lab action space type: {type(action_space)!r}"
            )
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        return {"action": Box(low=low, high=high, shape=action_space.shape, dtype=np.float32)}

    def _copy_space_dict(self, spaces):
        copied = {}
        for key, value in spaces.items():
            if isinstance(value, dict):
                copied[key] = self._copy_space_dict(value)
            else:
                copied[key] = value
        return copied

    def _prepare_action_tensor(self, action):
        if isinstance(action, dict):
            action = action.get("action", action)
        array = np.asarray(action, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        target_num_envs = max(int(getattr(self._request, "num_envs", 1) or 1), 1)
        if target_num_envs > 1:
            if array.shape[0] == 1:
                batched = np.zeros((target_num_envs, array.shape[-1]), dtype=np.float32)
                batched[0] = array[0]
                array = batched
            elif array.shape[0] != target_num_envs:
                raise ValueError(
                    f"IsaacLab single-agent wrapper expected either 1 action or {target_num_envs} actions, "
                    f"got shape={tuple(array.shape)}"
                )
        device = getattr(getattr(self._env, "unwrapped", self._env), "device", None)
        if torch is not None:
            return torch.as_tensor(array, dtype=torch.float32, device=device)
        return array

    @staticmethod
    def _is_invalid_render_video_frame(image):
        array = np.asarray(image)
        return array.size == 0 or not np.any(array)

    def _normalize_reset_output(self, reset_output):
        if isinstance(reset_output, tuple) and len(reset_output) >= 2:
            return reset_output[0], reset_output[1] or {}
        return reset_output, {}

    def _extract_state(self, obs_dict):
        if isinstance(obs_dict, dict) and "policy" in obs_dict:
            policy_obs = obs_dict["policy"]
        else:
            policy_obs = obs_dict
        chunks = []
        self._collect_state_chunks(policy_obs, chunks, prefix="")
        if not chunks:
            array = self._to_single_numpy(policy_obs)
            return array.astype(np.float32).reshape(-1)
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _collect_state_chunks(self, value, chunks, prefix):
        if isinstance(value, dict):
            for key in self._ordered_state_keys(value):
                nested_prefix = f"{prefix}.{key}" if prefix else key
                nested = value[key]
                if self._should_skip_state_key(key):
                    continue
                self._collect_state_chunks(nested, chunks, nested_prefix)
            return
        if isinstance(value, (list, tuple)):
            for idx, nested in enumerate(value):
                self._collect_state_chunks(nested, chunks, f"{prefix}[{idx}]")
            return
        array = self._to_single_numpy(value)
        if is_image_array(array):
            return
        chunks.append(array.astype(np.float32).reshape(-1))

    def _to_single_numpy(self, value):
        if torch is not None and torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        target_num_envs = max(int(getattr(self._request, "num_envs", 1) or 1), 1)
        if array.ndim >= 1 and target_num_envs > 1 and array.shape[0] == target_num_envs:
            array = array[0]
        elif array.ndim >= 1 and array.shape[0] == 1:
            array = array[0]
        return array

    def _looks_visual_key(self, key: str) -> bool:
        lowered = key.lower()
        return any(
            token in lowered
            for token in ("rgb", "image", "pixel", "camera", "depth", "segmentation")
        )

    def _should_skip_state_key(self, key: str) -> bool:
        lowered = key.lower()
        if lowered in ("last_joints", "last_joint", "previous_joints"):
            return True
        return self._looks_visual_key(key)

    def _ordered_state_keys(self, mapping: Dict) -> List[str]:
        keys = list(mapping.keys())
        preferred = [key for key in _GALAXEA_STATE_KEY_ORDER if key in mapping]
        remaining = sorted(key for key in keys if key not in preferred)
        return [*preferred, *remaining]

    def _resolve_image_tensor(self, obs_dict, extras, allow_placeholder):
        if not self._request.encoder or torch is None:
            return None

        start_time = time.perf_counter()
        try:
            image_tensor = self._image_provider.capture_tensor(
                obs_dict,
                extras,
                allow_placeholder=allow_placeholder,
            )
        except ImageAcquisitionError:
            if allow_placeholder:
                image_tensor = self._image_provider.placeholder_image_tensor()
            else:
                raise
        finally:
            self._timing_totals["TimeImagePostprocess"] += (time.perf_counter() - start_time)
        return image_tensor

    def _resolve_image(self, obs_dict, extras, allow_placeholder, image_tensor=None):
        try:
            if image_tensor is not None:
                return image_tensor.detach().cpu().numpy()
            return self._image_provider.capture(obs_dict, extras, allow_placeholder=allow_placeholder)
        except ImageAcquisitionError:
            if allow_placeholder:
                return self._image_provider.placeholder_image()
            raise

    def _build_timestep(self, reward, is_first, is_last, is_terminal):
        info = {"state": self._last_state.copy()}
        for key, value in self._extract_scalar_extras(self._last_extras).items():
            info[key] = value
        if self._last_extras:
            info["isaaclab_extras"] = self._last_extras
        return {
            "image": self._last_image.copy(),
            "state": self._last_state.copy(),
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "info": info,
        }

    def _extract_scalar_extras(self, extras):
        scalar_info = {}
        if not isinstance(extras, dict):
            return scalar_info
        for key, value in extras.items():
            if isinstance(value, dict):
                continue
            array = self._to_single_numpy(value)
            if array.ndim == 0:
                scalar_info[key] = array.item()
            elif array.ndim == 1 and array.shape[0] == 1:
                scalar_info[key] = array[0].item()
        return scalar_info

    def _cache_step(self, obs_dict, extras, state, image, image_tensor=None):
        self._last_obs_dict = obs_dict
        self._last_extras = extras or {}
        self._last_state = state.astype(np.float32)
        self._last_image = image.astype(np.uint8) if image.dtype != np.uint8 else image
        self._last_image_tensor = image_tensor

    def _to_scalar(self, value):
        if torch is not None and torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.ndim == 0:
            return float(array)
        return float(array.reshape(-1)[0])
