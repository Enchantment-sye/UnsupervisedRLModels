from __future__ import annotations

from contextlib import contextmanager
import os
import numpy as np

from envs.ogbench_scene_kitchen_like_eval import (
    SCENE_RESET_INFO_KEYS,
    calc_ogbench_scene_kitchen_like_metrics,
    reset_info_key,
)
from envs.wrappers import Box, EnvSpec


_STATE_SCENE_ID = "scene-v0"
_PIXEL_SCENE_ID = "visual-scene-v0"
_STATE_SCENE_OBS_DIM = 40
_ARM_MATERIAL_NAMES = (
    "ur5e/robotiq/metal",
    "ur5e/robotiq/silicone",
    "ur5e/robotiq/gray",
    "ur5e/robotiq/black",
    "ur5e/robotiq/pad_gray",
    "ur5e/black",
    "ur5e/jointgray",
    "ur5e/linkgray",
    "ur5e/lightblue",
)
_SCENE_ALIASES = {
    "scene",
    "scene-v0",
    "scene-play-v0",
    "visual-scene",
    "visual-scene-v0",
    "visual-scene-play-v0",
}


def _bootstrap_render_env():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def is_ogbench_task(task_name) -> bool:
    return isinstance(task_name, str) and task_name.startswith("ogbench_")


def resolve_ogbench_scene_env_id(task_name, encoder) -> str:
    if not is_ogbench_task(task_name):
        raise ValueError(f"OGBench task must start with 'ogbench_', got {task_name!r}.")
    suffix = task_name.split("_", 1)[1]
    if suffix not in _SCENE_ALIASES:
        raise NotImplementedError(
            f"Unsupported OGBench task {task_name!r}. Supported aliases are "
            "ogbench_scene, ogbench_scene-v0, ogbench_scene-play-v0, "
            "ogbench_visual-scene-v0, and ogbench_visual-scene-play-v0."
        )
    return _PIXEL_SCENE_ID if bool(int(encoder)) else _STATE_SCENE_ID


class OGBenchEnv:
    """Adapter from OGBench Gymnasium envs to METRA's dict timestep API."""

    def __init__(
        self,
        task_name="ogbench_scene",
        seed=None,
        action_repeat=1,
        size=(64, 64),
        video_size=None,
        video_source="blog",
        video_opaque_arm=True,
        encoder=0,
        flatten_obs=True,
    ):
        _bootstrap_render_env()
        try:
            import gymnasium as gym
            import ogbench  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OGBenchEnv requires gymnasium and ogbench. Run this under the "
                "OGBench conda environment."
            ) from exc

        self.name = str(task_name)
        self.env_id = resolve_ogbench_scene_env_id(task_name, encoder)
        self._size = tuple(size)
        self._video_size = _normalize_size(video_size, fallback=self._size)
        self._video_source = str(video_source or "blog")
        self._video_opaque_arm = bool(video_opaque_arm)
        self._video_capture_active = False
        self._uses_pixels = bool(int(encoder))
        make_kwargs = {"disable_env_checker": True}
        if self._uses_pixels:
            make_kwargs.update({"width": int(self._size[1]), "height": int(self._size[0])})
        try:
            self._env = gym.make(self.env_id, **make_kwargs)
        except TypeError:
            self._env = gym.make(self.env_id)
        self._seed = seed
        self._action_repeat = int(action_repeat)
        self.flatten_obs = bool(flatten_obs)
        self.last_reset_info = {}
        self._next_reset_seed = None
        self._done = False
        self._last_obs = None
        self._blank_image = np.zeros((*self._size, 3), dtype=np.uint8)
        self._state_dim = int(np.prod((*self._size, 3))) if self._uses_pixels else _STATE_SCENE_OBS_DIM

    @property
    def obs_space(self):
        return {
            "image": self._image_space(),
            "state": self._state_space(),
            "reward": Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": Box(0, 1, (), dtype=bool),
            "is_last": Box(0, 1, (), dtype=bool),
            "is_terminal": Box(0, 1, (), dtype=bool),
            "success": Box(0, 1, (), dtype=bool),
        }

    @property
    def act_space(self):
        action_space = self._env.action_space
        return {
            "action": Box(
                action_space.low,
                action_space.high,
                action_space.shape,
                action_space.dtype,
            )
        }

    @property
    def spec(self):
        return EnvSpec(obs_space=self.obs_space, act_space=self.act_space)

    def reset(self):
        reset_kwargs = {}
        seed = self._next_reset_seed if self._next_reset_seed is not None else self._seed
        self._next_reset_seed = None
        if seed is not None:
            reset_kwargs["seed"] = int(seed)
            self._seed = None
        np_random_state = None
        if seed is not None:
            # OGBench Scene samples task_id with global np.random in addition
            # to Gymnasium's env RNG. Temporarily seed both so evaluation
            # videos can start from the same semantic scene across skills.
            np_random_state = np.random.get_state()
            np.random.seed(int(seed))
            action_space_seed = getattr(getattr(self._env, "action_space", None), "seed", None)
            if callable(action_space_seed):
                action_space_seed(int(seed))
        try:
            obs, info = self._env.reset(**reset_kwargs)
        finally:
            if np_random_state is not None:
                np.random.set_state(np_random_state)
        self._done = False
        self._last_obs = obs
        self.last_reset_info = _copy_info(info)
        return self._build_timestep(
            obs,
            reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
            info=self.last_reset_info,
        )

    def step(self, action):
        raw_action = action.get("action", action) if isinstance(action, dict) else action
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = self._last_obs
        for _ in range(max(1, self._action_repeat)):
            obs, reward, cur_terminated, cur_truncated, cur_info = self._env.step(raw_action)
            total_reward += float(reward)
            info = _copy_info(cur_info)
            terminated = terminated or bool(cur_terminated)
            truncated = truncated or bool(cur_truncated)
            if terminated or truncated:
                break

        self._done = bool(terminated or truncated)
        self._last_obs = obs
        info = self._with_reset_info(info)
        return self._build_timestep(
            obs,
            reward=total_reward,
            is_first=False,
            is_last=self._done,
            is_terminal=terminated,
            info=info,
        )

    def calc_eval_metrics(self, trajectories, is_option_trajectories=True, coord_dims=None):
        _ = is_option_trajectories, coord_dims
        return calc_ogbench_scene_kitchen_like_metrics(trajectories)

    def close(self):
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def render(self, mode="rgb_array"):
        _ = mode
        return self.capture_video_frame(source=self._video_source)

    def capture_video_frame(self, source=None):
        camera, frame_size, opaque_arm = self._resolve_video_render_request(source)

        render = getattr(self._raw_env(), "render", None)
        if not callable(render):
            return self._blank_image.copy()
        with self._temporary_render_size(frame_size), self._temporary_opaque_arm(opaque_arm):
            try:
                frame = render(camera=camera) if camera is not None else render()
            except TypeError:
                frame = render()
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim == 1:
            frame = frame.reshape(_infer_flat_image_shape(frame))
        return _resize_image_nearest(frame, frame_size)

    def set_video_capture_active(self, active):
        """Switch only eval video rendering to the high-quality blog-style size."""
        self._video_capture_active = bool(active)
        target_size = self._video_size if self._video_capture_active else self._size
        self._set_raw_render_size(target_size)

    def set_next_reset_perturbation(self, seed, scale):
        # OGBench Scene video rollouts use this existing rollout hook only to
        # make reset states comparable across skills; no state perturbation is
        # applied here.
        _ = scale
        self._next_reset_seed = int(seed)

    def _build_timestep(self, obs, *, reward, is_first, is_last, is_terminal, info):
        info = _copy_info(info)
        state = self._extract_state(obs, info)
        image = self._extract_image(obs)
        info.setdefault("state", state.copy())
        return {
            "image": image,
            "state": state,
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "success": bool(info.get("success", False)),
            "info": info,
        }

    def _extract_state(self, obs, info):
        if "state" in info:
            return np.asarray(info["state"], dtype=np.float32).reshape(-1)
        if self._uses_pixels:
            return self._extract_image(obs).astype(np.float32, copy=False).reshape(-1)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _extract_image(self, obs):
        if self._uses_pixels:
            image = np.asarray(obs)
            if image.ndim == 1:
                image = image.reshape(_infer_flat_image_shape(image))
            image = image.astype(np.uint8, copy=False)
            return _resize_image_nearest(image, self._size)
        return self._blank_image.copy()

    def _image_space(self):
        return Box(0, 255, (*self._size, 3), dtype=np.uint8)

    def _state_space(self):
        return Box(-np.inf, np.inf, (self._state_dim,), dtype=np.float32)

    def _with_reset_info(self, info):
        info = _copy_info(info)
        for key in SCENE_RESET_INFO_KEYS:
            if key in self.last_reset_info:
                info[reset_info_key(key)] = _copy_value(self.last_reset_info[key])
        return info

    def _resolve_video_render_request(self, source):
        source = self._video_source if source is None else source
        source_key = str(source or "blog").lower()
        if source_key in ("observation", "pixels", "front_pixels"):
            camera = "front_pixels" if self._uses_pixels else "front"
            return camera, self._size, False
        if source_key in ("blog", "render", "front"):
            # The public OGBench page uses manipulation videos as visualizations,
            # not policy observations. Use the task camera and make the arm
            # opaque only during this eval-only render call so pixel training
            # still receives the official transparent-arm observation.
            return "front", self._video_size, self._video_opaque_arm
        return source, self._video_size, False

    @contextmanager
    def _temporary_opaque_arm(self, enabled):
        if not enabled or not self._uses_pixels:
            yield
            return

        model = getattr(self._raw_env(), "_model", None)
        if model is None:
            yield
            return

        originals = []
        for material_name in _ARM_MATERIAL_NAMES:
            try:
                material = model.material(material_name)
                rgba = material.rgba
            except Exception:
                continue
            originals.append((rgba, np.asarray(rgba).copy()))
            rgba[3] = 1.0

        try:
            yield
        finally:
            for rgba, original in originals:
                rgba[:] = original

    @contextmanager
    def _temporary_render_size(self, size):
        if self._video_capture_active:
            yield
            return

        old_size = self._raw_render_size()
        if old_size is None or old_size == tuple(size):
            yield
            return

        self._set_raw_render_size(size)
        try:
            yield
        finally:
            self._set_raw_render_size(old_size)

    def _raw_render_size(self):
        raw_env = self._raw_env()
        if not hasattr(raw_env, "_render_height") or not hasattr(raw_env, "_render_width"):
            return None
        return int(raw_env._render_height), int(raw_env._render_width)

    def _set_raw_render_size(self, size):
        raw_env = self._raw_env()
        if not hasattr(raw_env, "_render_height") or not hasattr(raw_env, "_render_width"):
            return
        size = tuple(size)
        if self._raw_render_size() == size:
            return
        self._close_raw_renderer()
        self._ensure_raw_offscreen_size(raw_env, size)
        raw_env._render_height = int(size[0])
        raw_env._render_width = int(size[1])

    def _close_raw_renderer(self):
        raw_env = self._raw_env()
        renderer = getattr(raw_env, "_renderer", None)
        if renderer is not None:
            close = getattr(renderer, "close", None)
            if callable(close):
                close()
        if hasattr(raw_env, "_renderer"):
            raw_env._renderer = None

    def _raw_env(self):
        return getattr(self._env, "unwrapped", self._env)

    def _ensure_raw_offscreen_size(self, raw_env, size):
        model = getattr(raw_env, "_model", None)
        if model is None:
            return
        try:
            visual_global = model.vis.global_
            visual_global.offheight = max(int(visual_global.offheight), int(size[0]))
            visual_global.offwidth = max(int(visual_global.offwidth), int(size[1]))
        except Exception:
            return


def _copy_info(info):
    if info is None:
        return {}
    return {key: _copy_value(value) for key, value in dict(info).items()}


def _copy_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return np.asarray(value).copy()
    except Exception:
        return value


def _infer_flat_image_shape(image):
    flat_size = int(np.asarray(image).size)
    channels = 3
    pixels = flat_size // channels
    side = int(round(np.sqrt(pixels)))
    if side * side * channels != flat_size:
        raise ValueError(f"Cannot infer HWC image shape from flat OGBench pixel obs of size {flat_size}.")
    return side, side, channels


def _normalize_size(size, fallback):
    if size is None:
        return tuple(fallback)
    if np.isscalar(size):
        value = int(size)
        if value <= 0:
            return tuple(fallback)
        return value, value
    values = tuple(size)
    if len(values) != 2:
        raise ValueError(f"Expected OGBench video_size to be scalar or (H, W), got {size!r}.")
    return int(values[0]), int(values[1])


def _resize_image_nearest(image, size):
    target_h, target_w = int(size[0]), int(size[1])
    if image.shape[:2] == (target_h, target_w):
        return image.copy()
    if image.ndim != 3:
        raise ValueError(f"OGBench pixel obs must be HWC image, got shape={image.shape}.")
    src_h, src_w = image.shape[:2]
    y_idx = np.floor(np.arange(target_h, dtype=np.float64) * src_h / target_h).astype(np.int64)
    x_idx = np.floor(np.arange(target_w, dtype=np.float64) * src_w / target_w).astype(np.int64)
    y_idx = np.clip(y_idx, 0, src_h - 1)
    x_idx = np.clip(x_idx, 0, src_w - 1)
    return image[y_idx][:, x_idx].copy()
