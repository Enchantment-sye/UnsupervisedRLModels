import re
from typing import Any, Iterable

try:
    import cv2
except ImportError:  # pragma: no cover - fallback below keeps the adapter usable.
    cv2 = None
try:
    import gym
except ImportError:
    import gymnasium as gym
import numpy as np

from envs.wrappers import Box, EnvSpec


_R1LITE_TASK_IDS = {
    "r1lite_base": "R1LiteBase-v0",
    "r1lite_block_hammer_beat": "R1LiteBlockHammerBeat-v0",
    "r1lite_block_handover": "R1LiteBlockHandover-v0",
    "r1lite_blocks_stack_easy": "R1LiteBlocksStackEasy-v0",
    "r1lite_blocks_stack_hard": "R1LiteBlocksStackHard-v0",
    "r1lite_bottle_adjust": "R1LiteBottleAdjust-v0",
    "r1lite_container_place": "R1LiteContainerPlace-v0",
    "r1lite_diverse_bottles_pick": "R1LiteDiverseBottlesPick-v0",
    "r1lite_dual_bottles_pick_easy": "R1LiteDualBottlesPickEasy-v0",
    "r1lite_dual_bottles_pick_hard": "R1LiteDualBottlesPickHard-v0",
    "r1lite_dual_shoes_place": "R1LiteDualShoesPlace-v0",
    "r1lite_empty_cup_place": "R1LiteEmptyCupPlace-v0",
    "r1lite_mug_hanging_easy": "R1LiteMugHangingEasy-v0",
    "r1lite_mug_hanging_hard": "R1LiteMugHangingHard-v0",
    "r1lite_pick_apple_messy": "R1LitePickAppleMessy-v0",
    "r1lite_put_apple_cabinet": "R1LitePutAppleCabinet-v0",
    "r1lite_shoe_place": "R1LiteShoePlace-v0",
    "r1lite_tool_adjust": "R1LiteToolAdjust-v0",
}

_STATE_SECTIONS = (
    "upper_body_observations",
    "lower_body_observations",
    "object_dict",
)

_PREFERRED_STATE_KEYS = (
    "left_arm_joint_position",
    "right_arm_joint_position",
    "left_arm_gripper_position",
    "right_arm_gripper_position",
    "left_arm_joint_velocity",
    "right_arm_joint_velocity",
    "left_arm_ee_pose",
    "right_arm_ee_pose",
    "chassis_joint_position",
    "torso_joint_position",
)

TRIVIEW_IMAGE_KEY = "rgb_left_right_head"
TRIVIEW_CAMERA_KEYS = ("rgb_left_hand", "rgb_right_hand", "rgb_head")
VIDEO_VIEW_DEFAULT = "default"
VIDEO_VIEW_ROBOT_FULL_BODY = "robot_full_body"

_ROBOT_FULL_BODY_CAMERA_POSITION = np.asarray([-2.2, -1.8, 1.05], dtype=np.float32)
_ROBOT_FULL_BODY_CAMERA_TARGET = np.asarray([0.0, -0.05, 0.75], dtype=np.float32)
_ROBOT_FULL_BODY_CAMERA_FOVY_DEG = 55.0


def resolve_galaxea_sim_env_id(task_name: str) -> str:
    """Resolve METRA's galaxea_* task suffix to a GalaxeaManipSim Gymnasium id."""
    if task_name.endswith("-v0"):
        return task_name

    key = task_name.strip().replace("-", "_")
    key = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
    key = re.sub(r"[^a-z0-9_]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")

    if key in _R1LITE_TASK_IDS:
        return _R1LITE_TASK_IDS[key]
    if not key.startswith("r1lite_"):
        candidate = f"r1lite_{key}"
        if candidate in _R1LITE_TASK_IDS:
            return _R1LITE_TASK_IDS[candidate]
    raise KeyError(
        f"Unknown GalaxeaManipSim task={task_name!r}. "
        f"Known R1 Lite tasks: {', '.join(sorted(_R1LITE_TASK_IDS))}"
    )


class GalaxeaSimEnv:
    """Adapter from GalaxeaManipSim Gymnasium envs to METRA's dict timestep API."""

    def __init__(
        self,
        name: str,
        seed: int | None = None,
        action_repeat: int = 1,
        size: tuple[int, int] = (64, 64),
        flatten_obs: int = 0,
        encoder: int = 0,
        headless: int = 1,
        obs_mode: str | None = None,
        image_key: str = "rgb_head",
        video_view_preset: str = VIDEO_VIEW_DEFAULT,
        controller_type: str = "bimanual_joint_position",
    ):
        try:
            import gymnasium as gymnasium
            import galaxea_sim.envs.base  # noqa: F401
            import galaxea_sim.envs.robotwin  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "GalaxeaManipSim backend requires the shangyy metra_galaxea environment "
                "with editable galaxea-sim installed. Try "
                "METRA_PYTHON=/home/shangyy/miniconda3/envs/metra_galaxea/bin/python."
            ) from exc

        self._env_id = resolve_galaxea_sim_env_id(name)
        self._seed = seed
        self._seed_applied = False
        self._action_repeat = max(int(action_repeat or 1), 1)
        self._size = tuple(size)
        self.flatten_obs = bool(flatten_obs)
        self.encoder = bool(encoder)
        self._image_key = image_key or "rgb_head"
        self._uses_triview = self._image_key == TRIVIEW_IMAGE_KEY
        self._video_view_preset = (video_view_preset or VIDEO_VIEW_DEFAULT).lower()
        if self._video_view_preset not in (VIDEO_VIEW_DEFAULT, VIDEO_VIEW_ROBOT_FULL_BODY):
            raise ValueError(f"Unsupported GalaxeaManipSim video view preset: {self._video_view_preset!r}")
        self._obs_mode = (obs_mode or ("image" if self.encoder else "state")).lower()
        if self._obs_mode not in ("state", "image"):
            raise ValueError(f"Unsupported GalaxeaManipSim obs_mode={self._obs_mode!r}")

        self._env = gymnasium.make(
            self._env_id,
            headless=bool(headless),
            obs_mode=self._obs_mode,
            controller_type=controller_type,
        )
        self._last_obs = None
        self._last_image = self._placeholder_image()
        self._last_named_state = {}
        self._obs_space = self._build_obs_space()
        self._act_space = self._build_act_space()

    @property
    def obs_space(self):
        return dict(self._obs_space)

    @property
    def act_space(self):
        return dict(self._act_space)

    @property
    def spec(self):
        return EnvSpec(obs_space=self.obs_space, act_space=self.act_space)

    @property
    def unwrapped(self):
        return getattr(self._env, "unwrapped", self._env)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self):
        reset_kwargs = {}
        if not self._seed_applied and self._seed is not None:
            reset_kwargs["seed"] = int(self._seed)
            self._seed_applied = True
        obs, info = self._env.reset(**reset_kwargs)
        self._last_obs = obs
        return self._build_timestep(obs, reward=0.0, is_first=True, is_last=False, is_terminal=False, info=info)

    def step(self, action):
        raw_action = action.get("action", action) if isinstance(action, dict) else action
        total_reward = 0.0
        done = False
        info = {}
        obs = self._last_obs
        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self._env.step(np.asarray(raw_action, dtype=np.float32))
            total_reward += float(reward)
            done = bool(terminated or truncated)
            if done:
                break
        self._last_obs = obs
        return self._build_timestep(
            obs,
            reward=total_reward,
            is_first=False,
            is_last=done,
            is_terminal=done,
            info=info,
        )

    def render(self, mode="offscreen"):
        image = self._resize_image(self._env.render())
        return image.reshape(-1) if self.flatten_obs else image

    def capture_video_frame(self, source=None):
        video_source = (source or "observation").lower()
        if video_source in ("render", "third_person"):
            self._apply_video_view_preset()
            return self._resize_image(self._env.render())
        if video_source != "observation":
            raise ValueError(f"Unsupported GalaxeaManipSim video frame source: {video_source!r}")
        return self._last_image.copy()

    def get_train_image_tensor(self):
        return self._last_image.reshape(-1).copy()

    def get_safety_state(self):
        return dict(self._last_named_state)

    def safety_denormalize_action(self, action):
        return action

    def safety_normalize_action(self, action):
        return action

    def safety_physical_action_bounds(self):
        space = self._act_space["action"]
        return np.asarray(space.low, dtype=np.float32).copy(), np.asarray(space.high, dtype=np.float32).copy()

    def close(self):
        if hasattr(self._env, "close"):
            return self._env.close()
        return None

    def _build_obs_space(self):
        state_dim = self._state_dim_from_space(getattr(self._env, "observation_space", None))
        state_space = Box(-np.inf, np.inf, (state_dim,), dtype=np.float32)
        image_depth = self._image_depth()
        image_shape = (
            (self._size[0] * self._size[1] * image_depth,)
            if self.flatten_obs
            else self._size + (image_depth,)
        )
        return {
            "image": Box(0, 255, image_shape, dtype=np.uint8),
            "state": state_space,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
            "info": {
                "state": state_space,
            },
        }

    def _build_act_space(self):
        action_space = getattr(self._env, "action_space")
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        return {"action": Box(low, high, action_space.shape, dtype=np.float32)}

    def _state_dim_from_space(self, observation_space) -> int:
        chunks = []
        if hasattr(observation_space, "spaces"):
            for section in _STATE_SECTIONS:
                section_space = observation_space.spaces.get(section)
                if section_space is not None:
                    self._collect_space_dims(section_space, chunks, prefix="")
        return int(sum(chunks)) if chunks else 1

    def _collect_space_dims(self, space, dims: list[int], prefix: str):
        if hasattr(space, "spaces"):
            keys = self._ordered_keys(space.spaces)
            for key in keys:
                if self._looks_visual_key(key) or key in ("language_instruction",):
                    continue
                nested_prefix = f"{prefix}.{key}" if prefix else key
                self._collect_space_dims(space.spaces[key], dims, nested_prefix)
            return
        shape = getattr(space, "shape", ())
        if len(shape) >= 3:
            return
        if shape is None:
            return
        dims.append(int(np.prod(shape)) if shape else 1)

    def _build_timestep(self, obs, *, reward, is_first, is_last, is_terminal, info):
        state = self._extract_state(obs)
        named_state = self._extract_named_state(obs)
        self._last_named_state = dict(named_state)
        image = self._extract_image(obs)
        success = bool((info or {}).get("success", False))
        return {
            "image": image.copy(),
            "state": state.copy(),
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "success": success,
            "info": {
                "state": state.copy(),
                "safety_state": named_state,
                "raw_galaxea_info": info or {},
                "success": success,
                "galaxea_sim_env_id": self._env_id,
                **(info or {}),
            },
        }

    def _extract_state(self, obs) -> np.ndarray:
        chunks = []
        for section in _STATE_SECTIONS:
            self._collect_state_chunks((obs or {}).get(section), chunks, prefix="")
        if not chunks:
            return np.zeros((1,), dtype=np.float32)
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _extract_named_state(self, obs) -> dict:
        named = {}
        for section in _STATE_SECTIONS:
            value = (obs or {}).get(section)
            self._collect_named_state(value, named, prefix="")
        return named

    def _collect_named_state(self, value: Any, named: dict, prefix: str):
        if value is None:
            return
        if isinstance(value, dict):
            for key in self._ordered_keys(value):
                if self._looks_visual_key(key) or key in ("language_instruction",):
                    continue
                nested_prefix = f"{prefix}.{key}" if prefix else key
                self._collect_named_state(value[key], named, nested_prefix)
            return
        if isinstance(value, (list, tuple)):
            for idx, nested in enumerate(value):
                self._collect_named_state(nested, named, f"{prefix}[{idx}]")
            return
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or array.ndim >= 3:
            return
        flat = array.astype(np.float32).reshape(-1)
        key = prefix.split(".")[-1] if prefix else ""
        canonical = self._canonical_safety_key(prefix, key)
        if canonical is not None:
            named[canonical] = flat.copy()
        if prefix and prefix not in named:
            named[prefix] = flat.copy()

    @staticmethod
    def _canonical_safety_key(path: str, key: str):
        lowered_path = path.lower()
        lowered_key = key.lower()
        aliases = {
            "left_arm_joint_position": ("left_arm_joint_position", "left_arm_joint_pos", "left_arm_qpos", "left_arm_joints"),
            "right_arm_joint_position": ("right_arm_joint_position", "right_arm_joint_pos", "right_arm_qpos", "right_arm_joints"),
            "left_arm_gripper_position": ("left_arm_gripper_position", "left_gripper_position", "left_gripper"),
            "right_arm_gripper_position": ("right_arm_gripper_position", "right_gripper_position", "right_gripper"),
            "left_arm_joint_velocity": ("left_arm_joint_velocity", "left_arm_joint_vel", "left_arm_qvel"),
            "right_arm_joint_velocity": ("right_arm_joint_velocity", "right_arm_joint_vel", "right_arm_qvel"),
            "left_arm_ee_pose": ("left_arm_ee_pose", "left_ee_pose"),
            "right_arm_ee_pose": ("right_arm_ee_pose", "right_ee_pose"),
            "chassis_joint_position": ("chassis_joint_position", "chassis_joint_pos"),
            "torso_joint_position": ("torso_joint_position", "torso_joint_pos"),
            "torso_joint_velocity": ("torso_joint_velocity", "torso_joint_vel"),
            "object_pose": ("object_pose", "pose"),
            "goal_pose": ("goal_pose", "target_pose"),
        }
        for canonical, candidates in aliases.items():
            if lowered_key in candidates or lowered_path.endswith(candidates):
                if canonical == "object_pose" and "object" not in lowered_path:
                    continue
                if canonical == "goal_pose" and ("goal" not in lowered_path and "target" not in lowered_path):
                    continue
                return canonical
        if "status" in lowered_key:
            return key
        if "effort" in lowered_key or "torque" in lowered_key:
            return key
        return None

    def _collect_state_chunks(self, value: Any, chunks: list[np.ndarray], prefix: str):
        if value is None:
            return
        if isinstance(value, dict):
            for key in self._ordered_keys(value):
                if self._looks_visual_key(key) or key in ("language_instruction",):
                    continue
                nested_prefix = f"{prefix}.{key}" if prefix else key
                self._collect_state_chunks(value[key], chunks, nested_prefix)
            return
        if isinstance(value, (list, tuple)):
            for idx, nested in enumerate(value):
                self._collect_state_chunks(nested, chunks, f"{prefix}[{idx}]")
            return
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            return
        if array.ndim >= 3:
            return
        chunks.append(array.astype(np.float32).reshape(-1))

    def _extract_image(self, obs) -> np.ndarray:
        if self._uses_triview:
            image = self._extract_triview_image(obs)
            if self.flatten_obs:
                image = image.reshape(-1)
            self._last_image = image.astype(np.uint8, copy=False)
            return self._last_image

        image = None
        upper = (obs or {}).get("upper_body_observations", {})
        if isinstance(upper, dict):
            image = upper.get(self._image_key)
            if image is None:
                for key in sorted(upper):
                    if key.startswith("rgb_"):
                        image = upper[key]
                        break
        if image is None:
            if self.encoder:
                image = self._env.render()
            else:
                image = self._placeholder_image()
        image = self._resize_image(image)
        if self.flatten_obs:
            image = image.reshape(-1)
        self._last_image = image.astype(np.uint8, copy=False)
        return self._last_image

    def _extract_triview_image(self, obs) -> np.ndarray:
        upper = (obs or {}).get("upper_body_observations", {})
        if not isinstance(upper, dict):
            raise KeyError(
                f"GalaxeaManipSim obs is missing upper_body_observations; "
                f"cannot build {TRIVIEW_IMAGE_KEY}."
            )

        missing = [key for key in TRIVIEW_CAMERA_KEYS if key not in upper]
        if missing:
            raise KeyError(
                f"GalaxeaManipSim obs is missing camera(s) {missing}; "
                f"{TRIVIEW_IMAGE_KEY} requires {', '.join(TRIVIEW_CAMERA_KEYS)}."
            )

        images = [self._resize_image(upper[key]) for key in TRIVIEW_CAMERA_KEYS]
        return np.concatenate(images, axis=-1)

    def _placeholder_image(self):
        image_depth = self._image_depth()
        shape = (
            (self._size[0] * self._size[1] * image_depth,)
            if self.flatten_obs
            else self._size + (image_depth,)
        )
        return np.zeros(shape, dtype=np.uint8)

    def _image_depth(self) -> int:
        return 3 * len(TRIVIEW_CAMERA_KEYS) if self._uses_triview else 3

    def _apply_video_view_preset(self):
        if self._video_view_preset == VIDEO_VIEW_DEFAULT:
            return
        if self._video_view_preset != VIDEO_VIEW_ROBOT_FULL_BODY:
            raise ValueError(f"Unsupported GalaxeaManipSim video view preset: {self._video_view_preset!r}")

        default_camera = getattr(self._env, "default_camera", None)
        camera_entity = getattr(default_camera, "entity", None)
        if camera_entity is None or not hasattr(camera_entity, "set_pose"):
            raise RuntimeError("GalaxeaManipSim robot_full_body video preset requires env.default_camera.entity.")

        camera_entity.set_pose(
            self._make_sapien_look_at_pose(
                _ROBOT_FULL_BODY_CAMERA_POSITION,
                _ROBOT_FULL_BODY_CAMERA_TARGET,
            )
        )
        if hasattr(default_camera, "set_fovy"):
            default_camera.set_fovy(np.deg2rad(_ROBOT_FULL_BODY_CAMERA_FOVY_DEG))

    @staticmethod
    def _make_sapien_look_at_pose(position, target):
        import sapien.core as sapien

        position = np.asarray(position, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        forward = target - position
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        left = np.cross(world_up, forward)
        if np.linalg.norm(left) < 1e-6:
            world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            left = np.cross(world_up, forward)
        left = left / (np.linalg.norm(left) + 1e-8)
        up = np.cross(forward, left)
        up = up / (np.linalg.norm(up) + 1e-8)

        rotation = np.stack([forward, left, up], axis=1)
        return sapien.Pose(position.tolist(), GalaxeaSimEnv._rotation_matrix_to_quaternion(rotation).tolist())

    @staticmethod
    def _rotation_matrix_to_quaternion(matrix):
        m = np.asarray(matrix, dtype=np.float64)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
        return quat / (np.linalg.norm(quat) + 1e-8)

    def _resize_image(self, image):
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        array = array[..., :3]
        target_h, target_w = self._size
        if array.shape[:2] == (target_h, target_w):
            return array
        if cv2 is not None and hasattr(cv2, "resize"):
            return cv2.resize(array, (target_w, target_h))
        try:
            from PIL import Image

            return np.asarray(Image.fromarray(array).resize((target_w, target_h)))
        except ImportError:
            y_idx = np.linspace(0, array.shape[0] - 1, target_h).astype(np.int64)
            x_idx = np.linspace(0, array.shape[1] - 1, target_w).astype(np.int64)
            return array[y_idx][:, x_idx]

    def _ordered_keys(self, mapping: dict | Iterable[str]):
        keys = list(mapping.keys() if hasattr(mapping, "keys") else mapping)
        preferred = [key for key in _PREFERRED_STATE_KEYS if key in keys]
        remaining = sorted(key for key in keys if key not in preferred)
        return [*preferred, *remaining]

    @staticmethod
    def _looks_visual_key(key: str) -> bool:
        lowered = str(key).lower()
        return any(token in lowered for token in ("rgb", "image", "depth", "camera", "segmentation"))
