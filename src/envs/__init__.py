import sys
import warnings
from pathlib import Path

_LEGACY_ENV_DIR = Path(__file__).resolve().parents[2] / "envs"
if _LEGACY_ENV_DIR.is_dir():
    legacy_path = str(_LEGACY_ENV_DIR)
    if legacy_path not in __path__:
        __path__.append(legacy_path)

from .wrappers import *  # noqa: F401,F403

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _get_config_attr(config, name, default=None):
    return getattr(config, name, default)


def task_requests_isaaclab(task_name):
    if not isinstance(task_name, str):
        return False
    if task_name.startswith("isaaclab:") or task_name.startswith("isaaclab_"):
        return True
    return False


def should_use_isaaclab_backend(config=None, *, task_name=None, env_backend=None):
    if config is not None:
        if task_name is None:
            task_name = _get_config_attr(config, "task", "")
        if env_backend is None:
            env_backend = _get_config_attr(config, "env_backend", "url")
    if env_backend == "isaaclab":
        return True
    return task_requests_isaaclab(task_name)


def normalize_env_backend_for_task(task_name, env_backend="url"):
    return "isaaclab" if should_use_isaaclab_backend(task_name=task_name, env_backend=env_backend) else env_backend


def _make_isaaclab_env(mode, config):
    from .isaaclab.factory import make_isaaclab_env

    return make_isaaclab_env(mode=mode, config=config)


def _make_official_metra_kitchen_env(config):
    from envs.kitchen.mykitchen import MyKitchenEnv

    env = MyKitchenEnv(
        action_repeat=1,
        width=_get_config_attr(config, "render_size", 64),
        log_per_goal=True,
    )
    env = NormalizeAction(env)
    env = TimeLimit(env, _get_config_attr(config, "time_limit", 0))
    if _get_config_attr(config, "framestack", 1) > 1:
        env = FrameStack(env, k=_get_config_attr(config, "framestack", 1))
    return env


def make_env(mode, config, llm_packages=None):
    if should_use_isaaclab_backend(config):
        env = _make_isaaclab_env(mode, config)
        env = NormalizeAction(env)
        env = TimeLimit(env, _get_config_attr(config, "time_limit", 0))
        if _get_config_attr(config, "framestack", 1) > 1:
            env = FrameStack(env, k=_get_config_attr(config, "framestack", 1))
        return env

    task_name = _get_config_attr(config, "task", "")
    if task_name in ("d4rl_kitchen", "kitchen", "metra_kitchen"):
        return _make_official_metra_kitchen_env(config)

    if "_" not in task_name:
        raise ValueError(
            f"Legacy env task must follow '<suite>_<task>' format, got: {task_name!r}"
        )
    suite, task = task_name.split("_", 1)

    if suite == "dmc":
        from .dmc import DMC, RandomVideoSource  # noqa: F401

        env = DMC(
            task,
            _get_config_attr(config, "action_repeat", 1),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            _get_config_attr(config, "dmc_camera", -1),
            flatten_obs=_get_config_attr(config, "flatten_obs", 1),
            render_image=bool(_get_config_attr(config, "encoder", 1)),
        )
        env = NormalizeAction(env)
    elif suite == "gym":
        from .gym_env import CarRacingEnv

        env = CarRacingEnv(
            task,
            _get_config_attr(config, "seed", 0),
            _get_config_attr(config, "action_repeat", 1),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            flatten_obs=_get_config_attr(config, "flatten_obs", 1),
        )
        env = NormalizeAction(env)
    elif suite == "dmcdriving":
        from .dmc import DMC, RandomVideoSource

        env = DMC(
            task,
            _get_config_attr(config, "action_repeat", 1),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            _get_config_attr(config, "dmc_camera", -1),
        )
        env = NormalizeAction(env)
        env = RandomVideoSource(
            env,
            _get_config_attr(config, "disvideo_dir"),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            total_frames=1000,
            grayscale=_get_config_attr(config, "grayscale", False),
        )
    elif suite == "d4rl":
        sys.path.append("lexa")
        if task == "kitchen":
            return _make_official_metra_kitchen_env(config)
        else:
            raise NotImplementedError(task)
    elif suite == "maze" or suite == "ball":
        from envs.maze.maze_interface import LocoMazeEnv

        env = LocoMazeEnv(
            task_type=suite,
            name=task,
            action_repeat=_get_config_attr(config, "action_repeat", 1),
            size=(_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
        )
        env = NormalizeAction(env)
    elif suite == "metaworld":
        from .metaworld import MetaWorld, ViewMetaWorld, MultiViewMetaWorld  # noqa: F401

        task = "-".join(task.split("_"))
        env = MetaWorld(
            task,
            _get_config_attr(config, "seed", 0),
            _get_config_attr(config, "action_repeat", 1),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            _get_config_attr(config, "camera", "corner"),
            flatten_obs=_get_config_attr(config, "flatten_obs", 1),
        )
        env = NormalizeAction(env)
    elif suite == "mvmetaworld":
        from .metaworld import MultiViewMetaWorld

        task = "-".join(task.split("_"))
        env = MultiViewMetaWorld(
            task,
            _get_config_attr(config, "seed", 0),
            _get_config_attr(config, "action_repeat", 1),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            _get_config_attr(config, "camera_keys"),
        )
        env = NormalizeAction(env)
    elif suite == "viewmetaworld":
        from .metaworld import ViewMetaWorld

        task = "-".join(task.split("_"))
        env = ViewMetaWorld(
            task,
            _get_config_attr(config, "seed", 0),
            _get_config_attr(config, "action_repeat", 1),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            _get_config_attr(config, "camera", "corner"),
            _get_config_attr(config, "viewpoint_mode"),
            _get_config_attr(config, "viewpoint_randomization_type"),
        )
        env = NormalizeAction(env)
    elif suite == "robodesk":
        from .robodesk import RoboDesk

        env = RoboDesk(
            task,
            reward="dense",
            action_repeat=_get_config_attr(config, "action_repeat", 1),
            render_size=(_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            time_limit=_get_config_attr(config, "time_limit", 200),
        )
        env = NormalizeAction(env)
        return env
    elif suite == "bigym":
        from .bigym_env import BiGymEnv

        env = BiGymEnv(
            task,
            _get_config_attr(config, "seed", 0),
            _get_config_attr(config, "action_repeat", 1),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            _get_config_attr(config, "camera", "corner"),
            flatten_obs=_get_config_attr(config, "flatten_obs", 1),
        )
        env = NormalizeAction(env)
    elif suite == "galaxea":
        from .galaxea_sim import GalaxeaSimEnv

        env = GalaxeaSimEnv(
            task,
            _get_config_attr(config, "seed", 0),
            _get_config_attr(config, "action_repeat", 1),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            flatten_obs=_get_config_attr(config, "flatten_obs", 0),
            encoder=_get_config_attr(config, "encoder", 0),
            headless=_get_config_attr(config, "galaxea_sim_headless", 1),
            obs_mode=_get_config_attr(config, "galaxea_sim_obs_mode", None),
            image_key=_get_config_attr(config, "galaxea_sim_image_key", "rgb_head"),
            video_view_preset=_get_config_attr(config, "galaxea_sim_video_view_preset", "default"),
            controller_type=_get_config_attr(config, "galaxea_sim_controller_type", "bimanual_joint_position"),
        )
        env = NormalizeAction(env)
    elif suite == "carla":
        from .carla import Carla

        env = Carla(
            ports=[_get_config_attr(config, "carla_port"), _get_config_attr(config, "carla_port") + 10],
            fix_weather=task,
            frame_skip=_get_config_attr(config, "action_repeat", 1),
            **_get_config_attr(config, "carla"),
        )
        env = NormalizeAction(env)
    elif suite == "dmcr":
        from .dmc_remastered import DMCRemastered

        env = DMCRemastered(
            task,
            _get_config_attr(config, "action_repeat", 1),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            _get_config_attr(config, "dmc_camera", -1),
            _get_config_attr(config, "dmcr_vary"),
        )
        env = NormalizeAction(env)
    elif suite == "minecraft":
        from .minecraft import Minecraft

        env = Minecraft(
            task,
            _get_config_attr(config, "seed", 0),
            _get_config_attr(config, "action_repeat", 1),
            (_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            _get_config_attr(config, "sim_size"),
            _get_config_attr(config, "eval_hard_reset_every") if mode == "eval" else _get_config_attr(config, "hard_reset_every"),
        )
        env = MultiDiscreteAction(env)
    elif suite == "debug":
        env = DummyEnv(
            name=task,
            seed=_get_config_attr(config, "seed", 0),
            action_repeat=_get_config_attr(config, "action_repeat", 1),
            size=(_get_config_attr(config, "render_size", 64), _get_config_attr(config, "render_size", 64)),
            flatten_obs=_get_config_attr(config, "flatten_obs", 1),
        )
        env = NormalizeAction(env)
    else:
        raise NotImplementedError(suite)

    env = TimeLimit(env, _get_config_attr(config, "time_limit", 0))
    if _get_config_attr(config, "framestack", 1) > 1:
        env = FrameStack(env, k=_get_config_attr(config, "framestack", 1))
    return env
