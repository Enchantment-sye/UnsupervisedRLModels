# # from .od_mujoco import OffDynamicsMujocoEnv
# # from .od_envs import *
# from .rlbench import RLBench

from pathlib import Path
import sys

from .wrappers import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
_SRC_ENV_DIR = Path(__file__).resolve().parents[1] / "src" / "envs"
if _SRC_ENV_DIR.is_dir():
    src_env_path = str(_SRC_ENV_DIR)
    if src_env_path not in __path__:
        __path__.append(src_env_path)


def _get_config_attr(config, name, default=None):
    return getattr(config, name, default)


def task_requests_isaaclab(task_name):
    if not isinstance(task_name, str):
        return False
    return task_name.startswith("isaaclab:") or task_name.startswith("isaaclab_")


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
        task_name = _get_config_attr(config, "task", "")
        if task_name in ("d4rl_kitchen", "kitchen", "metra_kitchen"):
            return _make_official_metra_kitchen_env(config)

        suite, task = task_name.split("_", 1)
        if suite == "dmc":
            from .dmc import DMC, RandomVideoSource
            env = DMC(
                task, config.action_repeat, (config.render_size, config.render_size), config.dmc_camera, flatten_obs=config.flatten_obs,
                render_image=bool(getattr(config, "encoder", 1)))
            env = NormalizeAction(env)
        elif suite == "gym":
            from .gym_env import CarRacingEnv
            env = CarRacingEnv(
                task, config.seed, config.action_repeat, (config.render_size, config.render_size), flatten_obs=config.flatten_obs
            )
            env = NormalizeAction(env)
        elif suite == "dmcdriving":
            from .dmc import DMC, RandomVideoSource
            env = DMC(
                task, config.action_repeat, (config.render_size, config.render_size), config.dmc_camera)
            env = NormalizeAction(env)
            env = RandomVideoSource(env, config.disvideo_dir, (config.render_size, config.render_size), total_frames=1000, grayscale=config.grayscale)
        elif suite == "d4rl":
            sys.path.append('lexa')
            if task == "kitchen":
                return _make_official_metra_kitchen_env(config)
        elif suite == "maze" or suite == "ball":
            from envs.maze.maze_interface import  LocoMazeEnv
            env = LocoMazeEnv(task_type=suite, name=task, action_repeat=config.action_repeat, size = (config.render_size, config.render_size))
            env = NormalizeAction(env)
        elif suite == "metaworld":
            from .metaworld import MetaWorld, ViewMetaWorld, MultiViewMetaWorld
            task = "-".join(task.split("_"))
            env = MetaWorld(
                task,
                config.seed,
                config.action_repeat,
                (config.render_size, config.render_size),
                config.camera,
                flatten_obs=config.flatten_obs
            )
            env = NormalizeAction(env)
        elif suite == "mvmetaworld":
            task = "-".join(task.split("_"))
            env = MultiViewMetaWorld(
                task,
                config.seed,
                config.action_repeat,
                (config.render_size, config.render_size),
                config.camera_keys,
            )
            env = NormalizeAction(env)
        elif suite == "viewmetaworld":
            task = "-".join(task.split("_"))
            env = ViewMetaWorld(
                task,
                config.seed,
                config.action_repeat,
                (config.render_size, config.render_size),
                config.camera,
                config.viewpoint_mode,
                config.viewpoint_randomization_type
            )
            env = NormalizeAction(env)
        elif suite == "robodesk":
            from .robodesk import RoboDesk
            env = RoboDesk(
                task,
                reward='dense',# if mode == 'train' else 'success',
                action_repeat = config.action_repeat,
                render_size = (config.render_size, config.render_size),
                time_limit = config.time_limit,
            )
            env = NormalizeAction(env)
            return env
        elif suite == "bigym":
            from .bigym_env import BiGymEnv
            env = BiGymEnv(
                task,
                config.seed,
                config.action_repeat,
                (config.render_size, config.render_size),
                config.camera,
                flatten_obs=config.flatten_obs
            )
            env = NormalizeAction(env)
        elif suite == "galaxea":
            from .galaxea_sim import GalaxeaSimEnv
            env = GalaxeaSimEnv(
                task,
                config.seed,
                config.action_repeat,
                (config.render_size, config.render_size),
                flatten_obs=getattr(config, "flatten_obs", 0),
                encoder=getattr(config, "encoder", 0),
                headless=getattr(config, "galaxea_sim_headless", 1),
                obs_mode=getattr(config, "galaxea_sim_obs_mode", None),
                image_key=getattr(config, "galaxea_sim_image_key", "rgb_head"),
                video_view_preset=getattr(config, "galaxea_sim_video_view_preset", "default"),
                controller_type=getattr(config, "galaxea_sim_controller_type", "bimanual_joint_position"),
            )
            env = NormalizeAction(env)
        elif suite == "carla":
            from .carla import Carla
            env = Carla(ports=[config.carla_port, config.carla_port + 10],
                             fix_weather=task, frame_skip=config.action_repeat, **config.carla)
            env = NormalizeAction(env)
        elif suite == "dmcr":
            from .dmc_remastered import DMCRemastered
            env = DMCRemastered(task, config.action_repeat, (config.render_size, config.render_size),
                                     config.dmc_camera, config.dmcr_vary)
            env = NormalizeAction(env)
        elif suite == "minecraft":
            from .minecraft import Minecraft
            env = Minecraft(task, config.seed, config.action_repeat, (config.render_size, config.render_size), config.sim_size, config.eval_hard_reset_every if mode == "eval" else config.hard_reset_every)
            env = MultiDiscreteAction(env)
        elif suite == "debug":
            env = DummyEnv(name=task, seed=config.seed, action_repeat=config.action_repeat, size=(config.render_size, config.render_size), flatten_obs=config.flatten_obs)
            env = NormalizeAction(env)
        else:
            raise NotImplementedError(suite)
        # if config.llm_acs_wrap:
        #     env = LLMActionWrapper(env, config, llm_packages, gap_frame=4)
        env = TimeLimit(env, config.time_limit)
        if config.framestack > 1:
            env = FrameStack(env, k=config.framestack)
        return env
