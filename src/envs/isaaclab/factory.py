import importlib
from dataclasses import dataclass
from typing import Optional

from .galaxea_overlay import (
    GALAXEA_RUNTIME_PACKAGES,
    GALAXEA_TASK_MODULE_CANDIDATES,
    activate_local_galaxea_overlay,
    ensure_galaxea_legacy_runtime,
    maybe_enable_galaxea_optional_extension_from_error,
)


@dataclass(frozen=True)
class IsaacLabEnvRequest:
    env_id: str
    mode: str
    encoder: int
    render_size: int
    flatten_obs: bool
    seed: int
    device: Optional[str]
    num_envs: int
    headless: bool
    enable_cameras: bool
    render_mode: Optional[str]
    image_source: str
    camera_key: Optional[str]
    video_source: str
    video_viewer_preset: str
    env_backend: str = "isaaclab"

    @property
    def wants_camera(self) -> bool:
        return self.image_source in ("auto", "camera") and self.enable_cameras

    @property
    def wants_render(self) -> bool:
        return self.image_source in ("auto", "render") or not self.encoder

    @property
    def wants_video_render(self) -> bool:
        return self.video_source == "render"


def _get_config_attr(config, name, default=None):
    return getattr(config, name, default)


def _resolve_task_identifier(config) -> str:
    task_name = _get_config_attr(config, "task", "")
    explicit_task = _get_config_attr(config, "isaaclab_task", "") or ""
    if explicit_task:
        return explicit_task
    if isinstance(task_name, str) and task_name.startswith("isaaclab:"):
        return task_name.split(":", 1)[1]
    if isinstance(task_name, str) and task_name.startswith("isaaclab_"):
        return task_name
    return ""


def _resolve_launcher_device(config) -> str:
    device = _get_config_attr(config, "device", None)
    if device in (None, ""):
        use_gpu = int(_get_config_attr(config, "use_gpu", 1) or 0)
        device = "cuda:0" if use_gpu else "cpu"
    device = str(device)
    if device == "cuda":
        return "cuda:0"
    return device


def resolve_isaaclab_request(config, mode: str) -> IsaacLabEnvRequest:
    from .registry import get_task_spec

    task_identifier = _resolve_task_identifier(config)
    if not task_identifier:
        raise ValueError(
            "Isaac Lab backend selected but no env id was provided. "
            "Use --task isaaclab_cartpole style names, or advanced overrides such as "
            "--isaaclab-task ENV_ID / --task isaaclab:ENV_ID."
        )
    task_spec = get_task_spec(task_identifier)

    image_source = _get_config_attr(config, "isaaclab_image_source", None)
    if image_source in (None, ""):
        image_source = (
            task_spec.default_image_source_encoder1
            if int(_get_config_attr(config, "encoder", 0))
            else task_spec.default_image_source_encoder0
        )
    image_source = str(image_source).lower()
    if image_source not in ("auto", "render", "camera"):
        raise ValueError(f"Unsupported isaaclab_image_source={image_source!r}")

    video_source = str(_get_config_attr(config, "isaaclab_video_source", "observation") or "observation").lower()
    if video_source not in ("observation", "render"):
        raise ValueError(f"Unsupported isaaclab_video_source={video_source!r}")

    video_viewer_preset = str(
        _get_config_attr(config, "isaaclab_video_viewer_preset", "inherit") or "inherit"
    ).lower()
    if video_viewer_preset not in ("inherit", "panorama_fixed"):
        raise ValueError(
            f"Unsupported isaaclab_video_viewer_preset={video_viewer_preset!r}"
        )

    num_envs = int(_get_config_attr(config, "isaaclab_num_envs", 1) or 1)
    if num_envs < 1:
        raise ValueError("isaaclab_num_envs must be >= 1")

    encoder = int(_get_config_attr(config, "encoder", 0))
    render_mode = _get_config_attr(config, "isaaclab_render_mode", "rgb_array")
    if render_mode is not None and str(render_mode).lower() in ("none", "off", "null"):
        render_mode = None
    headless = bool(int(_get_config_attr(config, "isaaclab_headless", 1)))

    enable_cameras = _get_config_attr(config, "isaaclab_enable_cameras", None)
    if enable_cameras is None:
        wants_render_rgb = image_source in ("auto", "render") or not encoder
        # Isaac Lab commonly requires cameras to be enabled for headless rgb_array rendering.
        enable_cameras = int(bool(encoder or (headless and render_mode is not None and wants_render_rgb)))
    if image_source == "camera":
        enable_cameras = 1

    return IsaacLabEnvRequest(
        env_id=task_spec.env_id,
        mode=mode,
        encoder=encoder,
        render_size=int(_get_config_attr(config, "render_size", 64)),
        flatten_obs=bool(_get_config_attr(config, "flatten_obs", 1)),
        seed=int(_get_config_attr(config, "seed", 0)),
        device=_resolve_launcher_device(config),
        num_envs=num_envs,
        headless=headless,
        enable_cameras=bool(int(enable_cameras)),
        render_mode=render_mode,
        image_source=image_source,
        camera_key=_get_config_attr(config, "isaaclab_camera_key", None) or None,
        video_source=video_source,
        video_viewer_preset=video_viewer_preset,
    )


def _set_attr_path(root, attr_path: str, value):
    if root is None:
        return False
    target = root
    parts = attr_path.split(".")
    for name in parts[:-1]:
        if not hasattr(target, name):
            return False
        target = getattr(target, name)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        return False
    setattr(target, leaf, value)
    return True


_GALAXEA_LEGACY_CFG_ATTR_OVERRIDES = {
    # Isaac Sim 5.1 no longer exposes the PhysX material ImprovePatchFriction
    # attribute, but the older Galaxea_Lab task stack still sets it by default.
    # Setting it to None makes the legacy spawner skip the unsupported field.
    "improve_patch_friction": None,
}


def _sanitize_galaxea_legacy_cfg_for_runtime(root, _visited=None) -> int:
    if root is None:
        return 0

    if _visited is None:
        _visited = set()

    root_id = id(root)
    if root_id in _visited:
        return 0
    _visited.add(root_id)

    changes = 0
    if isinstance(root, dict):
        for value in root.values():
            changes += _sanitize_galaxea_legacy_cfg_for_runtime(value, _visited)
        return changes

    if isinstance(root, (list, tuple, set, frozenset)):
        for value in root:
            changes += _sanitize_galaxea_legacy_cfg_for_runtime(value, _visited)
        return changes

    for attr_name, override_value in _GALAXEA_LEGACY_CFG_ATTR_OVERRIDES.items():
        if hasattr(root, attr_name):
            try:
                current_value = getattr(root, attr_name)
                if current_value != override_value:
                    setattr(root, attr_name, override_value)
                    changes += 1
            except Exception:
                pass

    state = getattr(root, "__dict__", None)
    if isinstance(state, dict):
        for value in state.values():
            if isinstance(value, (str, bytes, int, float, bool, type(None))):
                continue
            if callable(value):
                continue
            changes += _sanitize_galaxea_legacy_cfg_for_runtime(value, _visited)

    return changes


def _should_retry_gym_make_without_kwargs(exc: TypeError) -> bool:
    message = str(exc)
    if not any(
        needle in message
        for needle in (
            "unexpected keyword argument",
            "unexpected keyword arguments",
            "takes no keyword arguments",
        )
    ):
        return False
    return "cfg" in message or "render_mode" in message


def _ensure_env_id_registered(env_id: str):
    module_names = []
    if "Unitree-A1" in env_id:
        module_names.append("isaaclab_tasks.manager_based.locomotion.velocity.config.a1")
    elif "Unitree-Go1" in env_id:
        module_names.append("isaaclab_tasks.manager_based.locomotion.velocity.config.go1")
    elif "Unitree-Go2" in env_id:
        module_names.append("isaaclab_tasks.manager_based.locomotion.velocity.config.go2")
    elif env_id.startswith("Isaac-Velocity-") and "-H1" in env_id:
        module_names.append("isaaclab_tasks.manager_based.locomotion.velocity.config.h1")
    elif env_id.startswith("Isaac-Velocity-") and "-G1" in env_id:
        module_names.append("isaaclab_tasks.manager_based.locomotion.velocity.config.g1")
    elif env_id.startswith("Isaac-PickPlace-G1-InspireFTP"):
        module_names.append("isaaclab_tasks.manager_based.manipulation.pick_place")
    elif env_id.startswith("Isaac-PickPlace-Locomanipulation-G1") or env_id.startswith("Isaac-PickPlace-FixedBaseUpperBodyIK-G1"):
        module_names.append("isaaclab_tasks.manager_based.locomanipulation.pick_place")
    elif env_id.startswith("Isaac-R1-"):
        activate_local_galaxea_overlay(strict=True)
        module_names.extend(GALAXEA_TASK_MODULE_CANDIDATES)

    last_error = None
    import_errors = []
    attempted_optional_extensions = set()
    for module_name in module_names:
        while True:
            try:
                importlib.import_module(module_name)
                return
            except ImportError as exc:
                last_error = exc
                enabled_extension = None
                if env_id.startswith("Isaac-R1-"):
                    try:
                        enabled_extension = maybe_enable_galaxea_optional_extension_from_error(
                            exc,
                            attempted_extensions=attempted_optional_extensions,
                        )
                    except Exception as compat_exc:
                        import_errors.append(
                            f"{module_name}: optional legacy extension activation failed with "
                            f"{type(compat_exc).__name__}: {compat_exc}"
                        )
                        break
                if enabled_extension is not None:
                    continue
                import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
                break

    if module_names and last_error is not None:
        if env_id.startswith("Isaac-R1-"):
            detail = "; ".join(import_errors)
            optional_detail = ""
            if attempted_optional_extensions:
                optional_detail = (
                    " Optional compatibility extensions attempted during import: "
                    f"{', '.join(sorted(attempted_optional_extensions))}."
                )
            raise ImportError(
                "Failed to register the Galaxea R1 Isaac Lab task. "
                "The local /home/shangyy/Galaxea_Lab overlay was found, but its legacy task module "
                "could not be imported into the current Isaac Sim runtime. "
                "This usually means the deprecated omni.isaac.core compatibility extensions were not "
                f"enabled after AppLauncher startup. Candidate import errors: {detail}.{optional_detail}"
            ) from last_error
        raise last_error


def _import_optional_attr(module_names, attr_name):
    last_error = None
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ImportError(f"Could not import {attr_name} from any module.")


def _load_cfg_from_registry(env_id: str, request: IsaacLabEnvRequest):
    if env_id.startswith("Isaac-R1-"):
        ensure_galaxea_legacy_runtime(strict=True)
    _ensure_env_id_registered(env_id)

    try:
        load_cfg_from_registry = _import_optional_attr(
            (
                "isaaclab_tasks.utils.parse_cfg",
                "isaaclab_tasks.utils",
                "omni.isaac.lab_tasks.utils.parse_cfg",
                "omni.isaac.lab_tasks.utils",
            ),
            "load_cfg_from_registry",
        )
    except (ImportError, AttributeError):
        try:
            parse_env_cfg = _import_optional_attr(
                (
                    "isaaclab_tasks.utils.parse_cfg",
                    "isaaclab_tasks.utils",
                    "omni.isaac.lab_tasks.utils.parse_cfg",
                    "omni.isaac.lab_tasks.utils",
                ),
                "parse_env_cfg",
            )
        except (ImportError, AttributeError):
            return None
        return parse_env_cfg(
            task_name=env_id,
            use_gpu=str(request.device).startswith("cuda"),
            num_envs=request.num_envs,
            use_fabric=True,
        )

    return load_cfg_from_registry(env_id, "env_cfg_entry_point")


def _ensure_task_runtime_available():
    module_names = ("isaaclab_tasks", *GALAXEA_RUNTIME_PACKAGES)
    seen = set()
    last_error = None
    for module_name in module_names:
        if module_name in seen:
            continue
        seen.add(module_name)
        try:
            importlib.import_module(module_name)
            return
        except ImportError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ImportError("Isaac Lab task runtime is unavailable.")


def _apply_common_cfg_overrides(env_cfg, request):
    if env_cfg is None:
        return None

    _set_attr_path(env_cfg, "scene.num_envs", request.num_envs)
    _set_attr_path(env_cfg, "scene.enable_cameras", request.enable_cameras)
    _set_attr_path(env_cfg, "sim.device", request.device)
    _set_attr_path(env_cfg, "sim.enable_cameras", request.enable_cameras)
    _set_attr_path(env_cfg, "sim.render.enable_cameras", request.enable_cameras)
    if hasattr(env_cfg, "seed"):
        setattr(env_cfg, "seed", request.seed)
    if hasattr(env_cfg, "decimation") and getattr(env_cfg, "decimation", None) is None:
        setattr(env_cfg, "decimation", 1)
    if hasattr(env_cfg, "viewer") and hasattr(env_cfg.viewer, "resolution"):
        try:
            env_cfg.viewer.resolution = (request.render_size, request.render_size)
        except Exception:
            pass
    if hasattr(env_cfg, "enable_cameras"):
        try:
            setattr(env_cfg, "enable_cameras", request.enable_cameras)
        except Exception:
            pass
    return env_cfg


def make_isaaclab_env(mode, config):
    request = resolve_isaaclab_request(config, mode)

    from .floor_gradient import apply_floor_gradient_overlay
    from .launcher import acquire_isaaclab_app, run_with_isaaclab_startup_capture
    from .registry import get_task_spec
    from .scene_overrides import apply_white_dome_light_override

    task_spec = get_task_spec(request.env_id)
    if request.image_source == "camera" and not task_spec.supports_camera_obs:
        raise ValueError(
            f"{request.env_id} does not advertise camera observations; "
            "use --isaaclab-image-source render or auto instead."
        )
    if request.image_source == "render" and not task_spec.supports_render_rgb:
        raise ValueError(
            f"{request.env_id} does not advertise rgb rendering support; "
            "use --isaaclab-image-source camera or auto instead."
        )
    if request.encoder and request.image_source == "auto":
        if not task_spec.supports_camera_obs and not task_spec.supports_render_rgb:
            raise ValueError(
                f"{request.env_id} cannot satisfy encoder=1 because neither camera obs nor rgb render is available."
            )

    if request.env_id.startswith("Isaac-R1-"):
        activate_local_galaxea_overlay(strict=True)

    acquire_isaaclab_app(request)

    if request.env_id.startswith("Isaac-R1-"):
        ensure_galaxea_legacy_runtime(strict=True)

    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required for Isaac Lab backend.") from exc

    try:
        _ensure_task_runtime_available()
    except ImportError as exc:
        raise ImportError(
            "Isaac Lab backend requested but neither isaaclab_tasks nor omni.isaac.lab_tasks is installed."
        ) from exc

    env_cfg = run_with_isaaclab_startup_capture("load_cfg", _load_cfg_from_registry, request.env_id, request)
    env_cfg = _apply_common_cfg_overrides(env_cfg, request)
    if task_spec.cfg_builder is not None:
        env_cfg = task_spec.cfg_builder(env_cfg, request)
    if request.env_id.startswith("Isaac-R1-"):
        _sanitize_galaxea_legacy_cfg_for_runtime(env_cfg)

    gym_kwargs = {}
    if env_cfg is not None:
        gym_kwargs["cfg"] = env_cfg
    if request.render_mode is not None and task_spec.supports_render_rgb:
        gym_kwargs["render_mode"] = request.render_mode

    try:
        env = run_with_isaaclab_startup_capture("gym_make", gym.make, request.env_id, **gym_kwargs)
    except TypeError as exc:
        if not _should_retry_gym_make_without_kwargs(exc):
            raise
        env = run_with_isaaclab_startup_capture("gym_make_fallback", gym.make, request.env_id)

    overlay_info = run_with_isaaclab_startup_capture("floor_gradient", apply_floor_gradient_overlay, env, task_spec)
    dome_light_info = run_with_isaaclab_startup_capture("dome_light", apply_white_dome_light_override, env, task_spec)
    wrapped_env = task_spec.adapter_cls(env=env, task_spec=task_spec, request=request)
    if overlay_info is not None:
        setattr(wrapped_env, "_metra_floor_gradient_overlay", overlay_info)
    if dome_light_info is not None:
        setattr(wrapped_env, "_metra_dome_light_override", dome_light_info)
    return wrapped_env
