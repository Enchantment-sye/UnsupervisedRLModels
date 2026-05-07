from __future__ import annotations


VIEWER_PRESET_INHERIT = "inherit"
VIEWER_PRESET_ANYMAL_CLOSE = "anymal_close_follow"
VIEWER_PRESET_HUMANOID_CLOSE = "humanoid_close_follow"
VIEWER_PRESET_GALAXEA_WORKSTATION = "galaxea_workstation_follow"
VIEWER_PRESET_PANORAMA_FIXED = "panorama_fixed"


def _viewer_settings(
    *,
    eye,
    lookat,
    origin_type: str,
    asset_name: str = "robot",
    env_index: int = 0,
    body_name=None,
):
    return {
        "eye": tuple(float(x) for x in eye),
        "lookat": tuple(float(x) for x in lookat),
        "origin_type": origin_type,
        "asset_name": asset_name,
        "env_index": int(env_index),
        "body_name": body_name,
    }


ANYMAL_CLOSE_VIEWER_SETTINGS = _viewer_settings(
    eye=(1.55, 1.35, 0.95),
    lookat=(0.0, 0.0, 0.38),
    origin_type="asset_root",
)

HUMANOID_CLOSE_VIEWER_SETTINGS = _viewer_settings(
    eye=(2.1, 1.8, 1.55),
    lookat=(0.0, 0.0, 0.95),
    origin_type="asset_root",
)

GALAXEA_WORKSTATION_VIEWER_SETTINGS = _viewer_settings(
    eye=(1.55, 1.25, 1.35),
    lookat=(0.15, 0.0, 0.82),
    origin_type="asset_root",
)

GALAXEA_PANORAMA_FIXED_VIEWER_SETTINGS = _viewer_settings(
    eye=(1.95, 1.45, 2.35),
    lookat=(0.55, 0.0, 1.02),
    origin_type="world",
)


def _settings_copy(settings):
    if settings is None:
        return None
    copied = dict(settings)
    if copied.get("eye") is not None:
        copied["eye"] = tuple(copied["eye"])
    if copied.get("lookat") is not None:
        copied["lookat"] = tuple(copied["lookat"])
    return copied


def _disable_command_debug_vis(env_cfg):
    commands = getattr(env_cfg, "commands", None)
    if commands is None:
        return env_cfg

    for name in dir(commands):
        if name.startswith("_"):
            continue
        try:
            term_cfg = getattr(commands, name)
        except Exception:
            continue
        if hasattr(term_cfg, "debug_vis"):
            try:
                setattr(term_cfg, "debug_vis", False)
            except Exception:
                pass
    return env_cfg


def _set_cameras_enabled(env_cfg, enabled: bool):
    value = bool(enabled)
    _set_if_present(env_cfg, "scene.enable_cameras", value)
    _set_if_present(env_cfg, "sim.enable_cameras", value)
    _set_if_present(env_cfg, "sim.render.enable_cameras", value)
    if hasattr(env_cfg, "enable_cameras"):
        try:
            setattr(env_cfg, "enable_cameras", value)
        except Exception:
            pass
    return env_cfg


def _set_default_camera_keys(env_cfg, primary_key: str = "front_rgb"):
    # Galaxea tasks often expose camera observations under explicit keys such as
    # front_rgb / head_rgb instead of plain rgb.
    for attr_path in (
        "observations.policy.camera_key",
        "observations.policy.image_key",
        "observations.camera_key",
        "observations.image_key",
        "camera_key",
        "image_key",
    ):
        _set_if_present(env_cfg, attr_path, primary_key)
    return env_cfg


def _configure_galaxea_training_cameras(env_cfg, request):
    if env_cfg is None:
        return env_cfg
    if not getattr(request, "encoder", 0):
        return env_cfg
    if str(getattr(request, "mode", "train")).lower() != "train":
        return env_cfg
    if str(getattr(request, "image_source", "")).lower() != "camera":
        return env_cfg

    render_size = int(getattr(request, "render_size", 64))
    for attr_path in (
        "front_camera_cfg.height",
        "left_wrist_camera_cfg.height",
        "right_wrist_camera_cfg.height",
    ):
        _set_if_present(env_cfg, attr_path, render_size)
    for attr_path in (
        "front_camera_cfg.width",
        "left_wrist_camera_cfg.width",
        "right_wrist_camera_cfg.width",
    ):
        _set_if_present(env_cfg, attr_path, render_size)

    _set_if_present(env_cfg, "front_camera_cfg.data_types", ["rgb"])
    _set_if_present(env_cfg, "left_wrist_camera_cfg.data_types", ["rgb"])
    _set_if_present(env_cfg, "right_wrist_camera_cfg.data_types", ["rgb"])

    setattr(env_cfg, "_metra_front_camera_only", True)
    setattr(env_cfg, "_metra_rgb_only", True)
    setattr(env_cfg, "_metra_camera_native_size", render_size)
    return env_cfg


def _set_if_present(root, attr_path: str, value) -> bool:
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


def configure_viewer_camera(
    env_cfg,
    *,
    eye,
    lookat,
    origin_type: str = "asset_root",
    asset_name: str = "robot",
    env_index: int = 0,
):
    """Override the Isaac Lab viewer camera from metra without touching upstream task files."""
    if env_cfg is None or not hasattr(env_cfg, "viewer"):
        return env_cfg

    viewer = env_cfg.viewer
    _set_if_present(viewer, "eye", tuple(float(x) for x in eye))
    _set_if_present(viewer, "lookat", tuple(float(x) for x in lookat))
    _set_if_present(viewer, "origin_type", origin_type)
    _set_if_present(viewer, "asset_name", asset_name)
    _set_if_present(viewer, "env_index", int(env_index))
    _disable_command_debug_vis(env_cfg)
    return env_cfg


def apply_viewer_settings(env_cfg, settings):
    if settings is None:
        return env_cfg
    return configure_viewer_camera(
        env_cfg,
        eye=settings["eye"],
        lookat=settings["lookat"],
        origin_type=settings.get("origin_type", "asset_root"),
        asset_name=settings.get("asset_name", "robot"),
        env_index=settings.get("env_index", 0),
    )


def anymal_close_view_cfg_builder(env_cfg, request):
    # Keep the camera close to the robot root so render() frames are not dominated by empty terrain.
    return apply_viewer_settings(env_cfg, ANYMAL_CLOSE_VIEWER_SETTINGS)


def humanoid_close_view_cfg_builder(env_cfg, request):
    return apply_viewer_settings(env_cfg, HUMANOID_CLOSE_VIEWER_SETTINGS)


def galaxea_workstation_cfg_builder(env_cfg, request):
    env_cfg = apply_viewer_settings(env_cfg, GALAXEA_WORKSTATION_VIEWER_SETTINGS)
    env_cfg = _set_cameras_enabled(env_cfg, request.enable_cameras)
    if request.encoder:
        env_cfg = _set_default_camera_keys(env_cfg, primary_key=request.camera_key or "front_rgb")
        env_cfg = _configure_galaxea_training_cameras(env_cfg, request)
    return env_cfg


def get_default_viewer_preset_name(task_spec) -> str:
    cfg_builder = getattr(task_spec, "cfg_builder", None)
    if cfg_builder is galaxea_workstation_cfg_builder:
        return VIEWER_PRESET_GALAXEA_WORKSTATION
    if cfg_builder is anymal_close_view_cfg_builder:
        return VIEWER_PRESET_ANYMAL_CLOSE
    if cfg_builder is humanoid_close_view_cfg_builder:
        return VIEWER_PRESET_HUMANOID_CLOSE
    return VIEWER_PRESET_INHERIT


def resolve_viewer_preset_settings(task_spec, preset_name: str):
    resolved_preset = (preset_name or VIEWER_PRESET_INHERIT).lower()
    if resolved_preset == VIEWER_PRESET_INHERIT:
        resolved_preset = get_default_viewer_preset_name(task_spec)

    if resolved_preset == VIEWER_PRESET_ANYMAL_CLOSE:
        return resolved_preset, _settings_copy(ANYMAL_CLOSE_VIEWER_SETTINGS)
    if resolved_preset == VIEWER_PRESET_HUMANOID_CLOSE:
        return resolved_preset, _settings_copy(HUMANOID_CLOSE_VIEWER_SETTINGS)
    if resolved_preset == VIEWER_PRESET_GALAXEA_WORKSTATION:
        return resolved_preset, _settings_copy(GALAXEA_WORKSTATION_VIEWER_SETTINGS)
    if resolved_preset == VIEWER_PRESET_PANORAMA_FIXED:
        env_id = getattr(task_spec, "env_id", "") or ""
        if env_id.startswith("Isaac-R1-"):
            return resolved_preset, _settings_copy(GALAXEA_PANORAMA_FIXED_VIEWER_SETTINGS)
        return get_default_viewer_preset_name(task_spec), None
    return resolved_preset, None


def apply_task_viewer_preset(env_cfg, task_spec, preset_name: str):
    _, settings = resolve_viewer_preset_settings(task_spec, preset_name)
    return apply_viewer_settings(env_cfg, settings)
