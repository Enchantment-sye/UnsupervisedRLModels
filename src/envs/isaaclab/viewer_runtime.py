from __future__ import annotations

from contextlib import contextmanager
import numpy as np

from .cfg_builders import VIEWER_PRESET_INHERIT, resolve_viewer_preset_settings

RENDER_STABILITY_MAX_ATTEMPTS = 10
RENDER_STABILITY_REQUIRED_TRANSITIONS = 2
RENDER_STABILITY_MAD_THRESHOLD = 6.0


def _unwrap_isaac_env(env):
    candidate = getattr(env, "_env", env)
    return getattr(candidate, "unwrapped", candidate)


def _viewer_controller_for(env):
    isaac_env = _unwrap_isaac_env(env)
    return getattr(isaac_env, "viewport_camera_controller", None)


def _clone_frame(frame):
    if frame is None:
        return None
    return np.asarray(frame).copy()


def _is_invalid_render_frame(frame) -> bool:
    if frame is None:
        return True
    array = np.asarray(frame)
    return array.size == 0 or not np.any(array)


def _frame_mad(previous_frame, current_frame) -> float:
    previous = np.asarray(previous_frame, dtype=np.float32)
    current = np.asarray(current_frame, dtype=np.float32)
    return float(np.mean(np.abs(current - previous)))


def warmup_render_capture(
    capture_frame_fn,
    *,
    max_attempts: int = RENDER_STABILITY_MAX_ATTEMPTS,
    required_stable_transitions: int = RENDER_STABILITY_REQUIRED_TRANSITIONS,
    stable_mad_threshold: float = RENDER_STABILITY_MAD_THRESHOLD,
):
    last_valid_frame = None
    previous_valid_frame = None
    stable_transitions = 0

    for _ in range(max_attempts):
        frame = capture_frame_fn()
        if _is_invalid_render_frame(frame):
            continue

        frame = _clone_frame(frame)
        last_valid_frame = _clone_frame(frame)
        if previous_valid_frame is None:
            previous_valid_frame = frame
            continue

        if _frame_mad(previous_valid_frame, frame) <= stable_mad_threshold:
            stable_transitions += 1
        else:
            stable_transitions = 0
        previous_valid_frame = frame

        if stable_transitions >= required_stable_transitions:
            return frame, True

    return last_valid_frame, False


def apply_runtime_viewer_preset(env, preset_name: str) -> bool:
    task_spec = getattr(env, "_task_spec", None)
    resolved_preset, settings = resolve_viewer_preset_settings(task_spec, preset_name)
    if settings is None:
        return False

    controller = _viewer_controller_for(env)
    isaac_env = _unwrap_isaac_env(env)
    origin_type = settings.get("origin_type", "asset_root")

    if controller is not None:
        if origin_type == "world" and hasattr(controller, "update_view_to_world"):
            controller.update_view_to_world()
        elif origin_type == "env" and hasattr(controller, "update_view_to_env"):
            controller.update_view_to_env()
        elif origin_type == "asset_body" and hasattr(controller, "update_view_to_asset_body"):
            controller.update_view_to_asset_body(
                settings.get("asset_name", "robot"),
                settings.get("body_name"),
            )
        elif origin_type == "asset_root" and hasattr(controller, "update_view_to_asset_root"):
            controller.update_view_to_asset_root(settings.get("asset_name", "robot"))
        elif hasattr(controller, "update_view_to_world"):
            controller.update_view_to_world()

        if hasattr(controller, "update_view_location"):
            controller.update_view_location(eye=settings["eye"], lookat=settings["lookat"])
    else:
        sim = getattr(isaac_env, "sim", None)
        if sim is None or not hasattr(sim, "set_camera_view"):
            return False
        sim.set_camera_view(eye=settings["eye"], target=settings["lookat"])

    if hasattr(env, "_active_viewer_preset"):
        env._active_viewer_preset = resolved_preset
    return True


def reapply_active_viewer_preset(env) -> bool:
    preset_name = getattr(env, "_active_viewer_preset", None)
    if not preset_name or preset_name == VIEWER_PRESET_INHERIT:
        if hasattr(env, "default_viewer_preset_name"):
            try:
                preset_name = env.default_viewer_preset_name()
            except Exception:
                preset_name = VIEWER_PRESET_INHERIT
        else:
            preset_name = VIEWER_PRESET_INHERIT
    return apply_runtime_viewer_preset(env, preset_name)


@contextmanager
def temporary_video_viewer_preset(env, preset_name: str):
    if not preset_name or preset_name == VIEWER_PRESET_INHERIT:
        yield False
        return

    default_preset = VIEWER_PRESET_INHERIT
    if hasattr(env, "default_viewer_preset_name"):
        try:
            default_preset = env.default_viewer_preset_name()
        except Exception:
            default_preset = VIEWER_PRESET_INHERIT

    applied = apply_runtime_viewer_preset(env, preset_name)
    try:
        yield applied
    finally:
        if applied:
            apply_runtime_viewer_preset(env, default_preset)
