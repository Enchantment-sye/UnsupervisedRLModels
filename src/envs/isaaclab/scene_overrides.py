from __future__ import annotations

from typing import Dict, Optional


_DEFAULT_DOME_LIGHT_PATH = "/World/Light"


def _get_stage():
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac Lab stage is unavailable; cannot apply scene overrides.")
    return stage


def apply_white_dome_light_override(env, task_spec, *, light_path: str = _DEFAULT_DOME_LIGHT_PATH) -> Optional[Dict[str, object]]:
    if not getattr(task_spec, "env_id", "").startswith("Isaac-R1-"):
        return None

    from pxr import Gf, UsdLux

    stage = _get_stage()
    prim = stage.GetPrimAtPath(light_path)
    if not prim or not prim.IsValid():
        return None

    dome_light = UsdLux.DomeLight(prim)
    if not dome_light:
        return None

    color_attr = dome_light.GetColorAttr()
    if not color_attr.IsValid():
        color_attr = dome_light.CreateColorAttr()
    color_attr.Set(Gf.Vec3f(1.0, 1.0, 1.0))

    return inspect_dome_light(light_path)


def inspect_dome_light(light_path: str = _DEFAULT_DOME_LIGHT_PATH) -> Optional[Dict[str, object]]:
    from pxr import UsdLux

    stage = _get_stage()
    prim = stage.GetPrimAtPath(light_path)
    if not prim or not prim.IsValid():
        return None

    dome_light = UsdLux.DomeLight(prim)
    if not dome_light:
        return None

    color_value = dome_light.GetColorAttr().Get()
    intensity_value = dome_light.GetIntensityAttr().Get()

    if color_value is None:
        color = None
    else:
        color = tuple(float(channel) for channel in color_value)

    return {
        "path": light_path,
        "color": color,
        "intensity": float(intensity_value) if intensity_value is not None else None,
    }
