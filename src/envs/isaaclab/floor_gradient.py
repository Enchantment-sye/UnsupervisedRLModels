from __future__ import annotations

from typing import Dict, Optional

from envs.color_gradients import sample_gradient_rgb


_OVERLAY_ROOT = "/World/MetraFloorGradient"
_AXIS_TO_INDEX = {"x": 0, "y": 1}


def overlay_path_for_task(task_name: str) -> str:
    return f"{_OVERLAY_ROOT}/{task_name}"


def _get_stage():
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac Lab stage is unavailable; cannot apply floor gradient overlay.")
    return stage


def _validate_axes(overlay_spec):
    if overlay_spec.forward_axis not in _AXIS_TO_INDEX or overlay_spec.lateral_axis not in _AXIS_TO_INDEX:
        raise ValueError(
            f"Unsupported floor gradient axes forward={overlay_spec.forward_axis!r} lateral={overlay_spec.lateral_axis!r}"
        )
    if overlay_spec.forward_axis == overlay_spec.lateral_axis:
        raise ValueError("Floor gradient forward_axis and lateral_axis must differ.")


def _define_parent(stage, path: str):
    from pxr import UsdGeom

    return UsdGeom.Xform.Define(stage, path)


def _define_tile(stage, path: str, translation, scale, color):
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    cube.CreateDisplayOpacityAttr([1.0])
    xformable = UsdGeom.Xformable(cube)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xformable.AddScaleOp().Set(Gf.Vec3f(*scale))
    return cube


def apply_floor_gradient_overlay(env, task_spec) -> Optional[Dict[str, object]]:
    overlay_spec = getattr(task_spec, "floor_gradient_overlay", None)
    if overlay_spec is None:
        return None

    _validate_axes(overlay_spec)
    stage = _get_stage()
    _define_parent(stage, _OVERLAY_ROOT)
    task_path = overlay_path_for_task(task_spec.task_name)
    task_prim = stage.GetPrimAtPath(task_path)
    if task_prim and task_prim.IsValid():
        return inspect_floor_gradient_overlay(task_spec.task_name)

    _define_parent(stage, task_path)

    tile_length = overlay_spec.length / overlay_spec.tiles_x
    tile_width = overlay_spec.width / overlay_spec.tiles_y
    z_center = overlay_spec.origin[2] + overlay_spec.z_offset + overlay_spec.tile_height * 0.5

    forward_axis_idx = _AXIS_TO_INDEX[overlay_spec.forward_axis]
    lateral_axis_idx = _AXIS_TO_INDEX[overlay_spec.lateral_axis]

    for ix in range(overlay_spec.tiles_x):
        forward = -overlay_spec.length * 0.5 + (ix + 0.5) * tile_length
        u = (ix + 0.5) / overlay_spec.tiles_x
        for iy in range(overlay_spec.tiles_y):
            lateral = -overlay_spec.width * 0.5 + (iy + 0.5) * tile_width
            v = (iy + 0.5) / overlay_spec.tiles_y
            color = sample_gradient_rgb(overlay_spec.preset_name, u, v)
            coords = [overlay_spec.origin[0], overlay_spec.origin[1], z_center]
            coords[forward_axis_idx] = overlay_spec.origin[forward_axis_idx] + forward
            coords[lateral_axis_idx] = overlay_spec.origin[lateral_axis_idx] + lateral
            _define_tile(
                stage=stage,
                path=f"{task_path}/tile_{ix:03d}_{iy:03d}",
                translation=tuple(coords),
                scale=(tile_length, tile_width, overlay_spec.tile_height),
                color=color,
            )

    return inspect_floor_gradient_overlay(task_spec.task_name)


def inspect_floor_gradient_overlay(task_name: str) -> Optional[Dict[str, object]]:
    from pxr import UsdGeom

    stage = _get_stage()
    task_path = overlay_path_for_task(task_name)
    task_prim = stage.GetPrimAtPath(task_path)
    if not task_prim or not task_prim.IsValid():
        return None

    children = sorted(task_prim.GetChildren(), key=lambda prim: prim.GetName())
    tile_colors = []
    for child in children:
        cube = UsdGeom.Cube(child)
        if not cube:
            continue
        color_attr = cube.GetDisplayColorAttr()
        if not color_attr.IsValid():
            continue
        color_value = color_attr.Get()
        if not color_value:
            continue
        tile_colors.append(tuple(float(channel) for channel in color_value[0]))

    return {
        "path": task_path,
        "tile_count": len(tile_colors),
        "first_color": tile_colors[0] if tile_colors else None,
        "last_color": tile_colors[-1] if tile_colors else None,
    }
