from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Type


@dataclass(frozen=True)
class FloorGradientOverlaySpec:
    preset_name: str
    length: float = 120.0
    width: float = 24.0
    tiles_x: int = 48
    tiles_y: int = 16
    tile_height: float = 0.01
    z_offset: float = 0.001
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    forward_axis: str = "x"
    lateral_axis: str = "y"


@dataclass(frozen=True)
class IsaacLabTaskSpec:
    task_name: str
    env_id: str
    workflow_type: str
    obs_type: str
    action_type: str
    requires_cameras: bool
    supports_render_rgb: bool
    supports_camera_obs: bool
    camera_obs_key: Optional[str]
    adapter_cls: Type
    cfg_builder: Optional[Callable] = None
    default_num_envs: int = 1
    default_image_source_encoder0: str = "render"
    default_image_source_encoder1: str = "auto"
    floor_gradient_overlay: Optional[FloorGradientOverlaySpec] = None
    aliases: Tuple[str, ...] = ()
    notes: str = ""
