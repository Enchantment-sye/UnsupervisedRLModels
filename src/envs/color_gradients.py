from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class GradientPreset:
    name: str
    blue_channel: float = 128.0 / 255.0

    def sample(self, u: float, v: float) -> Tuple[float, float, float]:
        u = float(np.clip(u, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        return (u, v, self.blue_channel)


_PRESETS: Dict[str, GradientPreset] = {
    "dmc_quadruped_run_forward_color": GradientPreset(
        name="dmc_quadruped_run_forward_color",
    ),
}


def get_gradient_preset(name: str) -> GradientPreset:
    if name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS))
        raise KeyError(f"Unknown gradient preset {name!r}. Available: {available}")
    return _PRESETS[name]


def sample_gradient_rgb(name: str, u: float, v: float) -> Tuple[float, float, float]:
    return get_gradient_preset(name).sample(u, v)


def sample_gradient_rgb_uint8(name: str, u: float, v: float) -> np.ndarray:
    rgb = np.asarray(sample_gradient_rgb(name, u, v), dtype=np.float32)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
