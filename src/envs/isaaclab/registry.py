from typing import Dict, Iterable, List

from .base_spec import IsaacLabTaskSpec

_TASK_SPECS: Dict[str, IsaacLabTaskSpec] = {}
_TASK_NAME_TO_ENV_ID: Dict[str, str] = {}
_INITIALIZED = False


def register_task_spec(spec: IsaacLabTaskSpec) -> IsaacLabTaskSpec:
    existing = _TASK_SPECS.get(spec.env_id)
    if existing is not None and existing != spec:
        raise ValueError(f"Isaac Lab task already registered: {spec.env_id}")
    _TASK_SPECS[spec.env_id] = spec
    for task_name in (spec.task_name, *spec.aliases):
        existing_env_id = _TASK_NAME_TO_ENV_ID.get(task_name)
        if existing_env_id is not None and existing_env_id != spec.env_id:
            raise ValueError(f"Isaac Lab task alias already registered: {task_name}")
        _TASK_NAME_TO_ENV_ID[task_name] = spec.env_id
    return spec


def _ensure_registry_initialized():
    global _INITIALIZED
    if _INITIALIZED:
        return
    from .tasks import classic  # noqa: F401
    from .tasks import locomotion  # noqa: F401
    from .tasks import manipulation  # noqa: F401
    from .tasks import multirotor  # noqa: F401

    _INITIALIZED = True


def is_isaaclab_task_name(task_name: str) -> bool:
    _ensure_registry_initialized()
    return task_name in _TASK_NAME_TO_ENV_ID


def get_task_spec(identifier: str) -> IsaacLabTaskSpec:
    _ensure_registry_initialized()
    env_id = _TASK_NAME_TO_ENV_ID.get(identifier, identifier)
    if env_id not in _TASK_SPECS:
        available_task_names = ", ".join(sorted(_TASK_NAME_TO_ENV_ID))
        available_env_ids = ", ".join(sorted(_TASK_SPECS))
        raise KeyError(
            f"Unknown Isaac Lab task identifier={identifier!r}. "
            f"Available task names: {available_task_names}. "
            f"Available env ids: {available_env_ids}"
        )
    return _TASK_SPECS[env_id]


def list_task_specs() -> List[IsaacLabTaskSpec]:
    _ensure_registry_initialized()
    return [
        _TASK_SPECS[key]
        for key in sorted(_TASK_SPECS)
    ]


def iter_task_specs() -> Iterable[IsaacLabTaskSpec]:
    yield from list_task_specs()
