import importlib
import os
import sys
from dataclasses import dataclass
from typing import Tuple


DEFAULT_GALAXEA_LAB_PATH = "/home/shangyy/Galaxea_Lab"
GALAXEA_LEGACY_EXTENSION_NAMES = (
    "omni.isaac.lab",
    "omni.isaac.lab_assets",
    "omni.isaac.lab_tasks",
)
GALAXEA_TASK_MODULE_CANDIDATES = (
    "omni.isaac.lab_tasks.galaxea.direct.lift",
    "isaaclab_tasks.galaxea.direct.lift",
)
GALAXEA_RUNTIME_PACKAGES = (
    "omni.isaac.lab_tasks",
    "isaaclab_tasks",
)
GALAXEA_LEGACY_COMPAT_EXTENSIONS = (
    "omni.isaac.core",
    "omni.isaac.sensor",
    "omni.isaac.core_nodes",
    "omni.isaac.nucleus",
    "omni.isaac.cloner",
    "omni.isaac.version",
)
GALAXEA_LEGACY_RUNTIME_MODULES = (
    "omni.isaac.core",
    "omni.isaac.sensor",
    "omni.isaac.cloner",
)
GALAXEA_OPTIONAL_EXTENSION_BY_MODULE = {
    "omni.isaac.cloner": "omni.isaac.cloner",
    "isaacsim.core.cloner": "isaacsim.core.cloner",
    "omni.isaac.version": "omni.isaac.version",
    "omni.isaac.dynamic_control": "omni.isaac.dynamic_control",
    "omni.isaac.motion_generation": "omni.isaac.motion_generation",
    "omni.isaac.manipulators": "omni.isaac.manipulators",
    "omni.isaac.ui": "omni.isaac.ui",
    "omni.isaac.kit": "omni.isaac.kit",
}
GALAXEA_R1_ENV_IDS = (
    "Isaac-R1-Lift-Bin-IK-Rel-Direct-v0",
    "Isaac-R1-Multi-Fruit-IK-Abs-Direct-v0",
)


@dataclass(frozen=True)
class GalaxeaOverlayStatus:
    galaxea_lab_path: str
    extension_paths: Tuple[str, ...]
    missing_paths: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.galaxea_lab_path) and not self.missing_paths


@dataclass(frozen=True)
class GalaxeaLegacyRuntimeStatus:
    overlay_status: GalaxeaOverlayStatus
    enabled_extensions: Tuple[str, ...]
    imported_modules: Tuple[str, ...]
    errors: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.overlay_status.ok and not self.errors


def resolve_galaxea_lab_path(path: str | None = None) -> str:
    candidate = path or os.environ.get("METRA_GALAXEA_LAB_PATH") or DEFAULT_GALAXEA_LAB_PATH
    return os.path.abspath(os.path.expanduser(candidate))


def resolve_galaxea_extension_paths(path: str | None = None) -> Tuple[str, ...]:
    root = resolve_galaxea_lab_path(path)
    return tuple(
        os.path.join(root, "source", "extensions", extension_name)
        for extension_name in GALAXEA_LEGACY_EXTENSION_NAMES
    )


def galaxea_overlay_status(path: str | None = None) -> GalaxeaOverlayStatus:
    galaxea_lab_path = resolve_galaxea_lab_path(path)
    extension_paths = resolve_galaxea_extension_paths(galaxea_lab_path)
    missing_paths = tuple(ext_path for ext_path in extension_paths if not os.path.isdir(ext_path))
    return GalaxeaOverlayStatus(
        galaxea_lab_path=galaxea_lab_path,
        extension_paths=extension_paths,
        missing_paths=missing_paths,
    )


def activate_local_galaxea_overlay(path: str | None = None, *, strict: bool = False) -> GalaxeaOverlayStatus:
    status = galaxea_overlay_status(path)
    if strict and not status.ok:
        missing = ", ".join(status.missing_paths) or status.galaxea_lab_path
        raise FileNotFoundError(f"Galaxea overlay paths are unavailable: {missing}")

    if not status.ok:
        return status

    for ext_path in reversed(status.extension_paths):
        if ext_path not in sys.path:
            sys.path.insert(0, ext_path)

    existing_pythonpath = [
        item
        for item in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if item
    ]
    merged_pythonpath = []
    for ext_path in (*status.extension_paths, *existing_pythonpath):
        if ext_path not in merged_pythonpath:
            merged_pythonpath.append(ext_path)

    os.environ["METRA_GALAXEA_LAB_PATH"] = status.galaxea_lab_path
    os.environ["METRA_GALAXEA_OVERLAY_PATHS"] = os.pathsep.join(status.extension_paths)
    os.environ["PYTHONPATH"] = os.pathsep.join(merged_pythonpath)
    return status


def _resolve_enable_extension():
    from isaacsim.core.utils.extensions import enable_extension

    return enable_extension


def _extract_missing_module_name(exc: Exception) -> str | None:
    missing_module_name = getattr(exc, "name", None)
    if missing_module_name:
        return str(missing_module_name)

    message = str(exc)
    prefix = "No module named "
    if message.startswith(prefix):
        module_name = message[len(prefix):].strip()
        if module_name.startswith("'") and module_name.endswith("'"):
            module_name = module_name[1:-1]
        return module_name or None
    return None


def _format_legacy_runtime_error(errors: Tuple[str, ...]) -> str:
    detail = "; ".join(errors)
    return (
        "Galaxea_Lab legacy runtime is not ready inside the current Isaac Lab process. "
        "The local clone still targets the older omni.isaac.lab* stack, so the current "
        "Isaac Sim app must enable deprecated compatibility extensions first. "
        f"Details: {detail}"
    )


def _sanitize_legacy_physics_material_cfg(cfg) -> int:
    if cfg is None:
        return 0

    changes = 0
    if hasattr(cfg, "improve_patch_friction"):
        try:
            if getattr(cfg, "improve_patch_friction") is not None:
                setattr(cfg, "improve_patch_friction", None)
                changes += 1
        except Exception:
            pass
    return changes


def _patch_galaxea_legacy_material_runtime() -> Tuple[str, ...]:
    patched_targets = []

    def _patch_module_callable(module_name: str, attr_name: str, wrapper_factory):
        module = importlib.import_module(module_name)
        original = getattr(module, attr_name, None)
        if original is None:
            return
        if getattr(original, "_metra_galaxea_compat_patched", False):
            patched_targets.append(f"{module_name}.{attr_name}")
            return
        wrapped = wrapper_factory(original)
        wrapped._metra_galaxea_compat_patched = True
        setattr(module, attr_name, wrapped)
        patched_targets.append(f"{module_name}.{attr_name}")

    def _wrap_spawn_rigid_body_material(original):
        def patched(prim_path, cfg, *args, **kwargs):
            _sanitize_legacy_physics_material_cfg(cfg)
            return original(prim_path, cfg, *args, **kwargs)

        return patched

    def _wrap_spawn_ground_plane(original):
        def patched(prim_path, cfg, *args, **kwargs):
            _sanitize_legacy_physics_material_cfg(getattr(cfg, "physics_material", None))
            return original(prim_path, cfg, *args, **kwargs)

        return patched

    for module_name in (
        "omni.isaac.lab.sim.spawners.materials.physics_materials",
        "omni.isaac.lab.sim.spawners.materials",
    ):
        _patch_module_callable(module_name, "spawn_rigid_body_material", _wrap_spawn_rigid_body_material)

    for module_name in (
        "omni.isaac.lab.sim.spawners.from_files.from_files",
        "omni.isaac.lab.sim.spawners.from_files",
    ):
        _patch_module_callable(module_name, "spawn_ground_plane", _wrap_spawn_ground_plane)

    return tuple(patched_targets)


def _patch_galaxea_training_camera_runtime() -> Tuple[str, ...]:
    patched_targets = []

    def _patch_lift_env(module_name: str, class_name: str, *, multi_fruit: bool):
        module = importlib.import_module(module_name)
        env_cls = getattr(module, class_name, None)
        if env_cls is None:
            return
        if getattr(env_cls, "_metra_training_camera_runtime_patched", False):
            patched_targets.append(f"{module_name}.{class_name}")
            return

        original_setup_scene = env_cls._setup_scene
        original_get_observations = env_cls._get_observations

        def _front_camera_only_setup_scene(self):
            if not getattr(self.cfg, "_metra_front_camera_only", False):
                return original_setup_scene(self)

            if multi_fruit:
                self._object = [0] * 4
                self._robot = module.Articulation(self.cfg.robot_cfg)
                self._drop_height = 0.96

                object1_cfg = module.copy.deepcopy(self.cfg.carrot_cfg)
                object1_cfg.init_state.pos = (0.35, -0.35, self._drop_height)
                object1_cfg.spawn.scale = (0.3, 0.3, 0.3)
                self._object[0] = module.RigidObject(object1_cfg)
                self._object.append(module.RigidObject(object1_cfg))

                object2_cfg = module.copy.deepcopy(self.cfg.banana_cfg)
                object2_cfg.init_state.pos = (0.6, -0.6, self._drop_height)
                object2_cfg.spawn.scale = (0.3, 0.3, 0.3)
                self._object[1] = module.RigidObject(object2_cfg)
                self._object.append(module.RigidObject(object2_cfg))

                object3_cfg = module.copy.deepcopy(object2_cfg)
                object3_cfg.init_state.pos = (0.6, 0.6, self._drop_height)
                object3_cfg.prim_path = "/World/envs/env_.*/banana3"
                self._object[2] = module.RigidObject(object3_cfg)

                basket_cfg = module.copy.deepcopy(self.cfg.basket_cfg)
                basket_cfg.spawn.scale = (0.4, 0.4, 0.4)
                basket_cfg.init_state.pos = (0.45, 0.05, 1.05)
                basket_cfg.init_state.rot = (0.707, 0.0, 0.0, 0.707)
                basket_cfg.prim_path = "/World/envs/env_.*/basket"
                self._object[3] = module.RigidObject(basket_cfg)
            else:
                self._robot = module.Articulation(self.cfg.robot_cfg)
                self._object = module.RigidObject(self.cfg.object_cfg)

            if self.cfg.table_cfg.spawn is not None:
                if multi_fruit:
                    self.cfg.table_cfg.spawn.scale = (0.09, 0.09, 0.09)
                self.cfg.table_cfg.spawn.func(
                    self.cfg.table_cfg.prim_path,
                    self.cfg.table_cfg.spawn,
                    translation=self.cfg.table_cfg.init_state.pos,
                    orientation=self.cfg.table_cfg.init_state.rot,
                )

            if self.cfg.enable_camera:
                self._front_camera = module.Camera(self.cfg.front_camera_cfg)
                self.scene.sensors["front_camera"] = self._front_camera

            self._left_ee_frame = module.FrameTransformer(self.cfg.left_ee_frame_cfg)
            self._right_ee_frame = module.FrameTransformer(self.cfg.right_ee_frame_cfg)

            self.scene.articulations["robot"] = self._robot
            if multi_fruit:
                self.scene.rigid_objects["object0"] = self._object[0]
                self.scene.rigid_objects["object1"] = self._object[1]
                self.scene.rigid_objects["object2"] = self._object[2]
                self.scene.rigid_objects["object3"] = self._object[3]
            else:
                self.scene.rigid_objects["object"] = self._object
            self.scene.sensors["left_ee_frame"] = self._left_ee_frame
            self.scene.sensors["right_ee_frame"] = self._right_ee_frame
            self.scene.extras["table"] = module.XFormPrimView(
                self.cfg.table_cfg.prim_path,
                reset_xform_properties=False,
            )

            module.spawn_ground_plane(
                prim_path="/World/ground",
                cfg=module.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
            )
            self.scene.clone_environments(copy_from_source=False)

            intensity = 2000.0 if multi_fruit else 1000.0
            light_cfg = module.sim_utils.DomeLightCfg(
                intensity=intensity,
                color=(0.75, 0.75, 0.75),
            )
            light_cfg.func("/World/Light", light_cfg)

        def _front_camera_only_get_observations(self):
            if not getattr(self.cfg, "_metra_front_camera_only", False):
                return original_get_observations(self)

            left_ee_pos = self._left_ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
            right_ee_pos = self._right_ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
            left_ee_pose = module.torch.cat(
                [left_ee_pos, self._left_ee_frame.data.target_quat_w[..., 0, :]],
                dim=-1,
            )
            right_ee_pose = module.torch.cat(
                [right_ee_pos, self._right_ee_frame.data.target_quat_w[..., 0, :]],
                dim=-1,
            )

            joint_pos, joint_vel = self._process_joint_value()
            if multi_fruit:
                object_pos = self._object[self.object_id].data.root_pos_w - self.scene.env_origins
                object_quat = self._object[self.object_id].data.root_quat_w
            else:
                object_pos = self._object.data.root_pos_w - self.scene.env_origins
                object_quat = self._object.data.root_quat_w
            object_pose = module.torch.cat([object_pos, object_quat], dim=-1)

            obs = {
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "left_ee_pose": left_ee_pose,
                "right_ee_pose": right_ee_pose,
                "object_pose": object_pose,
                "goal_pose": module.torch.cat([self.goal_pos, self.goal_rot], dim=-1),
                "last_joints": joint_pos,
            }
            if self.cfg.enable_camera:
                obs["front_rgb"] = self._front_camera.data.output["rgb"].clone()[..., :3]
            return {"policy": obs}

        env_cls._setup_scene = _front_camera_only_setup_scene
        env_cls._get_observations = _front_camera_only_get_observations
        env_cls._metra_training_camera_runtime_patched = True
        patched_targets.append(f"{module_name}.{class_name}")

    _patch_lift_env(
        "omni.isaac.lab_tasks.galaxea.direct.lift.lift_env",
        "R1LiftEnv",
        multi_fruit=False,
    )
    _patch_lift_env(
        "omni.isaac.lab_tasks.galaxea.direct.lift.pick_fruit_env",
        "R1MultiFruitEnv",
        multi_fruit=True,
    )
    return tuple(patched_targets)


def ensure_galaxea_legacy_runtime(path: str | None = None, *, strict: bool = False) -> GalaxeaLegacyRuntimeStatus:
    overlay_status = activate_local_galaxea_overlay(path, strict=strict)
    if not overlay_status.ok:
        errors = (f"Galaxea overlay paths are unavailable: {', '.join(overlay_status.missing_paths)}",)
        if strict:
            raise FileNotFoundError(errors[0])
        return GalaxeaLegacyRuntimeStatus(
            overlay_status=overlay_status,
            enabled_extensions=(),
            imported_modules=(),
            errors=errors,
        )

    try:
        enable_extension = _resolve_enable_extension()
    except Exception as exc:
        error = (
            "Could not resolve isaacsim.core.utils.extensions.enable_extension. "
            "Call ensure_galaxea_legacy_runtime() only after AppLauncher has started the Isaac Sim app. "
            f"Original error: {type(exc).__name__}: {exc}"
        )
        if strict:
            raise RuntimeError(error) from exc
        return GalaxeaLegacyRuntimeStatus(
            overlay_status=overlay_status,
            enabled_extensions=(),
            imported_modules=(),
            errors=(error,),
        )

    enabled_extensions = []
    errors = []
    for extension_name in GALAXEA_LEGACY_COMPAT_EXTENSIONS:
        try:
            if enable_extension(extension_name):
                enabled_extensions.append(extension_name)
            else:
                errors.append(f"enable_extension({extension_name!r}) returned False")
        except Exception as exc:
            errors.append(f"{extension_name}: {type(exc).__name__}: {exc}")

    importlib.invalidate_caches()

    imported_modules = []
    for module_name in GALAXEA_LEGACY_RUNTIME_MODULES:
        try:
            importlib.import_module(module_name)
            imported_modules.append(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    try:
        _patch_galaxea_legacy_material_runtime()
    except Exception as exc:
        errors.append(f"legacy material compat patch: {type(exc).__name__}: {exc}")
    try:
        _patch_galaxea_training_camera_runtime()
    except Exception as exc:
        errors.append(f"legacy training camera patch: {type(exc).__name__}: {exc}")

    status = GalaxeaLegacyRuntimeStatus(
        overlay_status=overlay_status,
        enabled_extensions=tuple(enabled_extensions),
        imported_modules=tuple(imported_modules),
        errors=tuple(errors),
    )
    if strict and not status.ok:
        raise RuntimeError(_format_legacy_runtime_error(status.errors))
    return status


def maybe_enable_galaxea_optional_extension_from_error(
    exc: Exception,
    attempted_extensions: set[str] | None = None,
) -> str | None:
    missing_module_name = _extract_missing_module_name(exc)
    if not missing_module_name:
        return None

    extension_name = GALAXEA_OPTIONAL_EXTENSION_BY_MODULE.get(missing_module_name)
    if not extension_name:
        return None

    if attempted_extensions is not None and extension_name in attempted_extensions:
        return None

    enable_extension = _resolve_enable_extension()
    enabled = enable_extension(extension_name)
    if attempted_extensions is not None:
        attempted_extensions.add(extension_name)
    if not enabled:
        return None

    importlib.invalidate_caches()
    return extension_name
