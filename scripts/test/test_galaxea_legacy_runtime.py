from types import SimpleNamespace

import pytest

from src.envs.isaaclab import factory
from src.envs.isaaclab import galaxea_overlay as overlay


def _ok_overlay_status():
    return overlay.GalaxeaOverlayStatus(
        galaxea_lab_path="/home/shangyy/Galaxea_Lab",
        extension_paths=(
            "/home/shangyy/Galaxea_Lab/source/extensions/omni.isaac.lab",
            "/home/shangyy/Galaxea_Lab/source/extensions/omni.isaac.lab_assets",
            "/home/shangyy/Galaxea_Lab/source/extensions/omni.isaac.lab_tasks",
        ),
        missing_paths=(),
    )


def test_ensure_galaxea_legacy_runtime_reports_success(monkeypatch):
    monkeypatch.setattr(overlay, "activate_local_galaxea_overlay", lambda path=None, strict=False: _ok_overlay_status())
    monkeypatch.setattr(overlay, "_resolve_enable_extension", lambda: (lambda extension_name: True))

    imported = []

    def fake_import_module(module_name):
        imported.append(module_name)
        return object()

    monkeypatch.setattr(overlay.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(overlay.importlib, "invalidate_caches", lambda: None)

    status = overlay.ensure_galaxea_legacy_runtime()

    assert status.ok
    assert status.enabled_extensions == overlay.GALAXEA_LEGACY_COMPAT_EXTENSIONS
    assert status.imported_modules == overlay.GALAXEA_LEGACY_RUNTIME_MODULES
    assert imported[: len(overlay.GALAXEA_LEGACY_RUNTIME_MODULES)] == list(overlay.GALAXEA_LEGACY_RUNTIME_MODULES)


def test_ensure_galaxea_legacy_runtime_surfaces_missing_app_startup(monkeypatch):
    monkeypatch.setattr(overlay, "activate_local_galaxea_overlay", lambda path=None, strict=False: _ok_overlay_status())

    def raise_import_error():
        raise ImportError("isaacsim.core.utils.extensions is unavailable")

    monkeypatch.setattr(overlay, "_resolve_enable_extension", raise_import_error)

    status = overlay.ensure_galaxea_legacy_runtime()

    assert not status.ok
    assert "AppLauncher" in status.errors[0]


def test_r1_registration_error_mentions_legacy_compatibility(monkeypatch):
    monkeypatch.setattr(factory, "activate_local_galaxea_overlay", lambda path=None, strict=False: _ok_overlay_status())

    def fake_import_module(module_name):
        raise ImportError(f"missing dependency while importing {module_name}")

    monkeypatch.setattr(factory.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as exc_info:
        factory._ensure_env_id_registered("Isaac-R1-Lift-Bin-IK-Rel-Direct-v0")

    message = str(exc_info.value)
    assert "Failed to register the Galaxea R1 Isaac Lab task" in message
    assert "omni.isaac.core compatibility extensions" in message
    assert "omni.isaac.lab_tasks.galaxea.direct.lift" in message
    assert "isaaclab_tasks.galaxea.direct.lift" in message


def test_r1_registration_retries_when_known_optional_legacy_extension_is_missing(monkeypatch):
    monkeypatch.setattr(factory, "activate_local_galaxea_overlay", lambda path=None, strict=False: _ok_overlay_status())

    calls = []

    def fake_import_module(module_name):
        calls.append(module_name)
        if module_name == "omni.isaac.lab_tasks.galaxea.direct.lift" and calls.count(module_name) == 1:
            raise ModuleNotFoundError("No module named 'omni.isaac.cloner'")
        if module_name == "omni.isaac.lab_tasks.galaxea.direct.lift":
            return object()
        raise AssertionError(f"Unexpected import attempt: {module_name}")

    enabled = []

    def fake_enable_from_error(exc, attempted_extensions=None):
        assert isinstance(exc, ImportError)
        enabled.append(str(exc))
        if attempted_extensions is not None:
            attempted_extensions.add("omni.isaac.cloner")
        return "omni.isaac.cloner"

    monkeypatch.setattr(factory.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(factory, "maybe_enable_galaxea_optional_extension_from_error", fake_enable_from_error)

    factory._ensure_env_id_registered("Isaac-R1-Lift-Bin-IK-Rel-Direct-v0")

    assert calls.count("omni.isaac.lab_tasks.galaxea.direct.lift") == 2
    assert enabled


def test_galaxea_cfg_sanitizer_clears_legacy_patch_friction_recursively():
    cfg = SimpleNamespace(
        improve_patch_friction=True,
        sim=SimpleNamespace(
            physics_material=SimpleNamespace(improve_patch_friction=True),
        ),
        assets=[
            SimpleNamespace(
                spawn=SimpleNamespace(
                    physics_material=SimpleNamespace(improve_patch_friction=True),
                )
            )
        ],
    )

    changes = factory._sanitize_galaxea_legacy_cfg_for_runtime(cfg)

    assert changes == 3
    assert cfg.improve_patch_friction is None
    assert cfg.sim.physics_material.improve_patch_friction is None
    assert cfg.assets[0].spawn.physics_material.improve_patch_friction is None


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("got an unexpected keyword argument 'cfg'", True),
        ("unexpected keyword argument 'render_mode'", True),
        ("takes no keyword arguments", False),
        ("Attribute 'ImprovePatchFriction' does not exist on prim '/physicsScene/defaultMaterial'.", False),
    ),
)
def test_gym_make_fallback_only_retries_signature_mismatches(message, expected):
    assert factory._should_retry_gym_make_without_kwargs(TypeError(message)) is expected


def test_patch_galaxea_legacy_material_runtime_sanitizes_ground_plane_defaults(monkeypatch):
    calls = []

    def fake_spawn_rigid_body_material(prim_path, cfg, *args, **kwargs):
        calls.append(("material", prim_path, cfg.improve_patch_friction))
        return "material-ok"

    def fake_spawn_ground_plane(prim_path, cfg, *args, **kwargs):
        calls.append(("ground", prim_path, cfg.physics_material.improve_patch_friction))
        return cfg.physics_material.func("/World/ground/material", cfg.physics_material)

    fake_modules = {
        "omni.isaac.lab.sim.spawners.materials.physics_materials": SimpleNamespace(
            spawn_rigid_body_material=fake_spawn_rigid_body_material
        ),
        "omni.isaac.lab.sim.spawners.materials": SimpleNamespace(
            spawn_rigid_body_material=fake_spawn_rigid_body_material
        ),
        "omni.isaac.lab.sim.spawners.from_files.from_files": SimpleNamespace(
            spawn_ground_plane=fake_spawn_ground_plane
        ),
        "omni.isaac.lab.sim.spawners.from_files": SimpleNamespace(
            spawn_ground_plane=fake_spawn_ground_plane
        ),
    }

    monkeypatch.setattr(overlay.importlib, "import_module", lambda module_name: fake_modules[module_name])

    patched = overlay._patch_galaxea_legacy_material_runtime()

    cfg = SimpleNamespace(
        physics_material=SimpleNamespace(
            improve_patch_friction=True,
            func=fake_modules["omni.isaac.lab.sim.spawners.materials"].spawn_rigid_body_material,
        )
    )
    result = fake_modules["omni.isaac.lab.sim.spawners.from_files"].spawn_ground_plane("/World/ground", cfg)

    assert result == "material-ok"
    assert cfg.physics_material.improve_patch_friction is None
    assert calls == [
        ("ground", "/World/ground", None),
        ("material", "/World/ground/material", None),
    ]
    assert "omni.isaac.lab.sim.spawners.from_files.spawn_ground_plane" in patched
