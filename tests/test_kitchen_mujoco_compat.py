import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import envs.kitchen.mujoco_compat as mujoco_compat


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_compatible_model_wraps_classed_top_level_kettle_default(tmp_path):
    model_path = (
        tmp_path
        / "d4rl"
        / "kitchen"
        / "adept_envs"
        / "franka"
        / "assets"
        / "franka_kitchen_ee_ctrl.xml"
    )
    kettle_path = (
        tmp_path
        / "d4rl"
        / "kitchen"
        / "adept_models"
        / "kitchen"
        / "assets"
        / "kettle_asset.xml"
    )

    _write(
        model_path,
        """
<mujoco>
  <include file="../../../adept_models/kitchen/assets/kettle_asset.xml"/>
  <compiler meshdir="../../../adept_models/kitchen"
            texturedir="../../../adept_models/kitchen"/>
</mujoco>
""",
    )
    _write(
        kettle_path,
        """
<mujocoinclude>
  <asset/>
  <default class="kettle">
    <joint damping="2"/>
  </default>
</mujocoinclude>
""",
    )

    patched_model = Path(
        mujoco_compat.get_mujoco_compatible_kitchen_model(model_path, force=True)
    )
    patched_include = ET.parse(patched_model).getroot().find("include").get("file")
    patched_kettle = ET.parse(patched_include).getroot()
    direct_defaults = [child for child in list(patched_kettle) if child.tag == "default"]

    assert Path(patched_include).is_absolute()
    assert patched_model.stat().st_mode & 0o777 == 0o644
    assert Path(patched_include).stat().st_mode & 0o777 == 0o644
    assert ET.parse(patched_model).getroot().find("compiler").get("meshdir") == str(
        (model_path.parent / "../../../adept_models/kitchen").resolve()
    )
    assert "class" not in direct_defaults[0].attrib
    assert direct_defaults[0].find("default").get("class") == "kettle"


def test_patch_is_skipped_for_old_mujoco(monkeypatch, tmp_path):
    model_path = (
        tmp_path
        / "d4rl"
        / "kitchen"
        / "adept_envs"
        / "franka"
        / "assets"
        / "franka_kitchen_ee_ctrl.xml"
    )
    kettle_path = (
        tmp_path
        / "d4rl"
        / "kitchen"
        / "adept_models"
        / "kitchen"
        / "assets"
        / "kettle_asset.xml"
    )
    _write(
        model_path,
        '<mujoco><include file="../../../adept_models/kitchen/assets/kettle_asset.xml"/></mujoco>',
    )
    _write(kettle_path, '<mujocoinclude><default class="kettle"/></mujocoinclude>')
    monkeypatch.setattr(mujoco_compat.importlib.metadata, "version", lambda _: "3.1.5")

    model = mujoco_compat.get_mujoco_compatible_kitchen_model(model_path)

    assert model == str(model_path.resolve())


def test_real_d4rl_kitchen_model_loads_after_compat_patch():
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed")

    try:
        import mujoco
    except ImportError as exc:
        pytest.skip(f"mujoco cannot be imported in this runtime: {exc}")

    model_path = Path(
        "/home/shangyy/D4RL/d4rl/kitchen/adept_envs/franka/assets/franka_kitchen_ee_ctrl.xml"
    )
    if not model_path.exists():
        pytest.skip("local D4RL Kitchen model is not installed")

    patched_model = mujoco_compat.get_mujoco_compatible_kitchen_model(
        model_path,
        force=True,
    )
    model = mujoco.MjModel.from_xml_path(patched_model)

    assert model.nq == 30
    assert model.nv == 29
    assert model.nu == 9
