import hashlib
import importlib.metadata
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


_CACHE_DIR = Path(tempfile.gettempdir()) / f"metra_kitchen_mujoco_compat_{os.getuid()}"
_KETTLE_ASSET = "kettle_asset.xml"
_MUJOCO_PATCH_VERSION = (3, 2, 0)
_EGL_DEVICE_CANDIDATES = tuple(str(i) for i in range(16))
_CACHE_SCHEMA_VERSION = "v3"


def get_mujoco_compatible_kitchen_model(model_path, *, force=False):
    """Return a MuJoCo-3-compatible copy of the D4RL Kitchen model XML."""
    model_path = Path(model_path).resolve()
    if not force and not needs_kitchen_xml_patch():
        return str(model_path)

    kettle_asset = _find_included_file(model_path, _KETTLE_ASSET)
    cache_key = _cache_key(model_path, kettle_asset)
    patched_model = _CACHE_DIR / f"{model_path.stem}_{cache_key}.xml"
    patched_kettle = _CACHE_DIR / f"{_KETTLE_ASSET[:-4]}_{cache_key}.xml"

    if not force and _is_readable_file(patched_model) and _is_readable_file(patched_kettle):
        return str(patched_model)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _write_patched_kettle_asset(kettle_asset, patched_kettle, model_path)
    _write_patched_model(model_path, patched_model, patched_kettle, cache_key)
    return str(patched_model)


def configure_kitchen_mujoco_runtime():
    """Configure MuJoCo rendering before D4RL Kitchen imports dm_control."""
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

    if os.environ.get("MUJOCO_EGL_DEVICE_ID"):
        return os.environ["MUJOCO_EGL_DEVICE_ID"]

    requested = os.environ.get("METRA_KITCHEN_EGL_DEVICE_ID")
    if requested:
        if requested.strip().lower() != "auto":
            os.environ["MUJOCO_EGL_DEVICE_ID"] = requested
            return requested
        return _select_working_egl_device()

    for env_name in ("GL_DEVICE_ID", "CUDA_VISIBLE_DEVICES"):
        device = _first_device_from_env(os.environ.get(env_name))
        if device is not None:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = device
            return device

    return _select_working_egl_device()


def needs_kitchen_xml_patch():
    """MuJoCo 3.2+ rejects D4RL Kitchen's classed top-level default block."""
    try:
        version = importlib.metadata.version("mujoco")
    except importlib.metadata.PackageNotFoundError:
        return True
    return _version_tuple(version) >= _MUJOCO_PATCH_VERSION


def _select_working_egl_device():
    original = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    for device in _EGL_DEVICE_CANDIDATES:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = device
        if _egl_device_can_render(device):
            return device
    if original is None:
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    else:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = original
    return None


def _egl_device_can_render(device):
    code = (
        "import mujoco\n"
        "xml = '<mujoco><worldbody><geom type=\"sphere\" size=\"0.1\"/></worldbody></mujoco>'\n"
        "model = mujoco.MjModel.from_xml_string(xml)\n"
        "data = mujoco.MjData(model)\n"
        "context = mujoco.GLContext(64, 64)\n"
        "context.make_current()\n"
        "scene = mujoco.MjvScene(model, maxgeom=1000)\n"
        "camera = mujoco.MjvCamera()\n"
        "options = mujoco.MjvOption()\n"
        "viewport = mujoco.MjrRect(0, 0, 64, 64)\n"
        "render_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)\n"
        "mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)\n"
        "mujoco.mjr_render(viewport, scene, render_context)\n"
        "render_context.free()\n"
        "context.free()\n"
    )
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["MUJOCO_EGL_DEVICE_ID"] = str(device)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=15,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0


def _first_device_from_env(value):
    if not value:
        return None
    first = value.split(",", 1)[0].strip()
    if not first or first == "-1":
        return None
    return first


def _version_tuple(version):
    parts = []
    for part in version.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits or 0))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _is_readable_file(path):
    return path.is_file() and os.access(path, os.R_OK)


def _cache_key(*paths):
    digest = hashlib.sha1()
    digest.update(_CACHE_SCHEMA_VERSION.encode("utf-8"))
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _atomic_write_xml(tree, path):
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tree.write(tmp_path, encoding="unicode")
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_patched_model(model_path, patched_model, patched_kettle, cache_key):
    tree = ET.parse(model_path)
    root = tree.getroot()

    for include in root.iter("include"):
        include_file = include.get("file")
        if not include_file:
            continue
        resolved = (model_path.parent / include_file).resolve()
        if resolved.name == _KETTLE_ASSET:
            include.set("file", str(patched_kettle))
        else:
            include.set("file", str(_write_patched_include_file(resolved, model_path, cache_key)))

    for compiler in root.iter("compiler"):
        for attr in ("meshdir", "texturedir"):
            value = compiler.get(attr)
            if value and not os.path.isabs(value):
                compiler.set(attr, str((model_path.parent / value).resolve()))

    _atomic_write_xml(tree, patched_model)


def _write_patched_include_file(include_path, model_path, cache_key):
    include_path = Path(include_path).resolve()
    patched_include = _CACHE_DIR / f"{include_path.stem}_{cache_key}_{_path_digest(include_path)}.xml"
    if _is_readable_file(patched_include):
        return patched_include

    tree = ET.parse(include_path)
    root = tree.getroot()
    reference_dirs = (include_path.parent, model_path.parent)

    for elem in root.iter():
        if elem.tag in ("mesh", "texture", "hfield"):
            file_value = elem.get("file")
            if file_value:
                elem.set("file", str(_resolve_existing_resource_path(file_value, reference_dirs, model_path)))

        if elem.tag == "compiler":
            for attr in ("meshdir", "texturedir"):
                value = elem.get(attr)
                if value:
                    elem.set(attr, str(_resolve_existing_resource_path(value, reference_dirs, model_path)))

        if elem.tag == "include":
            include_file = elem.get("file")
            if include_file:
                elem.set("file", str(_resolve_existing_resource_path(include_file, reference_dirs, model_path)))

    _atomic_write_xml(tree, patched_include)
    return patched_include


def _write_patched_kettle_asset(kettle_asset, patched_kettle, model_path):
    tree = ET.parse(kettle_asset)
    root = tree.getroot()

    for index, child in enumerate(list(root)):
        if child.tag != "default" or "class" not in child.attrib:
            continue
        wrapper = ET.Element("default")
        root.remove(child)
        wrapper.append(child)
        root.insert(index, wrapper)

    reference_dirs = (kettle_asset.parent, model_path.parent)
    for elem in root.iter():
        if elem.tag in ("mesh", "texture", "hfield"):
            file_value = elem.get("file")
            if file_value:
                elem.set("file", str(_resolve_existing_resource_path(file_value, reference_dirs, model_path)))

    _atomic_write_xml(tree, patched_kettle)


def _find_included_file(model_path, filename):
    tree = ET.parse(model_path)
    root = tree.getroot()
    for include in root.iter("include"):
        include_file = include.get("file")
        if not include_file:
            continue
        resolved = (model_path.parent / include_file).resolve()
        if resolved.name == filename:
            return resolved
    raise FileNotFoundError(f"{filename} is not included by {model_path}")


def _path_digest(path):
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]


def _resolve_existing_resource_path(value, reference_dirs, model_path):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    candidates = [(Path(base) / path).resolve() for base in reference_dirs]
    candidates.extend(_resource_root_candidates(path, model_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resource_root_candidates(path, model_path):
    parts = path.parts
    roots = []
    for parent in (model_path.parent, *model_path.parents):
        if (parent / "third_party").exists() or (parent / "adept_models").exists():
            roots.append(parent)

    candidates = []
    for marker in ("third_party", "adept_models", "adept_envs"):
        if marker not in parts:
            continue
        suffix = Path(*parts[parts.index(marker):])
        candidates.extend((root / suffix).resolve() for root in roots)

    normalized_parts = [part for part in parts if part not in ("", ".", "..")]
    if normalized_parts:
        suffix = Path(*normalized_parts)
        if normalized_parts[0] == "kitchen":
            candidates.extend((root / "adept_models" / suffix).resolve() for root in roots)
        elif normalized_parts[0] == "franka":
            candidates.extend((root / "third_party" / suffix).resolve() for root in roots)
    return candidates
