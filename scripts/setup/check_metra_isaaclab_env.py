#!/usr/bin/env python3

import argparse
import importlib
import importlib.util
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.envs.isaaclab.galaxea_overlay import (
    GALAXEA_R1_ENV_IDS,
    GALAXEA_TASK_MODULE_CANDIDATES,
    activate_local_galaxea_overlay,
    ensure_galaxea_legacy_runtime,
    galaxea_overlay_status,
)


CORE_MODULES = ("isaaclab", "isaacsim", "isaaclab_assets", "isaaclab_tasks")
GALAXEA_MODULE_GROUPS = (
    ("omni.isaac.lab_tasks.galaxea.direct.lift", "Galaxea direct lift tasks (legacy namespace)"),
    ("isaaclab_tasks.galaxea.direct.lift", "Galaxea direct lift tasks"),
)
MAIN_COMPAT_MODULES = ("dm_control", "metaworld", "mujoco", "pybullet")
KITCHEN_MODULES = ("d4rl", "mujoco_py", "mjrl")
OPTIONAL_MODULES = ("bigym", "robodesk")
SPEC_ONLY_MODULES = {"isaaclab_assets", "isaaclab_tasks"}


def probe_module(module_name):
    result = {"name": module_name, "ok": False}
    if module_name in SPEC_ONLY_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            result["error_type"] = "ModuleNotFoundError"
            result["error"] = f"No module named '{module_name}'"
            return result
        result["ok"] = True
        result["file"] = spec.origin
        result["version"] = None
        return result

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised in environment checks
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result

    result["ok"] = True
    result["file"] = getattr(module, "__file__", None)
    result["version"] = getattr(module, "__version__", None)
    return result


def probe_cuda():
    result = {"name": "torch.cuda", "ok": False}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-specific
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result

    is_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if is_available else 0
    result["ok"] = is_available
    result["is_available"] = is_available
    result["device_count"] = device_count
    if is_available:
        result["devices"] = [torch.cuda.get_device_name(idx) for idx in range(device_count)]
    else:
        result["error_type"] = "CudaUnavailable"
        result["error"] = "torch.cuda.is_available() returned False"
    return result


def probe_galaxea():
    results = []
    for module_name, description in GALAXEA_MODULE_GROUPS:
        result = probe_module(module_name)
        result["description"] = description
        results.append(result)
    return results


def probe_galaxea_overlay():
    status = galaxea_overlay_status()
    result = {
        "name": "galaxea_overlay",
        "ok": status.ok,
        "galaxea_lab_path": status.galaxea_lab_path,
        "extension_paths": list(status.extension_paths),
    }
    if not status.ok:
        result["error_type"] = "MissingGalaxeaOverlay"
        result["error"] = ", ".join(status.missing_paths) or status.galaxea_lab_path
        result["missing_paths"] = list(status.missing_paths)
    return result


def probe_galaxea_runtime():
    result = {"name": "galaxea_runtime", "ok": False}
    app_launcher = None
    simulation_app = None
    try:
        overlay_status = activate_local_galaxea_overlay(strict=True)
        result["galaxea_lab_path"] = overlay_status.galaxea_lab_path
        result["extension_paths"] = list(overlay_status.extension_paths)

        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
        simulation_app = getattr(app_launcher, "app", None)
        runtime_status = ensure_galaxea_legacy_runtime(strict=True)
        result["enabled_extensions"] = list(runtime_status.enabled_extensions)
        result["imported_modules"] = list(runtime_status.imported_modules)

        loaded_module = None
        last_error = None
        for module_name in GALAXEA_TASK_MODULE_CANDIDATES:
            try:
                importlib.import_module(module_name)
                loaded_module = module_name
                break
            except Exception as exc:  # pragma: no cover - runtime-specific
                last_error = exc
        if loaded_module is None:
            if last_error is not None:
                raise last_error
            raise ImportError("No Galaxea task module candidate could be imported.")

        import gymnasium as gym

        registry = getattr(gym, "registry", {})
        missing_env_ids = [env_id for env_id in GALAXEA_R1_ENV_IDS if env_id not in registry]
        if missing_env_ids:
            raise KeyError(f"Missing Galaxea env ids after runtime bootstrap: {missing_env_ids}")

        result["ok"] = True
        result["loaded_module"] = loaded_module
        result["env_ids"] = list(GALAXEA_R1_ENV_IDS)
    except Exception as exc:  # pragma: no cover - runtime-specific
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        try:
            if hasattr(app_launcher, "close"):
                app_launcher.close()
            elif simulation_app is not None and hasattr(simulation_app, "close"):
                simulation_app.close()
        except Exception:
            pass
    return result


def classify_hint(result):
    if result["ok"]:
        return None
    error_text = result.get("error", "")
    module_name = result["name"]
    if module_name == "isaaclab":
        return (
            "isaaclab is unavailable. Install the local extension source from "
            "/home/shangyy/IsaacLab/source/isaaclab with the dedicated metra_isaaclab Python."
        )
    if module_name == "isaaclab_assets":
        return (
            "isaaclab_assets is unavailable. Install the local extension source from "
            "/home/shangyy/IsaacLab/source/isaaclab_assets with the dedicated metra_isaaclab Python."
        )
    if module_name == "isaaclab_tasks":
        return (
            "isaaclab_tasks is unavailable. Install the local extension source from "
            "/home/shangyy/IsaacLab/source/isaaclab_tasks with the dedicated metra_isaaclab Python."
        )
    if module_name == "torch.cuda":
        return (
            "CUDA is not visible inside metra_isaaclab. Check CUDA_VISIBLE_DEVICES, NVIDIA driver/toolkit "
            "visibility, and verify torch.cuda.is_available() with the dedicated Python."
        )
    if module_name == "galaxea_overlay":
        return (
            "Galaxea overlay paths are unavailable. Confirm METRA_GALAXEA_LAB_PATH points to "
            "/home/shangyy/Galaxea_Lab and source scripts/setup/activate_galaxea_overlay.sh."
        )
    if module_name == "galaxea_runtime":
        return (
            "Galaxea runtime bootstrap failed. Launch the check through scripts/setup/check_metra_isaaclab_env.sh "
            "after sourcing the overlay, and verify the deprecated omni.isaac.core compatibility extensions can be "
            "enabled before the local R1 task ids register."
        )
    if module_name in {name for name, _ in GALAXEA_MODULE_GROUPS}:
        return (
            "Galaxea task modules are unavailable. Activate the local /home/shangyy/Galaxea_Lab overlay or install "
            "its legacy extension packages into metra_isaaclab, then verify the old omni.isaac.core-based namespace "
            "can be imported after Isaac Sim startup."
        )
    if module_name in CORE_MODULES:
        return (
            f"{module_name} is unavailable. Create or repair the dedicated metra_isaaclab "
            "environment with scripts/setup/create_metra_isaaclab_env.sh."
        )
    if module_name in KITCHEN_MODULES and "missing MuJoCo" in error_text:
        return (
            "Kitchen/D4RL compatibility is blocked by a missing MuJoCo 2.10/2.1 runtime for mujoco_py. "
            "Set MUJOCO_PY_MUJOCO_PATH to a valid MuJoCo installation before reinstalling the kitchen layer."
        )
    if module_name == "d4rl":
        return "d4rl failed to import. Install the local source tree from /home/shangyy/D4RL inside metra_isaaclab."
    if module_name == "mujoco_py":
        return (
            "mujoco_py failed to import. Verify /home/shangyy/.mujoco/mujoco210 exists, "
            "/usr/lib/nvidia is on LD_LIBRARY_PATH, and reinstall mujoco-py==2.1.2.14 in metra_isaaclab."
        )
    if module_name == "mjrl":
        return "mjrl failed to import. Install the local source tree from /home/shangyy/mjrl inside metra_isaaclab."
    if module_name in OPTIONAL_MODULES:
        return f"{module_name} is optional and can be installed later from the optional extras layer."
    return None


def print_report(title, results):
    print(title)
    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        detail = result.get("file") or result.get("error", "")
        print(f"  - {result['name']}: {status} {detail}".rstrip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--require-cuda", action="store_true", default=False)
    parser.add_argument("--require-galaxea", action="store_true", default=False)
    args = parser.parse_args()

    results = []
    required_groups = [CORE_MODULES]
    if args.full:
        required_groups.extend((MAIN_COMPAT_MODULES, KITCHEN_MODULES))

    for group in required_groups:
        for module_name in group:
            results.append(probe_module(module_name))

    overlay_result = probe_galaxea_overlay()
    if args.require_galaxea:
        results.append(overlay_result)
        if overlay_result["ok"]:
            activate_local_galaxea_overlay(strict=True)

    if args.require_cuda:
        results.append(probe_cuda())
    galaxea_results = probe_galaxea()
    if args.require_galaxea:
        if all(result["ok"] for result in results):
            results.append(probe_galaxea_runtime())

    optional_results = [probe_module(module_name) for module_name in OPTIONAL_MODULES]
    if not args.require_galaxea:
        optional_results.append(overlay_result)
    optional_results.extend(galaxea_results)

    failing_required = [result for result in results if not result["ok"]]
    hints = []
    seen_hints = set()
    for hint in (classify_hint(result) for result in results + optional_results):
        if hint and hint not in seen_hints:
            seen_hints.add(hint)
            hints.append(hint)

    payload = {
        "python": sys.executable,
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "required": results,
        "optional": optional_results,
        "hints": hints,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.quiet and not failing_required:
        pass
    else:
        print_report("Required modules:", results)
        if optional_results:
            print_report("Optional modules:", optional_results)
        if hints:
            print("Hints:")
            for hint in hints:
                print(f"  - {hint}")

    if failing_required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
