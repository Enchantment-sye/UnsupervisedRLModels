import contextlib
import os
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class _LauncherState:
    app_launcher: Any
    simulation_app: Any
    settings: Dict[str, Any]
    ref_count: int = 1


_LOCK = threading.Lock()
_STATE: Optional[_LauncherState] = None


def _startup_is_quiet() -> bool:
    return os.environ.get("METRA_ISAACLAB_QUIET", "1").strip().lower() not in ("0", "false", "no", "off")


def _default_quiet_kit_args() -> str:
    return " ".join(
        [
            "--/app/enableStdoutOutput=0",
            "--/log/outputStreamLevel=Error",
            "--/log/channels/carb.*=error",
            "--/log/channels/omni.*=error",
            "--/log/channels/rtx.*=error",
            "--/log/channels/gpu.foundation.plugin=error",
            "--/log/channels/pxr.*=error",
        ]
    )


def _ensure_writable_tmpdir() -> str:
    current_tmp = tempfile.gettempdir()
    target_tmp = os.path.join(os.path.expanduser("~"), ".cache", "isaaclab_tmp")
    isaaclab_tmp = os.path.join(target_tmp, "isaaclab", "logs")

    if os.access(current_tmp, os.W_OK):
        existing_logs = os.path.join(current_tmp, "isaaclab", "logs")
        if not os.path.exists(existing_logs) or os.access(existing_logs, os.W_OK):
            return current_tmp

    os.makedirs(isaaclab_tmp, exist_ok=True)
    os.environ["TMPDIR"] = target_tmp
    tempfile.tempdir = target_tmp
    return target_tmp


def _startup_log_dir() -> str:
    base_tmp = _ensure_writable_tmpdir()
    log_dir = os.path.join(base_tmp, "isaaclab", "startup_logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


@contextlib.contextmanager
def _redirect_native_output(log_path: str):
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "a", buffering=1) as sink:
        try:
            try:
                original_stdout.flush()
                original_stderr.flush()
            except Exception:
                pass
            sys.stdout = sink
            sys.stderr = sink
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            try:
                sink.flush()
            except Exception:
                pass
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_with_isaaclab_startup_capture(section: str, callback, *args, **kwargs):
    if not _startup_is_quiet():
        return callback(*args, **kwargs)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(_startup_log_dir(), f"{section}_{timestamp}_{os.getpid()}.log")
    try:
        with _redirect_native_output(log_path):
            return callback(*args, **kwargs)
    except Exception:
        print(f"[error] Isaac Lab startup failed. See {log_path}", file=sys.stderr)
        raise


def _extract_request_settings(request) -> Dict[str, Any]:
    settings = {
        "headless": bool(request.headless),
        "enable_cameras": bool(request.enable_cameras),
        "device": request.device,
    }
    if _startup_is_quiet():
        settings["kit_args"] = _default_quiet_kit_args()
    return settings


def acquire_isaaclab_app(request):
    global _STATE
    settings = _extract_request_settings(request)
    with _LOCK:
        if _STATE is None:
            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            _ensure_writable_tmpdir()
            try:
                from isaaclab.app import AppLauncher
            except ImportError as exc:
                raise ImportError(
                    "Isaac Lab backend requested but isaaclab is not installed. "
                    "Install Isaac Lab and Isaac Sim-compatible packages in the target environment."
                ) from exc

            app_launcher = run_with_isaaclab_startup_capture("app_launcher", AppLauncher, dict(settings))
            simulation_app = getattr(app_launcher, "app", None)
            _STATE = _LauncherState(
                app_launcher=app_launcher,
                simulation_app=simulation_app,
                settings=settings,
                ref_count=1,
            )
            return simulation_app

        _STATE.ref_count += 1
        if _STATE.settings != settings:
            warnings.warn(
                "Isaac Lab app already initialized with different launcher settings; "
                "reusing the existing SimulationApp instance.",
                RuntimeWarning,
            )
        return _STATE.simulation_app


def release_isaaclab_app():
    global _STATE
    with _LOCK:
        if _STATE is None:
            return
        _STATE.ref_count -= 1
        if _STATE.ref_count > 0:
            return

        try:
            if hasattr(_STATE.app_launcher, "close"):
                _STATE.app_launcher.close()
            elif _STATE.simulation_app is not None and hasattr(_STATE.simulation_app, "close"):
                _STATE.simulation_app.close()
        finally:
            _STATE = None
