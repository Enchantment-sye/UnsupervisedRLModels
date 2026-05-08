import os
import sys

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from envs.kitchen import mujoco_compat


def test_configure_kitchen_mujoco_runtime_uses_explicit_device(monkeypatch):
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    monkeypatch.setenv("METRA_KITCHEN_EGL_DEVICE_ID", "3")

    assert mujoco_compat.configure_kitchen_mujoco_runtime() == "3"
    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "3"


def test_configure_kitchen_mujoco_runtime_auto_probes_devices(monkeypatch):
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    monkeypatch.delenv("GL_DEVICE_ID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("METRA_KITCHEN_EGL_DEVICE_ID", "auto")
    monkeypatch.setattr(mujoco_compat, "_EGL_DEVICE_CANDIDATES", ("0", "1"))
    monkeypatch.setattr(mujoco_compat, "_egl_device_can_render", lambda device: device == "1")

    assert mujoco_compat.configure_kitchen_mujoco_runtime() == "1"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"
