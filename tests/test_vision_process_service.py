"""Tests for process-isolated vision preview internals."""

import importlib.util
import sys
import time
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType("vision_process_test_package")
PACKAGE.__path__ = [str(ROOT / "tools/vision")]
sys.modules[PACKAGE.__name__] = PACKAGE


def load_module(name):
    qualified = f"{PACKAGE.__name__}.{name}"
    spec = importlib.util.spec_from_file_location(
        qualified, ROOT / "tools/vision" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


camera_module = load_module("camera")
session_module = load_module("session")
load_module("service")
process_module = load_module("process_service")

CameraConfig = camera_module.CameraConfig
CameraFormat = camera_module.CameraFormat
CameraFrame = camera_module.CameraFrame
ProcessVisionService = process_module.ProcessVisionService
VisionSessionState = session_module.VisionSessionState
run_child = process_module._vision_process_main


class FakeConnection:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def poll(self, timeout=0):
        return False

    def close(self):
        self.closed = True


class FakeCamera:
    def __init__(self, error=None):
        self.error = error
        self.closed = False
        self.sequence = 0

    def open(self):
        if self.error:
            raise self.error
        return CameraFormat(1280, 720, 25.0, "MJPG", "fake")

    def read(self):
        self.sequence += 1
        return CameraFrame(self.sequence, time.time(), time.monotonic(), object())

    def close(self):
        self.closed = True


class ClosingDisplay:
    def __init__(self):
        self.closed = False

    def show(self, image, *, sequence):
        return False

    def close(self):
        self.closed = True


class VisionProcessChildTests(unittest.TestCase):
    def test_child_reports_active_then_closed_and_releases(self):
        connection, camera, display = FakeConnection(), FakeCamera(), ClosingDisplay()
        run_child(
            connection,
            CameraConfig(),
            camera_factory=lambda: camera,
            display_factory=lambda: display,
        )
        self.assertEqual([item["event"] for item in connection.sent], ["active", "closed"])
        self.assertTrue(camera.closed)
        self.assertTrue(display.closed)
        self.assertTrue(connection.closed)

    def test_child_reports_open_error_and_releases(self):
        connection = FakeConnection()
        camera = FakeCamera(RuntimeError("camera busy"))
        run_child(
            connection,
            CameraConfig(),
            camera_factory=lambda: camera,
            display_factory=ClosingDisplay,
        )
        self.assertEqual(connection.sent, [{"event": "error", "error": "camera busy"}])
        self.assertTrue(camera.closed)


class VisionProcessParentTests(unittest.TestCase):
    def test_active_message_updates_format_and_state(self):
        service = ProcessVisionService()
        service.session.request_start()
        service._handle_message(
            {
                "event": "active",
                "format": {
                    "width": 1280,
                    "height": 720,
                    "fps": 25.0,
                    "fourcc": "MJPG",
                    "backend": "V4L2",
                },
            }
        )
        self.assertEqual(service.session.state, VisionSessionState.ACTIVE)
        self.assertEqual(service.camera_format.fps, 25.0)

    def test_error_message_preserves_reason(self):
        service = ProcessVisionService()
        service._handle_message({"event": "error", "error": "camera disconnected"})
        self.assertEqual(service.session.state, VisionSessionState.ERROR)
        self.assertEqual(service.session.error, "camera disconnected")

    def test_closed_message_does_not_hide_error(self):
        service = ProcessVisionService()
        service.session.mark_error("read failed")
        service._handle_message({"event": "closed"})
        self.assertEqual(service.session.state, VisionSessionState.ERROR)


if __name__ == "__main__":
    unittest.main()
