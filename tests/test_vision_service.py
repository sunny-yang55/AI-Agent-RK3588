"""Unit tests for background vision service lifecycle."""

import importlib.util
import sys
import threading
import time
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType("vision_service_test_package")
PACKAGE.__path__ = [str(ROOT / "tools/vision")]
sys.modules[PACKAGE.__name__] = PACKAGE


def load_module(name):
    path = ROOT / "tools/vision" / f"{name}.py"
    qualified = f"{PACKAGE.__name__}.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


camera_module = load_module("camera")
session_module = load_module("session")
service_module = load_module("service")
CameraFormat = camera_module.CameraFormat
CameraFrame = camera_module.CameraFrame
VisionService = service_module.VisionService
VisionSessionState = session_module.VisionSessionState


class FakeCamera:
    def __init__(self, *, open_error=None, read_error=None):
        self.open_error = open_error
        self.read_error = read_error
        self.opened = False
        self.closed = False
        self.sequence = 0

    def open(self):
        if self.open_error:
            raise self.open_error
        self.opened = True
        return CameraFormat(1280, 720, 25.0, "MJPG", "fake")

    def read(self):
        if self.read_error:
            raise self.read_error
        self.sequence += 1
        time.sleep(0.001)
        return CameraFrame(self.sequence, time.time(), time.monotonic(), object())

    def close(self):
        self.closed = True


class FakeDisplay:
    def __init__(self, *, close_after=None):
        self.close_after = close_after
        self.frames = 0
        self.closed = False

    def show(self, image, *, sequence):
        self.frames += 1
        return self.close_after is None or self.frames < self.close_after

    def close(self):
        self.closed = True


class BlockingCamera(FakeCamera):
    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def read(self):
        self.release.wait(1.0)
        return super().read()


class VisionServiceTests(unittest.TestCase):
    def test_start_and_stop_release_resources(self):
        camera, display = FakeCamera(), FakeDisplay()
        service = VisionService(
            camera_factory=lambda: camera,
            display_factory=lambda: display,
        )
        self.assertTrue(service.start())
        self.assertEqual(service.session.state, VisionSessionState.ACTIVE)
        self.assertEqual(service.camera_format.width, 1280)
        self.assertTrue(service.stop())
        self.assertTrue(camera.closed)
        self.assertTrue(display.closed)
        self.assertEqual(service.session.state, VisionSessionState.CLOSED)

    def test_repeated_start_is_idempotent(self):
        service = VisionService(
            camera_factory=FakeCamera,
            display_factory=FakeDisplay,
        )
        self.assertTrue(service.start())
        thread = service._thread
        self.assertTrue(service.start())
        self.assertIs(service._thread, thread)
        service.stop()

    def test_window_close_ends_session(self):
        camera, display = FakeCamera(), FakeDisplay(close_after=2)
        service = VisionService(
            camera_factory=lambda: camera,
            display_factory=lambda: display,
        )
        self.assertTrue(service.start())
        service._thread.join(1.0)
        self.assertFalse(service.is_running)
        self.assertTrue(camera.closed)
        self.assertEqual(service.session.state, VisionSessionState.CLOSED)

    def test_open_failure_is_reported_and_released(self):
        camera = FakeCamera(open_error=RuntimeError("camera busy"))
        service = VisionService(
            camera_factory=lambda: camera,
            display_factory=FakeDisplay,
        )
        self.assertFalse(service.start())
        self.assertEqual(service.session.state, VisionSessionState.ERROR)
        self.assertEqual(service.session.error, "camera busy")
        self.assertTrue(camera.closed)

    def test_read_failure_is_reported_and_released(self):
        camera, display = FakeCamera(read_error=RuntimeError("read failed")), FakeDisplay()
        service = VisionService(
            camera_factory=lambda: camera,
            display_factory=lambda: display,
        )
        service.start()
        service._thread.join(1.0)
        self.assertEqual(service.session.state, VisionSessionState.ERROR)
        self.assertTrue(camera.closed)
        self.assertTrue(display.closed)

    def test_stop_timeout_is_reported(self):
        camera = BlockingCamera()
        service = VisionService(
            camera_factory=lambda: camera,
            display_factory=FakeDisplay,
        )
        self.assertTrue(service.start())
        self.assertFalse(service.stop(timeout=0.001))
        self.assertEqual(service.session.state, VisionSessionState.ERROR)
        camera.release.set()
        service._thread.join(1.0)
        self.assertEqual(service.session.state, VisionSessionState.ERROR)


if __name__ == "__main__":
    unittest.main()
