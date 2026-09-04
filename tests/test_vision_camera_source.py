"""Unit tests for tools.vision.camera without physical camera hardware."""

import unittest

from tools.vision.camera import (
    CameraConfig,
    CameraNotOpenError,
    CameraOpenError,
    CameraReadError,
    OpenCVCameraSource,
    discover_camera_devices,
)


class FakeCV:
    CAP_V4L2 = 200
    CAP_PROP_FOURCC = 6
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38

    @staticmethod
    def VideoWriter_fourcc(*characters):
        value = 0
        for index, character in enumerate(characters):
            value |= ord(character) << (8 * index)
        return value


class FakeCapture:
    def __init__(self, opened=True, frames=None):
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False
        self.values = {
            FakeCV.CAP_PROP_FRAME_WIDTH: 1280.0,
            FakeCV.CAP_PROP_FRAME_HEIGHT: 720.0,
            FakeCV.CAP_PROP_FPS: 30.0,
            FakeCV.CAP_PROP_FOURCC: float(FakeCV.VideoWriter_fourcc(*"MJPG")),
        }

    def isOpened(self):
        return self.opened and not self.released

    def set(self, prop, value):
        if prop != FakeCV.CAP_PROP_BUFFERSIZE:
            self.values[prop] = float(value)
        return True

    def get(self, prop):
        return self.values.get(prop, 0.0)

    def getBackendName(self):
        return "V4L2"

    def read(self):
        if not self.frames:
            return False, None
        return self.frames.pop(0)

    def release(self):
        self.released = True


class OpenCVCameraSourceTests(unittest.TestCase):
    def make_source(self, capture):
        calls = []

        def factory(device, backend):
            calls.append((device, backend))
            return capture

        source = OpenCVCameraSource(
            CameraConfig(device="/dev/test-camera"),
            cv_module=FakeCV,
            capture_factory=factory,
        )
        return source, calls

    def test_open_applies_config_and_reports_negotiated_format(self):
        capture = FakeCapture()
        source, calls = self.make_source(capture)

        camera_format = source.open()

        self.assertEqual(calls, [("/dev/test-camera", FakeCV.CAP_V4L2)])
        self.assertTrue(source.is_open)
        self.assertEqual(camera_format.width, 1280)
        self.assertEqual(camera_format.height, 720)
        self.assertEqual(camera_format.fps, 30.0)
        self.assertEqual(camera_format.fourcc, "MJPG")
        self.assertEqual(camera_format.backend, "V4L2")

    def test_open_is_idempotent(self):
        capture = FakeCapture()
        source, calls = self.make_source(capture)

        first = source.open()
        second = source.open()

        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)

    def test_read_returns_sequenced_frames(self):
        first_image = object()
        second_image = object()
        capture = FakeCapture(frames=[(True, first_image), (True, second_image)])
        source, _ = self.make_source(capture)
        source.open()

        first = source.read()
        second = source.read()

        self.assertEqual(first.sequence, 1)
        self.assertEqual(second.sequence, 2)
        self.assertIs(first.image, first_image)
        self.assertIs(second.image, second_image)
        self.assertGreater(first.captured_at, 0)
        self.assertGreater(first.monotonic_at, 0)

    def test_read_before_open_raises(self):
        source, _ = self.make_source(FakeCapture())

        with self.assertRaises(CameraNotOpenError):
            source.read()

    def test_failed_open_releases_capture(self):
        capture = FakeCapture(opened=False)
        source, _ = self.make_source(capture)

        with self.assertRaises(CameraOpenError):
            source.open()

        self.assertTrue(capture.released)
        self.assertFalse(source.is_open)

    def test_failed_read_raises(self):
        source, _ = self.make_source(FakeCapture())
        source.open()

        with self.assertRaises(CameraReadError):
            source.read()

    def test_close_is_idempotent(self):
        capture = FakeCapture()
        source, _ = self.make_source(capture)
        source.open()

        source.close()
        source.close()

        self.assertTrue(capture.released)
        self.assertFalse(source.is_open)
        self.assertIsNone(source.negotiated_format)

    def test_context_manager_closes_capture(self):
        capture = FakeCapture(frames=[(True, object())])
        source, _ = self.make_source(capture)

        with source as opened_source:
            self.assertTrue(opened_source.is_open)
            opened_source.read()

        self.assertTrue(capture.released)
        self.assertFalse(source.is_open)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            CameraConfig(width=0)
        with self.assertRaises(ValueError):
            CameraConfig(fourcc="RGB")
        with self.assertRaises(ValueError):
            CameraConfig(buffer_size=0)

    def test_default_config_uses_automatic_discovery(self):
        self.assertEqual(CameraConfig().device, "auto")

    def test_discovery_prefers_override_and_deduplicates_nodes(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {"VISION_CAMERA_DEVICE": "/dev/video41"}), patch(
            "tools.vision.camera.glob.glob",
            side_effect=[
                ["/dev/v4l/by-id/camera-video-index0"],
                ["/sys/class/video4linux/video41"],
            ],
        ), patch(
            "tools.vision.camera.os.path.realpath",
            side_effect=lambda path: (
                "/sys/devices/usb/camera" if path.endswith("/device") else "/dev/video41"
            ),
        ):
            self.assertEqual(discover_camera_devices(), ["/dev/video41"])


if __name__ == "__main__":
    unittest.main()
