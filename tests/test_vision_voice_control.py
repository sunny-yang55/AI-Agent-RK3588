"""Tests for local voice control of the visual process."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TOOLS_PACKAGE = types.ModuleType("tools")
TOOLS_PACKAGE.__path__ = [str(ROOT / "tools")]
VISION_PACKAGE = types.ModuleType("tools.vision")
VISION_PACKAGE.__path__ = [str(ROOT / "tools/vision")]
sys.modules.setdefault("tools", TOOLS_PACKAGE)
sys.modules.setdefault("tools.vision", VISION_PACKAGE)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("tools.vision.session", ROOT / "tools/vision/session.py")
controller_module = load("vision_control_under_test", ROOT / "runtime/vision_control.py")
VisionVoiceController = controller_module.VisionVoiceController


class FakeService:
    def __init__(self, *, running=False, start_result=True, stop_result=True):
        self.running = running
        self.start_result = start_result
        self.stop_result = stop_result
        self.starts = 0
        self.stops = 0
        self.session = SimpleNamespace(error=None)
        self.description = "我看到1个人。"
        self.query = None

    @property
    def is_running(self):
        return self.running

    def start(self):
        self.starts += 1
        self.running = self.start_result
        return self.start_result

    def stop(self):
        self.stops += 1
        self.running = False
        return self.stop_result

    def describe(self, query=""):
        self.query = query
        return self.description


class VisionVoiceControllerTests(unittest.TestCase):
    def make_controller(self, service):
        spoken = []

        def speak(text, **kwargs):
            spoken.append((text, kwargs))

        return VisionVoiceController(service, speak), spoken

    def test_open_is_handled_locally(self):
        service = FakeService()
        controller, spoken = self.make_controller(service)
        self.assertTrue(controller.handle("小安打开摄像头"))
        self.assertEqual(service.starts, 1)
        self.assertEqual(spoken[0][0], "摄像头已打开。")

    def test_close_stops_running_service(self):
        service = FakeService(running=True)
        controller, spoken = self.make_controller(service)
        self.assertTrue(controller.handle("关闭摄像头"))
        self.assertEqual(service.stops, 1)
        self.assertEqual(spoken[0][0], "摄像头已关闭。")

    def test_unrelated_speech_is_not_consumed(self):
        service = FakeService()
        controller, spoken = self.make_controller(service)
        self.assertFalse(controller.handle("介绍一下安徽芜湖"))
        self.assertEqual(service.starts, 0)
        self.assertEqual(spoken, [])

    def test_start_failure_is_spoken(self):
        service = FakeService(start_result=False)
        service.session.error = "camera busy"
        controller, spoken = self.make_controller(service)
        self.assertTrue(controller.handle("打开摄像头"))
        self.assertIn("camera busy", spoken[0][0])
        self.assertTrue(spoken[0][1]["allow_interrupt"])

    def test_visual_question_speaks_latest_description(self):
        service = FakeService(running=True)
        controller, spoken = self.make_controller(service)
        self.assertTrue(controller.handle("前面有什么"))
        self.assertEqual(spoken[0][0], "我看到1个人。")
        self.assertEqual(service.query, "前面有什么")
        self.assertTrue(spoken[0][1]["allow_interrupt"])

    def test_visual_question_does_not_start_camera_when_closed(self):
        service = FakeService()
        controller, spoken = self.make_controller(service)
        self.assertFalse(controller.handle("这是什么"))
        self.assertEqual(service.starts, 0)
        self.assertEqual(spoken, [])

    def test_runtime_close_is_silent(self):
        service = FakeService(running=True)
        controller, spoken = self.make_controller(service)
        controller.close()
        self.assertEqual(service.stops, 1)
        self.assertEqual(spoken, [])

    def test_legacy_yolo_registration_is_opt_in(self):
        source = (ROOT / "tools/loader.py").read_text(encoding="utf-8")
        self.assertIn('AI_AGENT_LEGACY_VISION_TOOL", "0"', source)


if __name__ == "__main__":
    unittest.main()
