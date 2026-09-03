"""Tests for voice-facing vision session control."""

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/vision/session.py"
SPEC = importlib.util.spec_from_file_location("vision_session_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
vision_session = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vision_session
SPEC.loader.exec_module(vision_session)

VisionCommand = vision_session.VisionCommand
VisionSession = vision_session.VisionSession
VisionSessionState = vision_session.VisionSessionState
classify_vision_command = vision_session.classify_vision_command


class VisionCommandTests(unittest.TestCase):
    def test_explicit_open_phrases(self):
        for phrase in ("看", "小安看一下", "帮我看一看", "打开摄像头"):
            self.assertEqual(classify_vision_command(phrase), VisionCommand.OPEN)

    def test_close_takes_precedence(self):
        self.assertEqual(
            classify_vision_command("小安，不看了，关闭摄像头"),
            VisionCommand.CLOSE,
        )

    def test_visual_question_is_describe_command(self):
        for phrase in ("前面有什么", "你看到了什么", "这是什么", "看下前面"):
            self.assertEqual(classify_vision_command(phrase), VisionCommand.DESCRIBE)

    def test_visual_followup_requires_active_session(self):
        phrase = "还有一卷纸看到了吗"
        self.assertIsNone(classify_vision_command(phrase, active=False))
        self.assertEqual(
            classify_vision_command(phrase, active=True),
            VisionCommand.DESCRIBE,
        )

    def test_ambiguous_look_does_not_open(self):
        for phrase in ("我看这件事可以", "查看天气", "看起来不错"):
            self.assertIsNone(classify_vision_command(phrase), phrase)

    def test_short_close_requires_active_session(self):
        self.assertIsNone(classify_vision_command("关闭"))
        self.assertEqual(
            classify_vision_command("关闭", active=True),
            VisionCommand.CLOSE,
        )


class VisionSessionTests(unittest.TestCase):
    def test_normal_lifecycle(self):
        session = VisionSession()
        self.assertTrue(session.request_start())
        self.assertEqual(session.state, VisionSessionState.STARTING)
        session.mark_active()
        self.assertTrue(session.is_active)
        self.assertTrue(session.request_stop())
        self.assertEqual(session.state, VisionSessionState.STOPPING)
        session.mark_closed()
        self.assertEqual(session.state, VisionSessionState.CLOSED)

    def test_start_and_stop_are_idempotent(self):
        session = VisionSession()
        self.assertTrue(session.request_start())
        self.assertFalse(session.request_start())
        self.assertTrue(session.request_stop())
        self.assertFalse(session.request_stop())

    def test_error_can_restart(self):
        session = VisionSession()
        session.mark_error("camera unavailable")
        self.assertEqual(session.state, VisionSessionState.ERROR)
        self.assertEqual(session.error, "camera unavailable")
        self.assertTrue(session.request_start())
        self.assertIsNone(session.error)

    def test_invalid_active_transition_is_rejected(self):
        with self.assertRaises(RuntimeError):
            VisionSession().mark_active()


if __name__ == "__main__":
    unittest.main()
