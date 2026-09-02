import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "build-v1.3.4"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT))

from speech.asr.chunked_rknn_sensevoice_asr import _merge_text
from tools.speech.speech_tool import (
    SpeechTool, classify_local_command, classify_playback_command, normalize,
    strip_wake_word,
)


class WakeWordTests(unittest.TestCase):
    def test_wake_only(self):
        woke, command = strip_wake_word("小安，小安。")
        self.assertTrue(woke)
        self.assertEqual(command, "")

    def test_wake_and_command_same_utterance(self):
        woke, command = strip_wake_word("小安小安，介绍一下安徽芜湖。")
        self.assertTrue(woke)
        self.assertEqual(command, "介绍一下安徽芜湖")

    def test_normalize_asr_tags(self):
        self.assertEqual(normalize("<|zh|> 你好，小安！"), "你好小安")

    def test_wake_opens_non_expiring_conversation(self):
        with patch.dict(os.environ, {
            "AI_AGENT_ALWAYS_LISTEN": "0",
            "AI_AGENT_WAKE_WORD_ENABLED": "1",
            "AI_AGENT_CONTINUOUS_DIALOG_SECONDS": "15",
        }, clear=True):
            tool = SpeechTool()
        tool.open_conversation_window()
        self.assertEqual(tool._awake_until, float("inf"))

    def test_default_requires_literal_wake_once(self):
        with patch.dict(os.environ, {}, clear=True):
            tool = SpeechTool()
        self.assertEqual(tool.continuous_seconds, 0.0)
        self.assertFalse(tool.always_listen)
        self.assertTrue(tool.wake_required)

    def test_legacy_wake_gate_can_still_be_enabled(self):
        with patch.dict(os.environ, {
            "AI_AGENT_ALWAYS_LISTEN": "0",
            "AI_AGENT_WAKE_WORD_ENABLED": "1",
        }, clear=True):
            tool = SpeechTool()
        self.assertFalse(tool.always_listen)
        self.assertTrue(tool.wake_required)

    def test_always_listen_window_never_expires(self):
        with patch.dict(os.environ, {"AI_AGENT_ALWAYS_LISTEN": "1"}, clear=True):
            tool = SpeechTool()
        with patch("tools.speech.speech_tool.time.monotonic", return_value=100.0):
            tool.open_conversation_window()
        self.assertEqual(tool._awake_until, float("inf"))

    def test_normal_utterance_never_needs_wake_in_always_listen_mode(self):
        with patch.dict(os.environ, {"AI_AGENT_ALWAYS_LISTEN": "1"}, clear=True):
            tool = SpeechTool()
        tool.initialized = True
        tool.audio = SimpleNamespace(record=lambda: {"samples": [1]})
        tool.asr = SimpleNamespace(transcribe=lambda audio: "简单介绍一下安徽芜湖。")
        tool._awake_until = 0.0  # Regression: even stale legacy state cannot gate it.
        result = tool.listen()
        self.assertTrue(result.success)
        self.assertEqual(result.text, "简单介绍一下安徽芜湖")

    def test_sensevoice_homophone_does_not_wake(self):
        woke, command = strip_wake_word("向安，介绍一下安徽工程大学。")
        self.assertFalse(woke)
        self.assertEqual(command, "向安介绍一下安徽工程大学")

    def test_xiao_han_homophone_does_not_wake(self):
        woke, command = strip_wake_word("小韩你好。")
        self.assertFalse(woke)
        self.assertEqual(command, "小韩你好")

    def test_literal_wake_works_in_any_position(self):
        for phrase in ("小安", "你好小安", "小安小安", "请问小安在吗"):
            woke, _ = strip_wake_word(phrase)
            self.assertTrue(woke, phrase)

    def test_playback_stop_phrases_require_wake(self):
        self.assertEqual(
            classify_local_command("小安，停止", require_wake=True), "stop"
        )
        self.assertEqual(
            classify_local_command("小安，听一下", require_wake=True), "stop"
        )
        self.assertIsNone(classify_local_command("停止", require_wake=True))

    def test_playback_stops_on_any_text_containing_stop_character(self):
        for phrase in ("停止", "停", "请停一下", "不是让你停了吗", "小安停止"):
            self.assertEqual(classify_playback_command(phrase), "stop", phrase)

    def test_exit_is_global(self):
        self.assertEqual(classify_local_command("好的，再见"), "exit")

    def test_exit_variants_are_global(self):
        for phrase in ("小安再见", "再见小安", "再见"):
            self.assertEqual(classify_local_command(phrase), "exit")

    def test_late_stop_is_not_an_llm_request(self):
        self.assertEqual(classify_local_command("小安停止等一下"), "stop")

    def test_audio_overlap_text_is_deduplicated(self):
        self.assertEqual(_merge_text("请介绍安徽芜湖", "安徽芜湖的历史"), "请介绍安徽芜湖的历史")


if __name__ == "__main__":
    unittest.main()
