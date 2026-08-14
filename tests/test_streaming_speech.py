import importlib.util
import time
import unittest
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "runtime/streaming_speech.py"
spec = importlib.util.spec_from_file_location("streaming_speech", module_path)
streaming_speech = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(streaming_speech)
StreamingSpeechPlayer = streaming_speech.StreamingSpeechPlayer


class StreamingSpeechPlayerTests(unittest.TestCase):
    def test_speaks_complete_sentences_in_order(self):
        spoken = []
        player = StreamingSpeechPlayer(spoken.append, min_chars=4, max_chars=20)
        player.feed("安徽工程")
        player.feed("大学位于芜湖。它是一所")
        player.feed("工科院校！")
        player.finish()
        self.assertEqual(spoken, ["安徽工程大学位于芜湖。", "它是一所工科院校！"])

    def test_flushes_tail(self):
        spoken = []
        player = StreamingSpeechPlayer(spoken.append, min_chars=4, max_chars=20)
        player.feed("这是没有句号的结尾")
        player.finish()
        self.assertEqual(spoken, ["这是没有句号的结尾"])

    def test_generation_and_speech_overlap(self):
        spoken = []

        def slow_speak(text):
            time.sleep(0.05)
            spoken.append(text)

        player = StreamingSpeechPlayer(slow_speak, min_chars=4, max_chars=20)
        started = time.perf_counter()
        player.feed("第一句话。")
        feed_elapsed = time.perf_counter() - started
        player.feed("第二句话。")
        player.finish()
        self.assertLess(feed_elapsed, 0.04)
        self.assertEqual(spoken, ["第一句话。", "第二句话。"])

    def test_never_splits_long_text_without_sentence_end(self):
        spoken = []
        player = StreamingSpeechPlayer(spoken.append, min_chars=4, max_chars=10)
        player.feed("中国科学技术大学和中国科学院合肥物质科学研究院")
        self.assertEqual(spoken, [])
        player.feed("位于合肥。")
        player.finish()
        self.assertEqual(
            spoken,
            ["中国科学技术大学和中国科学院合肥物质科学研究院位于合肥。"],
        )

    def test_nonblocking_finish_then_wait(self):
        spoken = []

        def slow_speak(text):
            time.sleep(0.08)
            spoken.append(text)

        player = StreamingSpeechPlayer(slow_speak)
        player.feed("第一句话。第二句话。")
        started = time.perf_counter()
        player.finish(wait=False)
        self.assertLess(time.perf_counter() - started, 0.04)
        player.wait()
        self.assertEqual(spoken, ["第一句话。", "第二句话。"])


if __name__ == "__main__":
    unittest.main()
