import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "build-v1.3.4"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT))


class WholeTTSModeTests(unittest.TestCase):
    def test_runtime_defaults_to_whole_answer_mode_for_gapless_playback(self):
        source = (ROOT / "runtime/runtime_manager.py").read_text(encoding="utf-8")
        self.assertIn('AI_AGENT_TTS_MODE", "whole"', source)
        self.assertIn('tts_mode == "sentence"', source)

    def test_user_mode_prints_one_coherent_answer_block(self):
        source = (ROOT / "runtime/runtime_manager.py").read_text(encoding="utf-8")
        self.assertIn("ui.assistant(text)", source)
        self.assertIn("if not ui.USER_MODE", source)

    def test_edge_backend_is_interruptible(self):
        source = (ROOT / "speech/tts/edge_tts_backend.py").read_text(encoding="utf-8")
        self.assertIn("def stop(self)", source)
        self.assertIn("pygame.mixer.music.stop()", source)

    def test_long_edge_tts_is_chunked_and_parallel(self):
        source = (ROOT / "speech/tts/edge_tts_backend.py").read_text(encoding="utf-8")
        self.assertIn("def _split_text", source)
        self.assertIn("ThreadPoolExecutor", source)
        self.assertIn("AI_AGENT_EDGE_PARALLEL", source)
        self.assertIn("AI_AGENT_EDGE_RETRIES", source)

    def test_first_chunk_plays_without_waiting_for_all_chunks(self):
        source = (ROOT / "speech/tts/edge_tts_backend.py").read_text(encoding="utf-8")
        self.assertIn("for expected, future in enumerate(futures)", source)
        self.assertIn("future.result()", source)
        self.assertNotIn("as_completed", source)

    def test_decoded_duration_guards_against_early_segment_switch(self):
        source = (ROOT / "speech/tts/edge_tts_backend.py").read_text(encoding="utf-8")
        self.assertIn("pygame.mixer.Sound(filename)", source)
        self.assertIn("sound.get_length()", source)
        self.assertIn("time.monotonic() < deadline", source)

    def test_tts_engine_has_offline_failover_and_bounded_edge_retry(self):
        source = (ROOT / "speech/tts/tts_engine.py").read_text(encoding="utf-8")
        self.assertIn("Edge-TTS失败，自动回退 Piper", source)
        self.assertIn("Edge-TTS失败，自动回退 espeak-ng", source)
        self.assertIn("AI_AGENT_TTS_FAILOVER_RETRIES", source)
        self.assertIn("self._stop_requested.is_set()", source)

    def test_edge_failure_gets_bounded_final_retry(self):
        from speech.tts.tts_engine import TTSEngine
        import threading

        engine = TTSEngine.__new__(TTSEngine)
        engine.backend = Mock()
        engine.backend.speak.side_effect = [False, True]
        engine.backend.partial_output = False
        engine.backend_name = "edge"
        engine._active_backend = engine.backend
        engine._stop_requested = threading.Event()
        engine._playback_started = threading.Event()
        with patch.dict(os.environ, {"AI_AGENT_TTS_FAILOVER_RETRIES": "1"}), \
             patch("speech.tts.tts_engine.piper_available", return_value=False), \
             patch("speech.tts.tts_engine.shutil.which", return_value=None), \
             patch("speech.tts.tts_engine.time.sleep"):
            self.assertTrue(engine.speak("测试语音"))
        self.assertEqual(engine.backend.speak.call_count, 2)

    def test_user_barge_in_does_not_restart_failed_edge(self):
        from speech.tts.tts_engine import TTSEngine
        import threading

        engine = TTSEngine.__new__(TTSEngine)
        engine.backend = Mock()
        engine.backend_name = "edge"
        engine._active_backend = engine.backend
        engine._stop_requested = threading.Event()
        engine._playback_started = threading.Event()
        engine.backend.speak.side_effect = lambda text: engine._stop_requested.set() or False
        self.assertFalse(engine.speak("测试语音"))
        self.assertEqual(engine.backend.speak.call_count, 1)

    def test_edge_never_mixes_piper_inside_one_sentence(self):
        source = (ROOT / "speech/tts/edge_tts_backend.py").read_text(encoding="utf-8")
        self.assertNotIn("_piper_chunk", source)
        self.assertIn("Edge sentence synthesis failed", source)

    def test_offline_voice_fallback_is_opt_in(self):
        source = (ROOT / "speech/tts/tts_engine.py").read_text(encoding="utf-8")
        self.assertIn('AI_AGENT_TTS_OFFLINE_FALLBACK", "0"', source)

    def test_partial_edge_output_is_never_replayed_from_start(self):
        source = (ROOT / "speech/tts/tts_engine.py").read_text(encoding="utf-8")
        self.assertIn('getattr(self.backend, "partial_output", False)', source)
        self.assertIn("为避免重复，不再从头回退整篇", source)

    def test_piper_playback_is_interruptible(self):
        source = (ROOT / "speech/tts/piper_tts_backend.py").read_text(encoding="utf-8")
        self.assertIn("def stop(self)", source)
        self.assertIn("subprocess.Popen", source)
        self.assertIn("self._process.terminate()", source)

    def test_voice_prompt_requires_short_complete_answer(self):
        responder = (ROOT / "agent/responder.py").read_text(encoding="utf-8")
        adapter = (ROOT / "tools/llm/adapter.py").read_text(encoding="utf-8")
        self.assertIn("最多120个汉字", responder)
        self.assertIn("必须以完整句子结束", responder)
        self.assertIn("最多120个汉字", adapter)
        self.assertIn("接近长度限制时主动结束当前句", adapter)

    def test_edge_chunk_has_hard_timeout_and_nonblocking_shutdown(self):
        source = (ROOT / "speech/tts/edge_tts_backend.py").read_text(encoding="utf-8")
        self.assertIn("AI_AGENT_EDGE_TIMEOUT", source)
        self.assertIn("asyncio.wait_for", source)
        self.assertIn("pool.shutdown(wait=False, cancel_futures=True)", source)

    def test_barge_in_waits_for_real_playback(self):
        speech = (ROOT / "tools/speech/speech_tool.py").read_text(encoding="utf-8")
        engine = (ROOT / "speech/tts/tts_engine.py").read_text(encoding="utf-8")
        self.assertIn("wait_for_playback_start(stop_event)", speech)
        self.assertIn("播放已开始，启用语音打断监听", speech)
        self.assertIn("self._playback_started", engine)

    def test_playback_event_is_prepared_before_monitor_thread(self):
        speech = (ROOT / "tools/speech/speech_tool.py").read_text(encoding="utf-8")
        prepare = speech.index("self.tts.prepare_speak()")
        start = speech.index("monitor.start()")
        self.assertLess(prepare, start)

    def test_user_ui_and_debug_mode_exist(self):
        ui_source = (ROOT / "voice_ui.py").read_text(encoding="utf-8")
        run_source = (ROOT / "run_rk3588.sh").read_text(encoding="utf-8")
        self.assertIn("小安 · RK3588 智能语音助手", ui_source)
        self.assertIn("AI_AGENT_UI_MODE", ui_source)
        self.assertIn("voice-debug.log", ui_source)
        self.assertIn('AI_AGENT_UI_MODE:-user', run_source)

    def test_voice_prompt_is_shorter(self):
        responder = (ROOT / "agent/responder.py").read_text(encoding="utf-8")
        self.assertIn("最多120个汉字", responder)

    def test_user_ui_hides_noisy_recognition_state(self):
        speech = (ROOT / "tools/speech/speech_tool.py").read_text(encoding="utf-8")
        self.assertNotIn('ui.state("正在识别…", "◌")', speech)
        self.assertIn('if not ui.USER_MODE:', speech)

    def test_ui_displays_pre_rewrite_utterance(self):
        runtime = (ROOT / "runtime/runtime_manager.py").read_text(encoding="utf-8")
        preserve = runtime.index("display_text = text")
        rewrite = runtime.index("self.context_resolver.resolve")
        display = runtime.index("ui.user(display_text)")
        self.assertLess(preserve, rewrite)
        self.assertLess(rewrite, display)

    def test_non_interruptible_exit_has_clean_status(self):
        speech = (ROOT / "tools/speech/speech_tool.py").read_text(encoding="utf-8")
        runtime = (ROOT / "runtime/runtime_manager.py").read_text(encoding="utf-8")
        self.assertIn('if allow_interrupt else "正在播报…"', speech)
        self.assertIn('ui.state("语音助手已退出", "✓")', runtime)

    def test_default_response_token_guard_is_reduced(self):
        adapter = (ROOT / "tools/llm/adapter.py").read_text(encoding="utf-8")
        self.assertIn('AI_AGENT_MAX_RESPONSE_TOKENS", "240"', adapter)

    def test_llm_timeout_and_retry_are_bounded(self):
        adapter = (ROOT / "tools/llm/adapter.py").read_text(encoding="utf-8")
        self.assertIn('AI_AGENT_LLM_TIMEOUT", "20"', adapter)
        self.assertIn('AI_AGENT_LLM_RETRIES", "1"', adapter)
        self.assertIn("max_retries=0", adapter)
        self.assertIn("Never replay a stream", adapter)


if __name__ == "__main__":
    unittest.main()
