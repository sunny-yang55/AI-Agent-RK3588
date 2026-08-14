"""
TTS Engine

统一TTS调用入口

SpeechTool
    |
    ↓
TTSEngine
    |
    ↓
SpeechSynthesizer backend
"""

import logging
import os
import platform
import shutil
import threading
import time
import voice_ui as ui

from .piper_tts_backend import piper_available

logger = logging.getLogger(__name__)


class TTSEngine:

    def __init__(self):
        requested = os.getenv("AI_AGENT_TTS_BACKEND", "auto").strip().lower()
        if requested not in {"auto", "piper", "espeak", "edge", "none"}:
            raise ValueError(
                "AI_AGENT_TTS_BACKEND must be auto, piper, espeak, edge or none"
            )
        backend = self._select_backend(requested)
        self.backend = self._create_backend(backend)
        self._active_backend = self.backend
        self.backend_name = backend
        self._stop_requested = threading.Event()
        self._playback_started = threading.Event()
        self._install_playback_callback(self.backend)
        ui.debug(f"[系统] TTS后端: {self._backend_label(backend)}")

    def prepare_speak(self) -> None:
        """Reset per-turn events before the barge-in waiter is started."""
        self._stop_requested.clear()
        self._playback_started.clear()

    def _install_playback_callback(self, backend) -> None:
        setter = getattr(backend, "set_playback_started_callback", None)
        if setter is not None:
            setter(self._playback_started.set)

    def wait_for_playback_start(self, stop_event, poll_seconds=0.05) -> bool:
        while not stop_event.is_set():
            if self._playback_started.wait(poll_seconds):
                return True
        return False

    @staticmethod
    def _select_backend(requested: str) -> str:
        if requested != "auto":
            return requested
        if platform.system() == "Windows":
            return "edge"
        if piper_available():
            return "piper"
        if shutil.which("espeak-ng"):
            return "espeak"
        return "none"

    @staticmethod
    def _create_backend(name: str):
        if name == "piper":
            from .piper_tts_backend import PiperTTS

            return PiperTTS()
        if name == "espeak":
            from .espeak_tts_backend import EspeakTTS

            return EspeakTTS()
        if name == "edge":
            from .edge_tts_backend import EdgeTTS

            return EdgeTTS()
        if name == "none":
            from .mock_tts import MockTTS

            return MockTTS()
        raise AssertionError(f"unsupported TTS backend: {name}")

    @staticmethod
    def _backend_label(name: str) -> str:
        return {
            "piper": "Piper（离线）",
            "espeak": "espeak-ng（离线）",
            "edge": "Edge-TTS（在线普通话）",
            "none": "未启用，仅显示文字",
        }[name]

    def speak(self, text):

        if not text:

            return False

        self._active_backend = self.backend
        if self.backend.speak(text):
            return True
        if self._stop_requested.is_set():
            return False
        if getattr(self.backend, "partial_output", False):
            ui.debug("[TTS] 已播出部分内容；为避免重复，不再从头回退整篇")
            return False

        # Edge is an online service and can legitimately return no audio.
        # Prefer a configured local voice, then espeak-ng, and only then make
        # bounded final Edge attempts so a transient outage does not silently
        # discard an entire answer.
        if self.backend_name == "edge":
            if piper_available():
                ui.debug("[TTS] Edge-TTS失败，自动回退 Piper（离线）")
                fallback = self._create_backend("piper")
                self._install_playback_callback(fallback)
                self._active_backend = fallback
                if fallback.speak(text):
                    return True
            if shutil.which("espeak-ng"):
                ui.debug("[TTS] Edge-TTS失败，自动回退 espeak-ng（离线）")
                fallback = self._create_backend("espeak")
                self._playback_started.set()
                self._active_backend = fallback
                if fallback.speak(text):
                    return True

            attempts = max(0, int(os.getenv("AI_AGENT_TTS_FAILOVER_RETRIES", "2")))
            for attempt in range(1, attempts + 1):
                if self._stop_requested.is_set():
                    return False
                delay = min(2.0, 0.5 * attempt)
                ui.debug(f"[TTS] 离线后端不可用，等待 {delay:.1f}s 后重试 Edge {attempt}/{attempts}")
                time.sleep(delay)
                self._active_backend = self.backend
                if self.backend.speak(text):
                    ui.debug(f"[TTS] Edge 第 {attempt} 次容错重试成功")
                    return True

        ui.debug("[TTS] 所有可用播报后端均失败，本轮仅保留文字")
        return False

    def stop(self):
        self._stop_requested.set()
        stop = getattr(self._active_backend, "stop", None)
        if stop is not None:
            stop()
