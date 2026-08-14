"""Offline Piper backend for RK3588, enabled when a model is installed."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

from .alsa_player import play_wav
from .synthesizer import SpeechSynthesizer

logger = logging.getLogger(__name__)


def configured_piper_model() -> Path | None:
    value = os.getenv("PIPER_MODEL", "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def piper_available() -> bool:
    model = configured_piper_model()
    return (
        (shutil.which("piper") is not None or importlib.util.find_spec("piper") is not None)
        and model is not None
        and model.is_file()
    )


class PiperTTS(SpeechSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.executable = shutil.which("piper")
        self.model = configured_piper_model()
        self.module_available = importlib.util.find_spec("piper") is not None
        if not self.executable and not self.module_available:
            raise RuntimeError("piper is not installed")
        if self.model is None or not self.model.is_file():
            raise RuntimeError("PIPER_MODEL must point to an installed Piper .onnx model")
        self.voice = None
        self._stopped = threading.Event()
        self._process: subprocess.Popen | None = None
        self._playback_started = None
        if self.module_available:
            # Load once. Re-launching the Piper CLI for every streamed sentence
            # reloads a ~60 MB model and destroys first-sentence latency.
            from piper import PiperVoice

            self.voice = PiperVoice.load(str(self.model))
        logger.info("PiperTTS initialized with %s", self.model)

    def stop(self) -> None:
        self._stopped.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def set_playback_started_callback(self, callback) -> None:
        self._playback_started = callback

    def synthesize_to_file(self, text: str) -> str | None:
        """Synthesize without playing; caller owns the returned temporary file."""
        if not text or len(text.strip()) < 2:
            return None
        handle = tempfile.NamedTemporaryFile(
            prefix="ai_agent_piper_", suffix=".wav", delete=False
        )
        wav_path = handle.name
        handle.close()
        try:
            if self.voice is not None:
                with wave.open(wav_path, "wb") as wav_file:
                    self.voice.synthesize_wav(text, wav_file)
            else:
                # Compatibility with the archived standalone Piper binary.
                subprocess.run(
                    [
                        self.executable,
                        "--model",
                        str(self.model),
                        "--output_file",
                        wav_path,
                    ],
                    input=text,
                    text=True,
                    check=True,
                )
            return wav_path
        except Exception as exc:
            logger.warning("Piper TTS failed: %s", exc)
            try:
                os.remove(wav_path)
            except FileNotFoundError:
                pass
            return None

    def _play_interruptibly(self, wav_path: str) -> bool:
        player = shutil.which("aplay")
        if not player:
            raise RuntimeError("aplay is not installed; install alsa-utils")
        device = os.getenv("AI_AGENT_PLAYBACK_DEVICE", "default")
        self._process = subprocess.Popen(
            [player, "-q", "-D", device, wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self._playback_started is not None:
            self._playback_started()
        try:
            while self._process.poll() is None:
                if self._stopped.is_set():
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                    return False
                time.sleep(0.02)
            return self._process.returncode == 0
        finally:
            self._process = None

    def speak(self, text: str) -> bool:
        self._stopped.clear()
        wav_path = self.synthesize_to_file(text)
        if wav_path is None:
            return False
        try:
            return self._play_interruptibly(wav_path)
        except (OSError, RuntimeError) as exc:
            logger.warning("Piper playback failed: %s", exc)
            return False
        finally:
            try:
                os.remove(wav_path)
            except FileNotFoundError:
                pass
