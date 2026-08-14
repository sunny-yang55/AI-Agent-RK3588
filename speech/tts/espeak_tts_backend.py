"""Offline eSpeak NG backend for RK3588."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

from .alsa_player import play_wav
from .synthesizer import SpeechSynthesizer

logger = logging.getLogger(__name__)


class EspeakTTS(SpeechSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.executable = shutil.which("espeak-ng")
        if not self.executable:
            raise RuntimeError("espeak-ng is not installed")
        self.voice = os.getenv("ESPEAK_VOICE", "cmn")
        self.speed = int(os.getenv("ESPEAK_SPEED", "140"))
        self.amplitude = int(os.getenv("ESPEAK_AMPLITUDE", "200"))
        if not 80 <= self.speed <= 450:
            raise ValueError("ESPEAK_SPEED must be between 80 and 450")
        if not 0 <= self.amplitude <= 200:
            raise ValueError("ESPEAK_AMPLITUDE must be between 0 and 200")
        logger.info("EspeakTTS initialized")

    def speak(self, text: str) -> bool:
        if not text or len(text.strip()) < 2:
            return False
        handle = tempfile.NamedTemporaryFile(
            prefix="ai_agent_espeak_", suffix=".wav", delete=False
        )
        wav_path = handle.name
        handle.close()
        try:
            subprocess.run(
                [
                    self.executable,
                    "-v",
                    self.voice,
                    "-s",
                    str(self.speed),
                    "-a",
                    str(self.amplitude),
                    "-w",
                    wav_path,
                    text,
                ],
                check=True,
            )
            play_wav(wav_path)
            return True
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            logger.warning("eSpeak NG TTS failed: %s", exc)
            return False
        finally:
            try:
                os.remove(wav_path)
            except FileNotFoundError:
                pass
