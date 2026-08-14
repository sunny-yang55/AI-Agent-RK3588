"""
Mock TTS Backend（Mock后端）

当前:
模拟语音播放

未来:
替换:
- Windows TTS
- RK3588 TTS
"""

import logging

from .synthesizer import SpeechSynthesizer

logger = logging.getLogger(__name__)


class MockTTS(SpeechSynthesizer):

    def __init__(self):

        super().__init__()

        logger.info("MockTTS initialized")

    def speak(self, text):

        if not text:
            logger.warning("Empty text received")

            return False

        print("[TTS OUTPUT]")
        print(text)

        return True
