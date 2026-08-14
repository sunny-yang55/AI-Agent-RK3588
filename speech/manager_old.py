"""
Speech Manager

Speech v0.7.1

负责统一管理：

Microphone
    |
Audio
    |
ASR
    |
Text


未来扩展:

TTS
VAD
Streaming ASR
"""

import logging

from speech.asr import SenseVoiceASR
from speech.audio import MicrophoneInput

logger = logging.getLogger(__name__)


class SpeechManager:
    """
    Speech Runtime Manager

    统一语音入口
    """

    def __init__(self):

        logger.info("[Speech] Initializing...")

        # Audio backend

        self.audio = MicrophoneInput()

        # ASR backend

        self.asr = SenseVoiceASR()

        logger.info("[Speech] Ready")

    def listen(self):
        """
        采集语音并识别

        Returns:
            text:str
        """

        logger.info("[Speech] Recording...")

        audio = self.audio.record()

        logger.info("[Speech] Audio captured")

        text = self.asr.transcribe(audio)

        logger.info("[Speech] ASR result: %s", text)

        return text
