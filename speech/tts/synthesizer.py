"""
Speech Synthesizer Interface（TTS抽象接口）

负责:
1. 文本转语音接口抽象
2. 统一TTS调用入口

未来实现:
Windows:
    pyttsx3
    edge-tts

RK3588:
    piper
    espeak-ng
    离线TTS模型
"""

import logging

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """
    TTS抽象接口
    """

    def __init__(self):
        logger.info("SpeechSynthesizer initialized")

    def speak(self, text):
        """
        文本转语音

        Args:
            text:
                输入文本

        Returns:
            bool
        """

        raise NotImplementedError("TTS backend must implement speak()")
