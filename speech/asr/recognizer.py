"""
Speech Recognition Interface

负责:
1. 音频 -> 文本
2. 提供统一ASR接口

未来支持:
- Whisper
- FunASR
- SenseVoice
- RKNN ASR
"""

import logging

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    ASR抽象接口
    """

    def __init__(self):
        logger.info("SpeechRecognizer initialized")

    def transcribe(self, audio):
        """
        音频识别

        Args:
            audio:
                audio object

        Returns:
            text
        """

        raise NotImplementedError("ASR engine must implement transcribe()")
