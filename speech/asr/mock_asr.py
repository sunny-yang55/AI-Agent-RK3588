"""
Mock ASR Engine

用于:
- 架构测试
- Pipeline测试

未来替换:
Whisper/FunASR/RKNN
"""

import logging

from .recognizer import SpeechRecognizer

logger = logging.getLogger(__name__)


class MockASR(SpeechRecognizer):

    def __init__(self):

        super().__init__()

        logger.info("MockASR initialized")

    def transcribe(self, audio):

        data = audio.get("data")

        # MockAudioInput
        # data 是字符串
        if isinstance(data, str):

            return data

        # MicrophoneInput
        # data 是 numpy 音频数组
        if data is not None:

            return "你好，这是语音输入测试"

        return "无法识别语音"
