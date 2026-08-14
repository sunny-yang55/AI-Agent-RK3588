"""
Speech Interface Layer

定义:
1. Audio Input
2. ASR
3. TTS

真实实现:
- Windows
- RK3588
- Cloud

均通过这些接口接入。
"""

from abc import ABC, abstractmethod


class AudioInput(ABC):
    """
    音频输入接口
    """

    @abstractmethod
    def record(self):
        """
        获取音频数据
        """
        pass


class SpeechRecognizer(ABC):
    """
    ASR接口

    Audio -> Text
    """

    @abstractmethod
    def transcribe(self, audio):
        pass


class SpeechSynthesizer(ABC):
    """
    TTS接口

    Text -> Voice
    """

    @abstractmethod
    def speak(self, text):
        pass
