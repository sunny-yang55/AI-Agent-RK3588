"""
Audio Source Interface

所有音频输入后端统一接口

Backend:

Mock:
    MockAudioInput

Windows:
    MicrophoneInput

RK3588:
    ALSA / USB Mic
"""

from abc import ABC, abstractmethod


class AudioSource(ABC):

    @abstractmethod
    def record(self):
        """
        返回音频数据
        """
        pass
