from .input import AudioSource


class MockAudioInput(AudioSource):

    def record(self):

        return {
            "type": "audio",
            "data": "你好，这是语音输入测试",
            "sample_rate": 16000,
            "channels": 1,
        }
