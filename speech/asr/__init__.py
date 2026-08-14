"""ASR backends with lazy optional-dependency loading."""

from .recognizer import SpeechRecognizer

__all__ = ["MockASR", "SpeechRecognizer", "SenseVoiceASR", "RKNNSenseVoiceASR"]


def __getattr__(name: str):
    if name == "MockASR":
        from .mock_asr import MockASR

        return MockASR
    if name == "SenseVoiceASR":
        from .sensevoice_asr import SenseVoiceASR

        return SenseVoiceASR
    if name == "RKNNSenseVoiceASR":
        from .rknn_sensevoice_asr import RKNNSenseVoiceASR

        return RKNNSenseVoiceASR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
