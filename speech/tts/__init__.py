"""TTS backends with lazy optional-dependency loading."""

from .synthesizer import SpeechSynthesizer

__all__ = ["SpeechSynthesizer", "MockTTS", "EdgeTTS", "EspeakTTS", "PiperTTS"]


def __getattr__(name: str):
    if name == "MockTTS":
        from .mock_tts import MockTTS

        return MockTTS
    if name == "EdgeTTS":
        from .edge_tts_backend import EdgeTTS

        return EdgeTTS
    if name == "EspeakTTS":
        from .espeak_tts_backend import EspeakTTS

        return EspeakTTS
    if name == "PiperTTS":
        from .piper_tts_backend import PiperTTS

        return PiperTTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
