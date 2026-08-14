"""Speech subsystem public API.

Optional platform backends are imported lazily so importing ``speech`` on an
RK3588 does not require the Windows-only sounddevice/FunASR/Edge-TTS stack.
"""

from .pipeline import SpeechPipeline

__all__ = [
    "SpeechPipeline",
    "AudioSource",
    "MockAudioInput",
    "MicrophoneInput",
]


def __getattr__(name: str):
    if name in {"AudioSource", "MockAudioInput", "MicrophoneInput"}:
        from . import audio

        return getattr(audio, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
