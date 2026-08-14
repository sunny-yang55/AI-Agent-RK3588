"""Audio-input backends with lazy optional-dependency loading."""

from .input import AudioSource

__all__ = ["AudioSource", "ALSAMicrophoneInput", "MicrophoneInput", "MockAudioInput"]


def __getattr__(name: str):
    if name == "ALSAMicrophoneInput":
        from .alsa_microphone import ALSAMicrophoneInput

        return ALSAMicrophoneInput
    if name == "MicrophoneInput":
        from .microphone import MicrophoneInput

        return MicrophoneInput
    if name == "MockAudioInput":
        from .mock_audio import MockAudioInput

        return MockAudioInput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
