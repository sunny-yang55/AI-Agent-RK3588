"""
Voice Activity Detection
Backend: WebRTC VAD
Version: v1.1-rk3588

Purpose:
    Replace Silero VAD (Torch) with WebRTC VAD.
    Remove Torch dependency for RK3588 deployment.
"""

import numpy as np
import webrtcvad


class VoiceActivityDetector:
    """
    WebRTC Voice Activity Detector

    mode:
        0 = Least aggressive
        1 = Low
        2 = Medium (Recommended)
        3 = Most aggressive
    """

    def __init__(self, mode: int = 2):
        self.vad = webrtcvad.Vad(mode)

    def has_speech(
        self,
        audio,
        sample_rate: int = 16000,
    ) -> bool:
        """
        Parameters
        ----------
        audio : numpy.ndarray
            float32 waveform (-1 ~ 1)

        sample_rate : int
            Only supports:
                8000
                16000
                32000
                48000
        """

        if audio is None:
            return False

        if isinstance(audio, list):
            audio = np.asarray(audio, dtype=np.float32)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # float32 -> int16 PCM
        pcm16 = (audio * 32767).astype(np.int16)

        frame_ms = 30
        frame_length = int(sample_rate * frame_ms / 1000)

        if len(pcm16) < frame_length:
            return False

        for start in range(
            0,
            len(pcm16) - frame_length + 1,
            frame_length,
        ):

            frame = pcm16[start : start + frame_length]

            if self.vad.is_speech(
                frame.tobytes(),
                sample_rate,
            ):
                return True

        return False
