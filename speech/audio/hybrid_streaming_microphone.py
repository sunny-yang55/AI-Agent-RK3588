"""Streaming microphone for Zipformer partials and SenseVoice final pass."""

from __future__ import annotations

import collections
import os
import time
from typing import Any

import numpy as np
import webrtcvad

import voice_ui as ui
from .alsa_pcm_stream import ALSAStreamingMicrophone


class HybridStreamingMicrophoneInput:
    """Capture one utterance while showing non-authoritative partial text."""

    def __init__(self, partial_asr: Any, device: str | None = None) -> None:
        self.partial_asr = partial_asr
        self.device = device or os.getenv("AI_AGENT_AUDIO_DEVICE", "auto")
        self.sample_rate = 16000
        self.frame_ms = 30
        self.start_timeout = float(os.getenv("AI_AGENT_VAD_START_TIMEOUT", "3"))
        self.max_record_time = float(os.getenv("AI_AGENT_VAD_MAX_RECORD_SECONDS", "10"))
        self.end_silence = float(os.getenv("AI_AGENT_HYBRID_END_SILENCE", "0.70"))
        self.start_voice_ms = int(os.getenv("AI_AGENT_VAD_START_VOICE_MS", "210"))
        self.pre_roll_ms = int(os.getenv("AI_AGENT_VAD_PRE_ROLL_MS", "360"))
        self.vad = webrtcvad.Vad(int(os.getenv("AI_AGENT_VAD_MODE", "2")))

    def record(self, stop_event=None, *, quiet: bool = False):
        required_voice = max(1, self.start_voice_ms // self.frame_ms)
        required_silence = max(1, round(self.end_silence * 1000 / self.frame_ms))
        pre_roll = collections.deque(maxlen=max(1, self.pre_roll_ms // self.frame_ms))
        captured: list[np.ndarray] = []
        voice_run = silence_run = 0
        speech_started = False
        waiting_started = time.monotonic()
        speech_started_at = 0.0
        last_partial = ""

        microphone = ALSAStreamingMicrophone(
            self.device, sample_rate=self.sample_rate, frame_ms=self.frame_ms
        )
        try:
            with microphone:
                self.device = microphone.device
                for frame in microphone.frames(stop_event):
                    now = time.monotonic()
                    voiced = self.vad.is_speech(frame.pcm, self.sample_rate)
                    if not speech_started:
                        pre_roll.append(frame.waveform)
                        voice_run = voice_run + 1 if voiced else 0
                        if voice_run >= required_voice:
                            speech_started = True
                            speech_started_at = now
                            self.partial_asr.reset()
                            captured.extend(pre_roll)
                            if not quiet:
                                ui.recognition_start()
                            for waveform in pre_roll:
                                partial = self.partial_asr.accept_waveform(waveform)
                                if partial and partial != last_partial:
                                    last_partial = partial
                                    if not quiet:
                                        ui.recognition_partial(partial)
                            pre_roll.clear()
                        elif now - waiting_started >= self.start_timeout:
                            return None
                        continue

                    captured.append(frame.waveform)
                    partial = self.partial_asr.accept_waveform(frame.waveform)
                    if partial and partial != last_partial:
                        last_partial = partial
                        if not quiet:
                            ui.recognition_partial(partial)
                    silence_run = 0 if voiced else silence_run + 1
                    if silence_run >= required_silence:
                        break
                    if now - speech_started_at >= self.max_record_time:
                        break
        finally:
            microphone.stop()

        if not captured:
            return None
        # Partial finalization only improves the UI. SenseVoice below remains
        # the authoritative transcript sent to command handling and the LLM.
        partial = self.partial_asr.finalize()
        if partial and partial != last_partial and not quiet:
            ui.recognition_partial(partial)
        waveform = np.ascontiguousarray(np.concatenate(captured), dtype=np.float32)
        return {
            "type": "audio", "data": waveform,
            "sample_rate": self.sample_rate, "channels": 1,
            "partial_text": partial or last_partial,
        }
