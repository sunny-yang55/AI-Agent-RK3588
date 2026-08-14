"""Long-utterance adapter for the static time100 SenseVoice RKNN model."""

from __future__ import annotations

import os

import numpy as np
import voice_ui as ui

from .rknn_sensevoice_asr import RKNNSenseVoiceASR


def _merge_text(left: str, right: str, max_overlap: int = 12) -> str:
    """Remove a small textual overlap produced by adjacent audio windows."""
    if not left:
        return right
    if not right:
        return left
    upper = min(max_overlap, len(left), len(right))
    for size in range(upper, 0, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return left + right


class ChunkedRKNNSenseVoiceASR(RKNNSenseVoiceASR):
    """Recognize up to ten seconds using safe static-model windows."""

    def transcribe(self, audio: dict[str, object]) -> str:
        waveform = np.asarray(audio.get("data", []), dtype=np.float32).reshape(-1)
        sample_rate = int(audio.get("sample_rate", 16000))
        window_seconds = float(os.getenv("AI_AGENT_ASR_CHUNK_SECONDS", "4.8"))
        overlap_seconds = float(os.getenv("AI_AGENT_ASR_CHUNK_OVERLAP", "0.25"))
        window = int(window_seconds * sample_rate)
        overlap = int(overlap_seconds * sample_rate)
        if waveform.size <= window:
            return super().transcribe(audio)
        if not 0 <= overlap < window:
            raise ValueError("ASR chunk overlap must be smaller than its window")

        ui.debug(f"[SenseVoice RKNN] 长语音分段识别: {waveform.size / sample_rate:.2f}s")
        merged = ""
        start = 0
        part = 0
        while start < waveform.size:
            stop = min(start + window, waveform.size)
            chunk = waveform[start:stop]
            if chunk.size < int(0.2 * sample_rate):
                break
            part += 1
            text = super().transcribe({
                "type": "audio", "data": np.ascontiguousarray(chunk),
                "sample_rate": sample_rate, "channels": 1,
            })
            ui.debug(f"[SenseVoice RKNN] 分段 {part}: {text}")
            merged = _merge_text(merged, text)
            if stop >= waveform.size:
                break
            start = stop - overlap
        return merged
