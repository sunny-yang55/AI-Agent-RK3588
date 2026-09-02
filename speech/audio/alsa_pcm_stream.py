"""Low-latency raw PCM streaming from an ALSA capture device."""

from __future__ import annotations

import os
import select
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .alsa_device_detector import detect_capture_device


@dataclass(frozen=True)
class PCMFrame:
    pcm: bytes
    waveform: np.ndarray
    captured_at: float


class ALSAStreamingMicrophone:
    """Yield fixed-duration mono S16_LE frames without blocking forever."""

    def __init__(self, device: str | None = None, *, sample_rate: int = 16000,
                 frame_ms: int = 30) -> None:
        self.device = (device or os.getenv("AI_AGENT_AUDIO_DEVICE", "auto")).strip()
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        if frame_ms not in {10, 20, 30}:
            raise ValueError("frame_ms must be 10, 20 or 30")
        self.frame_bytes = sample_rate * frame_ms // 1000 * 2
        self.label: str | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> "ALSAStreamingMicrophone":
        if self._process is not None:
            return self
        if not self.device or self.device.casefold() == "auto":
            self.device, self.label = detect_capture_device(sample_rate=self.sample_rate)
        command = [
            "arecord", "-q", "-D", self.device, "-t", "raw", "-f", "S16_LE",
            "-r", str(self.sample_rate), "-c", "1", "-B", "100000", "-F", "30000",
        ]
        try:
            self._process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
        except FileNotFoundError as exc:
            raise RuntimeError("arecord is not installed; install alsa-utils") from exc
        return self

    def frames(self, stop_event: threading.Event | None = None) -> Iterator[PCMFrame]:
        self.start()
        assert self._process is not None and self._process.stdout is not None
        process = self._process
        fd = process.stdout.fileno()
        pending = bytearray()
        while process.poll() is None:
            if stop_event is not None and stop_event.is_set():
                break
            readable, _, _ = select.select([fd], [], [], 0.1)
            if not readable:
                continue
            chunk = os.read(fd, self.frame_bytes - len(pending))
            if not chunk:
                break
            pending.extend(chunk)
            if len(pending) != self.frame_bytes:
                continue
            pcm = bytes(pending)
            pending.clear()
            waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            yield PCMFrame(pcm, np.ascontiguousarray(waveform), time.monotonic())
        if process.poll() not in (None, 0):
            detail = ""
            if process.stderr is not None:
                detail = process.stderr.read().decode("utf-8", "replace").strip()
            raise RuntimeError(detail or "arecord exited unexpectedly")

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def __enter__(self) -> "ALSAStreamingMicrophone":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()
