"""VAD-driven ALSA microphone input for RK3588."""

from __future__ import annotations

import collections
import os
import select
import subprocess
import threading
import time

import numpy as np
import webrtcvad
import voice_ui as ui

from .alsa_device_detector import detect_capture_device
from .input import AudioSource


class ALSAMicrophoneInput(AudioSource):
    """Capture one utterance and stop after trailing silence."""

    def __init__(self, device: str | None = None, sample_rate: int = 16000,
                 *, start_timeout: float | None = None,
                 max_record_time: float | None = None,
                 end_silence: float | None = None) -> None:
        configured = device or os.getenv("AI_AGENT_AUDIO_DEVICE", "auto")
        self.device = configured.strip() or "auto"
        self._auto_device = self.device.casefold() == "auto"
        self._selected_label: str | None = None
        self.sample_rate = sample_rate
        self.start_timeout = start_timeout if start_timeout is not None else float(
            os.getenv("AI_AGENT_VAD_START_TIMEOUT", "3")
        )
        self.max_record_time = max_record_time if max_record_time is not None else float(
            os.getenv("AI_AGENT_VAD_MAX_RECORD_SECONDS", "10")
        )
        self.end_silence = end_silence if end_silence is not None else float(
            os.getenv("AI_AGENT_VAD_END_SILENCE", "0.7")
        )
        self.frame_ms = int(os.getenv("AI_AGENT_VAD_FRAME_MS", "30"))
        self.start_voice_ms = int(os.getenv("AI_AGENT_VAD_START_VOICE_MS", "180"))
        self.pre_roll_ms = int(os.getenv("AI_AGENT_VAD_PRE_ROLL_MS", "300"))
        self.vad = webrtcvad.Vad(int(os.getenv("AI_AGENT_VAD_MODE", "2")))
        self._no_pcm_count = 0
        if self.start_timeout <= 0 or self.max_record_time <= 0:
            raise ValueError("VAD timeouts must be positive")
        if self.frame_ms not in {10, 20, 30}:
            raise ValueError("AI_AGENT_VAD_FRAME_MS must be 10, 20 or 30")

    def _ensure_device(self, *, exclude: set[str] | None = None) -> None:
        if self.device.casefold() != "auto":
            return
        self.device, self._selected_label = detect_capture_device(
            sample_rate=self.sample_rate, exclude=exclude
        )
        ui.debug(f"[音频] 自动连接麦克风: {self._selected_label} ({self.device})")

    def _command(self) -> list[str]:
        return [
            "arecord", "-q", "-D", self.device, "-t", "raw", "-f", "S16_LE",
            "-r", str(self.sample_rate), "-c", "1", "-B", "100000", "-F", "30000",
        ]

    @staticmethod
    def _read_frame(process: subprocess.Popen[bytes], frame_bytes: int,
                    deadline: float,
                    stop_event: threading.Event | None = None) -> bytes | None:
        """Read one PCM frame without allowing ALSA to block the VAD clock."""
        assert process.stdout is not None
        pending = bytearray()
        fd = process.stdout.fileno()
        while len(pending) < frame_bytes:
            if stop_event is not None and stop_event.is_set():
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            readable, _, _ = select.select([fd], [], [], min(0.1, remaining))
            if not readable:
                if process.poll() is not None:
                    break
                continue
            chunk = os.read(fd, frame_bytes - len(pending))
            if not chunk:
                break
            pending.extend(chunk)
        return bytes(pending)

    def record(self, stop_event: threading.Event | None = None,
               *, quiet: bool = False) -> dict[str, object] | None:
        self._ensure_device()
        if not quiet:
            ui.debug(
                f"[音频] 等待讲话（{self.start_timeout:g} 秒内开口，"
                f"最长录音 {self.max_record_time:g} 秒）..."
            )
        frame_bytes = self.sample_rate * self.frame_ms // 1000 * 2
        pre_roll = collections.deque(maxlen=max(1, self.pre_roll_ms // self.frame_ms))
        required_voice = max(1, self.start_voice_ms // self.frame_ms)
        required_silence = max(1, int(self.end_silence * 1000 / self.frame_ms))
        process = None
        try:
            process = subprocess.Popen(
                self._command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
            assert process.stdout is not None
            captured: list[bytes] = []
            consecutive_voice = 0
            consecutive_silence = 0
            speech_started = False
            waiting_started = time.monotonic()
            waiting_deadline = waiting_started + self.start_timeout
            speech_started_at = 0.0
            while True:
                deadline = (
                    speech_started_at + self.max_record_time
                    if speech_started else waiting_deadline
                )
                frame = self._read_frame(process, frame_bytes, deadline, stop_event)
                if frame is None:
                    if stop_event is not None and stop_event.is_set():
                        return None
                    if speech_started:
                        if not quiet:
                            ui.debug(f"[VAD] 达到 {self.max_record_time:g} 秒录音上限")
                        break
                    if not quiet:
                        ui.debug("[VAD] 等待开口超时（ALSA 未返回音频数据）")
                    self._no_pcm_count += 1
                    if self._no_pcm_count >= 2:
                        if self._auto_device:
                            self.device = "auto"
                            self._selected_label = None
                        time.sleep(float(os.getenv("AI_AGENT_AUDIO_RECOVERY_DELAY", "0.25")))
                        self._no_pcm_count = 0
                    return None
                if len(frame) != frame_bytes:
                    detail = ""
                    if process.poll() is not None and process.stderr is not None:
                        detail = process.stderr.read().decode("utf-8", "replace").strip()
                    raise RuntimeError(detail or "arecord stopped before a complete frame")
                voiced = self.vad.is_speech(frame, self.sample_rate)
                if not speech_started:
                    pre_roll.append(frame)
                    consecutive_voice = consecutive_voice + 1 if voiced else 0
                    if consecutive_voice >= required_voice:
                        speech_started = True
                        speech_started_at = time.monotonic()
                        captured.extend(pre_roll)
                        pre_roll.clear()
                        if not quiet:
                            ui.debug("[VAD] 检测到讲话")
                    elif time.monotonic() - waiting_started >= self.start_timeout:
                        if not quiet:
                            ui.debug("[VAD] 等待开口超时")
                        return None
                    continue

                captured.append(frame)
                consecutive_silence = 0 if voiced else consecutive_silence + 1
                if consecutive_silence >= required_silence:
                    if not quiet:
                        ui.debug(f"[VAD] 静音 {self.end_silence:g} 秒，录音结束")
                    break
                if time.monotonic() - speech_started_at >= self.max_record_time:
                    if not quiet:
                        ui.debug(f"[VAD] 达到 {self.max_record_time:g} 秒录音上限")
                    break

            pcm = b"".join(captured)
            self._no_pcm_count = 0
            waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            return {
                "type": "audio", "data": np.ascontiguousarray(waveform),
                "sample_rate": self.sample_rate, "channels": 1,
            }
        except FileNotFoundError as exc:
            raise RuntimeError("arecord is not installed; install alsa-utils") from exc
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
