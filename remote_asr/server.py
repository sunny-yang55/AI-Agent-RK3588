"""Small HTTP bridge from RK3588 WAV input to the Windows SenseVoice backend."""

from __future__ import annotations

import io
import json
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np


MAX_WAV_BYTES = 20 * 1024 * 1024


def decode_wav(payload: bytes) -> dict[str, Any]:
    """Decode a 16-bit mono WAV into the audio dictionary used by SenseVoiceASR."""
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"invalid WAV file: {exc}") from exc

    if channels != 1:
        raise ValueError(f"WAV must be mono; received {channels} channels")
    if sample_rate != 16000:
        raise ValueError(f"WAV must be 16 kHz; received {sample_rate} Hz")
    if sample_width != 2:
        raise ValueError(f"WAV must use 16-bit PCM; received {sample_width * 8}-bit")

    waveform = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return {
        "type": "audio",
        "data": waveform,
        "sample_rate": sample_rate,
        "channels": channels,
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def create_server(host: str, port: int, recognizer: Any) -> ThreadingHTTPServer:
    """Create a server whose recognizer provides ``transcribe(audio)``."""

    class ASRHandler(BaseHTTPRequestHandler):
        def _respond(self, status: int, payload: dict[str, Any]) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._respond(404, {"ok": False, "error": "not found"})
                return
            self._respond(200, {"ok": True, "service": "sensevoice-asr"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/recognize":
                self._respond(404, {"ok": False, "error": "not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if content_length <= 0 or content_length > MAX_WAV_BYTES:
                self._respond(400, {"ok": False, "error": "invalid WAV size"})
                return

            try:
                audio = decode_wav(self.rfile.read(content_length))
                text = str(recognizer.transcribe(audio)).strip()
                self._respond(200, {"ok": True, "text": text})
            except ValueError as exc:
                self._respond(400, {"ok": False, "error": str(exc)})
            except Exception as exc:  # keep the service alive after one failed request
                self._respond(500, {"ok": False, "error": f"ASR failed: {exc}"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), ASRHandler)
