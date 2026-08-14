"""RK3588 client for the first-generation Windows ASR service."""

from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import urllib.request


class RemoteASRClient:
    def __init__(self, server_url: str, timeout: float = 60.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.server_url}/health", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def recognize_wav(self, wav_path: str | Path) -> str:
        wav_path = Path(wav_path)
        request = urllib.request.Request(
            f"{self.server_url}/recognize",
            data=wav_path.read_bytes(),
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                result = json.loads(exc.read().decode("utf-8"))
                message = result.get("error", f"HTTP {exc.code}")
            except (UnicodeDecodeError, ValueError):
                message = f"HTTP {exc.code}"
            raise RuntimeError(f"ASR request failed: {message}") from exc
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("ASR server returned invalid data") from exc

        if not result.get("ok"):
            raise RuntimeError(result.get("error", "ASR request failed"))
        return str(result.get("text", "")).strip()
