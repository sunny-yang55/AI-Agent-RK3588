"""Discover and validate ALSA capture devices without relying on card numbers."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


_CAPTURE_RE = re.compile(
    r"^card\s+(?P<card>\d+):\s*(?P<card_id>[^\s\[]+)\s*"
    r"\[(?P<card_name>[^]]+)\],\s*device\s+(?P<device>\d+):\s*"
    r"(?P<device_name>[^\[]*?)(?:\s*\[[^]]*\])?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ALSACaptureDevice:
    card: int
    card_id: str
    card_name: str
    device: int
    device_name: str

    @property
    def alsa_name(self) -> str:
        # ALSA card ids stay stable when numeric card ordering changes.
        return f"plughw:CARD={self.card_id},DEV={self.device}"

    @property
    def label(self) -> str:
        return f"{self.card_name} / {self.device_name}".strip(" /")


def list_capture_devices() -> list[ALSACaptureDevice]:
    """Return the devices reported by ``arecord -l``."""

    try:
        result = subprocess.run(
            ["arecord", "-l"],
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError("arecord is not installed; install alsa-utils") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"cannot enumerate ALSA capture devices: {detail}") from exc

    devices: list[ALSACaptureDevice] = []
    for raw_line in result.stdout.splitlines():
        match = _CAPTURE_RE.match(raw_line.strip())
        if not match:
            continue
        values = match.groupdict()
        devices.append(
            ALSACaptureDevice(
                card=int(values["card"]),
                card_id=values["card_id"],
                card_name=values["card_name"].strip(),
                device=int(values["device"]),
                device_name=values["device_name"].strip(),
            )
        )
    return devices


def _preference_terms() -> list[str]:
    configured = os.getenv(
        "AI_AGENT_MIC_PREFER",
        "microphone,mic,composite,headset,array,webcam,camera,usb",
    )
    return [item.strip().casefold() for item in configured.split(",") if item.strip()]


def _device_score(device: ALSACaptureDevice) -> tuple[int, int, int]:
    text = f"{device.card_id} {device.card_name} {device.device_name}".casefold()
    score = 0
    for position, term in enumerate(_preference_terms()):
        if term in text:
            score += 100 - min(position, 99)
    if "usb" in text:
        score += 20
    if any(term in text for term in ("hdmi", "loopback", "null")):
        score -= 1000
    # Preserve arecord's ordering only as the final tie breaker.
    return score, -device.card, -device.device


def probe_capture_device(
    device: str,
    *,
    sample_rate: int = 16000,
    seconds: float = 0.25,
) -> tuple[bool, str]:
    """Open a candidate and verify that it produces a valid mono PCM WAV."""

    handle = tempfile.NamedTemporaryFile(prefix="ai_agent_probe_", suffix=".wav", delete=False)
    wav_path = Path(handle.name)
    handle.close()
    try:
        timeout = max(2.0, seconds + 1.5)
        result = subprocess.run(
            [
                "arecord",
                "-q",
                "-D",
                device,
                "-f",
                "S16_LE",
                "-r",
                str(sample_rate),
                "-c",
                "1",
                "-d",
                str(max(1, int(round(seconds)))),
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        if result.returncode != 0:
            return False, (result.stderr or result.stdout or "arecord failed").strip()
        with wave.open(str(wav_path), "rb") as wav_file:
            valid = (
                wav_file.getnchannels() == 1
                and wav_file.getframerate() == sample_rate
                and wav_file.getsampwidth() == 2
                and wav_file.getnframes() > 0
            )
        return (True, "OK") if valid else (False, "invalid WAV format")
    except (OSError, subprocess.TimeoutExpired, wave.Error) as exc:
        return False, str(exc)
    finally:
        wav_path.unlink(missing_ok=True)


def detect_capture_device(
    *,
    sample_rate: int = 16000,
    exclude: set[str] | None = None,
) -> tuple[str, str]:
    """Select the highest-ranked capture device that passes a real probe."""

    blocked = exclude or set()
    candidates = sorted(list_capture_devices(), key=_device_score, reverse=True)
    candidates = [item for item in candidates if item.alsa_name not in blocked]
    if not candidates:
        raise RuntimeError("no ALSA capture device found; check the USB microphone")

    failures: list[str] = []
    for candidate in candidates:
        ok, detail = probe_capture_device(candidate.alsa_name, sample_rate=sample_rate)
        if ok:
            return candidate.alsa_name, candidate.label
        failures.append(f"{candidate.alsa_name}: {detail}")
    raise RuntimeError("no usable ALSA microphone; " + "; ".join(failures))
