"""Small ALSA playback helper shared by offline RK3588 TTS backends."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def playback_device() -> str:
    return os.getenv("AI_AGENT_PLAYBACK_DEVICE", "default")


def play_wav(path: str | Path) -> None:
    player = shutil.which("aplay")
    if not player:
        raise RuntimeError("aplay is not installed; install alsa-utils")
    device = playback_device()
    try:
        subprocess.run(
            [player, "-q", "-D", device, str(path)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"aplay failed for device {device!r} with exit code {exc.returncode}"
        ) from exc
