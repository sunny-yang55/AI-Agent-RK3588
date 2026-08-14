"""Small dependency-free user interface for the RK3588 voice runtime."""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime


USER_MODE = os.getenv("AI_AGENT_UI_MODE", "user").strip().lower() != "debug"
LOG_PATH = Path(__file__).resolve().parent / "logs" / "voice-debug.log"


def debug(message: str) -> None:
    if not USER_MODE:
        print(message, flush=True)
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}\n")
    except OSError:
        pass


def banner() -> None:
    print("\n╭──────────────────────────────────────────╮")
    print("│  小安 · RK3588 智能语音助手              │")
    print("│  直接说话｜“小安停止”打断｜“再见”退出    │")
    print("╰──────────────────────────────────────────╯\n")


def state(message: str, symbol: str = "●") -> None:
    print(f"{symbol} {message}", flush=True)


def user(text: str) -> None:
    print(f"\n你：{text}")


def assistant_start() -> None:
    print("\n小安：", end="", flush=True)


def completed() -> None:
    print("\n✓ 回答完成\n", flush=True)
